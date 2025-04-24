# models/prompt_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Modules ---

class SinusoidalPosEmb(nn.Module):
    """ Sinusoidal Position Embedding for Time """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): # time is expected to be alpha_t, shape (batch,)
        # Ensure time is on the correct device
        device = time.device
        # Clamp time to avoid issues near 0 or 1, although alpha_t should be in [0, 1]
        time = torch.clamp(time, 1e-5, 1.0 - 1e-5)
        # Use log(alpha_t) as input for potentially better numerical stability/spread
        # Or simply use alpha_t directly. Let's use alpha_t for now.
        # The paper doesn't specify the exact transformation for alpha_t -> time embedding input
        # Let's use a simple linear scale for demonstration. A log scale might be better.
        # time_input = -torch.log(time) # Alternative: use log SNR like in diffusion
        time_input = time * 1000 # Simple scaling

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time_input[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # If dim is odd, append a zero
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)

        return embeddings

class AdaptiveRMSNorm(nn.Module):
    """ Adaptive RMS Normalization with time conditioning """
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * dim) # Output scale and shift
        )
        self.eps = 1e-6

    def forward(self, x, time_emb):
        # Project time embedding to get scale and shift parameters
        scale_shift = self.time_mlp(time_emb)
        # Ensure scale_shift has the same number of dimensions as x for broadcasting
        # x shape: (batch, seq_len, dim) or (batch, dim)
        # time_emb shape: (batch, time_emb_dim) -> scale_shift shape: (batch, 2*dim)
        while len(scale_shift.shape) < len(x.shape):
            scale_shift = scale_shift.unsqueeze(1) # Add seq_len dim if needed: (batch, 1, 2*dim)

        scale, shift = scale_shift.chunk(2, dim=-1)

        # RMSNorm calculation
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        normalized_x = x * torch.rsqrt(variance + self.eps)

        # Apply scale and shift
        output = normalized_x * (1 + scale) + shift # Scale is applied multiplicatively around 1
        return output

class SwiGLU(nn.Module):
    """ SwiGLU Activation Function """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class TransformerBlock(nn.Module):
    """ A single Transformer Block with Adaptive RMSNorm and SwiGLU """
    def __init__(self, dim, n_heads, time_emb_dim, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"

        # Adaptive Norms
        self.norm1 = AdaptiveRMSNorm(dim, time_emb_dim)
        self.norm2 = AdaptiveRMSNorm(dim, time_emb_dim)

        # Attention Layer (using standard MultiheadAttention for simplicity)
        # Note: Query-Key RMSNorm mentioned in paper not implemented here, using standard MHA
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        # FeedForward Layer
        hidden_dim = dim * 4 # Standard expansion factor
        # SwiGLU needs output dim of 2 * hidden_dim for the gating mechanism
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        self.ff_dropout = nn.Dropout(dropout)


    def forward(self, x, time_emb, attn_mask=None):
        # x shape: (batch, seq_len, dim)
        # time_emb shape: (batch, time_emb_dim)

        # Self-Attention part
        residual = x
        normed_x = self.norm1(x, time_emb)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x, attn_mask=attn_mask, need_weights=False)
        x = residual + attn_output # TODO: Add dropout to attn_output? MHA has dropout param

        # FeedForward part
        residual = x
        normed_x = self.norm2(x, time_emb)
        ff_output = self.ff(normed_x)
        x = residual + self.ff_dropout(ff_output)

        return x

# --- Main Prompt Generator Module ---

class PromptGenerator(nn.Module):
    """
    Generates a sequence of k soft prompt embeddings from a sentence embedding
    and a noise level (alpha_t).
    """
    def __init__(
        self,
        sentence_emb_dim: int = 1024, # Sentence-T5-XL embedding dim
        decoder_emb_dim: int = 1280, # GPT-2 Large embedding dim
        k_soft_tokens: int = 8,
        n_layers: int = 6,
        n_heads: int = 12, # Should match decoder_emb_dim / head_dim
        dropout: float = 0.1
    ):
        super().__init__()
        self.k_soft_tokens = k_soft_tokens
        self.decoder_emb_dim = decoder_emb_dim
        self.sentence_emb_dim = sentence_emb_dim

        # 1. Initial Projection & Reshape
        self.input_proj = nn.Linear(sentence_emb_dim, k_soft_tokens * decoder_emb_dim)

        # 2. Time Embedding MLP for conditioning
        time_emb_dim = decoder_emb_dim * 4 # Standard practice: time dim = 4 * model dim
        self.time_embedder = SinusoidalPosEmb(decoder_emb_dim) # Output dim matches model dim
        self.time_mlp = nn.Sequential(
            nn.Linear(decoder_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 3. Positional Encoding for the k tokens
        self.positional_encoding = nn.Parameter(torch.randn(1, k_soft_tokens, decoder_emb_dim))

        # 4. Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_emb_dim,
                n_heads=n_heads,
                time_emb_dim=time_emb_dim,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

        # 5. Final Layer Norm (optional, but good practice)
        self.final_norm = nn.LayerNorm(decoder_emb_dim)

        logging.info(f"Initialized PromptGenerator: k={k_soft_tokens}, layers={n_layers}, heads={n_heads}, model_dim={decoder_emb_dim}")

    def forward(self, sentence_embedding, alpha_t):
        """
        Args:
            sentence_embedding (torch.Tensor): Shape (batch_size, sentence_emb_dim)
            alpha_t (torch.Tensor): Shape (batch_size,) - noise level schedule value

        Returns:
            torch.Tensor: Soft prompt embeddings, shape (batch_size, k_soft_tokens, decoder_emb_dim)
        """
        batch_size = sentence_embedding.shape[0]
        device = sentence_embedding.device

        # 1. Project and Reshape Sentence Embedding
        # (batch, sentence_emb_dim) -> (batch, k * decoder_emb_dim) -> (batch, k, decoder_emb_dim)
        projected_input = self.input_proj(sentence_embedding)
        x = projected_input.view(batch_size, self.k_soft_tokens, self.decoder_emb_dim)

        # 2. Calculate Time Embedding
        time_emb_sin = self.time_embedder(alpha_t) # (batch, decoder_emb_dim)
        time_emb = self.time_mlp(time_emb_sin)     # (batch, time_emb_dim)

        # 3. Add Positional Encoding
        x = x + self.positional_encoding[:, :self.k_soft_tokens, :]

        # 4. Pass through Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x, time_emb) # Pass time embedding to each block for adaptive norm

        # 5. Final Normalization
        x = self.final_norm(x)

        return x

# --- Example Usage (for testing the module) ---
if __name__ == '__main__':
    # Example parameters (adjust as needed)
    B = 4 # Batch size
    SENT_DIM = 1024
    DEC_DIM = 1280
    K = 8
    LAYERS = 6
    HEADS = 12 # 1280 / 12 is not integer, GPT2-Large uses 1280 dim / 20 heads = 64 head_dim. Let's adjust.
    HEADS = 20 # Make dim divisible by heads (1280 / 20 = 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create dummy input
    dummy_sent_emb = torch.randn(B, SENT_DIM).to(device)
    dummy_alpha_t = torch.rand(B).to(device) # Random alpha_t values between 0 and 1

    # Instantiate the model
    prompt_gen = PromptGenerator(
        sentence_emb_dim=SENT_DIM,
        decoder_emb_dim=DEC_DIM,
        k_soft_tokens=K,
        n_layers=LAYERS,
        n_heads=HEADS
    ).to(device)

    # Forward pass
    prompt_gen.train() # Set to train mode for testing dropout etc.
    with torch.no_grad(): # Use no_grad for simple shape checking
         soft_prompt = prompt_gen(dummy_sent_emb, dummy_alpha_t)

    logging.info(f"Input sentence embedding shape: {dummy_sent_emb.shape}")
    logging.info(f"Input alpha_t shape: {dummy_alpha_t.shape}")
    logging.info(f"Output soft prompt shape: {soft_prompt.shape}")

    # Check output shape
    expected_shape = (B, K, DEC_DIM)
    assert soft_prompt.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {soft_prompt.shape}"
    logging.info("PromptGenerator test completed successfully.")

    # Check number of parameters
    num_params = sum(p.numel() for p in prompt_gen.parameters() if p.requires_grad)
    logging.info(f"PromptGenerator total trainable parameters: {num_params:,}")

