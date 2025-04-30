# models/diffusion_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, Union

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Modules (Similar to prompt_generator.py) ---
# Consider refactoring these into a common module later

class SinusoidalPosEmb(nn.Module):
    """ Sinusoidal Position Embedding for Time (expects log SNR lambda_t) """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time_values): # time_values is lambda_t, shape (batch,)
        device = time_values.device
        half_dim = self.dim // 2
        # Scaling factor for embeddings - adjust if needed
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Use lambda_t directly as input for embeddings
        embeddings = time_values[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # Zero padding for odd dimensions
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

class AdaptiveRMSNorm(nn.Module):
    """ Adaptive RMS Normalization with time conditioning """
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        # Simple MLP for projecting time embedding to scale/shift
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * dim) # Output: scale and shift
        )
        self.eps = 1e-6

    def forward(self, x, time_emb):
        scale_shift = self.time_mlp(time_emb)
        # Ensure scale_shift has the same number of dimensions as x for broadcasting
        while len(scale_shift.shape) < len(x.shape):
            scale_shift = scale_shift.unsqueeze(1) # Add seq_len dim: (batch, 1, 2*dim)

        scale, shift = scale_shift.chunk(2, dim=-1)
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        normalized_x = x * torch.rsqrt(variance + self.eps)
        output = normalized_x * (1 + scale) + shift # Apply scale around 1
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
        if self.head_dim * n_heads != dim:
             raise ValueError("dim must be divisible by n_heads")

        self.norm1 = AdaptiveRMSNorm(dim, time_emb_dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = AdaptiveRMSNorm(dim, time_emb_dim)

        hidden_dim = dim * 4
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        self.ff_dropout = nn.Dropout(dropout) # Dropout after residual connection

    def forward(self, x, time_emb, attn_mask=None):
        # x shape: (batch, seq_len, dim)
        # time_emb shape: (batch, time_emb_dim)

        # Self-Attention part
        residual = x
        normed_x = self.norm1(x, time_emb)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x, attn_mask=attn_mask, need_weights=False)
        # Dropout on attention output before residual? Typically included in MHA layer.
        x = residual + attn_output

        # FeedForward part
        residual = x
        normed_x = self.norm2(x, time_emb)
        ff_output = self.ff(normed_x)
        x = residual + self.ff_dropout(ff_output) # Apply dropout after FF layers

        return x

# --- Configuration Class ---

class DiffusionTransformerConfig(PretrainedConfig):
    model_type = "dglm_diffusion_transformer"

    def __init__(
        self,
        sentence_emb_dim: int = 1024, # e.g., Sentence-T5-XL
        transformer_dim: int = 768,
        input_tokens: int = 64, # Number of tokens after projection/reshape
        input_proj_intermediate_dim: int = 96, # Dim per token after first projection
        output_proj_intermediate_dim: int = 96, # Dim per token before final projection
        n_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1, # Default dropout, check Table 8 if specified
        time_emb_multiplier: int = 4, # Multiplier for time embedding dimension
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sentence_emb_dim = sentence_emb_dim
        self.transformer_dim = transformer_dim
        self.input_tokens = input_tokens
        self.input_proj_intermediate_dim = input_proj_intermediate_dim
        self.output_proj_intermediate_dim = output_proj_intermediate_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.time_emb_multiplier = time_emb_multiplier

        # Calculated dimensions based on Table 8 interpretation
        self.input_proj_output_dim = input_tokens * input_proj_intermediate_dim # 64 * 96 = 6144
        self.concat_feature_dim = input_proj_intermediate_dim * 2 # 96 * 2 = 192
        self.output_proj_input_dim = input_tokens * output_proj_intermediate_dim # 64 * 96 = 6144


# --- Main Diffusion Transformer Module ---

class DiffusionTransformer(PreTrainedModel):
    """
    Implements the Transformer-based diffusion network from DGLM paper (Fig 4, Table 8).
    Predicts velocity 'v' for v-parameterization objective.
    """
    config_class = DiffusionTransformerConfig

    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__(config)
        self.config = config

        # --- Input Processing Layers ---
        # 1a. Project noisy latent z_t
        self.latent_proj = nn.Linear(config.sentence_emb_dim, config.input_proj_output_dim)
        # 1b. Project prefix embedding x_pref
        self.prefix_proj = nn.Linear(config.sentence_emb_dim, config.input_proj_output_dim)
        # 1c. Project concatenated features to transformer dim
        self.concat_proj = nn.Linear(config.concat_feature_dim, config.transformer_dim)

        # --- Time Embedding ---
        time_input_dim = config.transformer_dim # Time emb input dim matches model dim
        self.time_embedder = SinusoidalPosEmb(time_input_dim)
        time_emb_dim = time_input_dim * config.time_emb_multiplier
        self.time_mlp = nn.Sequential(
            nn.Linear(time_input_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # --- Transformer Body ---
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.transformer_dim,
                n_heads=config.n_heads,
                time_emb_dim=time_emb_dim,
                dropout=config.dropout
            ) for _ in range(config.n_layers)
        ])

        # --- Output Processing Layers ---
        # 2a. Project transformer output back to intermediate dim per token
        self.output_proj1 = nn.Linear(config.transformer_dim, config.output_proj_intermediate_dim)
        # 2b. Final projection from concatenated features to sentence embedding dim (velocity prediction)
        self.output_proj2 = nn.Linear(config.output_proj_input_dim, config.sentence_emb_dim)

        # --- Null Embedding for CFG ---
        self.null_embedding = nn.Parameter(torch.randn(1, config.sentence_emb_dim))

        # --- Final Layer Norm (good practice) ---
        self.final_norm = nn.LayerNorm(config.transformer_dim)

        logging.info(f"Initialized DiffusionTransformer: layers={config.n_layers}, heads={config.n_heads}, model_dim={config.transformer_dim}")

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TransformerBlock):
            module.gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.apply(lambda module: self._set_gradient_checkpointing(module, value=True))

    def forward(
        self,
        noisy_latent: torch.Tensor,        # Shape: (batch, sentence_emb_dim) - z_t
        prefix_embedding: torch.Tensor,    # Shape: (batch, sentence_emb_dim) - x_pref or null_embedding
        time_values: torch.Tensor,         # Shape: (batch,) - lambda_t (log SNR)
        return_dict: Optional[bool] = None,
        **kwargs # Allow passing other args if needed, though not used currently
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass of the diffusion model.

        Args:
            noisy_latent (torch.Tensor): The noisy continuation embedding z_t.
            prefix_embedding (torch.Tensor): The prefix embedding x_pref (potentially masked with null).
            time_values (torch.Tensor): The log SNR values lambda_t for time conditioning.
            return_dict (Optional[bool]): Whether to return a dictionary (not implemented yet) or just the tensor.

        Returns:
            torch.Tensor: The predicted velocity v_theta. Shape: (batch, sentence_emb_dim).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = noisy_latent.shape[0]
        device = noisy_latent.device

        # --- 1. Input Processing ---
        # 1a/b. Project latent and prefix, reshape
        latent_proj = self.latent_proj(noisy_latent).view(
            batch_size, self.config.input_tokens, self.config.input_proj_intermediate_dim
        ) # B, 64, 96
        prefix_proj = self.prefix_proj(prefix_embedding).view(
            batch_size, self.config.input_tokens, self.config.input_proj_intermediate_dim
        ) # B, 64, 96

        # 1c. Concatenate along feature dim
        concat_features = torch.cat([latent_proj, prefix_proj], dim=-1) # B, 64, 192

        # 1d. Project to transformer dimension
        transformer_input = self.concat_proj(concat_features) # B, 64, 768

        # --- 2. Time Embedding ---
        # Use lambda_t for time embedding
        time_emb_sin = self.time_embedder(time_values) # B, 768
        time_emb = self.time_mlp(time_emb_sin)         # B, time_emb_dim (e.g., 768*4)

        # --- 3. Transformer Body ---
        x = transformer_input
        for block in self.transformer_blocks:
            # Pass time embedding for adaptive norm conditioning
            x = block(x, time_emb) # B, 64, 768

        # --- 4. Output Processing ---
        # Apply final norm before output projection
        x = self.final_norm(x)

        # 4a. Project back to intermediate dim
        output_proj1 = self.output_proj1(x) # B, 64, 96

        # 4b. Reshape/Concatenate
        output_flat = output_proj1.view(batch_size, -1) # B, 6144

        # 4c. Final projection to predict velocity
        predicted_velocity = self.output_proj2(output_flat) # B, sentence_emb_dim

        if not return_dict:
            return predicted_velocity
        else:
            # Can return a more structured output if needed later
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=predicted_velocity)


# --- Example Usage (for testing the module) ---
if __name__ == '__main__':
    # Example configuration
    config = DiffusionTransformerConfig(
        sentence_emb_dim=1024,
        transformer_dim=768,
        n_layers=12,
        n_heads=12,
        # Other params use defaults
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Instantiate the model
    model = DiffusionTransformer(config).to(device)
    model.eval() # Set to eval mode for testing

    # Create dummy input
    B = 4 # Batch size
    dummy_latent = torch.randn(B, config.sentence_emb_dim).to(device)
    dummy_prefix = torch.randn(B, config.sentence_emb_dim).to(device)
    # Simulate lambda_t values (log SNR can range widely)
    dummy_lambda_t = torch.linspace(-10, 10, B).to(device) # Example range

    # Forward pass
    with torch.no_grad():
         predicted_v = model(dummy_latent, dummy_prefix, dummy_lambda_t)

    logging.info(f"Input noisy_latent shape: {dummy_latent.shape}")
    logging.info(f"Input prefix_embedding shape: {dummy_prefix.shape}")
    logging.info(f"Input time_values (lambda_t) shape: {dummy_lambda_t.shape}")
    logging.info(f"Output predicted_velocity shape: {predicted_v.shape}")

    # Check output shape
    expected_shape = (B, config.sentence_emb_dim)
    assert predicted_v.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {predicted_v.shape}"
    logging.info("DiffusionTransformer test completed successfully.")

    # Check number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"DiffusionTransformer total trainable parameters: {num_params:,}")
    logging.info(f"Null embedding shape: {model.null_embedding.shape}")


