# data/diffusion_collator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel
from typing import List, Dict, Any, Optional
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def passes_filter(text: str, tokenizer: PreTrainedTokenizer, threshold: float) -> bool:
    """
    Checks if the text passes the compressibility filter based on token/char ratio.
    """
    if not text or not isinstance(text, str): return False
    text = text.strip()
    if not text: return False
    try:
        num_chars = len(text)
        num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if num_tokens == 0: return False
        passes = num_tokens <= threshold * num_chars
        if num_chars < 20 and num_tokens < 5: passes = False
        return passes
    except Exception as e:
        logging.warning(f"Error processing text during filtering: {e}. Text snippet: {text[:100]}...")
        return False

def get_cosine_schedule_values(t: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculates alpha_t, sigma_t, and lambda_t based on a cosine schedule.
    alpha_t = cos^2 ( t * pi / 2 ) where t is normalized time in [0, 1].
    lambda_t = log(alpha_t^2 / sigma_t^2)

    Args:
        t (torch.Tensor): Tensor of normalized time values [0, 1]. Shape (batch_size,).

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing 'alpha_t', 'sigma_t', 'lambda_t'.
    """
    # Ensure t is on the correct device if needed, but calculations are simple
    # device = t.device
    # Calculate alpha_t based on the cosine schedule
    alpha_t = torch.cos(t * math.pi / 2.0) ** 2
    # Clamp to avoid numerical issues near 0 and 1
    alpha_t = torch.clamp(alpha_t, 1e-8, 1.0 - 1e-8) # Use a slightly larger epsilon

    sigma_t_sq = 1.0 - alpha_t
    sigma_t = torch.sqrt(sigma_t_sq)

    # Calculate log SNR (lambda_t)
    # Add small epsilon to avoid log(0) or division by zero
    lambda_t = torch.log(alpha_t / sigma_t_sq + 1e-8)

    return {'alpha_t': alpha_t, 'sigma_t': sigma_t, 'lambda_t': lambda_t}


# --- Data Collator Class ---

@dataclass
class DataCollatorForDiffusionTraining:
    """
    Data collator for training the diffusion model.
    Handles filtering, splitting, embedding, noising, target calculation, and CFG masking.
    """
    # Tokenizers
    decoder_tokenizer: PreTrainedTokenizer # Used for splitting text consistently
    filter_tokenizer: PreTrainedTokenizer # Usually GPT-2
    sentence_tokenizer: PreTrainedTokenizer # For Sentence-T5

    # Models / Embeddings
    sentence_encoder: PreTrainedModel # Frozen Sentence-T5
    null_embedding: nn.Parameter # The learnable null embedding from the diffusion model

    # Configuration
    prefix_len: int = 32
    filter_threshold: float = 0.3
    cfg_mask_probability: float = 0.1 # Probability of masking prefix for CFG

    def __call__(self, examples: List[str]) -> Dict[str, torch.Tensor]:
        """
        Processes a batch of raw text strings for diffusion model training.

        Args:
            examples: A list of raw text strings from the dataset.

        Returns:
            A dictionary containing the prepared batch (CPU tensors):
            - 'noisy_latent': Noisy continuation embedding (z_t).
            - 'prefix_embedding': Prefix embedding (x_pref) or null embedding.
            - 'time_values': Noise level lambda_t.
            - 'target_velocity': Target velocity v = alpha_t * x_cont + sigma_t * epsilon.
        """
        # Determine devices
        # Assume sentence_encoder and null_embedding are on the same target device
        model_device = self.null_embedding.device
        cpu_device = torch.device('cpu')

        batch_size = len(examples)
        processed_batch = []

        # --- 1. Filtering & Splitting ---
        texts_to_process = list(examples)
        while len(processed_batch) < batch_size and texts_to_process:
            text = texts_to_process.pop(0).strip()
            if not passes_filter(text, self.filter_tokenizer, self.filter_threshold): continue
            try:
                tokenized_full = self.decoder_tokenizer.encode(text, add_special_tokens=False)
                if not isinstance(tokenized_full, (list, tuple)): continue
            except Exception: continue
            if len(tokenized_full) <= self.prefix_len: continue

            prefix_tokens = tokenized_full[:self.prefix_len]
            continuation_tokens = tokenized_full[self.prefix_len:] # Take rest as continuation
            if not continuation_tokens: continue

            try:
                # Decode prefix and continuation separately
                prefix_text = self.decoder_tokenizer.decode(prefix_tokens, skip_special_tokens=True)
                continuation_text = self.decoder_tokenizer.decode(continuation_tokens, skip_special_tokens=True)
            except Exception: continue

            if not prefix_text.strip() or not continuation_text.strip(): continue

            processed_batch.append({
                "prefix_text": prefix_text,
                "continuation_text": continuation_text,
            })
        # --- End Filtering & Splitting Loop ---

        if not processed_batch:
             # Raise error if no valid examples found after filtering/splitting
             raise ValueError(f"Data collator could not produce a valid batch from input. "
                              f"All {batch_size} examples were filtered out. Check data or criteria.")

        current_batch_size = len(processed_batch)

        # --- 2. Sentence Embedding (Prefix & Continuation) ---
        prefix_texts = [item["prefix_text"] for item in processed_batch]
        continuation_texts = [item["continuation_text"] for item in processed_batch]

        all_texts = prefix_texts + continuation_texts
        try:
            # Tokenize all texts together on CPU
            sent_inputs = self.sentence_tokenizer(
                all_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            # Move to GPU for encoding
            sent_inputs = sent_inputs.to(model_device)

            with torch.no_grad(): # Ensure sentence encoder is frozen
                encoder_outputs = self.sentence_encoder.encoder(
                    input_ids=sent_inputs.input_ids,
                    attention_mask=sent_inputs.attention_mask,
                    return_dict=True
                )
                # Pool and move back to CPU
                all_embeddings = encoder_outputs.last_hidden_state.mean(dim=1).to(cpu_device)

            # Split embeddings back into prefix and continuation
            x_pref = all_embeddings[:current_batch_size]
            x_cont = all_embeddings[current_batch_size:]

        except Exception as e:
             logging.error(f"Error during sentence embedding: {e}", exc_info=True)
             raise ValueError("Failed to generate sentence embeddings.") from e

        # --- 3. Noise Sampling (Cosine Schedule) ---
        # Sample t uniformly from [0, 1] for the batch
        t = torch.rand(current_batch_size, device=cpu_device) # Normalized time
        schedule_values = get_cosine_schedule_values(t)
        alpha_t = schedule_values['alpha_t'].unsqueeze(-1) # Add dim for broadcasting
        sigma_t = schedule_values['sigma_t'].unsqueeze(-1) # Add dim for broadcasting
        lambda_t = schedule_values['lambda_t'] # Keep as (batch,)

        # --- 4. Noising Continuation Embedding ---
        epsilon = torch.randn_like(x_cont, device=cpu_device) # Noise on CPU
        z_t = alpha_t * x_cont + sigma_t * epsilon # Noisy latent (CPU)

        # --- 5. Calculate Target Velocity ---
        target_v = alpha_t * x_cont + sigma_t * epsilon # Target velocity (CPU)

        # --- 6. Classifier-Free Guidance Masking ---
        # Decide which prefixes to mask
        mask_indices = torch.rand(current_batch_size, device=cpu_device) < self.cfg_mask_probability
        # Clone prefix embeddings to avoid modifying the original tensor if needed elsewhere
        x_pref_masked = x_pref.clone()
        # Replace selected embeddings with the null embedding (move null embedding to CPU for this)
        null_emb_cpu = self.null_embedding.to(cpu_device).detach() # Detach just in case
        # Ensure null_emb_cpu has batch dimension for broadcasting if needed
        if null_emb_cpu.shape[0] == 1:
             null_emb_cpu = null_emb_cpu.expand(current_batch_size, -1) # Expand if necessary

        # Apply masking using the boolean mask
        x_pref_masked[mask_indices] = null_emb_cpu[mask_indices]

        # --- 7. Return Batch (all tensors on CPU) ---
        return {
            'noisy_latent': z_t.detach(),           # Shape: (batch, sent_emb_dim)
            'prefix_embedding': x_pref_masked.detach(), # Shape: (batch, sent_emb_dim)
            'time_values': lambda_t.detach(),       # Shape: (batch,) - Represents noise level
            'target_velocity': target_v.detach(),   # Shape: (batch, sent_emb_dim)
        }

# --- Example Usage (Illustrative) ---
# if __name__ == '__main__':
#     # This requires loading actual models, tokenizers, and the null embedding
#     # Setup omitted for brevity. Assume components are loaded and on correct devices.
#
#     # dummy_decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     # dummy_filter_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     # dummy_sent_encoder = AutoModel.from_pretrained("google/sentence-t5-base").eval().cuda()
#     # dummy_sent_tokenizer = AutoTokenizer.from_pretrained("google/sentence-t5-base")
#     # dummy_null_embedding = nn.Parameter(torch.randn(1, dummy_sent_encoder.config.d_model)).cuda()
#
#     # collator = DataCollatorForDiffusionTraining(
#     #     decoder_tokenizer=dummy_decoder_tokenizer,
#     #     filter_tokenizer=dummy_filter_tokenizer,
#     #     sentence_tokenizer=dummy_sent_tokenizer,
#     #     sentence_encoder=dummy_sent_encoder,
#     #     null_embedding=dummy_null_embedding,
#     #     # Add other config params if needed
#     # )
#
#     # dummy_examples = [
#     #     "This is the prefix part. This is the continuation part which should be long enough.",
#     #     "Another prefix example. Followed by another continuation example, also sufficiently long.",
#     #     "Short prefix. Very very very very very very very very very very very very very long continuation."
#     # ]
#
#     # batch = collator(dummy_examples)
#     # logging.info("Collator output keys:", batch.keys())
#     # for key, value in batch.items():
#     #     logging.info(f"Shape of {key}: {value.shape}, Device: {value.device}")


