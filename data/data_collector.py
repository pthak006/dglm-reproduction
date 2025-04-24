# data/data_collator.py

import torch
import torch.nn as nn
import random
import math
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel
from typing import List, Dict, Any, Optional
import logging

# Assuming PromptGenerator is in models.prompt_generator
# Adjust import path if needed
try:
    from models.prompt_generator import PromptGenerator
except ImportError:
    logging.error("Could not import PromptGenerator. Make sure models/prompt_generator.py exists.")
    # Define a dummy class if import fails, to allow script loading but fail at runtime
    class PromptGenerator(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, *args, **kwargs): raise NotImplementedError("PromptGenerator not loaded")


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def passes_filter(text: str, tokenizer: PreTrainedTokenizer, threshold: float) -> bool:
    """
    Checks if the text passes the compressibility filter based on token/char ratio.
    (Copied from the previously planned preprocess.py)
    """
    if not text or not isinstance(text, str):
        return False # Handle None or non-string input
    try:
        # Remove leading/trailing whitespace which can affect char count
        text = text.strip()
        if not text: return False

        num_chars = len(text)
        # Tokenize without adding special tokens for pure content length
        num_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        if num_tokens == 0:
            return False # Avoid division by zero

        # Paper's filter: drop if num_tokens > threshold * num_chars
        passes = num_tokens <= threshold * num_chars
        # Heuristic: filter very short potentially noisy strings
        if num_chars < 20 and num_tokens < 5:
             passes = False

        return passes

    except Exception as e:
        logging.warning(f"Error processing text during filtering: {e}. Text snippet: {text[:100]}...")
        return False

def sample_alpha_cosine(num_samples: int, s: float = 3.0, device: torch.device = 'cpu') -> torch.Tensor:
    """
    Samples alpha_t values from a scaled cosine schedule.
    alpha_t = cos^2 ( (t/T + s) / (1 + s) * pi / 2 )
    where t is uniform in [0, T]. We sample t/T uniformly in [0, 1].

    Args:
        num_samples: Number of alpha_t values to sample (batch size).
        s: Offset parameter from the paper (default 3.0).
        device: The torch device to create the tensor on.

    Returns:
        Tensor of shape (num_samples,) containing alpha_t values.
    """
    # Sample t/T uniformly from [0, 1]
    t_over_T = torch.rand(num_samples, device=device)

    # Calculate the argument for cosine
    cos_arg = (t_over_T + s) / (1 + s) * (math.pi / 2.0)

    # Calculate alpha_t
    alpha_t = torch.cos(cos_arg) ** 2

    # Clamp to avoid numerical issues near 0 and 1
    alpha_t = torch.clamp(alpha_t, 1e-5, 1.0 - 1e-5)
    return alpha_t


# --- Data Collator Class ---

@dataclass
class DataCollatorForDecoderFineTuning:
    """
    Data collator for fine-tuning the decoder and prompt generator.
    Handles dynamic filtering, splitting, noise augmentation, and batch preparation.
    """
    # Models & Tokenizers
    decoder_tokenizer: PreTrainedTokenizer
    filter_tokenizer: PreTrainedTokenizer # Usually GPT-2
    sentence_encoder: PreTrainedModel # Must have .encode() or similar method
    sentence_tokenizer: PreTrainedTokenizer
    prompt_generator: PromptGenerator
    decoder_embedding_layer: nn.Embedding # Pass decoder.get_input_embeddings()

    # Configuration
    prefix_len: int = 32
    k_soft_tokens: int = 8
    max_seq_len: int = 96 # Max length for decoder input (prefix + soft_prompt + continuation)
    filter_threshold: float = 0.3
    noise_schedule_s: float = 3.0 # Offset for cosine schedule

    # Padding values
    label_ignore_index: int = -100

    def __call__(self, examples: List[str]) -> Dict[str, torch.Tensor]:
        """
        Processes a batch of raw text strings.

        Args:
            examples: A list of raw text strings from the dataset.

        Returns:
            A dictionary containing the prepared batch:
            - 'input_embeds': Concatenated prefix + soft prompt embeddings.
            - 'attention_mask': Attention mask for the input embeddings.
            - 'labels': Labels for language modeling loss (masked).
        """
        device = self.decoder_embedding_layer.weight.device # Assume all models/layers are on the same device
        batch_size = len(examples) # Initial batch size (might be reduced by filtering)
        processed_batch = []

        # --- 1. Filtering & Splitting ---
        texts_to_process = list(examples) # Make a copy to potentially modify/retry
        while len(processed_batch) < batch_size and texts_to_process:
            text = texts_to_process.pop(0).strip() # Get next text

            # Apply filter
            if not passes_filter(text, self.filter_tokenizer, self.filter_threshold):
                # logging.debug(f"Text failed filter: {text[:100]}...")
                continue # Skip this example

            # Split into prefix and continuation based on decoder tokens
            # Need decoder tokenizer here
            tokenized_full = self.decoder_tokenizer.encode(text, add_special_tokens=False)

            if len(tokenized_full) <= self.prefix_len:
                # logging.debug(f"Text too short after tokenization: {len(tokenized_full)} tokens. Skipping.")
                continue # Not enough tokens for prefix + continuation

            prefix_tokens = tokenized_full[:self.prefix_len]
            # Limit continuation length based on max_seq_len and k_soft_tokens
            max_continuation_len = self.max_seq_len - self.prefix_len - self.k_soft_tokens
            continuation_tokens = tokenized_full[self.prefix_len : self.prefix_len + max_continuation_len]

            if not continuation_tokens:
                 # logging.debug("No continuation tokens left after splitting and length limit. Skipping.")
                 continue # Need at least one continuation token

            # Decode continuation tokens back to string for sentence encoder
            # Important: Use skip_special_tokens=True
            continuation_text = self.decoder_tokenizer.decode(continuation_tokens, skip_special_tokens=True)

            if not continuation_text.strip():
                # logging.debug("Continuation text is empty after decoding. Skipping.")
                continue

            processed_batch.append({
                "prefix_tokens": prefix_tokens,
                "continuation_tokens": continuation_tokens,
                "continuation_text": continuation_text,
            })

        if not processed_batch:
             # If filtering removed the entire batch, return an empty dict or handle appropriately
             logging.warning("Entire batch filtered out. Returning empty batch.")
             # Returning empty dict might cause issues in Trainer, consider raising error or padding?
             # For now, let's return something that won't break indexing but indicates emptiness
             return {
                 'input_embeds': torch.empty(0, self.max_seq_len, self.decoder_embedding_layer.embedding_dim, device=device),
                 'attention_mask': torch.empty(0, self.max_seq_len, device=device, dtype=torch.long),
                 'labels': torch.empty(0, self.max_seq_len, device=device, dtype=torch.long)
             }

        current_batch_size = len(processed_batch)

        # --- 2. Sentence Embedding for Continuations ---
        continuation_texts = [item["continuation_text"] for item in processed_batch]
        sent_inputs = self.sentence_tokenizer(
            continuation_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 # Typical max length for sentence encoders
        ).to(device)

        with torch.no_grad(): # Ensure sentence encoder is frozen
            # Sentence-T5 uses encoder-decoder, get encoder last hidden state
            # Or use a dedicated SentenceTransformer model wrapper if available
            # Assuming direct use of AutoModel for T5:
            sent_outputs = self.sentence_encoder(
                input_ids=sent_inputs.input_ids,
                attention_mask=sent_inputs.attention_mask,
                return_dict=True
            )
            # Mean pool the last hidden state of the encoder
            # Mask pooling based on attention mask might be better
            sent_embeddings = sent_outputs.last_hidden_state.mean(dim=1) # Shape: (batch, sent_emb_dim)

        # --- 3. Noise Augmentation ---
        alpha_t = sample_alpha_cosine(current_batch_size, s=self.noise_schedule_s, device=device) # Shape: (batch,)
        sigma_t_sq = 1.0 - alpha_t
        sigma_t = torch.sqrt(sigma_t_sq)
        epsilon = torch.randn_like(sent_embeddings) # Shape: (batch, sent_emb_dim)

        # Add dimension for broadcasting alpha_t and sigma_t
        alpha_t_unsqueezed = alpha_t.unsqueeze(-1) # Shape: (batch, 1)
        sigma_t_unsqueezed = sigma_t.unsqueeze(-1) # Shape: (batch, 1)

        noisy_sent_embeddings = alpha_t_unsqueezed * sent_embeddings + sigma_t_unsqueezed * epsilon

        # --- 4. Soft Prompt Generation ---
        # Prompt generator needs noisy embeddings and alpha_t
        soft_prompt_embeddings = self.prompt_generator(noisy_sent_embeddings, alpha_t) # Shape: (batch, k, dec_emb_dim)

        # --- 5. Prepare Decoder Input Embeddings & Attention Mask ---
        input_embeds_list = []
        attention_mask_list = []
        max_len_in_batch = 0 # Track max length dynamically for padding

        for i in range(current_batch_size):
            prefix_token_ids = torch.tensor(processed_batch[i]["prefix_tokens"], dtype=torch.long, device=device)
            # Get embeddings from the decoder's embedding layer
            prefix_embeds = self.decoder_embedding_layer(prefix_token_ids) # Shape: (prefix_len, dec_emb_dim)

            # Concatenate prefix embeddings and soft prompt embeddings
            combined_embeds = torch.cat([prefix_embeds, soft_prompt_embeddings[i]], dim=0) # Shape: (prefix_len + k, dec_emb_dim)
            input_embeds_list.append(combined_embeds)

            # Create attention mask (all 1s for the actual content)
            current_len = combined_embeds.shape[0]
            attention_mask_list.append(torch.ones(current_len, dtype=torch.long, device=device))
            max_len_in_batch = max(max_len_in_batch, current_len)

        # Pad input_embeds and attention_mask
        padded_input_embeds = torch.zeros(current_batch_size, max_len_in_batch, self.decoder_embedding_layer.embedding_dim, device=device)
        padded_attention_mask = torch.zeros(current_batch_size, max_len_in_batch, dtype=torch.long, device=device)

        for i in range(current_batch_size):
            seq_len = input_embeds_list[i].shape[0]
            padded_input_embeds[i, :seq_len] = input_embeds_list[i]
            padded_attention_mask[i, :seq_len] = attention_mask_list[i]

        # --- 6. Prepare Labels ---
        labels_list = []
        max_label_len = 0

        for i in range(current_batch_size):
            continuation_token_ids = torch.tensor(processed_batch[i]["continuation_tokens"], dtype=torch.long, device=device)
            # Labels: Ignore prefix and soft prompt, predict continuation
            prefix_ignore = torch.full((self.prefix_len,), self.label_ignore_index, dtype=torch.long, device=device)
            soft_prompt_ignore = torch.full((self.k_soft_tokens,), self.label_ignore_index, dtype=torch.long, device=device)

            label_seq = torch.cat([prefix_ignore, soft_prompt_ignore, continuation_token_ids])
            labels_list.append(label_seq)
            max_label_len = max(max_label_len, label_seq.shape[0])

        # Pad labels
        # Ensure max_label_len matches max_len_in_batch if decoder predicts for every input position
        final_max_len = max(max_len_in_batch, max_label_len)
        # Re-pad attention mask and input embeds if label length is longer (shouldn't happen with this logic)
        if final_max_len > max_len_in_batch:
             logging.warning(f"Max label length ({max_label_len}) exceeded max input length ({max_len_in_batch}). Adjusting padding.")
             # This indicates a potential issue in length calculation or assumptions
             # Re-pad inputs to match the required output length for labels
             padded_input_embeds = F.pad(padded_input_embeds, (0, 0, 0, final_max_len - max_len_in_batch), value=0)
             padded_attention_mask = F.pad(padded_attention_mask, (0, final_max_len - max_len_in_batch), value=0)


        padded_labels = torch.full((current_batch_size, final_max_len), self.label_ignore_index, dtype=torch.long, device=device)
        for i in range(current_batch_size):
            seq_len = labels_list[i].shape[0]
            padded_labels[i, :seq_len] = labels_list[i]


        # --- 7. Return Batch ---
        return {
            'input_embeds': padded_input_embeds,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels,
        }

# --- Example Usage (Illustrative - requires actual models/tokenizers) ---
# if __name__ == '__main__':
#     # This requires loading actual models and tokenizers, setup omitted for brevity
#     # Assume models/tokenizers are loaded and on the correct device
#     # dummy_decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     # dummy_filter_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     # dummy_sent_encoder = AutoModel.from_pretrained("google/sentence-t5-base") # Use base for faster testing
#     # dummy_sent_tokenizer = AutoTokenizer.from_pretrained("google/sentence-t5-base")
#     # dummy_decoder_model = AutoModelForCausalLM.from_pretrained("gpt2")
#     # dummy_prompt_gen = PromptGenerator(...) # Instantiate with correct dims

#     # collator = DataCollatorForDecoderFineTuning(
#     #     decoder_tokenizer=dummy_decoder_tokenizer,
#     #     filter_tokenizer=dummy_filter_tokenizer,
#     #     sentence_encoder=dummy_sent_encoder,
#     #     sentence_tokenizer=dummy_sent_tokenizer,
#     #     prompt_generator=dummy_prompt_gen,
#     #     decoder_embedding_layer=dummy_decoder_model.get_input_embeddings()
#     #     # Add other config params if needed
#     # )

#     # dummy_examples = [
#     #     "This is a reasonably long sentence that should pass the filter and be used for training.",
#     #     "Short.", # Should fail filter
#     #     "<html><body><code>" + "x="*500 + "</code></body></html>", # Should fail filter
#     #     "Another good example sentence, hopefully long enough to provide prefix and continuation."
#     # ]

#     # batch = collator(dummy_examples)
#     # logging.info("Collator output keys:", batch.keys())
#     # for key, value in batch.items():
#     #     logging.info(f"Shape of {key}: {value.shape}")

