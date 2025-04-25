# data/data_collator.py

import torch.nn.functional as F
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
    max_retries: int = 5  # Maximum number of retries to find a valid batch

    # Padding values
    label_ignore_index: int = -100
    
    # Logger
    logger: logging.Logger = logging.getLogger(__name__)

    # Modified __call__ to handle empty batches internally
    def __call__(self, examples: List[str]) -> Dict[str, torch.Tensor]:
        # Determine the target device for model inference (GPU)
        # Use try-except in case the layer hasn't been moved to a device yet
        try:
            model_device = self.decoder_embedding_layer.weight.device
        except AttributeError:
             self.logger.warning("decoder_embedding_layer.weight.device not accessible, defaulting model_device to CPU.")
             model_device = torch.device('cpu') # Fallback to CPU

        # Prepare the final batch on CPU for flexibility and efficiency
        cpu_device = torch.device('cpu')
        
        # Keep trying until we get a valid batch or exhaust retries
        retry_count = 0
        all_examples = list(examples)  # Start with provided examples
        
        while retry_count < self.max_retries:
            if not all_examples:
                self.logger.warning(f"Ran out of examples after {retry_count} retries. Cannot create a valid batch.")
                # Create a minimal valid batch with zeros to avoid training loop errors
                return self._create_empty_batch(cpu_device)
                
            # Take the next batch_size examples
            batch_size = min(len(all_examples), len(examples))
            current_examples = all_examples[:batch_size]
            all_examples = all_examples[batch_size:]  # Remove used examples
            
            processed_batch = []
            
            # --- 1. Filtering & Splitting ---
            texts_to_process = list(current_examples)  # Make a copy
            original_count = len(texts_to_process)
            filtered_count = 0
            tokenization_error_count = 0
            decode_error_count = 0
            length_issue_count = 0

            while texts_to_process:  # Process all provided examples
                text = texts_to_process.pop(0).strip()  # Get next text

                # Apply filter
                if not passes_filter(text, self.filter_tokenizer, self.filter_threshold):
                    # self.logger.debug(f"Text failed filter: {text[:100]}...")
                    filtered_count += 1
                    continue  # Skip this example

                # --- Robust Tokenization (Decoder) ---
                try:
                    tokenized_full = self.decoder_tokenizer.encode(text, add_special_tokens=False)
                    if not isinstance(tokenized_full, (list, tuple)) or not tokenized_full:
                        self.logger.warning(f"Decoder tokenizer returned invalid/empty output for text: {text[:100]}... Type: {type(tokenized_full)}. Skipping.")
                        tokenization_error_count += 1
                        continue
                except Exception as e:
                    self.logger.warning(f"Error during decoder tokenization for text: {text[:100]}... Error: {e}. Skipping.")
                    tokenization_error_count += 1
                    continue
                # --- End Robust Tokenization (Decoder) ---

                if len(tokenized_full) <= self.prefix_len:
                    # self.logger.debug(f"Text too short after tokenization: {len(tokenized_full)} tokens <= prefix_len {self.prefix_len}. Skipping.")
                    length_issue_count += 1
                    continue  # Not enough tokens for prefix + continuation

                # --- Assign prefix and continuation ---
                prefix_tokens = tokenized_full[:self.prefix_len]
                max_continuation_len = self.max_seq_len - self.prefix_len - self.k_soft_tokens
                if max_continuation_len <= 0:
                    self.logger.error(f"Calculated max_continuation_len ({max_continuation_len}) is non-positive. Check config. Skipping text: {text[:100]}...")
                    length_issue_count += 1
                    continue
                continuation_tokens = tokenized_full[self.prefix_len : self.prefix_len + max_continuation_len]

                if not continuation_tokens:
                    # self.logger.debug(f"No continuation tokens left after splitting/length limit for text: {text[:100]}... Skipping.")
                    length_issue_count += 1
                    continue  # Need at least one continuation token

                # --- Robust Decoding (Continuation) ---
                try:
                    continuation_text = self.decoder_tokenizer.decode(continuation_tokens, skip_special_tokens=True)
                except Exception as e:
                    self.logger.warning(f"Error during decoding continuation tokens: {continuation_tokens}. Error: {e}. Skipping text: {text[:100]}...")
                    decode_error_count += 1
                    continue
                # --- End Robust Decoding (Continuation) ---

                if not continuation_text.strip():
                    # self.logger.debug(f"Continuation text is empty after decoding for text: {text[:100]}... Skipping.")
                    decode_error_count += 1
                    continue

                # --- Append if all checks passed ---
                processed_batch.append({
                    "prefix_tokens": prefix_tokens,
                    "continuation_tokens": continuation_tokens,
                    "continuation_text": continuation_text,
                })
            # --- End Filtering & Splitting ---

            # Handle case where filtering/tokenization removed all examples
            if not processed_batch:
                self.logger.warning(f"Batch is empty after filtering/splitting. Original#: {original_count}, Filtered: {filtered_count}, TokenizeErrs: {tokenization_error_count}, LenIssues: {length_issue_count}, DecodeErrs: {decode_error_count}. Trying next batch.")
                retry_count += 1
                continue  # Try next batch

            # --- 2. Sentence Embedding (Individual Processing for Robustness) ---
            sent_embeddings_list = []
            valid_indices = []  # Keep track of original indices that processed successfully
            sent_encode_error_count = 0
            initial_processed_count = len(processed_batch)  # Count before sentence encoding

            # Get all continuation texts first
            continuation_texts_all = [item["continuation_text"] for item in processed_batch]

            for idx, text in enumerate(continuation_texts_all):
                try:
                    # Tokenize one text instance on CPU first
                    single_sent_input = self.sentence_tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=False,
                        max_length=512  # Sentence encoder max length
                    )

                    if 'input_ids' not in single_sent_input or single_sent_input.input_ids.numel() == 0:
                        # self.logger.debug(f"Sentence tokenizer returned empty/invalid output for text: {text[:100]}... Skipping item.")
                        sent_encode_error_count += 1
                        continue

                    single_sent_input = single_sent_input.to(model_device)

                    with torch.no_grad():
                        encoder_outputs = self.sentence_encoder.encoder(
                            input_ids=single_sent_input.input_ids,
                            attention_mask=single_sent_input.attention_mask,
                            return_dict=True
                        )
                        sent_embedding_gpu = encoder_outputs.last_hidden_state.mean(dim=1)
                        sent_embedding = sent_embedding_gpu.to(cpu_device)

                    sent_embeddings_list.append(sent_embedding)
                    valid_indices.append(idx)  # Record the original index

                except IndexError as ie:
                     self.logger.warning(f"Caught IndexError during sentence encoding for text: {text[:100]}... Error: {ie}. Skipping item.")
                     sent_encode_error_count += 1
                     continue
                except Exception as e:
                    self.logger.warning(f"Generic error during sentence encoding for text: {text[:100]}... Error: {e}. Skipping item.")
                    sent_encode_error_count += 1
                    continue

            # Check if *any* items survived the sentence encoding step
            if not valid_indices:
                self.logger.warning(f"Batch is empty after sentence encoding step. Initial#: {initial_processed_count}, Errors: {sent_encode_error_count}. Trying next batch.")
                retry_count += 1
                continue  # Try next batch

            # Filter the original processed_batch and update batch size
            processed_batch = [processed_batch[i] for i in valid_indices]
            current_batch_size = len(processed_batch)
            sent_embeddings = torch.cat(sent_embeddings_list, dim=0)

            # --- 3. Noise Augmentation (on CPU) ---
            alpha_t = sample_alpha_cosine(current_batch_size, s=self.noise_schedule_s, device=cpu_device)
            sigma_t_sq = 1.0 - alpha_t
            sigma_t = torch.sqrt(sigma_t_sq.clamp(min=1e-8))  # Add clamp for numerical stability
            epsilon = torch.randn_like(sent_embeddings, device=cpu_device)
            alpha_t_unsqueezed = alpha_t.unsqueeze(-1)
            sigma_t_unsqueezed = sigma_t.unsqueeze(-1)
            noisy_sent_embeddings = alpha_t_unsqueezed * sent_embeddings + sigma_t_unsqueezed * epsilon

            # --- 4. Soft Prompt Generation ---
            noisy_sent_embeddings_gpu = noisy_sent_embeddings.to(model_device)
            alpha_t_gpu = alpha_t.to(model_device)
            soft_prompt_embeddings_gpu = self.prompt_generator(noisy_sent_embeddings_gpu, alpha_t_gpu)
            soft_prompt_embeddings = soft_prompt_embeddings_gpu.to(cpu_device)

            # --- 5. Prepare Decoder Input Embeddings & Attention Mask (on CPU) ---
            input_embeds_list = []
            attention_mask_list = []
            max_len_in_batch = 0

            for i in range(current_batch_size):
                prefix_token_ids = torch.tensor(processed_batch[i]["prefix_tokens"], dtype=torch.long, device=cpu_device)
                prefix_token_ids_gpu = prefix_token_ids.to(model_device)
                prefix_embeds_gpu = self.decoder_embedding_layer(prefix_token_ids_gpu)
                prefix_embeds = prefix_embeds_gpu.to(cpu_device)

                combined_embeds = torch.cat([prefix_embeds.detach(), soft_prompt_embeddings[i].detach()], dim=0)
                input_embeds_list.append(combined_embeds)

                current_len = combined_embeds.shape[0]
                attention_mask_list.append(torch.ones(current_len, dtype=torch.long, device=cpu_device))
                max_len_in_batch = max(max_len_in_batch, current_len)

            # --- 6. Prepare Labels (on CPU) ---
            labels_list = []
            max_label_len = 0

            for i in range(current_batch_size):
                continuation_token_ids = torch.tensor(processed_batch[i]["continuation_tokens"], dtype=torch.long, device=cpu_device)
                prefix_ignore = torch.full((self.prefix_len,), self.label_ignore_index, dtype=torch.long, device=cpu_device)
                soft_prompt_ignore = torch.full((self.k_soft_tokens,), self.label_ignore_index, dtype=torch.long, device=cpu_device)

                label_seq = torch.cat([prefix_ignore, soft_prompt_ignore, continuation_token_ids])
                labels_list.append(label_seq)
                max_label_len = max(max_label_len, label_seq.shape[0])

            # --- 7. Pad Inputs, Attention Masks, and Labels (on CPU) ---
            final_max_len = max(max_len_in_batch, max_label_len)

            if final_max_len > self.max_seq_len:
                 self.logger.warning(f"Calculated final max length ({final_max_len}) exceeds configured max_seq_len ({self.max_seq_len}). Truncating.")
                 final_max_len = self.max_seq_len

            embedding_dim = self.decoder_embedding_layer.embedding_dim
            padded_input_embeds = torch.zeros(current_batch_size, final_max_len, embedding_dim, device=cpu_device)
            padded_attention_mask = torch.zeros(current_batch_size, final_max_len, dtype=torch.long, device=cpu_device)
            padded_labels = torch.full((current_batch_size, final_max_len), self.label_ignore_index, dtype=torch.long, device=cpu_device)

            for i in range(current_batch_size):
                input_seq_len = input_embeds_list[i].shape[0]
                len_to_pad_input = min(input_seq_len, final_max_len)
                padded_input_embeds[i, :len_to_pad_input] = input_embeds_list[i][:len_to_pad_input]

                mask_len = attention_mask_list[i].shape[0]
                len_to_pad_mask = min(mask_len, final_max_len)
                padded_attention_mask[i, :len_to_pad_mask] = attention_mask_list[i][:len_to_pad_mask]

                label_seq_len = labels_list[i].shape[0]
                len_to_pad_label = min(label_seq_len, final_max_len)
                padded_labels[i, :len_to_pad_label] = labels_list[i][:len_to_pad_label]

            # --- 8. Return Final Batch (all tensors on CPU) ---
            self.logger.debug(f"Successfully collated batch of size {current_batch_size}. Final sequence length: {final_max_len}")
            return {
                'input_embeds': padded_input_embeds,
                'attention_mask': padded_attention_mask,
                'labels': padded_labels,
            }
            
        # If we've exhausted all retries and still don't have a valid batch
        self.logger.warning(f"Failed to create a valid batch after {self.max_retries} retries.")
        return self._create_empty_batch(cpu_device)
    
    def _create_empty_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Create a minimal valid batch with a single example of zeros to avoid training loop errors."""
        embedding_dim = self.decoder_embedding_layer.embedding_dim
        min_seq_len = self.prefix_len + self.k_soft_tokens + 1  # At least one continuation token
        
        self.logger.warning(f"Creating minimal empty batch with sequence length {min_seq_len}")
        
        return {
            'input_embeds': torch.zeros(1, min_seq_len, embedding_dim, device=device),
            'attention_mask': torch.ones(1, min_seq_len, dtype=torch.long, device=device),
            'labels': torch.full((1, min_seq_len), self.label_ignore_index, dtype=torch.long, device=device),
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
