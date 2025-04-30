# scripts/train_diffusion.py

import os
import logging
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.cauchy import Cauchy
from torch.distributions.normal import Normal
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedModel,
    PreTrainedTokenizer
)
from huggingface_hub import HfFolder, whoami # For Hub integration

# Import custom modules (adjust paths if necessary)
try:
    from data.dataset import C4TrainingDataset
    from data.diffusion_collator import DataCollatorForDiffusionTraining, get_cosine_schedule_values
    from models.diffusion_network import DiffusionTransformer, DiffusionTransformerConfig
except ImportError as e:
    logging.error(f"Failed to import custom modules: {e}")
    logging.error("Ensure data/datasets.py, data/diffusion_collator.py, and models/diffusion_network.py exist.")
    exit(1)

# --- Loss Weighting Function ---

def loss_weighting_fn(lambda_t: torch.Tensor) -> torch.Tensor:
    """
    Calculates the loss weight w(lambda_t) based on Appendix C.
    Uses Cauchy for lambda_t < 0 and Normal for lambda_t >= 0.
    Parameters from paper: mu=0, sigma=2.4 for both distributions.
    Args:
        lambda_t (torch.Tensor): Log SNR values for the batch. Shape (batch_size,).
    Returns:
        torch.Tensor: Loss weights for the batch. Shape (batch_size,).
    """
    device = lambda_t.device
    mu = 0.0
    sigma = 2.4
    # Ensure sigma is a tensor for device placement if needed
    sigma_tensor = torch.tensor(sigma, device=device)

    # Define distributions
    # Cauchy location=mu, scale=sigma
    cauchy_dist = Cauchy(loc=mu, scale=sigma_tensor)
    # Normal loc=mu, scale=sigma
    normal_dist = Normal(loc=mu, scale=sigma_tensor)

    # Calculate PDF values (use log_prob for stability then exponentiate)
    log_pdf_cauchy = cauchy_dist.log_prob(lambda_t)
    log_pdf_normal = normal_dist.log_prob(lambda_t)

    # Get PDF values at lambda_t = 0 to find normalization constants
    # Use log_prob(tensor(0.0)) for consistency
    zero_tensor = torch.tensor(0.0, device=device)
    log_pdf_cauchy_at_0 = cauchy_dist.log_prob(zero_tensor)
    log_pdf_normal_at_0 = normal_dist.log_prob(zero_tensor)

    # Calculate normalized log PDFs (log(pdf(x)/pdf(0)) = log_prob(x) - log_prob(0))
    norm_log_pdf_cauchy = log_pdf_cauchy - log_pdf_cauchy_at_0
    norm_log_pdf_normal = log_pdf_normal - log_pdf_normal_at_0

    # Combine based on the condition lambda_t < 0
    weights = torch.where(
        lambda_t < 0,
        torch.exp(norm_log_pdf_cauchy), # Use normalized Cauchy PDF
        torch.exp(norm_log_pdf_normal)  # Use normalized Normal PDF
    )

    # Clamp weights to avoid potential numerical issues? Paper doesn't mention.
    # weights = torch.clamp(weights, min=1e-6) # Optional clamping

    return weights

# --- Custom Trainer for Diffusion Loss ---

class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the weighted v-prediction loss.
        """
        # Inputs from collator: noisy_latent, prefix_embedding, time_values (lambda_t), target_velocity
        noisy_latent = inputs.get("noisy_latent")
        prefix_embedding = inputs.get("prefix_embedding")
        lambda_t = inputs.get("time_values") # These are lambda_t values
        target_velocity = inputs.get("target_velocity")

        # Get model prediction (predicted velocity v_theta)
        # The model forward pass should accept these specific args
        outputs = model(
            noisy_latent=noisy_latent,
            prefix_embedding=prefix_embedding,
            time_values=lambda_t, # Pass lambda_t to the model if needed for time embedding
            return_dict=True # Ensure model returns dict-like or object
        )
        # Assuming the predicted velocity is the primary output or accessible via a key
        # Adjust based on the actual DiffusionTransformer output structure
        if isinstance(outputs, torch.Tensor):
             predicted_velocity = outputs
        elif hasattr(outputs, 'predicted_velocity'):
             predicted_velocity = outputs.predicted_velocity # Example if output is object
        elif isinstance(outputs, dict) and 'predicted_velocity' in outputs:
             predicted_velocity = outputs['predicted_velocity'] # Example if output is dict
        else:
             # Fallback or raise error if prediction format is unknown
             logging.error(f"Unexpected model output format: {type(outputs)}. Cannot extract predicted velocity.")
             # Try to use the first element if it's a tuple/list? Risky.
             if isinstance(outputs, (tuple, list)) and isinstance(outputs[0], torch.Tensor):
                 predicted_velocity = outputs[0]
             else:
                 raise TypeError(f"Cannot extract predicted velocity from model output of type {type(outputs)}")


        # Calculate MSE loss per element
        loss_mse = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
        # Reduce MSE across the embedding dimension, keep batch dimension
        loss_mse_reduced = loss_mse.mean(dim=list(range(1, loss_mse.dim()))) # Mean over all dims except batch

        # Calculate loss weights based on lambda_t
        weights = loss_weighting_fn(lambda_t)

        # Apply weights and calculate final batch loss (mean over batch)
        loss = (weights * loss_mse_reduced).mean()

        return (loss, outputs) if return_outputs else loss

# --- Main Training Function ---

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 1. Load Configuration ---
    logging.info(f"Loading base configuration from {args.config_path}")
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Get model names needed for tokenizers/embeddings
        decoder_model_name = config.get("auto_regressive_model_name", "gpt2-large") # For splitting consistency
        sentence_encoder_model_name = config.get("sentence_encoder_model_name", "google/sentence-t5-xl")
        filter_tokenizer_name = config.get("filter_tokenizer_name", "gpt2")
    except Exception as e:
        logging.error(f"Error loading config: {e}", exc_info=True)
        return

    # --- 2. Load Models and Tokenizers (Sentence Encoder part) ---
    logging.info("Loading Sentence Encoder and Tokenizers...")
    try:
        # Tokenizers needed by the collator
        decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        filter_tokenizer = AutoTokenizer.from_pretrained(filter_tokenizer_name)
        sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_encoder_model_name)

        # Sentence Encoder (frozen)
        sentence_encoder = AutoModel.from_pretrained(sentence_encoder_model_name).to(device)
        logging.info(f"Freezing Sentence Encoder ({sentence_encoder_model_name}) parameters.")
        for param in sentence_encoder.parameters():
            param.requires_grad = False
        sentence_encoder.eval()

    except Exception as e:
        logging.error(f"Error loading sentence encoder/tokenizers: {e}", exc_info=True)
        return

    # --- 3. Instantiate Diffusion Model ---
    logging.info("Instantiating Diffusion Transformer Model...")
    try:
        # Get embedding dimension from the loaded sentence encoder
        sent_emb_dim = sentence_encoder.config.d_model
        # Instantiate the actual DiffusionTransformer
        # diffusion_model = DiffusionTransformer(sentence_emb_dim=sent_emb_dim).to(device)
        config = DiffusionTransformerConfig(sentence_emb_dim=sent_emb_dim)
        diffusion_model = DiffusionTransformer(config).to(device)

    except Exception as e:
        logging.error(f"Error instantiating DiffusionTransformer: {e}", exc_info=True)
        return

    # --- 4. Load Dataset and Collator ---
    logging.info("Loading dataset and initializing data collator...")
    try:
        train_dataset = C4TrainingDataset(dataset_path=args.c4_subset_path)

        data_collator = DataCollatorForDiffusionTraining(
            decoder_tokenizer=decoder_tokenizer,
            filter_tokenizer=filter_tokenizer,
            sentence_tokenizer=sentence_tokenizer,
            sentence_encoder=sentence_encoder, # Pass frozen encoder
            null_embedding=diffusion_model.null_embedding, # Pass learnable param
            prefix_len=args.prefix_len,
            filter_threshold=args.filter_threshold,
            cfg_mask_probability=args.cfg_mask_probability
        )
    except Exception as e:
        logging.error(f"Error setting up dataset/collator: {e}", exc_info=True)
        return

    # --- 5. Hugging Face Hub Login ---
    # (Identical Hub login logic as in train_decoder.py)
    if args.push_to_hub:
        try:
            token = HfFolder.get_token()
            if token is None:
                logging.error("Push to Hub requested, but not logged in. Run `huggingface-cli login`.")
                return
            user_info = whoami(token=token)
            logging.info(f"Logged in to Hugging Face Hub as: {user_info['name']}")
            hub_model_id = f"{user_info['name']}/{args.hub_model_name}"
            logging.info(f"Will push model to: {hub_model_id}")
        except Exception as e:
            logging.error(f"Error checking Hugging Face login status: {e}")
            return
    else:
        hub_model_id = None

    # --- 6. Configure Training Arguments ---
    logging.info("Configuring Training Arguments...")
    # Calculate gradient accumulation (Target batch size 256 from Table 8)
    eff_batch_size = args.per_device_train_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_device_train_batch_size
    target_total_batch_size = 256
    if target_total_batch_size % eff_batch_size != 0:
        logging.warning(f"Target batch size {target_total_batch_size} not divisible by effective batch size {eff_batch_size}. Adjusting gradient accumulation.")
    gradient_accumulation_steps = max(1, target_total_batch_size // eff_batch_size)
    logging.info(f"Effective per-device batch size: {args.per_device_train_batch_size}")
    logging.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    logging.info(f"Target total batch size: {args.per_device_train_batch_size * gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else args.per_device_train_batch_size * gradient_accumulation_steps}")


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=args.report_to.split(',') if args.report_to else None,

        # Training strategy (from Table 8, with adjustments)
        max_steps=args.max_steps,
        learning_rate=args.learning_rate, # Use 1e-4 as default instead of 1e-3
        lr_scheduler_type=args.lr_scheduler_type, # Cosine decay
        warmup_steps=args.warmup_steps, # Default 1000
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm, # Use max_grad_norm (clip=1.0)
        optim="adamw_torch",
        adam_beta1=args.adam_beta1, # 0.9
        adam_beta2=args.adam_beta2, # 0.999
        weight_decay=args.weight_decay, # Use 0.01 as default instead of 0.04
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing, # Enable if needed

        # Logging and saving
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # Hub integration
        push_to_hub=args.push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy=args.hub_strategy,

        # Other args
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False, # Important for custom collators returning extra data
    )

    # --- 7. Instantiate Custom Trainer ---
    logging.info("Initializing Custom DiffusionTrainer...")
    trainer = DiffusionTrainer( # Use the custom trainer
        model=diffusion_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # --- 8. Start Training ---
    logging.info("Starting Diffusion Model training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        logging.info("Training finished.")

        # --- 9. Save Final Model & Stats ---
        logging.info("Saving final model...")
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        logging.info(f"Final diffusion model and state saved to {args.output_dir}")

        if args.push_to_hub:
             logging.info("Pushing final diffusion model to Hub...")
             trainer.push_to_hub(commit_message="Diffusion training completed")
             logging.info("Final diffusion model pushed to Hub.")

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)

# --- Argument Parser ---
if __name__ == "__main__":
    # Set multiprocessing start method for CUDA safety
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Note: Could not set multiprocessing start method ('spawn'): {e}")
        pass

    parser = argparse.ArgumentParser(description="Train Diffusion Model for DGLM.")

    # Paths
    parser.add_argument("--config_path", type=str, default="config/base_config.yaml", help="Path to base config YAML.")
    parser.add_argument("--c4_subset_path", type=str, default="data/raw/c4", help="Path to the downloaded C4 subset directory.")
    parser.add_argument("--output_dir", type=str, default="models/diffusion_model_trained", help="Directory to save checkpoints and final model.")

    # Data/Model Params
    parser.add_argument("--prefix_len", type=int, default=32, help="Length of prefix tokens for splitting.")
    parser.add_argument("--filter_threshold", type=float, default=0.3, help="Token/char ratio threshold for filtering.")
    parser.add_argument("--cfg_mask_probability", type=float, default=0.1, help="Probability of masking prefix for CFG training.")

    # Training Hyperparameters (Defaults based on Table 8, with adjustments noted)
    parser.add_argument("--max_steps", type=int, default=250000, help="Total training steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (Adjusted from 1e-3).")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for LR scheduler.") # Table 8 default
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU (Adjust based on memory).") # Target total 256
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (Adjusted from 0.04).")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")

    # Logging & Saving
    parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Integrations to report results to (e.g., 'tensorboard', 'wandb').")

    # Hub Integration
    parser.add_argument("--push_to_hub", action="store_true", help="Push model checkpoints and final model to Hub.")
    parser.add_argument("--hub_model_name", type=str, default="dglm-diffusion-transformer", help="Name for the model on the Hub.")
    parser.add_argument("--hub_strategy", type=str, default="checkpoint", help="Hub saving strategy.")

    # Other
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Number of workers for DataLoader.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")

    args = parser.parse_args()

    # --- Run Training ---
    train(args)
