# scripts/train_decoder.py

import os
import logging
import argparse
import yaml
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
    from models.prompt_generator import PromptGenerator
    from data.datasets import C4TrainingDataset
    from data.data_collator import DataCollatorForDecoderFineTuning
except ImportError as e:
    logging.error(f"Failed to import custom modules: {e}")
    logging.error("Ensure models/prompt_generator.py, data/datasets.py, data/data_collator.py exist.")
    exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# --- Combined Model Wrapper ---

class DecoderWithPromptGenerator(PreTrainedModel):
    """
    A wrapper model that combines the AutoRegressive Decoder and the PromptGenerator.
    This allows the Trainer to optimize both simultaneously and handle saving/loading.
    """
    # Use the config class of the base decoder model
    config_class = AutoModelForCausalLM.config_class

    def __init__(self, config, decoder: PreTrainedModel, prompt_generator: PromptGenerator):
        super().__init__(config)
        self.decoder = decoder
        self.prompt_generator = prompt_generator
        # Ensure the main config reflects the decoder's config for compatibility
        self.config = decoder.config

    def forward(
        self,
        input_embeds=None,
        attention_mask=None,
        labels=None,
        # Add any other arguments the underlying decoder might need
        **kwargs
    ):
        """
        Forward pass simply directs the inputs (prepared by the collator)
        to the underlying decoder model.
        """
        # The collator prepares input_embeds which includes the soft prompt.
        # The prompt_generator itself is used *within* the collator, not here.
        return self.decoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    # Implement saving/loading if needed, although Trainer might handle it
    # if submodules are correctly registered. Let's rely on Trainer for now.
    # If saving fails, we might need custom save_pretrained/from_pretrained.

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
        decoder_model_name = config.get("auto_regressive_model_name", "gpt2-large")
        sentence_encoder_model_name = config.get("sentence_encoder_model_name", "google/sentence-t5-xl")
        filter_tokenizer_name = config.get("filter_tokenizer_name", "gpt2") # Tokenizer for filtering
    except Exception as e:
        logging.error(f"Error loading config: {e}", exc_info=True)
        return

    # --- 2. Load Models and Tokenizers ---
    logging.info("Loading models and tokenizers...")
    try:
        decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if decoder_tokenizer.pad_token is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
            logging.info(f"Set decoder pad_token to eos_token ({decoder_tokenizer.eos_token})")

        filter_tokenizer = AutoTokenizer.from_pretrained(filter_tokenizer_name)

        sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_encoder_model_name)
        # Use AutoModel for Sentence T5 as we only need the encoder embeddings
        sentence_encoder = AutoModel.from_pretrained(sentence_encoder_model_name).to(device)

        decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name).to(device)

        # Freeze Sentence-T5
        logging.info(f"Freezing Sentence Encoder ({sentence_encoder_model_name}) parameters.")
        for param in sentence_encoder.parameters():
            param.requires_grad = False
        sentence_encoder.eval() # Set to eval mode

    except Exception as e:
        logging.error(f"Error loading models/tokenizers: {e}", exc_info=True)
        return

    # --- 3. Instantiate Prompt Generator ---
    logging.info("Instantiating PromptGenerator...")
    try:
        # Infer dimensions (adjust if needed based on actual loaded models)
        sent_emb_dim = sentence_encoder.config.d_model # e.g., 1024 for T5-XL
        dec_emb_dim = decoder.config.n_embd # e.g., 1280 for GPT2-Large
        n_heads = decoder.config.n_head # e.g., 20 for GPT2-Large

        prompt_generator = PromptGenerator(
            sentence_emb_dim=sent_emb_dim,
            decoder_emb_dim=dec_emb_dim,
            k_soft_tokens=args.k_soft_tokens,
            n_layers=args.prompt_gen_layers,
            n_heads=n_heads, # Use decoder's head count
            dropout=args.dropout
        ).to(device)
    except Exception as e:
        logging.error(f"Error instantiating PromptGenerator: {e}", exc_info=True)
        return

    # --- 4. Create Combined Model ---
    logging.info("Creating combined DecoderWithPromptGenerator model...")
    # Pass the decoder's config to the wrapper
    combined_model = DecoderWithPromptGenerator(
        config=decoder.config,
        decoder=decoder,
        prompt_generator=prompt_generator
    ).to(device)
    logging.info(f"Combined model created. Device: {combined_model.device}")

    # --- 5. Load Dataset and Collator ---
    logging.info("Loading dataset and initializing data collator...")
    try:
        train_dataset = C4TrainingDataset(dataset_path=args.c4_subset_path)

        data_collator = DataCollatorForDecoderFineTuning(
            decoder_tokenizer=decoder_tokenizer,
            filter_tokenizer=filter_tokenizer,
            sentence_encoder=sentence_encoder, # Pass the frozen model
            sentence_tokenizer=sentence_tokenizer,
            prompt_generator=prompt_generator, # Pass the trainable prompt generator
            decoder_embedding_layer=decoder.get_input_embeddings(), # Pass the embedding layer
            prefix_len=args.prefix_len,
            k_soft_tokens=args.k_soft_tokens,
            max_seq_len=args.max_seq_len,
            filter_threshold=args.filter_threshold,
            noise_schedule_s=args.noise_schedule_s
        )
    except Exception as e:
        logging.error(f"Error setting up dataset/collator: {e}", exc_info=True)
        return

    # --- 6. Hugging Face Hub Login (if pushing) ---
    if args.push_to_hub:
        try:
            token = HfFolder.get_token()
            if token is None:
                logging.error("Push to Hub requested, but not logged in. Please run `huggingface-cli login`.")
                # Optionally, prompt for login here, but CLI is safer
                return
            user_info = whoami(token=token)
            logging.info(f"Logged in to Hugging Face Hub as: {user_info['name']}")
            hub_model_id = f"{user_info['name']}/{args.hub_model_name}" # Construct full ID
            logging.info(f"Will push model to: {hub_model_id}")
        except Exception as e:
            logging.error(f"Error checking Hugging Face login status: {e}")
            return
    else:
        hub_model_id = None

    # --- 7. Configure Training Arguments ---
    logging.info("Configuring Training Arguments...")
    # Calculate gradient accumulation steps
    eff_batch_size = args.per_device_train_batch_size * torch.cuda.device_count() if torch.cuda.is_available() else args.per_device_train_batch_size
    target_total_batch_size = 64 # From paper (Table 7)
    if target_total_batch_size % eff_batch_size != 0:
        logging.warning(f"Target batch size {target_total_batch_size} not divisible by effective batch size {eff_batch_size}. Adjusting gradient accumulation.")
    gradient_accumulation_steps = max(1, target_total_batch_size // eff_batch_size)
    logging.info(f"Effective per-device batch size: {args.per_device_train_batch_size}")
    logging.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    logging.info(f"Target total batch size: {args.per_device_train_batch_size * gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else args.per_device_train_batch_size * gradient_accumulation_steps}")


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"), # Log to subfolder
        report_to=args.report_to.split(',') if args.report_to else None, # wandb or tensorboard

        # Training strategy
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=args.gradient_clipping,
        optim="adamw_torch", # Use PyTorch AdamW
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(), # Enable FP16 if CUDA is available

        # Logging and saving
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, # Limit total checkpoints

        # Hub integration
        push_to_hub=args.push_to_hub,
        hub_model_id=hub_model_id, # Use the constructed ID
        hub_strategy=args.hub_strategy, # "checkpoint", "end", "all_checkpoints"

        # Other args
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers, # Can speed up data loading
        # evaluation_strategy="no", # No eval setup yet
        # dataloader_pin_memory=True, # Might improve GPU transfer speed
        remove_unused_columns=False, # Important for custom collators
    )

    # --- 8. Instantiate Trainer ---
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=combined_model, # Use the wrapper model
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=None, # No eval dataset configured yet
        data_collator=data_collator,
        # tokenizer=decoder_tokenizer # Optional: pass tokenizer for auto-saving
    )

    # --- 9. Start Training ---
    logging.info("Starting training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        logging.info("Training finished.")

        # --- 10. Save Final Model & Stats ---
        logging.info("Saving final model...")
        trainer.save_model() # Saves to output_dir
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        logging.info(f"Final model and state saved to {args.output_dir}")

        # Final push to Hub if enabled
        if args.push_to_hub:
             logging.info("Pushing final model to Hub...")
             trainer.push_to_hub(commit_message="Training completed")
             logging.info("Final model pushed to Hub.")

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        # Optionally save checkpoint on interrupt
        # trainer.save_model(os.path.join(args.output_dir, "interrupt_checkpoint"))
        # logging.info("Saved interrupt checkpoint.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Decoder and Prompt Generator for DGLM.")

    # Paths
    parser.add_argument("--config_path", type=str, default="config/base_config.yaml", help="Path to base config YAML.")
    parser.add_argument("--c4_subset_path", type=str, default="data/raw/c4", help="Path to the downloaded C4 subset directory.")
    parser.add_argument("--output_dir", type=str, default="models/decoder_promptgen_finetuned", help="Directory to save checkpoints and final model.")

    # Model & Data Params (Match Table 7 and Collator)
    parser.add_argument("--k_soft_tokens", type=int, default=8, help="Number of soft prompt tokens.")
    parser.add_argument("--prompt_gen_layers", type=int, default=6, help="Number of layers in PromptGenerator Transformer.")
    parser.add_argument("--prefix_len", type=int, default=32, help="Length of prefix tokens.")
    parser.add_argument("--max_seq_len", type=int, default=96, help="Max sequence length for decoder input.")
    parser.add_argument("--filter_threshold", type=float, default=0.3, help="Token/char ratio threshold for filtering.")
    parser.add_argument("--noise_schedule_s", type=float, default=3.0, help="Offset 's' for cosine noise schedule.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for PromptGenerator.")

    # Training Hyperparameters (Match Table 7)
    parser.add_argument("--max_steps", type=int, default=250000, help="Total training steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warmup steps for LR scheduler.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU.") # Adjust based on memory
    # gradient_accumulation_steps calculated automatically based on target=64
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="AdamW beta2.") # Note: Paper table 8 has 0.999 for diffusion? Table 7 has 0.99. Using 0.99.
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Logging & Saving
    parser.add_argument("--logging_steps", type=int, default=100, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Integrations to report results to (e.g., 'tensorboard', 'wandb', 'none'). Comma-separated.")

    # Hub Integration
    parser.add_argument("--push_to_hub", action="store_true", help="Push model checkpoints and final model to Hugging Face Hub.")
    parser.add_argument("--hub_model_name", type=str, default="dglm-decoder-promptgen", help="Name for the model on the Hub (username prefixed automatically).")
    parser.add_argument("--hub_strategy", type=str, default="checkpoint", help="Hub saving strategy ('checkpoint', 'end', 'all_checkpoints').")

    # Other
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Number of workers for DataLoader.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")


    args = parser.parse_args()

    # --- Run Training ---
    train(args)
