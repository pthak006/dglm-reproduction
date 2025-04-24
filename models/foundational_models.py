# models/foundational_models.py

import yaml
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---

def load_config(config_path="config/base_config.yaml"):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise

# --- Model Loading Functions ---

def load_sentence_encoder(model_name: str, device: str):
    """
    Loads the Sentence Encoder model and tokenizer.
    Uses AutoModel for Sentence-T5 as it's primarily used for embeddings.
    """
    logging.info(f"Loading Sentence Encoder: {model_name} onto device: {device}")
    try:
        # === ACTUAL LOADING CODE FOR SENTENCE ENCODER ===
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        # =================================================

        model.eval() # Set to evaluation mode by default
        logging.info(f"Sentence Encoder {model_name} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load Sentence Encoder {model_name}: {e}")
        raise

def load_auto_regressive_model(model_name: str, device: str):
    """
    Loads the Auto-regressive model (for causal LM) and tokenizer.
    Uses AutoModelForCausalLM for models like GPT-2 used for generation.
    """
    logging.info(f"Loading Auto-regressive Model: {model_name} onto device: {device}")
    try:
        # === ACTUAL LOADING CODE FOR AUTO-REGRESSIVE MODEL ===
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        # ====================================================

        # GPT-2 Tokenizer requires a padding token for batch processing if not set
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
             tokenizer.pad_token = tokenizer.eos_token
             # Also update model config if necessary (often handled internally, but good practice)
             model.config.pad_token_id = tokenizer.pad_token_id
             logging.info(f"Set pad_token to eos_token ({tokenizer.eos_token}) for {model_name} tokenizer and model config.")
        elif tokenizer.pad_token is None:
             logging.warning(f"Tokenizer for {model_name} has no pad_token or eos_token. Batch generation might fail without padding.")


        model.eval() # Set to evaluation mode by default, change during training
        logging.info(f"Auto-regressive Model {model_name} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load Auto-regressive Model {model_name}: {e}")
        raise

# --- Example Usage ---

if __name__ == "__main__":
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    # Example for MPS (M1/M2 Macs) - uncomment if needed
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")

    # Load configuration
    try:
        # Ensure the config file exists relative to where you run the script
        # Or provide an absolute path
        script_dir = os.path.dirname(__file__) # Gets directory of the script
        config_path = os.path.join(script_dir, '..', 'config', 'base_config.yaml') # Navigate up and into config
        config = load_config(config_path=config_path)

        # Load Sentence Encoder
        encoder_model_name = config.get("sentence_encoder_model_name")
        if encoder_model_name:
            sentence_encoder, sentence_tokenizer = load_sentence_encoder(encoder_model_name, device)
            logging.info(f"Successfully retrieved Sentence Encoder: {type(sentence_encoder)}")
        else:
            logging.warning("sentence_encoder_model_name not found in config.")

        # Load Auto-regressive Model
        ar_model_name = config.get("auto_regressive_model_name")
        if ar_model_name:
            ar_model, ar_tokenizer = load_auto_regressive_model(ar_model_name, device)
            logging.info(f"Successfully retrieved Auto-regressive Model: {type(ar_model)}")
        else:
            logging.warning("auto_regressive_model_name not found in config.")

        logging.info("Model loading example finished.")

    except FileNotFoundError as e:
        logging.error(f"Setup failed: {e}")
    except Exception as e:
        logging.error(f"An error occurred during model loading test: {e}", exc_info=True)