# scripts/train_classifiers.py

import os
import logging
import argparse
import yaml
import joblib # For saving sklearn models
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
# Suppress convergence warnings from sklearn for cleaner logs (optional)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- Embedding Generation ---

def generate_embeddings(
    dataset: Dataset,
    text_column: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128 # Shorter max length for classifiers might be sufficient
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates Sentence-T5 embeddings for the text column of a dataset.

    Args:
        dataset (Dataset): The Hugging Face dataset.
        text_column (str): The name of the column containing text.
        model (PreTrainedModel): The frozen Sentence-T5 model (on target device).
        tokenizer (PreTrainedTokenizer): The Sentence-T5 tokenizer.
        device (torch.device): The device to run the model on (e.g., 'cuda').
        batch_size (int): Batch size for processing.
        max_length (int): Max sequence length for tokenizer.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - embeddings (np.ndarray): NumPy array of shape (num_samples, embed_dim).
            - labels (np.ndarray): NumPy array of corresponding labels.
    """
    embeddings_list = []
    labels_list = []
    model.eval() # Ensure model is in eval mode

    # Check if 'label' column exists, handle potential variations
    label_column = 'label'
    if label_column not in dataset.column_names:
         # Try common alternatives or raise error
         if 'labels' in dataset.column_names:
             label_column = 'labels'
         elif 'score' in dataset.column_names: # Example for toxicity scores
             label_column = 'score'
             logging.warning(f"Using '{label_column}' as label column. Ensure this is intended.")
         else:
              raise ValueError(f"Could not find a suitable label column in dataset: {dataset.column_names}")


    total_batches = (len(dataset) + batch_size - 1) // batch_size
    logging.info(f"Generating embeddings for {len(dataset)} examples in batches of {batch_size}...")

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Embeddings"):
        batch_texts = dataset[i : i + batch_size][text_column]
        batch_labels = dataset[i : i + batch_size][label_column]

        # Filter out potential None or non-string entries if any exist
        valid_indices = [j for j, txt in enumerate(batch_texts) if isinstance(txt, str) and txt.strip()]
        if not valid_indices:
            logging.warning(f"Skipping empty or invalid batch at index {i}")
            continue

        filtered_texts = [batch_texts[j] for j in valid_indices]
        filtered_labels = [batch_labels[j] for j in valid_indices]

        try:
            inputs = tokenizer(
                filtered_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            with torch.no_grad():
                # Use encoder output and mean pooling
                encoder_outputs = model.encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True
                )
                batch_embeddings = encoder_outputs.last_hidden_state.mean(dim=1)

            embeddings_list.append(batch_embeddings.cpu().numpy())
            labels_list.extend(filtered_labels) # Use extend for list of labels

        except Exception as e:
            logging.error(f"Error processing batch starting at index {i}: {e}", exc_info=True)
            # Optionally skip the batch or raise the error depending on desired robustness
            # For now, let's skip
            continue

    if not embeddings_list:
        raise RuntimeError("Failed to generate any embeddings. Check data and model.")

    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.array(labels_list)

    logging.info(f"Generated embeddings shape: {all_embeddings.shape}")
    logging.info(f"Generated labels shape: {all_labels.shape}")

    return all_embeddings, all_labels


# --- Classifier Training & Evaluation ---

def train_and_evaluate_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    attribute_name: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    l2_reg: float = 1e-3, # From Appendix C
    use_class_weight_balanced: bool = False # Set True for toxicity
) -> None:
    """
    Trains, evaluates, and saves a Logistic Regression classifier.

    Args:
        embeddings (np.ndarray): Feature embeddings.
        labels (np.ndarray): Corresponding labels.
        attribute_name (str): Name of the attribute (e.g., 'toxicity', 'sentiment').
        output_dir (str): Directory to save the trained model.
        test_size (float): Proportion of data to use for the test set.
        random_state (int): Random seed for splitting.
        l2_reg (float): L2 regularization strength (C = 1 / l2_reg).
        use_class_weight_balanced (bool): Whether to use balanced class weights.
    """
    logging.info(f"--- Training Classifier for: {attribute_name} ---")
    logging.info(f"Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")

    # Handle potential label preprocessing (e.g., for toxicity scores)
    if attribute_name == 'toxicity':
         # Example: Convert scores > 0.5 to label 1 (toxic), else 0
         threshold = 0.5
         logging.info(f"Applying threshold {threshold} to toxicity labels.")
         labels = (labels >= threshold).astype(int)
         unique_labels, counts = np.unique(labels, return_counts=True)
         logging.info(f"Label distribution after thresholding: {dict(zip(unique_labels, counts))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Define model
    C_value = 1.0 / l2_reg if l2_reg > 0 else float('inf') # Calculate C
    class_weight = 'balanced' if use_class_weight_balanced else None
    model = LogisticRegression(
        C=C_value,
        solver='lbfgs', # Default solver
        max_iter=1000, # Increase max_iter for potential convergence issues
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1 # Use all available CPU cores
    )

    # Train model
    logging.info(f"Training Logistic Regression (C={C_value:.1e}, class_weight={class_weight})...")
    model.fit(X_train, y_train)
    logging.info("Training complete.")

    # Evaluate model
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of class 1
    y_pred = model.predict(X_test)

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        logging.info(f"Test Set AUC ROC: {auc:.4f}")
    except ValueError as e:
        # Handle cases where AUC cannot be calculated (e.g., only one class in y_test)
        logging.warning(f"Could not calculate AUC ROC: {e}")
        auc = float('nan')

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Set Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, f"{attribute_name}_classifier.joblib")
    try:
        joblib.dump(model, model_save_path)
        logging.info(f"Classifier saved to: {model_save_path}")
    except Exception as e:
        logging.error(f"Failed to save classifier model: {e}", exc_info=True)

    logging.info(f"--- Finished Classifier for: {attribute_name} ---")


# --- Main Execution ---

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Config and Sentence Encoder ---
    logging.info(f"Loading base configuration from {args.config_path}")
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        sentence_encoder_model_name = config.get("sentence_encoder_model_name", "google/sentence-t5-xl")
    except Exception as e:
        logging.error(f"Error loading config: {e}", exc_info=True)
        return

    logging.info(f"Loading Sentence Encoder: {sentence_encoder_model_name}")
    try:
        sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_encoder_model_name)
        sentence_encoder = AutoModel.from_pretrained(sentence_encoder_model_name).to(device)
        sentence_encoder.eval() # Set to eval mode
        for param in sentence_encoder.parameters(): # Ensure frozen
            param.requires_grad = False
    except Exception as e:
        logging.error(f"Error loading Sentence Encoder: {e}", exc_info=True)
        return

    # --- Define Classifier Tasks ---
    # Structure: { 'attribute_name': { 'dataset_configs': [ {path, text_col, label_col, split} ], 'is_toxic': bool } }
    classifier_tasks = {
        "toxicity": {
            # Note: Verify 'jigsaw_toxicity_pred' structure and suitability
            "dataset_configs": [
                {"path": os.path.join(args.raw_data_dir, "jigsaw_toxicity"), "text_col": "comment_text", "label_col": "toxic", "split": "train"}
            ],
            "use_class_weight_balanced": True,
            "preprocess_label_func": lambda labels: (np.array(labels) >= 0.5).astype(int) # Example: Binarize toxicity score
        },
        "sentiment": {
            # Combine Amazon Polarity and SST-2 as per paper? Or train separately?
            # Let's combine for now.
            "dataset_configs": [
                {"path": os.path.join(args.raw_data_dir, "amazon_polarity"), "text_col": "content", "label_col": "label", "split": "train"},
                {"path": os.path.join(args.raw_data_dir, "sst2"), "text_col": "sentence", "label_col": "label", "split": "train"}
            ],
            "use_class_weight_balanced": False,
            "preprocess_label_func": None # Labels should be 0/1 already
        },
        # Optional: Add AG News for topic classification
        "topic": {
            "dataset_configs": [
                 {"path": os.path.join(args.raw_data_dir, "ag_news"), "text_col": "text", "label_col": "label", "split": "train"}
            ],
            "use_class_weight_balanced": False, # Typically not needed for AG News
            "preprocess_label_func": None
        }
    }

    # --- Process Each Task ---
    for attribute, task_config in classifier_tasks.items():
        logging.info(f"Processing attribute: {attribute}")
        all_embeddings_list = []
        all_labels_list = []
        combined_dataset = None

        # Load and combine datasets if multiple are specified
        datasets_to_combine = []
        for ds_config in task_config["dataset_configs"]:
            try:
                logging.info(f"Loading dataset from: {ds_config['path']}, split: {ds_config['split']}")
                ds = load_from_disk(ds_config['path'])
                # Select the correct split (e.g., 'train')
                if isinstance(ds, dict):
                    split_ds = ds.get(ds_config['split'])
                    if split_ds is None:
                        raise ValueError(f"Split '{ds_config['split']}' not found in {ds_config['path']}")
                else: # Assume it's a single Dataset object
                    split_ds = ds

                # Rename columns to standard 'text' and 'label' for consistency
                if ds_config['text_col'] != 'text':
                     split_ds = split_ds.rename_column(ds_config['text_col'], 'text')
                if ds_config['label_col'] != 'label':
                     split_ds = split_ds.rename_column(ds_config['label_col'], 'label')

                # Ensure required columns exist after renaming
                if 'text' not in split_ds.column_names or 'label' not in split_ds.column_names:
                     raise ValueError(f"Required columns ('text', 'label') not found in {ds_config['path']} after potential rename. Columns: {split_ds.column_names}")

                datasets_to_combine.append(split_ds)
                logging.info(f"Loaded {len(split_ds)} examples from {ds_config['path']}")

            except FileNotFoundError:
                logging.error(f"Dataset not found at {ds_config['path']}. Skipping this source for {attribute}.")
                continue
            except Exception as e:
                logging.error(f"Error loading or processing dataset {ds_config['path']}: {e}", exc_info=True)
                continue

        if not datasets_to_combine:
            logging.error(f"No valid datasets loaded for attribute '{attribute}'. Skipping training.")
            continue

        # Combine datasets if more than one was loaded successfully
        if len(datasets_to_combine) > 1:
            logging.info(f"Combining {len(datasets_to_combine)} datasets for {attribute}...")
            # Ensure features match before concatenating if possible, otherwise let concatenate handle it
            # Note: Concatenate might fail if features mismatch significantly. Consider mapping features first if needed.
            try:
                combined_dataset = concatenate_datasets(datasets_to_combine)
                logging.info(f"Combined dataset size: {len(combined_dataset)}")
            except Exception as e:
                 logging.error(f"Failed to concatenate datasets for {attribute}: {e}. Skipping training.")
                 continue
        elif datasets_to_combine:
            combined_dataset = datasets_to_combine[0]
        else: # Should not happen due to earlier check, but for safety
             continue


        # Generate embeddings for the combined dataset
        try:
            embeddings, labels = generate_embeddings(
                dataset=combined_dataset,
                text_column='text', # Use standardized column name
                model=sentence_encoder,
                tokenizer=sentence_tokenizer,
                device=device,
                batch_size=args.embedding_batch_size
            )
        except Exception as e:
            logging.error(f"Failed to generate embeddings for {attribute}: {e}", exc_info=True)
            continue # Skip to next attribute if embedding generation fails

        # Preprocess labels if needed
        preprocess_func = task_config.get("preprocess_label_func")
        if preprocess_func:
            try:
                labels = preprocess_func(labels)
            except Exception as e:
                 logging.error(f"Failed to preprocess labels for {attribute}: {e}. Skipping training.")
                 continue

        # Train and evaluate
        try:
            train_and_evaluate_classifier(
                embeddings=embeddings,
                labels=labels,
                attribute_name=attribute,
                output_dir=args.output_dir,
                test_size=args.test_size,
                random_state=args.seed,
                l2_reg=args.l2_reg,
                use_class_weight_balanced=task_config["use_class_weight_balanced"]
            )
        except Exception as e:
            logging.error(f"Failed during training/evaluation for {attribute}: {e}", exc_info=True)
            continue # Skip to next attribute

    logging.info("Classifier training process finished.")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train attribute classifiers on Sentence-T5 embeddings.")

    # Paths
    parser.add_argument("--config_path", type=str, default="config/base_config.yaml", help="Path to base config YAML (to find S-T5 model).")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw", help="Directory containing the downloaded raw attribute datasets.")
    parser.add_argument("--output_dir", type=str, default="models/attribute_classifiers", help="Directory to save trained classifier models.")

    # Parameters
    parser.add_argument("--embedding_batch_size", type=int, default=64, help="Batch size for generating Sentence-T5 embeddings.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for the test set.")
    parser.add_argument("--l2_reg", type=float, default=1e-3, help="L2 regularization strength for Logistic Regression.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # --- Run Main Function ---
    main(args)

