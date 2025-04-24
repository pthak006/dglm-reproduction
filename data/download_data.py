# data/download_data.py

import os
import logging
from datasets import load_dataset, Dataset, Features, Value
import argparse
import itertools # To limit the iterator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataset Configuration ---
# Added 'limit' key for datasets where we only want a subset
# Set limit for C4 based on our pessimistic estimate
C4_RAW_LIMIT = 30_000_000

DATASETS_TO_DOWNLOAD = {
    "c4": {
        "id": "allenai/c4",
        "config": "en",
        "limit": C4_RAW_LIMIT, # Limit the number of raw instances to download/save
        "needs_processing": f"Subset of {C4_RAW_LIMIT:,} raw instances. Needs filtering during training."
    },
    "openwebtext": {"id": "stas/openwebtext-10k", "config": None, "limit": None, "needs_processing": "Sample version."},
    "jigsaw_toxicity": {"id": "jigsaw_toxicity_pred", "config":"jigsaw_toxicity_pred", "limit": None, "needs_processing": "Check if this matches the paper's 'Unintended Bias' version."},
    "real_toxicity_prompts": {"id": "allenai/real-toxicity-prompts", "config": None, "limit": None, "needs_processing": None},
    "amazon_polarity": {"id": "amazon_polarity", "config": None, "limit": None, "needs_processing": None},
    "sst2": {"id": "glue", "config": "sst2", "limit": None, "needs_processing": None},
    "ag_news": {"id": "ag_news", "config": None, "limit": None, "needs_processing": None},
}

# --- Download Function ---

def download_datasets(datasets_info, download_dir="data/raw"):
    """Downloads specified datasets using the Hugging Face datasets library."""
    os.makedirs(download_dir, exist_ok=True)
    logging.info(f"Starting dataset downloads. Target directory: {download_dir}")

    all_successful = True
    for name, info in datasets_info.items():
        dataset_id = info["id"]
        config_name = info.get("config")
        limit = info.get("limit") # Get the limit if specified
        needs_processing = info.get("needs_processing")
        save_path = os.path.join(download_dir, name)

        if os.path.exists(save_path) and os.listdir(save_path):
             logging.info(f"Dataset '{name}' ({dataset_id}) appears to exist at {save_path}. Skipping download.")
             if needs_processing:
                 logging.warning(f"Reminder: Dataset '{name}' needs further processing: {needs_processing}")
             continue

        logging.info(f"Attempting to download dataset: '{name}' ({dataset_id}{'/' + config_name if config_name else ''})")
        try:
            if limit:
                # --- Handle limited download using streaming ---
                logging.info(f"Streaming dataset '{name}' and taking the first {limit:,} instances.")
                # Load in streaming mode
                streaming_dataset = load_dataset(
                    dataset_id,
                    name=config_name,
                    streaming=True,
                    trust_remote_code=True
                )
                # Assuming 'train' split, adjust if needed
                split_to_use = streaming_dataset.get('train', streaming_dataset) # Default to the dataset itself if no 'train' split

                # Take the first 'limit' examples
                limited_iterator = itertools.islice(split_to_use, limit)

                # Convert the limited iterator back to a Dataset object to save
                # This will process 'limit' items and might take time/memory
                logging.info(f"Collecting {limit:,} instances from the stream...")
                limited_data = list(limited_iterator) # Collect items into a list

                if not limited_data:
                     raise ValueError(f"Failed to retrieve any data from the stream for {name}. Check dataset/split name.")

                # Infer features from the first example
                # Note: This assumes features are consistent. Might need explicit Features definition for robustness.
                try:
                    inferred_features = Features.from_dict(limited_data[0])
                    logging.info(f"Inferred features for saving: {inferred_features}")
                except Exception as e:
                     logging.error(f"Could not infer features from first item: {limited_data[0]}. Error: {e}")
                     # Define minimal features manually if inference fails
                     logging.warning("Falling back to basic 'text' feature definition.")
                     inferred_features = Features({'text': Value('string'), 'timestamp': Value('string'), 'url': Value('string')}) # Example for C4

                dataset_to_save = Dataset.from_list(limited_data, features=inferred_features)
                logging.info(f"Collected {len(limited_data):,} instances. Saving subset to disk...")

            else:
                # --- Handle full download ---
                logging.info(f"Downloading full dataset '{name}'.")
                dataset_to_save = load_dataset(
                    dataset_id,
                    name=config_name,
                    trust_remote_code=True
                )

            # Save the dataset (either full or the collected subset)
            dataset_to_save.save_to_disk(save_path)
            logging.info(f"Successfully downloaded and saved dataset '{name}' to {save_path}")
            if needs_processing:
                logging.warning(f"Reminder: Dataset '{name}' needs further processing: {needs_processing}")

        except Exception as e:
            logging.error(f"Failed to download dataset '{name}' ({dataset_id}): {e}", exc_info=True)
            all_successful = False

    if all_successful:
        logging.info("All specified datasets downloaded or already present.")
    else:
        logging.warning("Some datasets failed to download. Check logs for details.")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for DGLM project.")
    parser.add_argument(
        "--download_dir",
        type=str,
        default="data/raw",
        help="Directory to save the downloaded datasets."
    )
    args = parser.parse_args()

    download_datasets(DATASETS_TO_DOWNLOAD, args.download_dir)

    # --- Post-Download Notes ---
    logging.info("="*50)
    logging.info("POST-DOWNLOAD NOTES:")
    logging.info(f"1. C4: Attempted to download and save only the first {C4_RAW_LIMIT:,} raw instances.")
    logging.info("   This subset still needs dynamic filtering during training.")
    logging.info("2. Check Jigsaw: Verify if 'jigsaw_toxicity_pred' corresponds to the paper's dataset.")
    logging.info("3. Disk Space: Monitor usage. The C4 subset will still be large, but much smaller than the full dataset.")
    logging.info("4. Integrity: If downloads were interrupted, remove the respective folder and re-run.")
    logging.info("="*50)
