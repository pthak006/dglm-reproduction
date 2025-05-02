# data/download_data.py

import os
import logging
from datasets import load_dataset, Dataset, Features, Value, DatasetInfo
import argparse
import itertools # To limit the iterator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataset Configuration ---
# Set limit for C4 based on our pessimistic estimate
# Using 30M as discussed, adjust if needed
C4_RAW_LIMIT = 50_000_000

DATASETS_TO_DOWNLOAD = {
    "openwebtext": {"id": "stas/openwebtext-10k", "config": None, "limit": None, "needs_processing": "Sample version."},
    # "jigsaw_toxicity": {"id": "jigsaw_toxicity_pred", "config":"jigsaw_toxicity_pred", "limit": None, "needs_processing": "Check if this matches the paper's 'Unintended Bias' version."},
    "real_toxicity_prompts": {"id": "allenai/real-toxicity-prompts", "config": None, "limit": None, "needs_processing": None},
    "amazon_polarity": {"id": "amazon_polarity", "config": None, "limit": None, "needs_processing": None},
    "sst2": {"id": "glue", "config": "sst2", "limit": None, "needs_processing": None},
    "ag_news": {"id": "ag_news", "config": None, "limit": None, "needs_processing": None},
    "c4": {
        "id": "allenai/c4",
        "config": "en",
        "limit": C4_RAW_LIMIT, # Limit the number of raw instances to download/save
        "needs_processing": f"Subset of {C4_RAW_LIMIT:,} raw instances. Needs filtering during training."
        # Define expected features for C4 to help from_generator
        # Check allenai/c4 features on Hugging Face Hub if these are wrong
        ,"features": Features({'text': Value('string'), 'timestamp': Value('string'), 'url': Value('string')})
    },
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
        explicit_features = info.get("features") # Get explicit features if provided
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
                # --- Handle limited download using streaming and from_generator ---
                logging.info(f"Streaming dataset '{name}' and saving the first {limit:,} instances.")

                # Define a generator function that yields the limited items
                def limited_generator():
                    logging.info(f"Starting stream for {name}...")
                    streaming_dataset = load_dataset(
                        dataset_id,
                        name=config_name,
                        streaming=True,
                        trust_remote_code=True
                    )
                    split_to_use = streaming_dataset.get('train', streaming_dataset)
                    limited_iterator = itertools.islice(split_to_use, limit)
                    count = 0
                    for item in limited_iterator:
                        yield item
                        count += 1
                        if count % 1_000_000 == 0: # Log progress every million items
                             logging.info(f"-> Yielded {count:,} items for {name}...")
                    logging.info(f"Finished yielding {count:,} items for {name}.")

                # Create the dataset using from_generator
                # Provide features if known, otherwise it tries to infer (which can fail)
                logging.info(f"Creating dataset for '{name}' using from_generator...")
                if explicit_features:
                     logging.info(f"Using explicitly defined features for {name}: {explicit_features}")
                     dataset_to_save = Dataset.from_generator(limited_generator, features=explicit_features)
                else:
                     # Attempt inference (less reliable for complex/nested features)
                     logging.warning(f"No explicit features defined for {name}. Attempting inference by from_generator.")
                     dataset_to_save = Dataset.from_generator(limited_generator)

                logging.info(f"Dataset object created for '{name}'. Saving subset to disk...")

            else:
                # --- Handle full download ---
                logging.info(f"Downloading full dataset '{name}'.")
                dataset_to_save = load_dataset(
                    dataset_id,
                    name=config_name,
                    trust_remote_code=True
                )
                # If it's a DatasetDict, potentially select only 'train' split or save all
                if isinstance(dataset_to_save, dict):
                     logging.warning(f"Downloaded '{name}' is a DatasetDict. Saving all splits found: {list(dataset_to_save.keys())}")
                     # save_to_disk handles DatasetDict directly
                elif not isinstance(dataset_to_save, Dataset):
                     raise TypeError(f"Expected Dataset or DatasetDict, but got {type(dataset_to_save)}")


            # Save the dataset (either full or the generated subset)
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
    logging.info(f"1. C4: Attempted to download and save only the first {C4_RAW_LIMIT:,} raw instances using a memory-efficient method.")
    logging.info("   This subset still needs dynamic filtering during training.")
    logging.info("2. Check Jigsaw: Verify if 'jigsaw_toxicity_pred' corresponds to the paper's dataset.")
    logging.info("3. Disk Space: Monitor usage. The C4 subset will still be large, but much smaller than the full dataset.")
    logging.info("4. Integrity: If downloads were interrupted, remove the respective folder and re-run.")
    logging.info("="*50)
