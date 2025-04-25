# data/datasets.py

import os
import logging
from torch.utils.data import IterableDataset
from datasets import load_from_disk, Dataset
import random

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class C4TrainingDataset(IterableDataset):
    """
    An iterable dataset that loads and yields text instances from the
    pre-downloaded C4 subset.
    """
    def __init__(self, dataset_path: str, text_field: str = "text"):
        """
        Args:
            dataset_path (str): Path to the directory containing the saved C4 subset
                                (output of download_data.py for 'c4').
            text_field (str): The name of the column containing the text data.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.text_field = text_field

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found at: {dataset_path}")

        try:
            # Load the dataset structure. Actual data loading is deferred.
            logging.info(f"Loading dataset structure from {self.dataset_path}...")
            # Use memory mapping for potentially large datasets if supported
            self.dataset = load_from_disk(self.dataset_path, keep_in_memory=False)
            logging.info("Dataset structure loaded successfully.")

            # Verify the text field exists
            if self.text_field not in self.dataset.column_names:
                 raise ValueError(f"Text field '{self.text_field}' not found in dataset columns: {self.dataset.column_names}")

            # Get dataset size if possible (might not work for all iterable datasets)
            try:
                 self.dataset_size = len(self.dataset)
                 logging.info(f"Dataset size: {self.dataset_size:,} examples.")
            except TypeError:
                 self.dataset_size = None
                 logging.warning("Could not determine dataset size (possibly streaming or non-standard).")


        except Exception as e:
            logging.error(f"Failed to load dataset from {self.dataset_path}: {e}", exc_info=True)
            raise

    def __iter__(self):
        """Yields raw text strings from the dataset."""
        logging.debug(f"Creating iterator for C4TrainingDataset: {self.dataset_path}")
        try:
            # Create a fresh iterator for the dataset each time __iter__ is called
            # This allows multiple epochs or multiple workers if configured correctly
            # Note: For true multi-worker support with IterableDataset, specific worker
            # handling (e.g., splitting shards) might be needed depending on the setup.
            # This basic implementation assumes single-worker or relies on PyTorch's
            # DataLoader handling if workers > 0.
            iterator = iter(self.dataset)
            for instance in iterator:
                text = instance.get(self.text_field)
                if text: # Yield only non-empty text
                    yield text
                else:
                    logging.debug(f"Skipping instance with empty text field: {instance}")
        except Exception as e:
            logging.error(f"Error during dataset iteration: {e}", exc_info=True)
            # Decide whether to raise or just log and stop iteration
            raise StopIteration from e # Stop iteration on error

# --- Example Usage (for testing the dataset class) ---
if __name__ == '__main__':
    # Assume the limited C4 subset was saved here by download_data.py
    # Adjust path if necessary
    DEFAULT_C4_SUBSET_PATH = "data/raw/c4"

    logging.info("Testing C4TrainingDataset...")

    if not os.path.isdir(DEFAULT_C4_SUBSET_PATH):
        logging.error(f"Test dataset path not found: {DEFAULT_C4_SUBSET_PATH}")
        logging.error("Please run download_data.py first or provide the correct path.")
    else:
        try:
            # Instantiate the dataset
            c4_dataset = C4TrainingDataset(dataset_path=DEFAULT_C4_SUBSET_PATH)

            # Iterate through a few examples
            logging.info("Fetching first 5 examples:")
            count = 0
            for text_instance in c4_dataset:
                logging.info(f"Example {count + 1}:\n'''\n{text_instance[:200]}...\n'''") # Print first 200 chars
                count += 1
                if count >= 5:
                    break

            if count == 0:
                 logging.warning("Could not retrieve any examples. Is the dataset empty or text field incorrect?")

            logging.info("Dataset iteration test completed.")

        except Exception as e:
            logging.error(f"An error occurred during dataset testing: {e}", exc_info=True)

