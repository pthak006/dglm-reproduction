# Diffusion Guided Language Modeling (DGLM) - Reproduction

This repository contains an implementation attempting to reproduce the results from the paper "Diffusion Guided Language Modeling" by Lovelace et al.[cite: 1].

The goal is to combine the fluency of auto-regressive models with the controllable generation capabilities of diffusion models via learned semantic proposals.

## Project Structure

dglm-reproduction/
├── README.md                 # Project overview, setup, usage
├── requirements.txt          # Python package dependencies
├── config/                   # Configuration files
├── data/                     # Data loading and preprocessing scripts
├── models/                   # Model definitions
├── scripts/                  # Main training, generation, evaluation scripts
├── utils/                    # Utility functions
└── notebooks/                # Jupyter notebooks (optional)

## Initial Setup

Follow these steps to set up the project environment and install necessary dependencies.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Replace with your repo URL
    cd dglm-reproduction
    ```

2.  **Create Conda Environment:**
    This project uses Conda for environment management. Create a new environment named `dglm-env` (or your preferred name) with a specific Python version (e.g., 3.10 or higher is recommended):
    ```bash
    conda create --name dglm-env python=3.10
    ```
    *(Confirm the creation process when prompted).*

3.  **Activate Conda Environment:**
    Before installing dependencies or running scripts, activate the environment:
    ```bash
    conda activate dglm-env
    ```
    *(Your terminal prompt should change to indicate you are now in the `(dglm-env)` environment).*

4.  **Install Dependencies:**
    Install the required Python packages listed in `requirements.txt` using pip within your activated Conda environment:
    ```bash
    pip install -r requirements.txt
    ```

## Loading Foundational Models (Implementation Step 1)

This project relies on pre-trained models from Hugging Face. The script `models/foundational_models.py` handles loading the core Sentence Encoder (e.g., Sentence-T5) and the base Auto-regressive model (e.g., GPT-2 Large) needed for subsequent steps.

1.  **Configuration:**
    Ensure your `config/base_config.yaml` file specifies the correct Hugging Face model identifiers:
    ```yaml
    # config/base_config.yaml
    sentence_encoder_model_name: "google/sentence-t5-xl" # Or other Sentence-T5 variant if needed
    auto_regressive_model_name: "gpt2-large"        # Or other base AR model if needed
    # ... other configurations ...
    ```

2.  **Running the Loader Script:**
    You can test loading the models by running the script directly from the project's root directory. Make sure your `dglm-env` Conda environment is activated.
    ```bash
    python models/foundational_models.py
    ```

    * **Note:** The first time you run this script (or any script that uses these models), the `transformers` library will automatically download the model weights and tokenizer files from the Hugging Face Hub and store them in a local cache (~/.cache/huggingface/hub by default). This might take some time and disk space depending on the model sizes and your internet connection. Subsequent runs will load the models directly from the cache.

This script confirms that your environment can access and load the necessary base models required for the next steps in the DGLM implementation plan.

## Data Acquisition (Implementation Step 2a)

The next step involves acquiring the datasets required for training and evaluation, as listed in the DGLM paper[cite: 112, 117, 119, 120, 151]. The script `data/download_data.py` uses the Hugging Face `datasets` library to download these datasets into a local directory (default: `data/raw/`).

1.  **Datasets Downloaded:**
    The script attempts to download:
    * C4 (`allenai/c4`) - *Note: Requires significant disk space and subsequent subsetting.*
    * OpenWebText (`stas/openwebtext-10k`) - *Note: Sample version.*
    * Jigsaw Toxicity (`jigsaw_toxicity_pred`) - *Note: Verify this matches the paper's exact dataset.*
    * RealToxicityPrompts (`allenai/real-toxicity-prompts`)
    * Amazon Polarity (`amazon_polarity`)
    * SST-2 (`glue/sst2`)
    * AG News (`ag_news`)

2.  **Running the Download Script:**
    Make sure your `dglm-env` Conda environment is active and run the script from the project's root directory:
    ```bash
    python data/download_data.py
    ```
    You can specify a different download directory using the `--download_dir` argument:
    ```bash
    python data/download_data.py --download_dir /path/to/your/large/storage
    ```

3.  **Important Notes:**
    * **Disk Space:** Be aware that C4, in particular, is very large (hundreds of GB). Ensure you have sufficient disk space before running the script.
    * **C4 Subsetting Required:** This script downloads the *full* C4 dataset. You **must** implement separate logic (likely in `data/preprocess.py`) to filter this down to the 10 million instances mentioned in the paper[cite: 112].
    * **Jigsaw Dataset:** Double-check if the downloaded `jigsaw_toxicity_pred` dataset corresponds to the "Jigsaw Unintended Bias" dataset cited in the paper[cite: 119]. You may need to find an alternative source or configuration if it doesn't align.
    * **Interrupted Downloads:** If a download is interrupted, the script might leave a partial folder in the `download_dir`. To ensure a clean download, delete the specific dataset's folder within `download_dir` and re-run the script.

This script handles the initial acquisition of the datasets. The next crucial part of Step 2 is implementing the preprocessing logic.
