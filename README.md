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

## Step 3: Fine-tuning Decoder & Prompt Generator (Semantic Conditioning)

This stage implements Section 3.1 of the DGLM paper, where the auto-regressive decoder (e.g., GPT-2 Large) is fine-tuned simultaneously with a `PromptGenerator` model. The goal is to teach the decoder to generate text continuations based on semantic information provided via soft prompts, simulating the eventual output of the diffusion model.

**Components Involved:**

* **`models/prompt_generator.py`**: Defines the Transformer-based Prompt Generator architecture.
* **`data/datasets.py`**: Contains `C4TrainingDataset` to stream C4 data.
* **`data/data_collator.py`**: Includes `DataCollatorForDecoderFineTuning`, which dynamically filters C4 data, applies noise augmentation to ground-truth continuation embeddings (from a frozen Sentence-T5), generates soft prompts using the `PromptGenerator`, and prepares batches for the decoder.
* **`scripts/train_decoder.py`**: Orchestrates the fine-tuning process using the Hugging Face `Trainer` and includes the `DecoderWithPromptGenerator` wrapper class to manage joint optimization.

**Running the Fine-tuning Script:**

1.  **Activate Environment:**
    ```bash
    conda activate dglm-env
    ```
2.  **Login (Optional but Recommended):**
    * For WandB logging: `wandb login`
    * For pushing to Hub: `huggingface-cli login`
3.  **Execute Training:**
    Run the training script from the project root directory. Adjust `--per_device_train_batch_size` based on your GPU memory.
    ```bash
    python scripts/train_decoder.py \
        --output_dir models/decoder_promptgen_finetuned_run1 \
        --c4_subset_path data/raw/c4 \
        --per_device_train_batch_size 2 \
        --report_to wandb \
        --logging_steps 100 \
        --save_steps 5000 \
        --max_steps 250000 \
        # --push_to_hub \ # Uncomment to enable pushing to Hub
        # --hub_model_name dglm-decoder-promptgen-v1 # Optional: specific Hub name
    ```
    * Monitor GPU memory; decrease batch size (e.g., to 1) if needed. Gradient accumulation is handled automatically to target an effective batch size near 64.
    * Check other arguments in the script (e.g., learning rate, warmup steps) and adjust if necessary.

**Monitoring Training (WandB Example):**

* If using `--report_to wandb`, the script will output a URL like `https://wandb.ai/<your-username>/<project-name>/runs/<run-id>`.
* Open this URL in your browser.
* Navigate to the "Charts" section and observe the `train/loss` plot. The loss should decrease over training steps.

**Expected Outcome:**

* Training logs and checkpoints will be saved locally in the specified `--output_dir`.
* If `--push_to_hub` is enabled, checkpoints and the final model (containing both the fine-tuned decoder and the trained prompt generator) will be uploaded to your Hugging Face Hub repository.
* This fine-tuned decoder and prompt generator are essential inputs for the subsequent steps involving the diffusion model.

## Step 4: Training the Diffusion Model

This stage implements Section 3.2 of the DGLM paper, training the Transformer-based diffusion model that learns to predict semantic embeddings of text continuations, conditioned on prefix embeddings. This model operates in the Sentence-T5 latent space and uses a v-prediction objective with a custom loss weighting.

**Components Involved:**

* **`models/diffusion_network.py`**: Defines the `DiffusionTransformer` architecture, including input/output processing, time conditioning, and the learnable null embedding for Classifier-Free Guidance (CFG).
* **`data/datasets.py`**: Reuses `C4TrainingDataset` to stream C4 data.
* **`data/diffusion_collator.py`**: Includes `DataCollatorForDiffusionTraining`, which filters C4 data, gets prefix/continuation embeddings from Sentence-T5, applies noise using a cosine schedule, calculates the target velocity (`v`), performs CFG masking, and prepares batches for the diffusion model.
* **`scripts/train_diffusion.py`**: Orchestrates the training process, including the custom `DiffusionTrainer` subclass that implements the weighted v-prediction loss function.

**Running the Diffusion Training Script:**

1.  **Activate Environment:**
    ```bash
    conda activate dglm-env
    ```
2.  **Login (Optional but Recommended):**
    * For WandB logging: `wandb login`
    * For pushing to Hub: `huggingface-cli login`
3.  **Execute Training:**
    Run the training script from the project root directory. Adjust `--per_device_train_batch_size` and consider enabling `--gradient_checkpointing` based on your GPU memory. The target effective batch size is 256 (using gradient accumulation).
    ```bash
    python -m scripts.train_diffusion \
        --output_dir models/diffusion_model_trained_run1 \
        --c4_subset_path data/raw/c4 \
        --per_device_train_batch_size 4 \
        --gradient_checkpointing \
        --report_to wandb \
        --logging_steps 100 \
        --save_steps 10000 \
        --max_steps 250000 \
        # --push_to_hub \ # Uncomment to enable pushing to Hub
        # --hub_model_name dglm-diffusion-transformer-v1 # Optional: specific Hub name
    ```
    * Monitor GPU memory closely; batch size might need to be 1 or 2.
    * Check hyperparameters in the script against Table 8 in the paper.

**Monitoring Training (WandB Example):**

* If using `--report_to wandb`, open the run URL provided in the terminal logs.
* Monitor the `train/loss` chart, which represents the custom weighted v-prediction loss. It should decrease over time.

**Expected Outcome:**

* Training logs and checkpoints of the `DiffusionTransformer` (including the null embedding) will be saved locally in the specified `--output_dir`.
* If `--push_to_hub` is enabled, checkpoints and the final model will be uploaded to your Hugging Face Hub repository.
* This trained diffusion model is ready to generate semantic proposals during inference.

## Step 5: Training Attribute Classifiers

This stage implements Section 3.3 and Appendix C, training simple Logistic Regression classifiers to predict attributes (like toxicity, sentiment, topic) directly from Sentence-T5 embeddings. These classifiers enable plug-and-play guidance during diffusion sampling.

**Components Involved:**

* **`scripts/train_classifiers.py`**: Contains logic to:
    * Load attribute-specific datasets (Jigsaw, Amazon Polarity, SST-2, AG News) downloaded previously.
    * Generate Sentence-T5 embeddings for the text data in these datasets using the frozen Sentence-T5 model.
    * Train `scikit-learn` Logistic Regression models using these embeddings and the corresponding labels.
    * Evaluate the classifiers (optional but recommended).
    * Save the trained classifiers using `joblib`.

**Running the Classifier Training Script:**

1.  **Activate Environment:**
    ```bash
    conda activate dglm-env
    ```
2.  **Verify Datasets:** Ensure the raw attribute datasets exist in the `--raw_data_dir` (default: `data/raw`) and that the column names specified within the script match the actual dataset structure.
3.  **Execute Training:**
    Run the script from the project root directory.
    ```bash
    python scripts/train_classifiers.py \
        --raw_data_dir data/raw \
        --output_dir models/attribute_classifiers \
        # --embedding_batch_size 64 # Optional: Adjust batch size for embedding generation
    ```

**Expected Outcome:**

* Trained `scikit-learn` Logistic Regression models saved as `.joblib` files (e.g., `toxicity_classifier.joblib`, `sentiment_classifier.joblib`) in the specified `--output_dir`.
* These saved classifiers will be loaded during the inference stage to guide the diffusion model's proposal generation.

