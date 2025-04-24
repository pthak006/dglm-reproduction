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

## Citation

Lovelace, J., Kishore, V., Chen, Y., & Weinberger, K. Q. (2024). *Diffusion Guided Language Modeling*. arXiv preprint arXiv:2408.04220. [cite: 1]
