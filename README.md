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

## Usage (To be added)

*(This section will later describe how to run data preprocessing, training, generation, and evaluation scripts).*

## Citation

Lovelace, J., Kishore, V., Chen, Y., & Weinberger, K. Q. (2024). *Diffusion Guided Language Modeling*. arXiv preprint arXiv:2408.04220. [cite: 1]
