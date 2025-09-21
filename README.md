# Domain Name Generator

A fine-tuned LLM system for generating domain name suggestions with systematic evaluation, edge case discovery, and iterative improvement.

## Project Overview

This project implements a domain name suggestion system using:
- **Fine-tuned open-source LLM** for domain generation - Mistral 7B-v0.3
- **LLM-as-a-Judge evaluation** framework for systematic scoring using one of OpenAI's models.
- **Iterative improvement** through edge case discovery and dataset augmentation
- **Safety guardrails** for inappropriate content filtering

## Setup Instructions

### Prerequisites
- Python 3.13.5 or higher
- Git
- An OpenAI API key
- A Hugging Face API key
- CUDA-MPS compatible GPU (optional)
- Access to Mistral 7B-v0.3 [Hugging Face Hub](https://huggingface.co/mistralai/Mistral-7B-v0.3)

>**Note for CUDA Users**: If you're using CUDA-enabled GPUs, you may need to install a specific version of PyTorch that's compatible with your CUDA version. Please visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to generate the correct installation command for your system.


### 1. Clone and Navigate to Repository
```bash
git clone https://github.com/ohmatheus/domain.yours.git
cd domain.yours
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables
The project uses environment variables for API credentials and configuration. Set up your environment by copying the example file:

```bash
# Copy the example environment file
cp .env_example .env

# Edit .env with your actual credentials
# The .env file is already in .gitignore and won't be tracked
```

#### Required Environment Variables

The project's settings system (defined in `src/settings.py`) supports the following variables:

- **`HUGGINGFACE_API_TOKEN`**: Your Hugging Face API token for accessing Mistral
- **`OPENAI_API_KEY`**: Your OpenAI API key for LLM-as-a-Judge evaluation
- **`DEVICE`** (Optional): Specify the device for model inference (e.g., "cuda", "cpu", "mps")

#### Example .env file:
```bash
HUGGINGFACE_API_TOKEN=your_hf_token_here
OPENAI_API_KEY=your_openai_api_key_here
DEVICE=cuda
```

## Training

The project includes a command-line training system that allows you to train domain generation models on different dataset versions.

### Training Scripts

- **`src/main.py`**: Main command-line interface for training operations
- **`src/train.py`**: Core training functionality with dataset validation and model configuration

### Training Commands

#### Train a Specific Version
Train a model on a specific dataset version (e.g., v1, v2, v3):

```bash
python main.py train --version v1
```

```bash
python main.py train --version v2
```

#### Train All Available Versions
Train models on all available dataset versions automatically:

```bash
python main.py train --version all
```

#### Advanced Options
Stop training all versions if one fails (when using `--version all`):

```bash
python main.py train --version all --stop-on-error
```

### Dataset Requirements

The training system expects dataset files to follow this naming convention:
- `data/dataset_v1.csv`
- `data/dataset_v2.csv`
- `data/dataset_v3.csv`
- etc.

Each dataset file must contain:
- `description`: Business description column
- `suggestions`: JSON array of 5 domain suggestions (or empty array)

### Training Configuration

The training uses the following configuration:
- **Model**: Mistral-7B-Instruct-v0.3
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Training Framework**: SFT (Supervised Fine-Tuning) with TRL
- **Validation Split**: 10% of training data
- **Early Stopping**: Enabled with patience of 2 epochs
- **Output**: Models saved to `models/model_{version}/`

## Evaluation

The project includes a command-line evaluation system that allows you to evaluate trained models against a test dataset using the LLM-as-a-Judge framework.

### Evaluation Scripts

- **`src/main.py`**: Main command-line interface for evaluation operations
- **`src/model_eval.py`**: Core evaluation functionality with domain generation and scoring

### Evaluation Commands

#### Evaluate a Specific Model Version
Evaluate a trained model on a specific version (e.g., v1, v2, v3):

```bash
python main.py eval --version v1
```

```bash
python main.py eval --version v2
```

#### Evaluate All Available Models
Evaluate all available trained models automatically:

```bash
python main.py eval --version all
```

#### Advanced Options
Stop evaluating all models if one fails (when using `--version all`):

```bash
python main.py eval --version all --stop-on-error
```

### Evaluation Requirements

The evaluation system expects:
- **Trained models**: Located in `models/model_{version}/` directories with valid config.json files
- **Test dataset**: `data/test_set.csv` file with business descriptions
- **Environment variables**: OpenAI API key for LLM-as-a-Judge evaluation (see Setup Instructions)

### Evaluation Process

1. **Model Loading**: Loads the trained LoRA adapter and merges it with the base Mistral model
2. **Domain Generation**: Generates domain suggestions for each test case
3. **LLM Evaluation**: Uses GPT-4 to score each generated domain on multiple criteria
4. **Results Export**: Saves detailed evaluation results to `data/model_{version}-results.csv`

### Evaluation Output

The evaluation results include:
- **Domain-level scores**: Relevance, creativity, brandability, and conciseness (1-5 scale)
- **Category counts**: Good, OK, random words, too long, failures, and inappropriate domains
- **Overall metrics**: Average scores and appropriateness classification
- **Detailed CSV**: Complete results saved for further analysis

## Analysis and Model Improvement

The project includes comprehensive analysis notebooks that demonstrate the iterative improvement process:

### Analysis Notebooks

- **`notebooks/analyse_v1.ipynb`**: Detailed analysis of model v1 performance
- **`notebooks/analyse_v2.ipynb`**: Comparative analysis of model v2 improvements
