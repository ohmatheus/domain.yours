# Domain Name Generator - ML Interview Project

A fine-tuned LLM system for generating domain name suggestions with systematic evaluation, edge case discovery, and iterative improvement.

## Project Overview

This project implements a domain name suggestion system using:
- **Fine-tuned open-source LLM** for domain generation - Mistral 7B-v0.1
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
- Access to Mistral 7B-v0.1 [Hugging Face Hub](https://huggingface.co/mistralai/Mistral-7B-v0.1)

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
- **`DOMAIN_COUNT`** (Optional): Number of domain name suggestions to generate (default: 5)

#### Example .env file:
```bash
HUGGINGFACE_API_TOKEN=your_hf_token_here
OPENAI_API_KEY=your_openai_api_key_here
DEVICE=cuda
```
