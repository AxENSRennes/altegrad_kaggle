#!/usr/bin/env python3
"""Stage 2 training script for Kaggle TPU via kaggle kernels push."""

import subprocess
import os

# === Setup ===
os.chdir("/kaggle/working")

# Configure W&B authentication
os.environ["WANDB_API_KEY"] = "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"

# Install git-lfs
subprocess.run(["apt-get", "install", "-y", "git-lfs"], check=True)
subprocess.run(["git", "lfs", "install"], check=True)

# Clone the code repository
subprocess.run([
    "git", "clone", "https://github.com/AxENSRennes/altegrad_kaggle.git"
], check=True)

# Clone HuggingFace checkpoints (contains stage1 checkpoint + GNN)
subprocess.run([
    "git", "clone",
    "https://huggingface.co/Moinada/altegrad-mol-caption",
    "altegrad_kaggle/mol-caption-code/hf_checkpoints"
], check=True)

# Install dependencies
# Note: torch and torch_xla are pre-installed on Kaggle TPU VMs
subprocess.run([
    "pip", "install", "-q",
    "accelerate", "transformers", "peft", "trl", "wandb", "nltk"
], check=True)

# Install PyTorch Geometric (compatible with pre-installed torch version)
subprocess.run([
    "pip", "install", "-q",
    "torch_geometric"
], check=True)

subprocess.run([
    "pip", "install", "-q",
    "pyg_lib", "torch_scatter", "torch_sparse",
    "-f", "https://data.pyg.org/whl/torch-2.1.0+cpu.html"
], check=True)

# Change to the training directory
os.chdir("/kaggle/working/altegrad_kaggle/mol-caption-code")

# === Run Stage 2 Training ===
subprocess.run([
    "accelerate", "launch",
    "--config_file", "accelerate_config_tpu.yaml",
    "run.py",
    "--mode", "full",
    "--hardware", "tpu",
    "--skip-stage1",
    "--wandb"
], check=True)

print("Training complete!")
print("Outputs saved to /kaggle/working/")
