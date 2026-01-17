#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for Molecule Captioning Stages.
Runs a single batch of Stage 1 and Stage 2 to verify dimensionality and pipeline.
"""

import os
# CRITICAL: This MUST be set before any other imports
os.environ["NPY_DISABLE_ARRAY_API"] = "1"

import torch
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from config import get_config
from model_wrapper import create_model
from train_stage1 import train_stage1
from train_stage2 import train_stage2
from dataset_caption import prepare_alignment_dataloaders

def test_pipeline():
    print("=== Pipeline Verification Started ===")
    
    # 1. Setup Config (Quick mode)
    config = get_config(mode="quick")
    config.use_wandb = False  # Disable wandb for test
    config.stage1_epochs = 1
    config.stage2_epochs = 1
    config.train_subset = 40  # Small subset (>= stage1_batch_size)
    config.val_subset = 10
    config.num_workers = 0    # DISABLE MULTIPROCESSING FOR TEST
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 2. Create Model
    print("\n[Step 1/3] Creating Model...")
    model = create_model(config, device=device)
    print("Model created successfully.")

    # 3. Test Stage 1 (Alignment)
    print("\n[Step 2/3] Testing Stage 1 Alignment...")
    try:
        metrics_s1, final_s1_step = train_stage1(model, config, logger=None)
        print(f"Stage 1 execution: SUCCESS (Final Step: {final_s1_step})")
    except Exception as e:
        print(f"Stage 1 execution: FAILED")
        import traceback
        traceback.print_exc()
        return

    # 4. Test Stage 2 (SFT)
    print("\n[Step 3/3] Testing Stage 2 SFT...")
    try:
        # Ensure checkpoint exists for stage 2 load (or skip load)
        metrics_s2 = train_stage2(model, config, logger=None, load_stage1=False)
        print("Stage 2 execution: SUCCESS")
    except Exception as e:
        print(f"Stage 2 execution: FAILED")
        import traceback
        traceback.print_exc()
        return

    print("\n=== Pipeline Verification Completed Successfully ===")

if __name__ == "__main__":
    test_pipeline()
