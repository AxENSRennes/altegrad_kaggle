#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for molecular captioning training and inference.

Usage:
    # Quick test (~5 min)
    python run.py --mode quick

    # Medium test (~1 hour)
    python run.py --mode medium

    # Full training (~9 hours)
    python run.py --mode full

    # Inference only
    python run.py --inference --checkpoint outputs/stage2_best.pt
"""

import argparse
import torch

from config import get_config
from model_wrapper import create_model
from train_stage1 import train_stage1
from train_stage2 import train_stage2
from inference import run_inference
from utils import set_seed, WandBLogger, ensure_dir
from report import print_config_summary


def train_full_pipeline(
    mode: str = "quick",
    use_wandb: bool = False,
    skip_stage1: bool = False,
):
    """
    Run the full training pipeline.

    Args:
        mode: Experiment mode ("quick", "medium", "full")
        use_wandb: Whether to log to W&B
        skip_stage1: Skip Stage 1 alignment (use for continuing training)
    """
    # Setup
    config = get_config(mode=mode)
    config.use_wandb = use_wandb
    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 60}")
    print(f"Molecular Captioning Training Pipeline")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    print_config_summary(config)

    # Ensure output directory exists
    ensure_dir(config.output_dir + "/")

    # Create model
    print("\nCreating model...")
    model = create_model(config, device=device)
    model.print_trainable_parameters()

    # Setup W&B
    logger = WandBLogger(enabled=config.use_wandb)
    if config.use_wandb:
        logger.init(config.wandb_project, config, tags=[mode])

    # Stage 1: Alignment
    if not skip_stage1:
        print("\n" + "=" * 60)
        print("STAGE 1: Alignment Training")
        print("=" * 60)
        stage1_metrics, stage1_final_step = train_stage1(model, config, logger)
        print(f"Stage 1 complete: val_loss={stage1_metrics['val_loss']:.4f}")
    else:
        print("\nSkipping Stage 1 (using existing checkpoint)")
        stage1_final_step = 0

    # Stage 2: SFT
    print("\n" + "=" * 60)
    print("STAGE 2: Supervised Fine-Tuning")
    print("=" * 60)
    stage2_metrics = train_stage2(model, config, logger, load_stage1=True, start_step=stage1_final_step)
    print(f"Stage 2 complete: bleu4={stage2_metrics['bleu4']:.2f}")

    # Inference (full mode only)
    if mode == "full":
        print("\n" + "=" * 60)
        print("INFERENCE: Generating Submission")
        print("=" * 60)
        run_inference(config)

    # Cleanup
    if logger:
        logger.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best checkpoint: {config.stage2_checkpoint_path}")
    if mode == "full":
        print(f"Submission: {config.submission_path}")
    print("=" * 60 + "\n")

    return stage2_metrics


def main():
    parser = argparse.ArgumentParser(description="Molecular Captioning Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "medium", "full"],
        help="Experiment mode"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 alignment training"
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference only"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output CSV for inference"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit inference to first N molecules (for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    args = parser.parse_args()

    if args.inference:
        # Inference only
        config = get_config(mode=args.mode)
        run_inference(
            config, 
            checkpoint_path=args.checkpoint, 
            output_path=args.output, 
            limit=args.limit,
            batch_size=args.batch_size
        )
    else:
        # Full training
        train_full_pipeline(
            mode=args.mode,
            use_wandb=args.wandb,
            skip_stage1=args.skip_stage1,
        )


if __name__ == "__main__":
    main()
