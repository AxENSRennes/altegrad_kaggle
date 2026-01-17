#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for Molecular Captioning training pipeline.

Supports multiple hardware backends via Accelerate:
- GPU: Standard CUDA training with optional 4-bit quantization
- TPU: TPU v5e-8 training via torch_xla
- CPU: Local testing without GPU

Usage:
    # GPU training (default)
    python run.py --mode quick

    # CPU testing (laptop)
    python run.py --mode quick --hardware cpu

    # TPU training (Kaggle)
    accelerate launch --config_file accelerate_config_tpu.yaml run.py --mode quick --hardware tpu

    # Inference only
    python run.py --inference --checkpoint outputs/stage2_best.pt
"""

import os
# CRITICAL: This MUST be set before any other imports
os.environ["NPY_DISABLE_ARRAY_API"] = "1"

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
    hardware: str = "auto",
    stage2_epochs: int = None,
    stage2_batch_size: int = None,
    stage2_grad_accum: int = None,
    stage2_lr_proj: float = None,
    stage2_lr_lora: float = None,
):
    """
    Run the full training pipeline.

    Args:
        mode: Experiment mode ("quick", "medium", "full")
        use_wandb: Whether to log to W&B
        skip_stage1: Skip Stage 1 alignment (use for continuing training)
        hardware: Hardware mode ("auto", "gpu", "tpu", "cpu")
        stage2_epochs: Override Stage 2 epochs
        stage2_batch_size: Override Stage 2 batch size per device
        stage2_grad_accum: Override Stage 2 gradient accumulation steps
        stage2_lr_proj: Override Stage 2 projector learning rate
        stage2_lr_lora: Override Stage 2 LoRA learning rate
    """
    # Setup config
    config = get_config(mode=mode)
    config.use_wandb = use_wandb
    config.hardware_mode = hardware
    config.detect_hardware()

    # Apply CLI overrides for Stage 2
    if stage2_epochs is not None:
        config.stage2_epochs = stage2_epochs
    if stage2_batch_size is not None:
        config.stage2_batch_size = stage2_batch_size
    if stage2_grad_accum is not None:
        config.stage2_grad_accum = stage2_grad_accum
    if stage2_lr_proj is not None:
        config.stage2_lr_proj = stage2_lr_proj
    if stage2_lr_lora is not None:
        config.stage2_lr_lora = stage2_lr_lora

    set_seed(config.seed)

    # Determine device based on hardware mode
    if config.hardware_mode == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(f"Using TPU device: {device}")
        except ImportError:
            print("Warning: torch_xla not available, falling back to CPU")
            device = "cpu"
            config.hardware_mode = "cpu"
    elif config.hardware_mode == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            config.hardware_mode = "cpu"

    print(f"\n{'=' * 60}")
    print(f"Molecular Captioning Training Pipeline")
    print(f"Mode: {mode}")
    print(f"Hardware: {config.hardware_mode}")
    print(f"Device: {device}")
    print(f"Quantization: {config.use_quantization}")
    print(f"{'=' * 60}\n")

    print_config_summary(config)

    # Ensure output directory exists
    ensure_dir(config.output_dir + "/")

    # Create model
    print("\nCreating model...")
    model = create_model(config, device=str(device))
    model.print_trainable_parameters()

    # Setup W&B
    logger = WandBLogger(enabled=config.use_wandb)
    if config.use_wandb:
        logger.init(config.wandb_project, config, tags=[mode, config.hardware_mode])

    # Stage 1: Alignment
    if not skip_stage1:
        print("\n" + "=" * 60)
        print("STAGE 1: Alignment Training")
        print("=" * 60)
        stage1_metrics, _ = train_stage1(model, config, logger)
        print(f"Stage 1 complete: val_loss={stage1_metrics['val_loss']:.4f}")
    else:
        print("\nSkipping Stage 1 (using existing checkpoint)")

    # Stage 2: SFT
    print("\n" + "=" * 60)
    print("STAGE 2: Supervised Fine-Tuning")
    print("=" * 60)

    stage2_metrics = train_stage2(model, config, logger, load_stage1=True)

    print(f"Stage 2 complete: bleu4={stage2_metrics.get('bleu4', 0.0):.2f}")

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
        "--hardware",
        type=str,
        default="auto",
        choices=["auto", "gpu", "tpu", "cpu"],
        help="Hardware mode (auto, gpu, tpu, cpu)"
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
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode at inference (model reasons before answering)"
    )
    # Stage 2 training overrides
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        help="Override Stage 2 epochs"
    )
    parser.add_argument(
        "--stage2-batch-size",
        type=int,
        help="Override Stage 2 batch size per device"
    )
    parser.add_argument(
        "--stage2-grad-accum",
        type=int,
        help="Override Stage 2 gradient accumulation steps"
    )
    parser.add_argument(
        "--stage2-lr-proj",
        type=float,
        help="Override Stage 2 projector learning rate"
    )
    parser.add_argument(
        "--stage2-lr-lora",
        type=float,
        help="Override Stage 2 LoRA learning rate"
    )
    args = parser.parse_args()

    if args.inference:
        # Inference only
        config = get_config(mode=args.mode)
        config.hardware_mode = args.hardware
        config.detect_hardware()
        run_inference(
            config,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            limit=args.limit,
            batch_size=args.batch_size,
            enable_thinking=args.thinking,
        )
    else:
        # Full training
        train_full_pipeline(
            mode=args.mode,
            use_wandb=args.wandb,
            skip_stage1=args.skip_stage1,
            hardware=args.hardware,
            stage2_epochs=args.stage2_epochs,
            stage2_batch_size=args.stage2_batch_size,
            stage2_grad_accum=args.stage2_grad_accum,
            stage2_lr_proj=args.stage2_lr_proj,
            stage2_lr_lora=args.stage2_lr_lora,
        )


if __name__ == "__main__":
    main()
