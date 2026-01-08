#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Generative Molecular Captioning with Qwen3-0.6B.

Supports three experiment modes:
- "quick": ~5 min test (500 samples, 1 epoch each stage)
- "medium": ~1 hour (5000 samples, 2 epochs each stage)
- "full": ~9 hours (all data, 3+5 epochs)
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def is_kaggle() -> bool:
    """Detect if running on Kaggle."""
    return os.path.exists("/kaggle/input")


@dataclass
class Config:
    """Configuration for molecular captioning training."""

    # Experiment mode: "quick" (5min), "medium" (1h), "full" (9h)
    experiment_mode: str = "quick"

    # === Paths ===
    # Auto-detect Kaggle vs local
    data_dir: str = field(default_factory=lambda: (
        "/kaggle/input/altegrad-2024" if is_kaggle()
        else "data"
    ))
    gnn_checkpoint: str = field(default_factory=lambda: (
        "/kaggle/input/gnn-checkpoints/gnn_v4_best.pt" if is_kaggle()
        else "checkpoints/gnn_v4_best.pt"
    ))
    output_dir: str = field(default_factory=lambda: (
        "/kaggle/working" if is_kaggle()
        else "outputs"
    ))

    # === Model Configuration ===
    llm_name: str = "Qwen/Qwen3-0.6B"
    gnn_hidden: int = 512
    gnn_layers: int = 5
    gnn_out_dim: int = 768  # GNN output embedding dimension
    proj_hidden: int = 1024  # Projector hidden dimension
    llm_hidden: int = 1024  # Qwen3-0.6B hidden size

    # === LoRA Configuration ===
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # === Stage 1: Alignment Training ===
    stage1_batch_size: int = 32
    stage1_grad_accum: int = 2
    stage1_lr: float = 1e-3
    stage1_epochs: int = 3
    stage1_warmup_steps: int = 100

    # === Stage 2: SFT Training ===
    stage2_batch_size: int = 8
    stage2_grad_accum: int = 8
    stage2_lr_proj: float = 5e-4
    stage2_lr_lora: float = 2e-4
    stage2_epochs: int = 5
    stage2_warmup_steps: int = 200
    eval_every_n_steps: int = 100

    # === General Training ===
    max_seq_length: int = 256
    num_workers: int = 4
    seed: int = 42
    use_amp: bool = True
    grad_clip_norm: float = 1.0

    # === Logging ===
    use_wandb: bool = True
    wandb_project: str = "mol-caption-gen"
    log_every_n_steps: int = 10

    # === Subset for quick experiments ===
    train_subset: Optional[int] = None  # Set by apply_mode()
    val_subset: Optional[int] = None

    def apply_mode(self):
        """Apply experiment mode presets."""
        if self.experiment_mode == "quick":
            self.stage1_epochs = 1
            self.stage2_epochs = 1
            self.train_subset = 500
            self.val_subset = 100
            self.eval_every_n_steps = 50
            self.stage1_warmup_steps = 10
            self.stage2_warmup_steps = 20
        elif self.experiment_mode == "medium":
            self.stage1_epochs = 2
            self.stage2_epochs = 2
            self.train_subset = 5000
            self.val_subset = 500
            self.eval_every_n_steps = 100
        elif self.experiment_mode == "full":
            self.stage1_epochs = 3
            self.stage2_epochs = 5
            self.train_subset = None  # Use all data
            self.val_subset = None
            self.eval_every_n_steps = 200
        else:
            raise ValueError(f"Unknown experiment mode: {self.experiment_mode}")

        return self

    @property
    def train_graphs_path(self) -> str:
        return os.path.join(self.data_dir, "train_graphs.pkl")

    @property
    def val_graphs_path(self) -> str:
        return os.path.join(self.data_dir, "validation_graphs.pkl")

    @property
    def test_graphs_path(self) -> str:
        return os.path.join(self.data_dir, "test_graphs.pkl")

    @property
    def stage1_checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "stage1_best.pt")

    @property
    def stage2_checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "stage2_best.pt")

    @property
    def submission_path(self) -> str:
        return os.path.join(self.output_dir, "submission.csv")


# Prompt template for caption generation
PROMPT_TEMPLATE = """<|user|>
Molecule Structure: <|graph|>
SMILES: {smiles}
Task: Describe the molecule's chemical properties and functional groups.
<|assistant|>
"""


def get_config(mode: str = "quick", **kwargs) -> Config:
    """Create a config with the specified mode and overrides."""
    config = Config(experiment_mode=mode, **kwargs)
    config.apply_mode()
    return config
