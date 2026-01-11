#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# CRITICAL: This MUST be set before any other imports
os.environ["NPY_DISABLE_ARRAY_API"] = "1"

import random
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    """Create directory for a file path if it doesn't exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def graph_to_smiles(graph) -> str:
    """Proxy for rdkit_utils.graph_to_smiles for backward compatibility."""
    from rdkit_utils import graph_to_smiles as g2s
    return g2s(graph)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Any] = None,
):
    """Save a training checkpoint."""
    ensure_dir(path)

    checkpoint = {
        "epoch": epoch,
        "metrics": metrics or {},
    }

    # Handle different model types
    if hasattr(model, "projector"):
        # MolCaptionModel - save projector and LoRA
        checkpoint["projector_state_dict"] = model.projector.state_dict()
        if hasattr(model, "llm") and hasattr(model.llm, "state_dict"):
            # Save LoRA adapter weights only
            checkpoint["lora_state_dict"] = {
                k: v for k, v in model.llm.state_dict().items()
                if "lora" in k.lower()
            }
    else:
        # Regular model
        checkpoint["state_dict"] = model.state_dict()

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = vars(config) if hasattr(config, "__dict__") else config

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        if "projector_state_dict" in checkpoint and hasattr(model, "projector"):
            model.projector.load_state_dict(checkpoint["projector_state_dict"])
            if "lora_state_dict" in checkpoint and hasattr(model, "llm"):
                # Load LoRA weights
                model.llm.load_state_dict(checkpoint["lora_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(num_params: int) -> str:
    """Format parameter count for display."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def get_grad_norm(model: nn.Module) -> float:
    """Calculate the global L2 norm of gradients for trainable parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5


class WandBLogger:
    """Simple W&B logging wrapper."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._wandb = None

    def init(self, project: str, config: Any, tags: list = None):
        """Initialize W&B run."""
        if not self.enabled:
            return

        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=project,
                config=vars(config) if hasattr(config, "__dict__") else config,
                tags=tags or [],
            )
        except ImportError:
            print("wandb not installed, logging disabled")
            self.enabled = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled and self._wandb is not None:
            self._wandb.log(data, step=step)

    def log_table(self, table_name: str, columns: list, data: list, step: Optional[int] = None):
        """Log a data table to W&B."""
        if self.enabled and self._wandb is not None:
            table = self._wandb.Table(columns=columns, data=data)
            self._wandb.log({table_name: table}, step=step)

    def finish(self):
        """Finish W&B run."""
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
