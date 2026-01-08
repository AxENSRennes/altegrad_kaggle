#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for molecular captioning.

Includes:
- SMILES reconstruction from graph features
- L2 normalization
- Checkpoint save/load
- W&B logging helpers
"""

import os
import random
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np


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
    """
    Reconstruct SMILES from PyG graph using RDKit.

    The graph features follow the encoding from data_utils.py:
    - x[:, 0]: atomic_num (0-118)
    - x[:, 3]: formal_charge (index in range(-5, 7), so subtract 5 to get actual charge)
    - edge_attr[:, 0]: bond_type index

    Args:
        graph: PyTorch Geometric Data object

    Returns:
        SMILES string or empty string if reconstruction fails
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
    except Exception:
        # Catch any import errors including numpy 2.x incompatibility (_ARRAY_API not found)
        return ""

    try:
        mol = RWMol()

        # Bond type mapping (from data_utils.py e_map)
        BOND_TYPES = {
            0: Chem.BondType.UNSPECIFIED,
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.QUADRUPLE,
            5: Chem.BondType.QUINTUPLE,
            6: Chem.BondType.HEXTUPLE,
            7: Chem.BondType.ONEANDAHALF,
            8: Chem.BondType.TWOANDAHALF,
            9: Chem.BondType.THREEANDAHALF,
            10: Chem.BondType.FOURANDAHALF,
            11: Chem.BondType.FIVEANDAHALF,
            12: Chem.BondType.AROMATIC,
            13: Chem.BondType.IONIC,
            14: Chem.BondType.HYDROGEN,
            15: Chem.BondType.THREECENTER,
            16: Chem.BondType.DATIVEONE,
            17: Chem.BondType.DATIVE,
            18: Chem.BondType.DATIVEL,
            19: Chem.BondType.DATIVER,
            20: Chem.BondType.OTHER,
            21: Chem.BondType.ZERO,
        }

        # Add atoms
        # x[:, 0] = atomic_num, x[:, 3] = formal_charge (offset by 5)
        for i in range(graph.x.size(0)):
            atomic_num = int(graph.x[i, 0].item())
            if atomic_num == 0:
                atomic_num = 6  # Default to carbon
            atom = Chem.Atom(atomic_num)

            # Formal charge: index 3, stored as value + 5 (range -5 to +6)
            if graph.x.size(1) > 3:
                formal_charge = int(graph.x[i, 3].item()) - 5
                atom.SetFormalCharge(formal_charge)

            mol.AddAtom(atom)

        # Add bonds (deduplicate bidirectional edges)
        added_bonds = set()
        if graph.edge_index is not None and graph.edge_index.numel() > 0:
            for j in range(graph.edge_index.size(1)):
                src = int(graph.edge_index[0, j].item())
                dst = int(graph.edge_index[1, j].item())

                # Skip self-loops and duplicate edges
                if src == dst:
                    continue
                bond_key = (min(src, dst), max(src, dst))
                if bond_key in added_bonds:
                    continue
                added_bonds.add(bond_key)

                # Get bond type
                if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
                    bond_type_idx = int(graph.edge_attr[j, 0].item())
                    bond_type = BOND_TYPES.get(bond_type_idx, Chem.BondType.SINGLE)
                else:
                    bond_type = Chem.BondType.SINGLE

                mol.AddBond(src, dst, bond_type)

        # Convert to SMILES
        mol = mol.GetMol()
        smiles = Chem.MolToSmiles(mol)
        return smiles if smiles else ""

    except Exception:
        return ""


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

    def finish(self):
        """Finish W&B run."""
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
