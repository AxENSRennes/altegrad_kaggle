#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Molecular Captioning with GNN + Qwen3-0.6B

Two-stage generative captioning system:
- Stage 1: Alignment training (projector only)
- Stage 2: SFT training (projector + LoRA)

Usage:
    from mol_caption_code import train_full_pipeline
    train_full_pipeline(mode="quick")
"""

from config import Config, get_config, SYSTEM_PROMPT, USER_PROMPT_FORMAT
from model_gnn import MolGNN, load_gnn_checkpoint
from model_projector import SolidBridgeProjector
from model_wrapper import MolCaptionModel, create_model
from dataset_caption import MolCaptionDatasetTRL, prepare_trl_dataloaders
from train_stage1 import train_stage1
from train_stage2 import train_stage2
from metrics import compute_metrics
from utils import set_seed, WandBLogger

__all__ = [
    "Config",
    "get_config",
    "SYSTEM_PROMPT",
    "USER_PROMPT_FORMAT",
    "MolGNN",
    "load_gnn_checkpoint",
    "SolidBridgeProjector",
    "MolCaptionModel",
    "create_model",
    "MolCaptionDatasetTRL",
    "prepare_trl_dataloaders",
    "train_stage1",
    "train_stage2",
    "compute_metrics",
    "set_seed",
    "WandBLogger",
]
