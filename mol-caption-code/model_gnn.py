#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolGNN: Graph Neural Network encoder for molecular graphs.

Copied and adapted from train_gcn_v5.py with the following architecture:
- AtomEncoder: Embedding for 9 atomic features
- EdgeEncoder: Embedding for 3 edge features
- GINEConv layers with residual connections
- AttentionalAggregation for graph-level pooling
- L2 normalized output embeddings
"""

import math
import pickle
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation


# Default hyperparameters (can be overridden via config)
DEFAULT_DROPOUT = 0.1
DEFAULT_LOGIT_SCALE_INIT = 1 / 0.07
DEFAULT_LOGIT_SCALE_MAX = 100.0


def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def infer_cardinalities_from_graphs(path: str) -> Tuple[List[int], List[int]]:
    """
    Scan graph pickle file to determine cardinality (num_classes) for each feature.
    Used to initialize embedding layers with correct vocabulary sizes.

    Args:
        path: Path to pickle file containing list of PyG graph objects

    Returns:
        Tuple of (atom_cardinalities, edge_cardinalities)
        - atom_cardinalities: list of 9 ints for each atom feature
        - edge_cardinalities: list of 3 ints for each edge feature
    """
    with open(path, "rb") as f:
        graphs = pickle.load(f)

    max_x = torch.zeros(9, dtype=torch.long)
    max_e = torch.zeros(3, dtype=torch.long)

    for g in graphs:
        if g.x is not None and g.x.numel() > 0:
            max_x = torch.maximum(max_x, g.x.max(dim=0).values.long())
        if g.edge_attr is not None and g.edge_attr.numel() > 0:
            max_e = torch.maximum(max_e, g.edge_attr.max(dim=0).values.long())

    # Add 2 for safety margin
    return (max_x + 2).tolist(), (max_e + 2).tolist()


class AtomEncoder(nn.Module):
    """
    Encoder for atomic features.

    Takes 9 categorical features per atom and produces a hidden representation.
    Features: atomic_num, chirality, degree, formal_charge, num_hs,
              num_radical_electrons, hybridization, is_aromatic, is_in_ring
    """

    def __init__(self, cardinalities: List[int], hidden: int, dropout: float = DEFAULT_DROPOUT):
        """
        Args:
            cardinalities: List of vocabulary sizes for each of the 9 features
            hidden: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, 48) for c in cardinalities])
        self.proj = nn.Sequential(
            nn.Linear(9 * 48, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, 9] (long tensor)

        Returns:
            Hidden representations [num_nodes, hidden]
        """
        # Concatenate embeddings from each feature
        embedded = torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)], dim=-1)
        return self.proj(embedded)


class EdgeEncoder(nn.Module):
    """
    Encoder for edge (bond) features.

    Takes 3 categorical features per edge and produces a hidden representation.
    Features: bond_type, stereo, is_conjugated
    """

    def __init__(self, cardinalities: List[int], hidden: int, dropout: float = DEFAULT_DROPOUT):
        """
        Args:
            cardinalities: List of vocabulary sizes for each of the 3 features
            hidden: Output hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, 48) for c in cardinalities])
        self.proj = nn.Sequential(
            nn.Linear(48, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e: Edge features [num_edges, 3] (long tensor)

        Returns:
            Hidden representations [num_edges, hidden]
        """
        # Sum embeddings from each feature (more memory efficient than concat)
        embedded = sum(emb(e[:, i]) for i, emb in enumerate(self.embs))
        return self.proj(embedded)


class MolGNN(nn.Module):
    """
    Molecular Graph Neural Network encoder.

    Architecture:
    1. AtomEncoder + EdgeEncoder for initial embeddings
    2. Multiple GINEConv layers with residual connections and LayerNorm
    3. AttentionalAggregation for graph-level pooling
    4. MLP readout to output embedding dimension
    5. L2 normalization of final embeddings
    """

    def __init__(
        self,
        atom_cardinalities: List[int],
        edge_cardinalities: List[int],
        hidden: int = 512,
        out_dim: int = 768,
        num_layers: int = 5,
        dropout: float = DEFAULT_DROPOUT,
    ):
        """
        Args:
            atom_cardinalities: Vocabulary sizes for atom features
            edge_cardinalities: Vocabulary sizes for edge features
            hidden: Hidden dimension for GNN layers
            out_dim: Output embedding dimension (typically 768 for BERT-like)
            num_layers: Number of GINEConv layers
            dropout: Dropout rate
        """
        super().__init__()

        self.atom_enc = AtomEncoder(atom_cardinalities, hidden, dropout)
        self.edge_enc = EdgeEncoder(edge_cardinalities, hidden, dropout)

        # GINEConv layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))

        # Attentional aggregation for graph-level pooling
        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
        )

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final projection to output dimension
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

        # Learnable temperature for contrastive learning (optional, used in training)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(DEFAULT_LOGIT_SCALE_INIT)))

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Encode a batch of molecular graphs.

        Args:
            batch: PyG Batch object containing batched graphs

        Returns:
            L2-normalized graph embeddings [batch_size, out_dim]
        """
        # Encode atoms
        h = self.atom_enc(batch.x.long())

        # Encode edges (handle empty edge case)
        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            e = self.edge_enc(batch.edge_attr.long())
        else:
            e = torch.zeros(batch.edge_index.size(1), h.size(-1), device=h.device)

        # Apply GINEConv layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.gelu(norm(conv(h, batch.edge_index, e)))

        # Graph-level pooling
        g = self.pool(h, batch.batch)

        # Readout and projection
        g = self.readout(g)
        g = self.proj(g)

        # L2 normalize
        return l2norm(g)

    def scale(self) -> torch.Tensor:
        """Return clamped temperature scale for contrastive loss."""
        return torch.clamp(self.logit_scale.exp(), max=DEFAULT_LOGIT_SCALE_MAX)


def load_gnn_checkpoint(config, device: str = "cpu") -> MolGNN:
    """
    Load a pre-trained MolGNN from checkpoint.

    If checkpoint is not found locally, attempts to download from HF Hub.

    Args:
        config: Config object with data_dir, gnn_checkpoint, gnn_hidden, gnn_layers
        device: Device to load model to

    Returns:
        Loaded MolGNN model
    """
    from pathlib import Path

    checkpoint_path = Path(config.gnn_checkpoint)

    # Download from HF Hub if not found locally
    if not checkpoint_path.exists():
        try:
            import sys
            sys.path.insert(0, str(checkpoint_path.parent.parent))
            from hf_checkpoint import download_checkpoint
            print(f"Checkpoint not found locally, downloading from HF Hub...")
            download_checkpoint(
                filename=checkpoint_path.name,
                local_dir=str(checkpoint_path.parent),
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Checkpoint {config.gnn_checkpoint} not found locally and failed to download from HF Hub: {e}"
            )

    # Infer cardinalities from training graphs
    atom_card, edge_card = infer_cardinalities_from_graphs(config.train_graphs_path)

    # Create model
    gnn = MolGNN(
        atom_cardinalities=atom_card,
        edge_cardinalities=edge_card,
        hidden=config.gnn_hidden,
        out_dim=config.gnn_out_dim,
        num_layers=config.gnn_layers,
    )

    # Load checkpoint
    checkpoint = torch.load(config.gnn_checkpoint, map_location=device)
    gnn.load_state_dict(checkpoint["state_dict"], strict=False)

    return gnn.to(device)
