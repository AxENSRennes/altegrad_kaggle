from __future__ import annotations

"""
Lightweight GINEConv graph encoder for molecule graphs.

The encoder expects node features as categorical indices following ``data_utils.x_map``
and edge features following ``data_utils.e_map``. Categorical fields are embedded,
summed, and passed through a stack of GINEConv layers with residual connections and
dropout. A global readout (mean by default) produces graph-level embeddings that can
be paired with text embeddings in contrastive training.
"""

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

from data_utils import x_map, e_map


@dataclass
class GINEConfig:
    hidden_dim: int = 512
    num_layers: int = 5
    dropout: float = 0.1
    readout: Literal["mean", "sum", "attn"] = "attn"
    residual: bool = True
    normalize: bool = False


class GINEEncoder(nn.Module):
    """
    Small GINE-based encoder that operates on PyG Batch objects.
    """

    def __init__(self, cfg: GINEConfig):
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings per field
        self.node_embs = nn.ModuleList([nn.Embedding(len(vals), cfg.hidden_dim) for vals in x_map.values()])
        self.edge_embs = nn.ModuleList([nn.Embedding(len(vals), cfg.hidden_dim) for vals in e_map.values()])

        self.in_ln = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            mlp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            conv = GINEConv(nn=mlp, train_eps=True, edge_dim=cfg.hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(cfg.hidden_dim))

        self.attn_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim // 2, 1),
            )
        )
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.cfg.hidden_dim

    def reset_parameters(self):
        for emb in list(self.node_embs) + list(self.edge_embs):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.attn_pool.reset_parameters()

    def _embed_nodes(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, num_fields] of categorical indices
        parts = [emb(x[:, i].long()) for i, emb in enumerate(self.node_embs)]
        h = torch.stack(parts, dim=0).sum(dim=0)
        return self.in_ln(h)

    def _embed_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr: [E, num_edge_fields] of categorical indices
        parts = [emb(edge_attr[:, i].long()) for i, emb in enumerate(self.edge_embs)]
        return torch.stack(parts, dim=0).sum(dim=0)

    def _readout(self, node_feats: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        if self.cfg.readout == "sum":
            return global_add_pool(node_feats, batch_idx)
        if self.cfg.readout == "attn":
            return self.attn_pool(node_feats, batch_idx)
        return global_mean_pool(node_feats, batch_idx)

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: PyG Batch with attributes x, edge_index, edge_attr, batch.

        Returns:
            graph_embeddings: [B, hidden_dim]
            node_embeddings:  [N, hidden_dim]
        """
        x = self._embed_nodes(batch.x)
        edge_attr = self._embed_edges(batch.edge_attr)

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, batch.edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new if self.cfg.residual else h_new

        graph_emb = self._readout(h, batch.batch)
        if self.cfg.normalize:
            graph_emb = F.normalize(graph_emb, p=2, dim=-1)

        return graph_emb, h
