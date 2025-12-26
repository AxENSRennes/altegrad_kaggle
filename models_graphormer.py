# models_graphormer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import warnings

_WARNED_FALLBACK_SP = False


# ----------------------------
# Utilities: per-graph shortest path distances
# ----------------------------
def _shortest_path_dist(num_nodes: int, edge_index: torch.Tensor, max_dist: int) -> torch.Tensor:
    """
    Compute unweighted shortest path distances (BFS) for a single graph.
    Returns [num_nodes, num_nodes] with values in [0..max_dist], where
    unreachable distances are clamped to max_dist.
    """
    device = edge_index.device
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)

    dist = torch.full((num_nodes, num_nodes), fill_value=max_dist, device=device, dtype=torch.long)
    for s in range(num_nodes):
        dist[s, s] = 0
        # BFS
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            du = dist[s, u].item()
            if du >= max_dist:
                continue
            for v in adj[u]:
                if dist[s, v].item() > du + 1:
                    dist[s, v] = du + 1
                    q.append(v)
    return dist


def batch_graph_to_padded(batch: Batch, max_dist: int):
    """
    Convert a PyG Batch into padded tensors:
      - x_pad: [B, Nmax, F]   (categorical indices)
      - node_mask: [B, Nmax]  (True where node exists)
      - dist_pad: [B, Nmax, Nmax] shortest-path distances clamped to max_dist
      - edge_types_pad: [B, Nmax, Nmax] bond_type index for direct edges else 0 ("UNSPECIFIED")
    """
    global _WARNED_FALLBACK_SP
    device = batch.x.device

    graphs = batch.to_data_list()
    B = len(graphs)
    Nmax = max(g.num_nodes for g in graphs)

    # IMPORTANT: grab cached arrays from the Batch (PyG stores non-tensors here as lists)
    # Cached [n,n] tensors stored as Python lists on the Batch.
    sp_list = getattr(batch, "_spatial_pos", None) or getattr(batch, "_spatial_pos_list", None)
    et_list = getattr(batch, "_edge_type_mat", None) or getattr(batch, "_edge_type_mat_list", None)

    def _to_long_tensor(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.long)
        return torch.as_tensor(x, device=device, dtype=torch.long)

    Fdim = graphs[0].x.size(-1)
    x_pad = torch.zeros((B, Nmax, Fdim), device=device, dtype=graphs[0].x.dtype)
    node_mask = torch.zeros((B, Nmax), device=device, dtype=torch.bool)

    dist_pad = torch.full((B, Nmax, Nmax), fill_value=max_dist, device=device, dtype=torch.long)
    bond_pad = torch.zeros((B, Nmax, Nmax), device=device, dtype=torch.long)

    for i, g in enumerate(graphs):
        n = g.num_nodes
        x_pad[i, :n] = g.x
        node_mask[i, :n] = True

        # --- shortest path distances ---
        cached_sp = _to_long_tensor(sp_list[i]) if sp_list is not None else None
        if cached_sp is None:
            if not _WARNED_FALLBACK_SP:
                warnings.warn(
                    "Fallback to on-the-fly shortest path computation detected. "
                    "This is slow and indicates missing cached '_spatial_pos'. "
                    "Please rerun prepare_graphormer_cache.py.",
                    category=UserWarning,
                    stacklevel=2,
                )
                _WARNED_FALLBACK_SP = True
            dist = _shortest_path_dist(n, g.edge_index.to(device), max_dist=max_dist)
        else:
            dist = cached_sp

        dist_pad[i, :n, :n] = dist

        # --- direct-edge bond types ---
        cached_et = _to_long_tensor(et_list[i]) if et_list is not None else None
        if cached_et is not None:
            bond_pad[i, :n, :n] = cached_et
        else:
            if hasattr(g, "edge_attr") and g.edge_attr is not None:
                bond_type = g.edge_attr[:, 0].long()
                ei = g.edge_index
                u = ei[0].long()
                v = ei[1].long()
                bond_pad[i, u, v] = bond_type.to(device)

    return x_pad, node_mask, dist_pad, bond_pad


# ----------------------------
# Graphormer-style attention block (with additive attention bias)
# ----------------------------
class BiasMultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,                        # [B, N, D]
        key_padding_mask: torch.Tensor,         # [B, N] True = pad
        attn_bias: torch.Tensor,                # [B, H, N, N] additive bias
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = attn_bias.size(1)

        # Convert per-head bias into the 3D attn_mask format expected by PyTorch: (B*H, N, N)
        attn_mask = attn_bias.reshape(B * H, N, N).to(dtype=torch.float32)

        # IMPORTANT:
        # To avoid dtype mismatch warnings (bool key_padding_mask + float attn_mask),
        # we fold padding into the bias and pass key_padding_mask=None.
        if key_padding_mask is not None:
            # Mask out keys that are padding: add large negative bias on padded key positions
            # key_padding_mask: [B, N] -> [B, 1, 1, N] -> broadcast over heads and queries
            pad_bias = key_padding_mask[:, None, None, :].to(attn_mask.dtype) * (-1e4)
            pad_bias = pad_bias.expand(B, H, N, N).reshape(B * H, N, N)
            attn_mask = attn_mask + pad_bias

        out, _ = self.mha(
            x, x, x,
            key_padding_mask=None,   # padding handled in attn_mask
            attn_mask=attn_mask,
            need_weights=False,
        )
        return out



class GraphormerEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = BiasMultiheadSelfAttention(dim, heads, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, key_padding_mask=pad_mask, attn_bias=attn_bias)
        h2 = self.ln2(x)
        x = x + self.ff(h2)
        return x


@dataclass
class GraphormerConfig:
    dim: int = 512
    layers: int = 6
    heads: int = 8
    dropout: float = 0.1
    max_dist: int = 12

    # categorical sizes from your baseline maps
    atomic_num: int = 119
    chirality: int = 9
    degree: int = 11
    formal_charge: int = 12  # -5..+6 => 12 values
    num_hs: int = 9
    num_radical_electrons: int = 5
    hybridization: int = 8
    is_aromatic: int = 2
    is_in_ring: int = 2

    bond_type: int = 22  # per e_map
    stereo: int = 6
    is_conjugated: int = 2


class GraphormerEncoder(nn.Module):
    """
    Graphormer-ish:
      - node feature embeddings (sum of 9 categorical embeddings)
      - attention bias from:
          * shortest-path distance embedding
          * direct-edge bond type embedding
    Outputs:
      encoder_hidden_states: [B, Nmax, D]
      encoder_attention_mask: [B, Nmax]  (1 = keep, 0 = pad)
    """
    def __init__(self, cfg: GraphormerConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim

        # Node feature embeddings: 9 categorical fields
        self.emb_atomic = nn.Embedding(cfg.atomic_num, D)
        self.emb_chirality = nn.Embedding(cfg.chirality, D)
        self.emb_degree = nn.Embedding(cfg.degree, D)
        self.emb_formal = nn.Embedding(cfg.formal_charge, D)
        self.emb_numhs = nn.Embedding(cfg.num_hs, D)
        self.emb_rad = nn.Embedding(cfg.num_radical_electrons, D)
        self.emb_hybrid = nn.Embedding(cfg.hybridization, D)
        self.emb_arom = nn.Embedding(cfg.is_aromatic, D)
        self.emb_ring = nn.Embedding(cfg.is_in_ring, D)

        self.in_ln = nn.LayerNorm(D)
        self.in_drop = nn.Dropout(cfg.dropout)

        # Attention biases (scalar per head, per pair)
        # We make embeddings to heads channels and use them as additive bias.
        self.dist_bias = nn.Embedding(cfg.max_dist + 1, cfg.heads)
        self.bond_bias = nn.Embedding(cfg.bond_type, cfg.heads)

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(dim=D, heads=cfg.heads, dropout=cfg.dropout)
            for _ in range(cfg.layers)
        ])
        self.out_ln = nn.LayerNorm(D)

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch.x.device
        x_pad, node_mask, dist_pad, bond_pad = batch_graph_to_padded(batch, max_dist=self.cfg.max_dist)

        # Node embedding sum
        # x_pad: [B, N, 9] => indices
        atomic = x_pad[..., 0].long()
        chir = x_pad[..., 1].long()
        deg = x_pad[..., 2].long()
        formal = x_pad[..., 3].long()  # already index 0..11 in your preprocessing
        numhs = x_pad[..., 4].long()
        rad = x_pad[..., 5].long()
        hyb = x_pad[..., 6].long()
        arom = x_pad[..., 7].long()
        ring = x_pad[..., 8].long()

        h = (
            self.emb_atomic(atomic)
            + self.emb_chirality(chir)
            + self.emb_degree(deg)
            + self.emb_formal(formal)
            + self.emb_numhs(numhs)
            + self.emb_rad(rad)
            + self.emb_hybrid(hyb)
            + self.emb_arom(arom)
            + self.emb_ring(ring)
        )
        h = self.in_drop(self.in_ln(h))

        # pad mask for attention: True means PAD for nn.MultiheadAttention
        pad_mask = ~node_mask  # [B, N]

        # Build additive attention bias: [B, heads, N, N]
        dist_pad = dist_pad.long()
        bond_pad = bond_pad.long()
        dist_pad = dist_pad.clamp(0, self.cfg.max_dist)
        bond_pad = bond_pad.clamp(0, self.cfg.bond_type - 1)


        dist_bias = self.dist_bias(dist_pad)               # [B, N, N, heads]
        bond_bias = self.bond_bias(bond_pad)               # [B, N, N, heads]
        attn_bias = (dist_bias + bond_bias).permute(0, 3, 1, 2).contiguous()  # [B, heads, N, N]

        # Mask out attention to/from padding by adding large negative bias
        # (attn_mask is additive, so use -1e4)
        neg = torch.full_like(attn_bias, -1e4)
        # where either query or key is pad -> block
        q_pad = pad_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]
        k_pad = pad_mask.unsqueeze(1).unsqueeze(1)   # [B,1,1,N]
        attn_bias = torch.where(q_pad | k_pad, neg, attn_bias)

        for layer in self.layers:
            h = layer(h, pad_mask=pad_mask, attn_bias=attn_bias)

        h = self.out_ln(h)
        encoder_attention_mask = node_mask.long()  # [B, N] 1 keep, 0 pad
        return h, encoder_attention_mask
