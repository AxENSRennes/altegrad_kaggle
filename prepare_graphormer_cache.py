#!/usr/bin/env python3
"""
Precompute Graphormer-style caches for each molecule graph:
- spatial_pos: [n, n] shortest-path distances clamped to max_dist
- edge_type_mat: [n, n] direct-edge bond type indices (else 0)

Writes new cached .pkl files to avoid expensive per-batch Python BFS in training.
"""

from __future__ import annotations
import os
import pickle
from typing import List
import numpy as np

import torch
from tqdm.auto import tqdm


# -----------------------
# Config
# -----------------------
INPUT_FILES = {
    "train": "data/train_graphs.pkl",
    "validation": "data/validation_graphs.pkl",
    "test": "data/test_graphs.pkl",
}

OUTPUT_FILES = {
    "train": "data/train_graphs_cached.pkl",
    "validation": "data/validation_graphs_cached.pkl",
    "test": "data/test_graphs_cached.pkl",
}

MAX_DIST = 12          # must match GraphormerConfig.max_dist used in training
BOND_TYPE_VOCAB = 22   # must match e_map['bond_type'] length
DIST_DTYPE = torch.uint8  # uint8 is enough for max_dist<=255 and saves RAM/disk


def shortest_path_all_pairs(num_nodes: int, edge_index: torch.Tensor, max_dist: int) -> torch.Tensor:
    """
    Unweighted all-pairs shortest paths using BFS from each node.
    Returns [n, n] with values in [0..max_dist] (unreachable clamped to max_dist).
    Runs on CPU.
    """
    # Build adjacency list (CPU python list)
    ei = edge_index.cpu()
    src = ei[0].tolist()
    dst = ei[1].tolist()

    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)

    # dist init
    dist = torch.full((num_nodes, num_nodes), fill_value=max_dist, dtype=torch.int16)
    for s in range(num_nodes):
        dist[s, s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            du = int(dist[s, u].item())
            if du >= max_dist:
                continue
            for v in adj[u]:
                if dist[s, v].item() > du + 1:
                    dist[s, v] = du + 1
                    q.append(v)

    # clamp + cast to compact dtype
    dist = dist.clamp(0, max_dist).to(DIST_DTYPE)
    return dist


def make_edge_type_mat(num_nodes: int, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    """
    Build dense [n, n] matrix where entry (u,v) is bond_type for direct edge u->v else 0.
    edge_attr[:, 0] is assumed to be bond_type index (as in your dataset).
    """
    mat = torch.zeros((num_nodes, num_nodes), dtype=torch.uint8)
    if edge_attr is None:
        return mat

    ei = edge_index.cpu().long()
    bond_type = edge_attr[:, 0].cpu().long().clamp(0, BOND_TYPE_VOCAB - 1)
    u = ei[0]
    v = ei[1]
    mat[u, v] = bond_type.to(torch.uint8)
    return mat


def process_split(split: str, in_path: str, out_path: str):
    if not os.path.exists(in_path):
        print(f"[skip] Missing: {in_path}")
        return

    print(f"\nLoading {split} graphs from: {in_path}")
    with open(in_path, "rb") as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs")

    # Precompute caches
    for g in tqdm(graphs, desc=f"Precomputing {split}", dynamic_ncols=True):
        n = int(g.num_nodes)

        if not hasattr(g, "spatial_pos") or g.spatial_pos is None:
            sp = shortest_path_all_pairs(n, g.edge_index, MAX_DIST)
            g.spatial_pos = sp.to(torch.uint8)

        if not hasattr(g, "edge_type_mat") or g.edge_type_mat is None:
            edge_attr = getattr(g, "edge_attr", None)
            et = make_edge_type_mat(n, g.edge_index, edge_attr)
            g.edge_type_mat = et.to(torch.uint8)

    print(f"Writing cached {split} graphs to: {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


def main():
    for split, in_path in INPUT_FILES.items():
        out_path = OUTPUT_FILES[split]
        process_split(split, in_path, out_path)

    print("\nAll splits processed.")
    print("Update your training script to use *_graphs_cached.pkl.")


if __name__ == "__main__":
    main()
