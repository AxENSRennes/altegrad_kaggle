#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrieval + Reranking 
"""

version_embed = "v1"
version_gnn = "v4"

# =========================================================
# IMPORTS
# =========================================================
import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from train_gcn_v3 import MolGNN, infer_cardinalities_from_graphs
from data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_GRAPHS = "data/train_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"
TRAIN_EMB_CSV = f"data/train_embeddings_{version_embed}.csv"

GNN_CKPT = f"checkpoints/gnn_{version_gnn}.pt"

BATCH_SIZE = 128
TOP_K = 150

W_GT  = 0.60
W_GG  = 0.25
W_DEN = 0.15

CENTER_EMB = True


# =========================================================
# UTILS
# =========================================================
def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


@torch.no_grad()
def l2norm(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)


@torch.no_grad()
def maybe_center(x, mean=None):
    if mean is None:
        mean = x.mean(dim=0, keepdim=True)
    return x - mean, mean


# =========================================================
# MAIN
# =========================================================
@torch.no_grad()
def main():
    print(f"Device: {DEVICE}")

    # -----------------------------------------------------
    # Load train text embeddings
    # -----------------------------------------------------
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    train_ids = list(train_emb.keys())

    train_txt = torch.stack([train_emb[i] for i in train_ids]).to(DEVICE)
    train_txt = l2norm(train_txt)

    if CENTER_EMB:
        train_txt, txt_mean = maybe_center(train_txt)
        train_txt = l2norm(train_txt)
    else:
        txt_mean = None

    train_id2desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    emb_dim = train_txt.size(1)

    print(f"Train captions: {len(train_ids)} | emb_dim={emb_dim}")

    # -----------------------------------------------------
    # Load GNN
    # -----------------------------------------------------
    ckpt = torch.load(GNN_CKPT, map_location=DEVICE)
    h = ckpt["hparams"]

    atom_card, edge_card = infer_cardinalities_from_graphs(TRAIN_GRAPHS)

    gnn = MolGNN(
        atom_card=atom_card,
        edge_card=edge_card,
        hidden=h["hidden"],
        out_dim=emb_dim,
        layers=h["layers"],
    ).to(DEVICE)

    gnn.load_state_dict(ckpt["state_dict"])
    gnn.eval()

    print("✓ GNN loaded")

    # -----------------------------------------------------
    # Precompute train graph embeddings (aligned)
    # -----------------------------------------------------
    print("Precomputing train graph embeddings...")

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    id2pos = {i: j for j, i in enumerate(train_ids)}
    train_graph_emb = torch.zeros(len(train_ids), emb_dim, device=DEVICE)

    for graphs in train_dl:
        graphs = graphs.to(DEVICE)
        g = gnn(graphs)

        if CENTER_EMB and txt_mean is not None:
            g = l2norm(g - txt_mean)

        for i, gid in enumerate(graphs.id):
            if gid in id2pos:
                train_graph_emb[id2pos[gid]] = g[i]

    print("✓ Train graph embeddings ready")

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------
    test_ds = PreprocessedGraphDataset(TEST_GRAPHS)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # -----------------------------------------------------
    # Retrieval + reranking
    # -----------------------------------------------------
    results = []
    ptr = 0

    for graphs in test_dl:
        graphs = graphs.to(DEVICE)
        q = gnn(graphs)

        if CENTER_EMB and txt_mean is not None:
            q = l2norm(q - txt_mean)

        sims = q @ train_txt.T
        _, topk_idx = sims.topk(TOP_K, dim=-1)

        for b in range(topk_idx.size(0)):
            cand = topk_idx[b]

            cand_txt = train_txt[cand]
            cand_g   = train_graph_emb[cand]

            s_gt = zscore(cand_txt @ q[b])
            s_gg = zscore(cand_g @ q[b])

            sim_mat = cand_txt @ cand_txt.T
            s_den = zscore(sim_mat.mean(dim=1))

            score = W_GT * s_gt + W_GG * s_gg + W_DEN * s_den
            best = score.argmax().item()

            best_id = train_ids[cand[best]]
            results.append({
                "ID": test_ds.ids[ptr],
                "description": train_id2desc[best_id],
            })

            ptr += 1

    # -----------------------------------------------------
    # Save submission
    # -----------------------------------------------------
    df = pd.DataFrame(results)
    out = f"outputs/submission_v4.csv"
    ensure_dir(out)
    df.to_csv(out, index=False)

    print(f"\n✓ Saved {len(df)} predictions to {out}")

if __name__ == "__main__":
    main()
