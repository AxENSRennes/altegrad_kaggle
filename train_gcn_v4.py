#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strong contrastive graph encoder (WITH validation eval + EARLY STOPPING)

- Contrastive CLIP-style loss with memory queue
- Validation used ONLY for offline selection (MRR / Recall@k), NEVER for backprop
- Early stopping based on VAL MRR (NOT on loss)
- Saves best checkpoint automatically
- Logs train / val metrics to CSV
"""

version_embed = "v1"
version_gnn   = "v4"

# =========================================================
# IMPORTS
# =========================================================
import os
import math
import random
import pickle
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset,
    collate_fn,
)

# =========================================================
# CONFIG
# =========================================================
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"

TRAIN_EMB_CSV = f"data/train_embeddings_{version_embed}.csv"
VAL_EMB_CSV   = f"data/validation_embeddings_{version_embed}.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ===== OPTIMIZED HYPERPARAMETERS =====
BATCH_SIZE = 256
EPOCHS = 60
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

HIDDEN = 512
LAYERS = 5
DROPOUT = 0.1

QUEUE_SIZE = 65536
LOGIT_SCALE_INIT = 1 / 0.07
LOGIT_SCALE_MAX = 100.0

USE_AMP = True

# ===== EARLY STOPPING =====
PATIENCE = 6
MIN_DELTA = 1e-4

CKPT_PATH = f"checkpoints/gnn_{version_gnn}.pt"
BEST_CKPT_PATH = f"checkpoints/gnn_{version_gnn}_best.pt"

LOG_CSV = f"logs/log_{version_gnn}.csv"

# =========================================================
# UTILS
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def l2norm(x: torch.Tensor, eps: float = 1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def infer_cardinalities_from_graphs(path: str):
    with open(path, "rb") as f:
        graphs = pickle.load(f)

    max_x = torch.zeros(9, dtype=torch.long)
    max_e = torch.zeros(3, dtype=torch.long)

    for g in graphs:
        if g.x is not None and g.x.numel() > 0:
            max_x = torch.maximum(max_x, g.x.max(dim=0).values.long())
        if g.edge_attr is not None and g.edge_attr.numel() > 0:
            max_e = torch.maximum(max_e, g.edge_attr.max(dim=0).values.long())

    return (max_x + 2).tolist(), (max_e + 2).tolist()

# =========================================================
# MEMORY QUEUE
# =========================================================
class MemoryQueue(nn.Module):
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.size = size
        self.register_buffer("queue", l2norm(torch.randn(size, dim)))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, x: torch.Tensor):
        b = x.size(0)
        ptr = int(self.ptr)

        if b >= self.size:
            self.queue.copy_(x[-self.size:])
            self.ptr.zero_()
            return

        end = ptr + b
        if end <= self.size:
            self.queue[ptr:end] = x
        else:
            self.queue[ptr:] = x[: self.size - ptr]
            self.queue[: end - self.size] = x[self.size - ptr :]

        self.ptr[0] = end % self.size

# =========================================================
# MODEL
# =========================================================
class AtomEncoder(nn.Module):
    def __init__(self, card, hidden):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, 48) for c in card])
        self.proj = nn.Sequential(
            nn.Linear(9 * 48, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.proj(torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)], dim=-1))


class EdgeEncoder(nn.Module):
    def __init__(self, card, hidden):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, 48) for c in card])
        self.proj = nn.Sequential(
            nn.Linear(48, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, e):
        return self.proj(sum(emb(e[:, i]) for i, emb in enumerate(self.embs)))


class MolGNN(nn.Module):
    def __init__(self, atom_card, edge_card, hidden, out_dim, layers):
        super().__init__()

        self.atom_enc = AtomEncoder(atom_card, hidden)
        self.edge_enc = EdgeEncoder(edge_card, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

        self.logit_scale = nn.Parameter(torch.tensor(math.log(LOGIT_SCALE_INIT)))

    def forward(self, batch: Batch):
        h = self.atom_enc(batch.x.long())

        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            e = self.edge_enc(batch.edge_attr.long())
        else:
            e = torch.zeros(batch.edge_index.size(1), h.size(-1), device=h.device)

        for conv, ln in zip(self.convs, self.norms):
            h = h + F.gelu(ln(conv(h, batch.edge_index, e)))

        g = self.pool(h, batch.batch)
        g = self.readout(g)
        g = self.proj(g)
        return l2norm(g)

    def scale(self):
        return torch.clamp(self.logit_scale.exp(), max=LOGIT_SCALE_MAX)

# =========================================================
# LOSS
# =========================================================
def clip_loss(mol, txt, queue, scale):
    all_txt = torch.cat([txt, queue], dim=0)
    logits = mol @ all_txt.T * scale
    labels = torch.arange(mol.size(0), device=mol.device)
    return F.cross_entropy(logits, labels)

# =========================================================
# EVALUATION
# =========================================================
@torch.no_grad()
def eval_retrieval(model, loader):
    model.eval()
    all_g, all_t = [], []

    for graphs, txt in loader:
        graphs = graphs.to(DEVICE)
        txt = l2norm(txt.to(DEVICE))
        all_g.append(model(graphs))
        all_t.append(txt)

    G = torch.cat(all_g, dim=0)
    T = torch.cat(all_t, dim=0)

    sims = G @ T.T
    ranks = sims.argsort(dim=-1, descending=True)

    target = torch.arange(G.size(0), device=DEVICE)
    pos = (ranks == target[:, None]).nonzero()[:, 1] + 1

    return {
        "MRR": (1.0 / pos.float()).mean().item(),
        "R@1": (pos <= 1).float().mean().item(),
        "R@5": (pos <= 5).float().mean().item(),
    }

# =========================================================
# TRAIN
# =========================================================
def train_epoch(model, loader, optim, scaler, queue):
    model.train()
    total = 0.0

    for graphs, txt in loader:
        graphs = graphs.to(DEVICE)
        txt = l2norm(txt.to(DEVICE))

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            mol = model(graphs)
            loss = clip_loss(mol, txt, queue.queue, model.scale())

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optim)
        scaler.update()

        queue.enqueue(txt)
        total += loss.item() * graphs.num_graphs

    return total / len(loader.dataset)

# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    for p in [TRAIN_GRAPHS, VAL_GRAPHS, TRAIN_EMB_CSV, VAL_EMB_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    ensure_dir(LOG_CSV)
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss",
            "logit_scale",
            "val_MRR",
            "val_R@1",
            "val_R@5",
            "patience",
        ])

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV)
    emb_dim = len(next(iter(train_emb.values())))

    atom_card, edge_card = infer_cardinalities_from_graphs(TRAIN_GRAPHS)

    train_dl = DataLoader(
        PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    val_dl = DataLoader(
        PreprocessedGraphDataset(VAL_GRAPHS, val_emb),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    model = MolGNN(atom_card, edge_card, HIDDEN, emb_dim, LAYERS).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    queue = MemoryQueue(emb_dim, QUEUE_SIZE).to(DEVICE)

    print("Warming up queue (train only)...")
    with torch.no_grad():
        for _, txt in train_dl:
            queue.enqueue(l2norm(txt.to(DEVICE)))
            if queue.ptr.item() >= QUEUE_SIZE:
                break

    best_mrr = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for ep in range(EPOCHS):
        loss = train_epoch(model, train_dl, optim, scaler, queue)
        scale = model.scale().item()
        val_scores = eval_retrieval(model, val_dl)
        val_mrr = val_scores["MRR"]

        improved = val_mrr > best_mrr + MIN_DELTA
        if improved:
            best_mrr = val_mrr
            best_epoch = ep + 1
            epochs_no_improve = 0
            ensure_dir(BEST_CKPT_PATH)
            torch.save(
                {"state_dict": model.state_dict(),
                 "best": {"epoch": best_epoch, "val_mrr": best_mrr}},
                BEST_CKPT_PATH,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {ep+1:02d}/{EPOCHS} | "
            f"loss={loss:.4f} | scale={scale:.2f} | "
            f"val MRR={val_mrr:.4f} | "
            f"patience={epochs_no_improve}/{PATIENCE}"
        )

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                ep + 1,
                loss,
                scale,
                val_scores["MRR"],
                val_scores["R@1"],
                val_scores["R@5"],
                epochs_no_improve,
            ])

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping after {PATIENCE} epochs without improvement.")
            break

    ensure_dir(CKPT_PATH)
    torch.save(
        {"state_dict": model.state_dict(),
         "best": {"epoch": best_epoch, "val_mrr": best_mrr}},
        CKPT_PATH,
    )

    print(f"\nBest val MRR = {best_mrr:.4f} @ epoch {best_epoch}")
    print(f"Logs saved to: {LOG_CSV}")


if __name__ == "__main__":
    main()
