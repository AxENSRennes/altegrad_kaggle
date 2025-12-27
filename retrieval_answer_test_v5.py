#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrieval + Neural Reranking

Stage 1: Fast TOP-K retrieval with cosine (graph->text)
Stage 2: Train a lightweight neural reranker (MLP) on-the-fly using TRAIN+VAL only
         (no GNN training, no backprop through GNN)
Inference: Rerank TOP-K with the MLP, output best training caption
"""

import os
import math
import random
import pickle
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation

from data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)

# =========================
# CONFIG
# =========================
version_embed = "v1"
version_gnn   = "v4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

SEED = 42

TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

TRAIN_EMB_CSV = f"data/train_embeddings_{version_embed}.csv"
VAL_EMB_CSV   = f"data/validation_embeddings_{version_embed}.csv"

# Prefer BEST checkpoint
GNN_CKPT = f"checkpoints/gnn_{version_gnn}_best.pt"
if not os.path.exists(GNN_CKPT):
    GNN_CKPT = f"checkpoints/gnn_{version_gnn}.pt"

OUT_CSV = "outputs/submission_v7.csv"

# Retrieval
BATCH_SIZE_EMB = 128
TOPK = 300

# Reranker training (fast)
RERANK_EPOCHS = 3
RERANK_LR = 2e-3
RERANK_BS = 64              # number of queries per step (each query uses TOPK candidates)
NEG_K = 64                  # candidates used in list (pos + NEG_K-1 negatives); sampled from TOPK
TRAIN_QUERY_SUBSAMPLE = 12000  # number of train queries used to train reranker (speed/quality tradeoff)
VAL_QUERY_SUBSAMPLE   = 2000   # number of val queries for early stopping
PATIENCE = 2

CENTER_EMB = True  # often helps with retrieval stability

# =========================
# UTILS
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

@torch.no_grad()
def l2norm(x: torch.Tensor, eps: float = 1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def maybe_center(x, mean=None):
    if mean is None:
        mean = x.mean(dim=0, keepdim=True)
    return x - mean, mean

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

# =========================
# GNN MODEL (compatible with your v4_final)
# =========================
DROPOUT_DEFAULT = 0.1

class AtomEncoder(nn.Module):
    def __init__(self, card, hidden, dropout=DROPOUT_DEFAULT):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, 48) for c in card])
        self.proj = nn.Sequential(
            nn.Linear(9 * 48, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)], dim=-1))

class EdgeEncoder(nn.Module):
    def __init__(self, card, hidden, dropout=DROPOUT_DEFAULT):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, 48) for c in card])
        self.proj = nn.Sequential(
            nn.Linear(48, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, e):
        return self.proj(sum(emb(e[:, i]) for i, emb in enumerate(self.embs)))

class MolGNN(nn.Module):
    def __init__(self, atom_card, edge_card, hidden, out_dim, layers, dropout=DROPOUT_DEFAULT):
        super().__init__()

        self.atom_enc = AtomEncoder(atom_card, hidden, dropout=dropout)
        self.edge_enc = EdgeEncoder(edge_card, hidden, dropout=dropout)

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
            nn.Dropout(dropout),
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

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

# =========================
# RERANKER
# =========================
class NeuralReranker(nn.Module):
    """
    Scores (q, cand_txt, cand_graph) with rich features:
    - cos(q, t), cos(q, g)
    - elementwise product q*t, q*g
    - abs diff |q-t|, |q-g|
    - (t*g) cosine as a proxy of caption-structure consistency
    """
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.15):
        super().__init__()
        # Feature dim:
        # q,t,g : not concatenated raw (too big), we use interactions only
        # q*t (dim) + |q-t| (dim) + q*g (dim) + |q-g| (dim) + 3 scalars
        feat_dim = 4 * dim + 3
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, q, t, g):
        # q: [B,D], t:[B,K,D], g:[B,K,D]
        B, K, D = t.shape
        q_ = q[:, None, :].expand(B, K, D)

        qt = q_ * t
        qg = q_ * g
        dqt = (q_ - t).abs()
        dqg = (q_ - g).abs()

        cos_qt = (q_ * t).sum(-1, keepdim=True)   # since normalized
        cos_qg = (q_ * g).sum(-1, keepdim=True)
        cos_tg = (t * g).sum(-1, keepdim=True)

        feats = torch.cat([qt, dqt, qg, dqg, cos_qt, cos_qg, cos_tg], dim=-1)
        score = self.net(feats).squeeze(-1)  # [B,K]
        return score

# =========================
# PRECOMPUTE EMBEDDINGS
# =========================
@torch.no_grad()
def embed_graphs(gnn, graphs_pkl, ids_order, emb_dim, txt_mean=None):
    ds = PreprocessedGraphDataset(graphs_pkl)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_EMB, shuffle=False, collate_fn=collate_fn)

    id2pos = {i: j for j, i in enumerate(ids_order)}
    out = torch.zeros(len(ids_order), emb_dim, device=DEVICE)

    for batch in dl:
        batch = batch.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            g = gnn(batch)
        if (txt_mean is not None):
            g = l2norm(g - txt_mean)
        for i, gid in enumerate(batch.id):
            if gid in id2pos:
                out[id2pos[gid]] = g[i]
    return out

@torch.no_grad()
def embed_graphs_in_order(gnn, graphs_pkl, txt_mean=None):
    ds = PreprocessedGraphDataset(graphs_pkl)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_EMB, shuffle=False, collate_fn=collate_fn)

    all_ids = []
    all_emb = []

    for batch in dl:
        all_ids.extend(list(batch.id))
        batch = batch.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            g = gnn(batch)
        if (txt_mean is not None):
            g = l2norm(g - txt_mean)
        all_emb.append(g.detach().cpu())

    return all_ids, l2norm(torch.cat(all_emb, dim=0)).to(DEVICE)

# =========================
# TOPK RETRIEVAL
# =========================
@torch.no_grad()
def topk_candidates(q_emb, train_txt, topk=TOPK):
    # q_emb [B,D], train_txt [N,D]
    sims = q_emb @ train_txt.T
    _, idx = sims.topk(topk, dim=-1)
    return idx

# =========================
# RERANKER TRAINING DATA BUILDER
# =========================
@dataclass
class ListBatch:
    q: torch.Tensor      # [B,D]
    t: torch.Tensor      # [B,K,D]
    g: torch.Tensor      # [B,K,D]
    y: torch.Tensor      # [B] index of positive in K

@torch.no_grad()
def build_list_batch(
    q_ids, q_emb, q_true_pos,  # q_true_pos: position in train_ids for the true caption (for train/val)
    train_txt, train_g,        # [N,D]
    topk_idx,                  # [B,TOPK]
    K=NEG_K
):
    # For each query: sample K candidates from TOPK, force include positive
    B = q_emb.size(0)
    D = q_emb.size(1)

    t_out = torch.empty((B, K, D), device=DEVICE)
    g_out = torch.empty((B, K, D), device=DEVICE)
    y_out = torch.empty((B,), dtype=torch.long, device=DEVICE)

    for b in range(B):
        cand = topk_idx[b].tolist()

        pos = q_true_pos[b]
        # ensure pos appears in list; if not, force it by replacing last element
        if pos not in cand:
            cand[-1] = pos

        # sample K-1 negatives from cand excluding pos
        neg_pool = [c for c in cand if c != pos]
        if len(neg_pool) < (K - 1):
            # pad by random negatives from full space
            extra = torch.randint(0, train_txt.size(0), (K - 1 - len(neg_pool),), device="cpu").tolist()
            neg_pool.extend(extra)

        negs = random.sample(neg_pool, K - 1) if len(neg_pool) >= (K - 1) else neg_pool[: (K - 1)]
        chosen = [pos] + negs
        random.shuffle(chosen)

        y = chosen.index(pos)

        chosen_t = train_txt[torch.tensor(chosen, device=DEVICE)]
        chosen_g = train_g[torch.tensor(chosen, device=DEVICE)]

        t_out[b] = chosen_t
        g_out[b] = chosen_g
        y_out[b] = y

    return ListBatch(q=q_emb, t=t_out, g=g_out, y=y_out)

# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    for p in [TRAIN_GRAPHS, VAL_GRAPHS, TEST_GRAPHS, TRAIN_EMB_CSV, VAL_EMB_CSV, GNN_CKPT]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    # ---- Load train/val text embeddings (E5) ----
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb   = load_id2emb(VAL_EMB_CSV)

    train_ids = list(train_emb.keys())
    val_ids   = list(val_emb.keys())

    train_txt = torch.stack([train_emb[i] for i in train_ids]).to(DEVICE)
    val_txt   = torch.stack([val_emb[i] for i in val_ids]).to(DEVICE)

    train_txt = l2norm(train_txt)
    val_txt   = l2norm(val_txt)

    txt_mean = None
    if CENTER_EMB:
        train_txt, txt_mean = maybe_center(train_txt)
        train_txt = l2norm(train_txt)
        val_txt   = l2norm(val_txt - txt_mean)

    emb_dim = train_txt.size(1)
    print(f"Train text: {len(train_ids)} | Val text: {len(val_ids)} | D={emb_dim}")

    # ---- Load descriptions for final submission ----
    train_id2desc = load_descriptions_from_graphs(TRAIN_GRAPHS)

    # ---- Load GNN ----
    ckpt = torch.load(GNN_CKPT, map_location=DEVICE)
    hparams = ckpt.get("hparams", {})
    hidden = int(hparams.get("hidden", 512))
    layers = int(hparams.get("layers", 5))
    dropout = float(hparams.get("dropout", 0.1))

    atom_card, edge_card = infer_cardinalities_from_graphs(TRAIN_GRAPHS)
    gnn = MolGNN(atom_card, edge_card, hidden, emb_dim, layers, dropout=dropout).to(DEVICE)
    gnn.load_state_dict(ckpt["state_dict"], strict=False)
    gnn.eval()
    print(f"✓ GNN loaded from {GNN_CKPT}")

    # ---- Precompute train graph embeddings aligned to train_ids (needed for reranker features) ----
    print("Precomputing train graph embeddings...")
    train_g = embed_graphs(gnn, TRAIN_GRAPHS, train_ids, emb_dim, txt_mean=txt_mean)
    train_g = l2norm(train_g)
    print("✓ Train graph embeddings ready")

    # ---- Precompute val graph embeddings in same val order (for reranker validation) ----
    print("Precomputing val graph embeddings...")
    val_order_ids, val_g = embed_graphs_in_order(gnn, VAL_GRAPHS, txt_mean=txt_mean)
    # Map each val molecule to its "positive caption" index within TRAIN captions:
    # Here, we assume the positive for val query is its own caption (in val split), but we are retrieving from TRAIN captions.
    # For listwise training, we still need a target index in TRAIN space: we approximate by retrieving its own caption embedding among TRAIN captions
    # by nearest neighbor in text space (usually exact or very close for identical style).
    # If your split keeps unique IDs/captions across train/val, this is still a strong proxy.
    print("Building val positives (text NN into train captions)...")
    # For each val text, find nearest train text index
    sims_tv = val_txt @ train_txt.T
    val_pos_in_train = sims_tv.argmax(dim=-1)  # [n_val]
    print("✓ Val positives ready")

    # ---- Build train queries subset (train graphs as queries, positives are their own captions) ----
    print("Preparing train queries for reranker...")
    # We need train graph embeddings in dataset order to match queries.
    train_order_ids, train_q = embed_graphs_in_order(gnn, TRAIN_GRAPHS, txt_mean=txt_mean)
    # Map each query id to its caption position in train_ids
    id2trainpos = {tid: i for i, tid in enumerate(train_ids)}
    train_pos = torch.tensor([id2trainpos.get(i, -1) for i in train_order_ids], device=DEVICE)
    ok_mask = train_pos >= 0
    train_q = train_q[ok_mask]
    train_pos = train_pos[ok_mask]
    print(f"Train queries usable: {train_q.size(0)}")

    # Subsample for speed
    n_train = min(TRAIN_QUERY_SUBSAMPLE, train_q.size(0))
    idx_train = torch.randperm(train_q.size(0), device=DEVICE)[:n_train]
    train_q = train_q[idx_train]
    train_pos = train_pos[idx_train]

    n_val = min(VAL_QUERY_SUBSAMPLE, val_g.size(0))
    idx_val = torch.randperm(val_g.size(0), device=DEVICE)[:n_val]
    val_q = val_g[idx_val]
    val_pos = val_pos_in_train[idx_val]

    # ---- Precompute TOPK indices for train/val queries (fast matmul) ----
    print("Computing TOPK candidates for reranker training...")
    train_topk = topk_candidates(train_q, train_txt, topk=TOPK)  # [n_train, TOPK]
    val_topk   = topk_candidates(val_q,   train_txt, topk=TOPK)  # [n_val, TOPK]
    print("✓ TOPK ready")

    # ---- Train neural reranker ----
    reranker = NeuralReranker(emb_dim, hidden=512, dropout=0.15).to(DEVICE)
    opt = torch.optim.AdamW(reranker.parameters(), lr=RERANK_LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val = -1.0
    bad = 0

    print("\nTraining reranker...")
    for ep in range(1, RERANK_EPOCHS + 1):
        reranker.train()
        # shuffle queries
        perm = torch.randperm(train_q.size(0), device=DEVICE)
        train_loss = 0.0
        n_batches = 0

        for s in range(0, train_q.size(0), RERANK_BS):
            idx = perm[s : s + RERANK_BS]
            q = train_q[idx]
            pos = train_pos[idx]
            topk_idx = train_topk[idx]

            batch = build_list_batch(
                q_ids=None, q_emb=q, q_true_pos=pos,
                train_txt=train_txt, train_g=train_g,
                topk_idx=topk_idx, K=NEG_K
            )

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                scores = reranker(batch.q, batch.t, batch.g)        # [B,K]
                loss = F.cross_entropy(scores, batch.y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # ---- Validate (Recall@1 on list batches) ----
        reranker.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for s in range(0, val_q.size(0), RERANK_BS):
                q = val_q[s : s + RERANK_BS]
                pos = val_pos[s : s + RERANK_BS]
                topk_idx = val_topk[s : s + RERANK_BS]

                batch = build_list_batch(
                    q_ids=None, q_emb=q, q_true_pos=pos,
                    train_txt=train_txt, train_g=train_g,
                    topk_idx=topk_idx, K=NEG_K
                )
                scores = reranker(batch.q, batch.t, batch.g)
                pred = scores.argmax(dim=-1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.numel()

        val_r1 = correct / max(total, 1)
        print(f"  ep {ep}/{RERANK_EPOCHS} | train_loss={train_loss:.4f} | val_list_R@1={val_r1:.4f}")

        if val_r1 > best_val + 1e-4:
            best_val = val_r1
            bad = 0
            best_state = {k: v.detach().cpu() for k, v in reranker.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("  early stop reranker")
                break

    # load best
    reranker.load_state_dict(best_state, strict=True)
    reranker.eval()
    print(f"✓ Reranker ready (best val_list_R@1={best_val:.4f})\n")

    # ---- Inference on TEST ----
    print("Running test inference...")
    test_ds = PreprocessedGraphDataset(TEST_GRAPHS)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE_EMB, shuffle=False, collate_fn=collate_fn)

    results = []
    ptr = 0

    with torch.no_grad():
        for batch in test_dl:
            batch = batch.to(DEVICE)

            # Multi-query stabilization (tiny jitter average)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                q1 = gnn(batch)
            if txt_mean is not None:
                q1 = l2norm(q1 - txt_mean)

            # Small noise (helps ranking stability a bit)
            q2 = l2norm(q1 + 0.03 * torch.randn_like(q1))
            q = l2norm(0.5 * q1 + 0.5 * q2)

            # Stage 1 TOPK
            idx_topk = topk_candidates(q, train_txt, topk=TOPK)  # [B,TOPK]

            # Stage 2 rerank (use a larger candidate list than NEG_K at inference)
            # We score full TOPK in chunks to keep memory safe.
            B = q.size(0)
            best_idx = torch.zeros(B, dtype=torch.long, device=DEVICE)
            best_score = torch.full((B,), -1e9, device=DEVICE)

            # gather candidate tensors once
            cand_t_all = train_txt[idx_topk]          # [B,TOPK,D]
            cand_g_all = train_g[idx_topk]            # [B,TOPK,D]

            chunk = 64
            for s in range(0, TOPK, chunk):
                t_chunk = cand_t_all[:, s:s+chunk, :]
                g_chunk = cand_g_all[:, s:s+chunk, :]
                scores = reranker(q, t_chunk, g_chunk)  # [B,chunk]
                sc_max, sc_arg = scores.max(dim=-1)
                better = sc_max > best_score
                best_score[better] = sc_max[better]
                best_idx[better] = (s + sc_arg[better]).long()

            chosen_train_pos = idx_topk[torch.arange(B, device=DEVICE), best_idx]  # [B]
            for b in range(B):
                train_pos = int(chosen_train_pos[b].item())
                best_train_id = train_ids[train_pos]
                results.append({
                    "ID": test_ds.ids[ptr],
                    "description": train_id2desc[best_train_id],
                })
                ptr += 1

    df = pd.DataFrame(results)
    ensure_dir(OUT_CSV)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Saved {len(df)} predictions to {OUT_CSV}")

if __name__ == "__main__":
    main()
