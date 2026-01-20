from __future__ import annotations

"""
Train a listwise MLP reranker for GINE retrieval.

Pipeline:
  1) Build a train-only retrieval database (text embeddings + descriptions).
  2) Mine top-K negatives for each query by cosine similarity.
  3) Train an MLP with listwise softmax loss over (pos + K negs).
  4) Validate by reranking accuracy / Recall@K / MRR.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data_utils import PreprocessedGraphDataset, batch_graphs_with_cache, load_id2emb
from models_gine import GINEConfig, GINEEncoder
from train_gine_contrastive import load_checkpoint, normalize_emb_keys, select_device


# -----------------------
# Config
# -----------------------
@dataclass
class RerankerConfig:
    train_graph_path: str = "data/train_graphs.pkl"
    val_graph_path: str | None = "data/validation_graphs.pkl"
    train_text_emb_csv: str = "data/train_embeddings.csv"
    val_text_emb_csv: str | None = "data/validation_embeddings.csv"
    ckpt_path: str = "ckpt_gine_contrastive/best.pt"
    save_dir: str = "ckpt_mlp_reranker"
    batch_size: int = 128
    eval_batch_size: int = 256
    encode_batch_size: int = 128
    retrieval_batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    topk: int = 30
    num_workers: int = 0
    seed: int = 42
    use_amp: bool = False
    dropout: float = 0.1
    hidden_dims: Tuple[int, ...] = (512, 256)
    include_graph: bool = True
    include_text: bool = True
    include_mul: bool = True
    include_abs: bool = True
    include_cos: bool = True
    metric_for_best: str = "mrr"  # "mrr" or "acc"
    log_every: int = 100


# -----------------------
# Utilities
# -----------------------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class GraphIdDataset(Dataset):
    """
    Returns (graph, id).
    """

    def __init__(self, graph_path: str):
        self.base = PreprocessedGraphDataset(graph_path, emb_dict=None)
        self.graphs = self.base.graphs
        self.ids = self.base.ids

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.ids[idx]


def graph_collate(batch):
    graphs, ids = zip(*batch)
    batch_graph = batch_graphs_with_cache(list(graphs))
    return batch_graph, list(ids)


class CandidateDataset(Dataset):
    """
    Returns (query_graph_emb, pos_text_emb, neg_indices).
    """

    def __init__(self, query_embs: torch.Tensor, pos_embs: torch.Tensor, neg_indices: torch.Tensor):
        if query_embs.size(0) != pos_embs.size(0) or query_embs.size(0) != neg_indices.size(0):
            raise ValueError("Query, pos, and neg tensors must align on the first dimension.")
        self.query_embs = query_embs
        self.pos_embs = pos_embs
        self.neg_indices = neg_indices

    def __len__(self):
        return self.query_embs.size(0)

    def __getitem__(self, idx):
        return self.query_embs[idx], self.pos_embs[idx], self.neg_indices[idx]


class MLPReranker(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
        include_graph: bool = True,
        include_text: bool = True,
        include_mul: bool = True,
        include_abs: bool = True,
        include_cos: bool = True,
    ):
        super().__init__()
        self.include_graph = include_graph
        self.include_text = include_text
        self.include_mul = include_mul
        self.include_abs = include_abs
        self.include_cos = include_cos

        in_dim = 0
        if include_graph:
            in_dim += embed_dim
        if include_text:
            in_dim += embed_dim
        if include_mul:
            in_dim += embed_dim
        if include_abs:
            in_dim += embed_dim
        if include_cos:
            in_dim += 1
        if in_dim == 0:
            raise ValueError("At least one feature component must be enabled.")

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def build_features(self, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        g: [B, D], t: [B, C, D]
        returns [B, C, F]
        """
        g_exp = g.unsqueeze(1)
        feats: List[torch.Tensor] = []
        if self.include_graph:
            feats.append(g_exp.expand(-1, t.size(1), -1))
        if self.include_text:
            feats.append(t)
        if self.include_mul:
            feats.append(g_exp * t)
        if self.include_abs:
            feats.append((g_exp - t).abs())
        if self.include_cos:
            feats.append((g_exp * t).sum(dim=-1, keepdim=True))
        return torch.cat(feats, dim=-1)

    def forward(self, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feats = self.build_features(g, t)
        bsz, cands, feat_dim = feats.shape
        scores = self.mlp(feats.view(bsz * cands, feat_dim)).view(bsz, cands)
        return scores


# -----------------------
# Embedding pipeline
# -----------------------
def load_graph_encoder(ckpt_path: Path, device: str):
    state, gine_cfg, train_cfg, best_loss = load_checkpoint(ckpt_path, device)
    gine_cfg = gine_cfg or GINEConfig()

    encoder = GINEEncoder(gine_cfg).to(device)
    enc_state = {k.replace("graph_encoder.", "", 1): v for k, v in state.items() if k.startswith("graph_encoder.")}
    encoder.load_state_dict(enc_state, strict=True)
    return encoder, train_cfg, state


def build_projections(state: dict):
    if "graph_proj.weight" in state:
        g_w = state["graph_proj.weight"]
        g_b = state.get("graph_proj.bias", None)
        graph_proj = nn.Linear(g_w.size(1), g_w.size(0), bias=g_b is not None)
        graph_proj.weight.data.copy_(g_w)
        if g_b is not None:
            graph_proj.bias.data.copy_(g_b)
    else:
        graph_proj = nn.Identity()

    if "text_proj.weight" in state:
        t_w = state["text_proj.weight"]
        t_b = state.get("text_proj.bias", None)
        text_proj = nn.Linear(t_w.size(1), t_w.size(0), bias=t_b is not None)
        text_proj.weight.data.copy_(t_w)
        if t_b is not None:
            text_proj.bias.data.copy_(t_b)
    else:
        text_proj = nn.Identity()

    return graph_proj, text_proj


def encode_graphs(encoder: GINEEncoder, graph_proj: nn.Module, dl, device: str):
    encoder.eval()
    graph_proj.eval()
    ids_all: List[int] = []
    embs: List[torch.Tensor] = []
    with torch.no_grad():
        for graphs, ids in tqdm(dl, desc="graph_enc", dynamic_ncols=True):
            graphs = graphs.to(device)
            g_emb, _ = encoder(graphs)
            g_emb = graph_proj(g_emb)
            g_emb = F.normalize(g_emb, p=2, dim=-1)
            embs.append(g_emb.cpu())
            ids_all.extend(ids)
    return ids_all, torch.cat(embs, dim=0)


def project_text_embeddings(id2emb: dict, text_proj: nn.Module, device: str, batch_size: int):
    ids = list(id2emb.keys())
    text_proj.eval()
    out: List[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(ids), batch_size), desc="text_proj", dynamic_ncols=True):
            chunk_ids = ids[i : i + batch_size]
            chunk = torch.stack([id2emb[j] for j in chunk_ids], dim=0)
            chunk = text_proj(chunk.to(device))
            chunk = F.normalize(chunk, p=2, dim=-1)
            out.append(chunk.cpu())
    return ids, torch.cat(out, dim=0)


def project_text_for_ids(ids: List[int], id2emb: dict, text_proj: nn.Module, device: str, batch_size: int):
    text_proj.eval()
    out: List[torch.Tensor] = []
    norm_ids = [normalize_id(i) for i in ids]
    with torch.no_grad():
        for i in tqdm(range(0, len(norm_ids), batch_size), desc="pos_text_proj", dynamic_ncols=True):
            chunk_ids = norm_ids[i : i + batch_size]
            missing = [j for j in chunk_ids if j not in id2emb]
            if missing:
                sample = ", ".join(str(j) for j in missing[:5])
                raise KeyError(f"Missing {len(missing)} embeddings. Sample IDs: {sample}")
            chunk = torch.stack([id2emb[j] for j in chunk_ids], dim=0)
            chunk = text_proj(chunk.to(device))
            chunk = F.normalize(chunk, p=2, dim=-1)
            out.append(chunk.cpu())
    return torch.cat(out, dim=0)


def mine_topk_indices(
    query_ids: List[int],
    query_embs: torch.Tensor,
    db_ids: List[int],
    db_embs: torch.Tensor,
    topk: int,
    device: str,
    batch_size: int,
    exclude_self: bool = True,
):
    db_id_to_idx = {normalize_id(i): idx for idx, i in enumerate(db_ids)}
    max_k = db_embs.size(0) - (1 if exclude_self else 0)
    if topk > max_k:
        raise ValueError(f"topk={topk} exceeds available candidates ({max_k}).")

    db_embs_device = db_embs.to(device)
    all_indices: List[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(query_ids), batch_size), desc="mine_topk", dynamic_ncols=True):
            chunk_ids = query_ids[i : i + batch_size]
            chunk = query_embs[i : i + batch_size].to(device)
            sims = torch.matmul(chunk, db_embs_device.t())
            if exclude_self:
                for row, qid in enumerate(chunk_ids):
                    db_idx = db_id_to_idx.get(normalize_id(qid))
                    if db_idx is not None:
                        sims[row, db_idx] = -float("inf")
            top_idx = torch.topk(sims, k=topk, dim=1).indices.cpu()
            all_indices.append(top_idx)
    return torch.cat(all_indices, dim=0)


# -----------------------
# Training / eval
# -----------------------
def train_epoch(
    model: MLPReranker,
    dl: DataLoader,
    db_embs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: str,
    cfg: RerankerConfig,
    scaler: torch.cuda.amp.GradScaler | None = None,
):
    model.train()
    total_loss = 0.0
    total = 0
    use_amp = scaler is not None

    for step, (q_emb, pos_emb, neg_idx) in enumerate(tqdm(dl, desc="train", dynamic_ncols=True)):
        q_emb = q_emb.to(device)
        pos_emb = pos_emb.to(device)
        neg_emb = db_embs[neg_idx].to(device)
        candidates = torch.cat([pos_emb.unsqueeze(1), neg_emb], dim=1)
        targets = torch.zeros(q_emb.size(0), dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                scores = model(q_emb, candidates)
                loss = F.cross_entropy(scores, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            scores = model(q_emb, candidates)
            loss = F.cross_entropy(scores, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * q_emb.size(0)
        total += q_emb.size(0)
        if cfg.log_every > 0 and (step + 1) % cfg.log_every == 0:
            avg = total_loss / max(1, total)
            print(f"[train] step={step + 1} avg_loss={avg:.4f}")

    return total_loss / max(1, total)


def eval_epoch(
    model: MLPReranker,
    dl: DataLoader,
    db_embs: torch.Tensor,
    device: str,
):
    model.eval()
    total = 0
    sum_acc = 0.0
    sum_rec1 = 0.0
    sum_rec5 = 0.0
    sum_rec10 = 0.0
    sum_mrr = 0.0

    with torch.no_grad():
        for q_emb, pos_emb, neg_idx in tqdm(dl, desc="val", dynamic_ncols=True):
            q_emb = q_emb.to(device)
            pos_emb = pos_emb.to(device)
            neg_emb = db_embs[neg_idx].to(device)
            candidates = torch.cat([pos_emb.unsqueeze(1), neg_emb], dim=1)

            scores = model(q_emb, candidates)
            pos_scores = scores[:, 0]
            ranks = 1 + (scores > pos_scores.unsqueeze(1)).sum(dim=1)

            total += q_emb.size(0)
            sum_acc += (ranks == 1).sum().item()
            sum_rec1 += (ranks <= 1).sum().item()
            sum_rec5 += (ranks <= 5).sum().item()
            sum_rec10 += (ranks <= 10).sum().item()
            sum_mrr += (1.0 / ranks.float()).sum().item()

    if total == 0:
        return {"acc": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    return {
        "acc": sum_acc / total,
        "recall@1": sum_rec1 / total,
        "recall@5": sum_rec5 / total,
        "recall@10": sum_rec10 / total,
        "mrr": sum_mrr / total,
    }


def save_checkpoint(path: Path, model: MLPReranker, cfg: RerankerConfig, best_metric: float):
    payload = {
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "best_metric": best_metric,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[info] Saved checkpoint to {path}")


# -----------------------
# Main
# -----------------------
def main():
    cfg = RerankerConfig()
    device = select_device()
    seed_all(cfg.seed)

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    train_graph_path = Path(cfg.train_graph_path)
    if not train_graph_path.exists():
        raise FileNotFoundError(f"Train graphs not found: {train_graph_path}")

    if not Path(cfg.train_text_emb_csv).exists():
        raise FileNotFoundError(f"Train embeddings CSV not found: {cfg.train_text_emb_csv}")

    encoder, _, state = load_graph_encoder(ckpt_path, device)
    encoder.eval()
    graph_proj, text_proj = build_projections(state)
    graph_proj = graph_proj.to(device)
    text_proj = text_proj.to(device)
    graph_proj.eval()
    text_proj.eval()

    train_graph_ds = GraphIdDataset(cfg.train_graph_path)
    train_graph_dl = DataLoader(
        train_graph_ds,
        batch_size=cfg.encode_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=graph_collate,
    )
    train_ids, train_graph_embs = encode_graphs(encoder, graph_proj, train_graph_dl, device)

    train_id2emb = normalize_emb_keys(load_id2emb(cfg.train_text_emb_csv))
    train_db_ids, train_db_embs = project_text_embeddings(
        train_id2emb, text_proj, device, cfg.encode_batch_size
    )
    db_id_to_idx = {normalize_id(i): idx for idx, i in enumerate(train_db_ids)}

    missing_train = [i for i in train_ids if normalize_id(i) not in db_id_to_idx]
    if missing_train:
        sample = ", ".join(str(i) for i in missing_train[:5])
        raise KeyError(f"Missing {len(missing_train)} train embeddings. Sample IDs: {sample}")

    train_pos_indices = torch.tensor([db_id_to_idx[normalize_id(i)] for i in train_ids], dtype=torch.long)
    train_pos_embs = train_db_embs[train_pos_indices]

    train_neg_indices = mine_topk_indices(
        train_ids,
        train_graph_embs,
        train_db_ids,
        train_db_embs,
        cfg.topk,
        device,
        cfg.retrieval_batch_size,
        exclude_self=True,
    )

    train_dataset = CandidateDataset(train_graph_embs, train_pos_embs, train_neg_indices)
    train_dl = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_dl = None
    if cfg.val_graph_path and cfg.val_text_emb_csv:
        val_graph_path = Path(cfg.val_graph_path)
        if val_graph_path.exists() and Path(cfg.val_text_emb_csv).exists():
            val_graph_ds = GraphIdDataset(cfg.val_graph_path)
            val_graph_dl = DataLoader(
                val_graph_ds,
                batch_size=cfg.encode_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=graph_collate,
            )
            val_ids, val_graph_embs = encode_graphs(encoder, graph_proj, val_graph_dl, device)

            val_id2emb = normalize_emb_keys(load_id2emb(cfg.val_text_emb_csv))
            val_pos_embs = project_text_for_ids(
                val_ids, val_id2emb, text_proj, device, cfg.encode_batch_size
            )

            val_neg_indices = mine_topk_indices(
                val_ids,
                val_graph_embs,
                train_db_ids,
                train_db_embs,
                cfg.topk,
                device,
                cfg.retrieval_batch_size,
                exclude_self=True,
            )

            val_dataset = CandidateDataset(val_graph_embs, val_pos_embs, val_neg_indices)
            val_dl = DataLoader(
                val_dataset,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
            )
        else:
            print("[warn] Validation paths not found; skipping validation.")

    embed_dim = train_graph_embs.size(1)
    if train_db_embs.size(1) != embed_dim:
        raise ValueError(
            f"Embedding dims mismatch: graph={embed_dim} text={train_db_embs.size(1)}. "
            "Check your checkpoint projections."
        )

    model = MLPReranker(
        embed_dim=embed_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        include_graph=cfg.include_graph,
        include_text=cfg.include_text,
        include_mul=cfg.include_mul,
        include_abs=cfg.include_abs,
        include_cos=cfg.include_cos,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.use_amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric = -math.inf
    best_path = Path(cfg.save_dir) / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        print(f"[info] Epoch {epoch}/{cfg.epochs}")
        train_loss = train_epoch(model, train_dl, train_db_embs, optimizer, device, cfg, scaler)
        print(f"[info] Train loss: {train_loss:.4f}")

        if val_dl is not None:
            metrics = eval_epoch(model, val_dl, train_db_embs, device)
            metric_val = metrics.get(cfg.metric_for_best, metrics["mrr"])
            print(
                f"[info] Val acc={metrics['acc']:.4f} "
                f"R@1={metrics['recall@1']:.4f} "
                f"R@5={metrics['recall@5']:.4f} "
                f"R@10={metrics['recall@10']:.4f} "
                f"MRR={metrics['mrr']:.4f}"
            )
            if metric_val > best_metric:
                best_metric = metric_val
                save_checkpoint(best_path, model, cfg, best_metric)
        else:
            if train_loss > best_metric:
                best_metric = train_loss
                save_checkpoint(best_path, model, cfg, best_metric)


if __name__ == "__main__":
    main()
