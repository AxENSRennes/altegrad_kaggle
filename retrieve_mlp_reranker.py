from __future__ import annotations

"""
Rerank test queries with a trained MLP reranker.

Pipeline:
  1) Encode test graphs with the GINE encoder + projection head.
  2) Build a train+val text database (embeddings + descriptions).
  3) Retrieve top-K candidates by cosine similarity.
  4) Rerank candidates with a trained MLP reranker.
  5) Write submission-style CSV.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data_utils import (
    PreprocessedGraphDataset,
    batch_graphs_with_cache,
    load_descriptions_from_graphs,
    load_id2emb,
)
from models_gine import GINEConfig, GINEEncoder
from train_gine_contrastive import load_checkpoint, normalize_emb_keys, select_device
from train_mlp_reranker import MLPReranker


# -----------------------
# Config
# -----------------------
@dataclass
class RerankConfig:
    test_graph_path: str = "data/test_graphs.pkl"
    ckpt_gine: str = "ckpt_gine_contrastive/best.pt"
    ckpt_reranker: str = "ckpt_mlp_reranker/best.pt"
    text_emb_csvs: Tuple[str, ...] = ("data/train_embeddings.csv", "data/validation_embeddings.csv")
    text_graph_paths: Tuple[str, ...] = ("data/train_graphs.pkl", "data/validation_graphs.pkl")
    topk: int = 30
    batch_size: int = 128
    encode_batch_size: int = 128
    retrieval_batch_size: int = 256
    num_workers: int = 0
    out_csv: str = "test_reranked.csv"
    submission_mode: bool = True


# -----------------------
# Utilities
# -----------------------
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


def load_retrieval_database(csv_paths: Tuple[str, ...], graph_paths: Tuple[str, ...]):
    id2emb = {}
    for path in csv_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing embeddings CSV: {path}")
        id2emb.update(normalize_emb_keys(load_id2emb(path)))

    id2desc = {}
    for path in graph_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing graph file: {path}")
        raw = load_descriptions_from_graphs(path)
        fixed = {}
        for k, v in raw.items():
            fixed[normalize_id(k)] = v
        id2desc.update(fixed)

    missing_desc = [i for i in id2emb if normalize_id(i) not in id2desc]
    if missing_desc:
        sample = ", ".join(str(i) for i in missing_desc[:5])
        raise KeyError(f"Missing {len(missing_desc)} descriptions. Sample IDs: {sample}")

    ids = list(id2emb.keys())
    texts = [id2desc[normalize_id(i)] for i in ids]
    return ids, texts, id2emb


def retrieve_topk(
    query_embs: torch.Tensor,
    db_embs: torch.Tensor,
    topk: int,
    device: str,
    batch_size: int,
):
    db_embs_device = db_embs.to(device)
    all_idx: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, query_embs.size(0), batch_size), desc="retrieve_topk", dynamic_ncols=True):
            chunk = query_embs[i : i + batch_size].to(device)
            sims = torch.matmul(chunk, db_embs_device.t())
            scores, idx = torch.topk(sims, k=topk, dim=1)
            all_idx.append(idx.cpu())
            all_scores.append(scores.cpu())
    return torch.cat(all_idx, dim=0), torch.cat(all_scores, dim=0)


def load_reranker(path: Path, embed_dim: int):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    hidden_dims = tuple(cfg.get("hidden_dims", (512, 256)))
    dropout = float(cfg.get("dropout", 0.1))
    include_graph = bool(cfg.get("include_graph", True))
    include_text = bool(cfg.get("include_text", True))
    include_mul = bool(cfg.get("include_mul", True))
    include_abs = bool(cfg.get("include_abs", True))
    include_cos = bool(cfg.get("include_cos", True))

    model = MLPReranker(
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        include_graph=include_graph,
        include_text=include_text,
        include_mul=include_mul,
        include_abs=include_abs,
        include_cos=include_cos,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, cfg


def rerank_candidates(
    model: MLPReranker,
    query_embs: torch.Tensor,
    db_embs: torch.Tensor,
    topk_idx: torch.Tensor,
    device: str,
    batch_size: int,
):
    model.eval()
    best_idx: List[torch.Tensor] = []
    best_scores: List[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, query_embs.size(0), batch_size), desc="rerank", dynamic_ncols=True):
            q = query_embs[i : i + batch_size].to(device)
            cand_idx = topk_idx[i : i + batch_size]
            cand_emb = db_embs[cand_idx].to(device)
            scores = model(q, cand_emb)
            local_best = scores.argmax(dim=1)
            row = torch.arange(scores.size(0), device=device)
            best = cand_idx.to(device)[row, local_best]
            best_idx.append(best.cpu())
            best_scores.append(scores.max(dim=1).values.cpu())
    return torch.cat(best_idx, dim=0), torch.cat(best_scores, dim=0)


# -----------------------
# Main
# -----------------------
def main():
    cfg = RerankConfig()
    device = select_device()

    ckpt_gine = Path(cfg.ckpt_gine)
    if not ckpt_gine.exists():
        raise FileNotFoundError(f"GINE checkpoint not found: {ckpt_gine}")
    ckpt_reranker = Path(cfg.ckpt_reranker)
    if not ckpt_reranker.exists():
        raise FileNotFoundError(f"Reranker checkpoint not found: {ckpt_reranker}")

    test_graph_path = Path(cfg.test_graph_path)
    if not test_graph_path.exists():
        raise FileNotFoundError(f"Test graphs not found: {test_graph_path}")

    # Encode test graphs
    encoder, _, state = load_graph_encoder(ckpt_gine, device)
    graph_proj, text_proj = build_projections(state)
    graph_proj = graph_proj.to(device)
    text_proj = text_proj.to(device)

    test_ds = GraphIdDataset(cfg.test_graph_path)
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.encode_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=graph_collate,
    )
    test_ids, test_graph_embs = encode_graphs(encoder, graph_proj, test_dl, device)

    # Load retrieval database (train + val)
    db_ids, db_texts, id2emb = load_retrieval_database(cfg.text_emb_csvs, cfg.text_graph_paths)
    db_ids, db_embs = project_text_embeddings(id2emb, text_proj, device, cfg.encode_batch_size)

    # Stage 1: cosine retrieval
    topk_idx, _ = retrieve_topk(
        test_graph_embs, db_embs, cfg.topk, device, cfg.retrieval_batch_size
    )

    # Stage 2: MLP reranking
    embed_dim = test_graph_embs.size(1)
    if db_embs.size(1) != embed_dim:
        raise ValueError(
            f"Embedding dims mismatch: graph={embed_dim} text={db_embs.size(1)}. "
            "Check your checkpoint projections."
        )
    reranker, _ = load_reranker(ckpt_reranker, embed_dim)
    reranker = reranker.to(device)

    best_idx, best_scores = rerank_candidates(
        reranker, test_graph_embs, db_embs, topk_idx, device, cfg.batch_size
    )

    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        if cfg.submission_mode:
            writer.writerow(["ID", "description"])
            for idx, pred_i in zip(test_ids, best_idx.tolist()):
                writer.writerow([idx, db_texts[pred_i]])
        else:
            writer.writerow(["ID", "description", "rerank_score"])
            for idx, pred_i, score in zip(test_ids, best_idx.tolist(), best_scores.tolist()):
                writer.writerow([idx, db_texts[pred_i], f"{score:.4f}"])
    print(f"[info] wrote reranked results to {out_path}")


if __name__ == "__main__":
    main()
