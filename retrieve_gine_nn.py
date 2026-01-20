from __future__ import annotations

"""
Nearest-neighbor retrieval with the GINE encoder.

This script:
  - loads the best contrastive checkpoint
  - encodes query graphs
  - loads a retrieval database of descriptions (train+val by default)
  - retrieves the nearest description for each graph by cosine similarity
  - writes a CSV comparing reference vs retrieved descriptions.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data_utils import (
    PreprocessedGraphDataset,
    batch_graphs_with_cache,
    load_descriptions_from_graphs,
    load_id2emb,
)
from models_gine import GINEConfig, GINEEncoder
from train_gine_contrastive import FrozenTextEncoder, load_checkpoint, normalize_emb_keys, select_device


@dataclass
class RetrievalConfig:
    val_graph_path: str = "data/test_graphs_func_groups.pkl"
    ckpt_path: str = "ckpt_gine_contrastive/best.pt"
    text_emb_csvs: Tuple[str, ...] = ("data/train_embeddings.csv", "data/validation_embeddings.csv")
    text_graph_paths: Tuple[str, ...] = ("data/train_graphs_func_groups.pkl", "data/validation_graphs_func_groups.pkl")
    batch_size: int = 64
    num_workers: int = 0
    out_csv: str = "test_retrieval.csv"
    submission_mode: bool = True


class GraphWithTextDataset(Dataset):
    """
    Returns (graph, description, id).
    """

    def __init__(self, graph_path: str):
        self.base = PreprocessedGraphDataset(graph_path, emb_dict=None)
        self.graphs = self.base.graphs
        self.ids = self.base.ids

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        desc = getattr(g, "description", "")
        return g, desc, self.ids[idx]


def collate_fn(batch):
    graphs, descs, ids = zip(*batch)
    batch_graph = batch_graphs_with_cache(list(graphs))
    return batch_graph, list(descs), list(ids)


def load_graph_encoder(ckpt_path: Path, device: str):
    state, gine_cfg, train_cfg, best_loss = load_checkpoint(ckpt_path, device)
    gine_cfg = gine_cfg or GINEConfig()

    encoder = GINEEncoder(gine_cfg).to(device)
    # Filter graph_encoder.* keys
    enc_state = {k.replace("graph_encoder.", "", 1): v for k, v in state.items() if k.startswith("graph_encoder.")}
    encoder.load_state_dict(enc_state, strict=True)

    return encoder, train_cfg, state


def build_projections(state: dict):
    # Graph projection
    if "graph_proj.weight" in state:
        g_w = state["graph_proj.weight"]
        g_b = state.get("graph_proj.bias", None)
        graph_proj = nn.Linear(g_w.size(1), g_w.size(0), bias=g_b is not None)
        graph_proj.weight.data.copy_(g_w)
        if g_b is not None:
            graph_proj.bias.data.copy_(g_b)
    else:
        graph_proj = nn.Identity()

    # Text projection
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


def encode_graphs(encoder: GINEEncoder, dl, device: str) -> Tuple[List[int], torch.Tensor]:
    encoder.eval()
    ids_all: List[int] = []
    embs: List[torch.Tensor] = []
    with torch.no_grad():
        for graphs, _, ids in tqdm(dl, desc="graph_enc", dynamic_ncols=True):
            graphs = graphs.to(device)
            g_emb, _ = encoder(graphs)
            embs.append(g_emb.cpu())
            ids_all.extend(ids)
    return ids_all, torch.cat(embs, dim=0)


def encode_texts(texts: List[str], text_encoder: FrozenTextEncoder, device: str, batch_size: int) -> torch.Tensor:
    text_encoder.model.to(device)
    all_embs: List[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="text_enc", dynamic_ncols=True):
        chunk = texts[i : i + batch_size]
        embs = text_encoder.encode(chunk, device)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)

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
            try:
                fixed[int(k)] = v
            except (ValueError, TypeError):
                fixed[k] = v
        id2desc.update(fixed)

    missing_desc = [i for i in id2emb if i not in id2desc]
    if missing_desc:
        sample = ", ".join(str(i) for i in missing_desc[:5])
        raise KeyError(f"Missing {len(missing_desc)} descriptions for embeddings. Sample IDs: {sample}")

    ids = list(id2emb.keys())
    text_embs = torch.stack([id2emb[i] for i in ids], dim=0)
    texts = [id2desc[i] for i in ids]
    return ids, texts, text_embs


def load_text_embeddings(ids: List[int], csv_paths: Tuple[str, ...]) -> torch.Tensor:
    id2emb = {}
    for path in csv_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing embeddings CSV: {path}")
        id2emb.update(normalize_emb_keys(load_id2emb(path)))
    missing = [i for i in ids if i not in id2emb]
    if missing:
        sample = ", ".join(str(i) for i in missing[:5])
        raise KeyError(f"Missing {len(missing)} embeddings in {csv_paths}. Sample IDs: {sample}")
    return torch.stack([id2emb[i] for i in ids], dim=0)


def main():
    cfg = RetrievalConfig()
    device = select_device()

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Data
    val_ds = GraphWithTextDataset(cfg.val_graph_path)
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    # Load encoder and text encoder config
    encoder, train_cfg, state = load_graph_encoder(ckpt_path, device)
    graph_proj, text_proj = build_projections(state)
    graph_proj = graph_proj.to(device)
    text_proj = text_proj.to(device)
    text_model = "bert-base-uncased"
    max_len = 128
    if isinstance(train_cfg, dict):
        text_model = train_cfg.get("text_model", text_model)
        max_len = train_cfg.get("max_text_len", max_len)

    # Encode texts and graphs
    all_texts = [getattr(g, "description", "") for g in val_ds.graphs]
    ids, graph_embs = encode_graphs(encoder, val_dl, device)
    if cfg.text_emb_csvs and cfg.text_graph_paths:
        retrieval_ids, retrieval_texts, text_embs = load_retrieval_database(
            cfg.text_emb_csvs, cfg.text_graph_paths
        )
    else:
        retrieval_texts = all_texts
        text_encoder = FrozenTextEncoder(text_model, max_len)
        if text_encoder.tokenizer.pad_token is None:
            text_encoder.tokenizer.pad_token = text_encoder.tokenizer.eos_token or text_encoder.tokenizer.cls_token
        text_embs = encode_texts(retrieval_texts, text_encoder, device, cfg.batch_size)

    # Apply projections and cosine similarity
    graph_embs = graph_proj(graph_embs.to(device))
    text_embs = text_proj(text_embs.to(device))
    graph_embs = torch.nn.functional.normalize(graph_embs, p=2, dim=-1)
    text_embs = torch.nn.functional.normalize(text_embs, p=2, dim=-1)
    sims = torch.matmul(graph_embs, text_embs.t())  # [N, N]
    top_idx = sims.argmax(dim=1)
    top_sim = sims.max(dim=1).values

    # Write CSV
    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        if cfg.submission_mode:
            writer.writerow(["ID", "description"])
            for idx, pred_i in zip(ids, top_idx.tolist()):
                writer.writerow([idx, retrieval_texts[pred_i]])
        else:
            writer.writerow(["ID", "ref_description", "retrieved_description", "cosine_sim"])
            for idx, ref_desc, pred_i, sim in zip(ids, all_texts, top_idx.tolist(), top_sim.tolist()):
                writer.writerow([idx, ref_desc, retrieval_texts[pred_i], f"{sim:.4f}"])
    print(f"[info] wrote retrieval results to {out_path}")


if __name__ == "__main__":
    main()
