from __future__ import annotations

"""
Compute graph embeddings for train/validation/test splits using a trained GINE encoder.
Outputs CSV files with columns: ID, embedding (comma-separated floats).
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_utils import batch_graphs_with_cache
from graph2text_utils import GraphOnlyDataset, select_device
from models_gine import GINEConfig, GINEEncoder


@dataclass
class GraphEmbeddingConfig:
    train_graph_path: str = "data/train_graphs.pkl"
    val_graph_path: str = "data/validation_graphs.pkl"
    test_graph_path: str = "data/test_graphs.pkl"
    train_out_csv: str = "data/train_graph_embeddings.csv"
    val_out_csv: str = "data/validation_graph_embeddings.csv"
    test_out_csv: str = "data/test_graph_embeddings.csv"
    ckpt_path: str = "ckpt_gine_contrastive/best.pt"
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frozen_gine_encoder(ckpt_path: str, device: str) -> GINEEncoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    gine_cfg = GINEConfig(**ckpt["gine_cfg"]) if "gine_cfg" in ckpt else GINEConfig()
    encoder = GINEEncoder(gine_cfg).to(device)
    state = ckpt.get("model_state", ckpt)
    enc_state = {k.replace("graph_encoder.", "", 1): v for k, v in state.items() if k.startswith("graph_encoder.")}
    if not enc_state:
        raise ValueError(f"No graph_encoder weights found in {ckpt_path}")
    encoder.load_state_dict(enc_state, strict=True)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


def collate_graphs(batch):
    return batch_graphs_with_cache(list(batch))


def write_embeddings_csv(path: Path, ids: List[str], embs: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "embedding"])
        for idx, emb in zip(ids, embs):
            writer.writerow([idx, ",".join(f"{x:.6f}" for x in emb.tolist())])
    print(f"[info] wrote {len(ids)} rows -> {path}")


@torch.inference_mode()
def process_split(
    split_name: str,
    dataset: GraphOnlyDataset,
    encoder: GINEEncoder,
    device: str,
    cfg: GraphEmbeddingConfig,
    out_csv: str,
    use_amp: bool,
):
    dl = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_graphs,
    )

    all_ids: List[str] = []
    all_embs: List[torch.Tensor] = []

    pbar = tqdm(dl, desc=f"encode_{split_name}", dynamic_ncols=True)
    offset = 0
    for batch_graph in pbar:
        batch_graph = batch_graph.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            graph_emb, _ = encoder(batch_graph)
        all_embs.append(graph_emb.cpu())
        batch_ids = dataset.ids[offset : offset + graph_emb.size(0)]
        all_ids.extend([str(i) for i in batch_ids])
        offset += graph_emb.size(0)

    embs_cat = torch.cat(all_embs, dim=0)
    write_embeddings_csv(Path(out_csv), all_ids, embs_cat)


def main():
    cfg = GraphEmbeddingConfig()
    device = select_device()
    seed_all(cfg.seed)
    use_amp = device == "cuda"

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"GINE checkpoint not found: {ckpt_path}")
    encoder = load_frozen_gine_encoder(str(ckpt_path), device)

    if Path(cfg.train_graph_path).exists():
        train_ds = GraphOnlyDataset(cfg.train_graph_path)
        process_split("train", train_ds, encoder, device, cfg, cfg.train_out_csv, use_amp)
    else:
        print(f"[warn] train graph file not found: {cfg.train_graph_path}")

    if Path(cfg.val_graph_path).exists():
        val_ds = GraphOnlyDataset(cfg.val_graph_path)
        process_split("val", val_ds, encoder, device, cfg, cfg.val_out_csv, use_amp)
    else:
        print(f"[warn] val graph file not found: {cfg.val_graph_path}")

    if Path(cfg.test_graph_path).exists():
        test_ds = GraphOnlyDataset(cfg.test_graph_path)
        process_split("test", test_ds, encoder, device, cfg, cfg.test_out_csv, use_amp)
    else:
        print(f"[warn] test graph file not found: {cfg.test_graph_path}")


if __name__ == "__main__":
    main()
