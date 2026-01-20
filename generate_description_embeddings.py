from __future__ import annotations

"""
Compute and store description embeddings for train/val graphs using a frozen E5 encoder.
Outputs CSV files with columns: ID, embedding (comma-separated floats).
"""

import csv
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class EmbeddingConfig:
    train_graph_path: str = "data/train_graphs_cached.pkl"
    val_graph_path: str = "data/validation_graphs_cached.pkl"
    train_out_csv: str = "data/train_embeddings.csv"
    val_out_csv: str = "data/validation_embeddings.csv"
    text_model: str = "intfloat/e5-large-v2"
    max_len: int = 300
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    prefix: str = "passage: "


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class GraphDescriptionDataset(Dataset):
    """
    Yields (graph_id, description_text) from a pickled list of graphs.
    """

    def __init__(self, pkl_path: str):
        self.pkl_path = pkl_path
        with open(pkl_path, "rb") as f:
            self.graphs = pickle.load(f)
        self.ids = [getattr(g, "id", i) for i, g in enumerate(self.graphs)]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> Tuple[int, str]:
        g = self.graphs[idx]
        desc = getattr(g, "description", "")
        return int(self.ids[idx]), desc


def collate_fn(batch: List[Tuple[int, str]]):
    ids, texts = zip(*batch)
    return list(ids), list(texts)


def encode_batch(
    texts: Sequence[str],
    tokenizer,
    model,
    device: str,
    max_len: int,
    use_amp: bool,
    prefix: str,
) -> torch.Tensor:
    if prefix:
        texts = [prefix + t for t in texts]
    tok = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            out = model(**tok)
    emb = mean_pool(out.last_hidden_state, tok["attention_mask"])
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


def write_embeddings_csv(path: Path, ids: List[int], embs: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "embedding"])
        for idx, emb in zip(ids, embs):
            writer.writerow([idx, ",".join(f"{x:.6f}" for x in emb.tolist())])
    print(f"[info] wrote {len(ids)} rows -> {path}")


def process_split(
    split_name: str,
    dataset: GraphDescriptionDataset,
    tokenizer,
    model,
    cfg: EmbeddingConfig,
    device: str,
    out_csv: str,
    use_amp: bool,
):
    dl = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    all_ids: List[int] = []
    all_embs: List[torch.Tensor] = []

    pbar = tqdm(dl, desc=f"encode_{split_name}", dynamic_ncols=True)
    for ids, texts in pbar:
        embs = encode_batch(texts, tokenizer, model, device, cfg.max_len, use_amp, cfg.prefix)
        all_ids.extend(ids)
        all_embs.append(embs.cpu())

    embs_cat = torch.cat(all_embs, dim=0)
    write_embeddings_csv(Path(out_csv), all_ids, embs_cat)


def main():
    cfg = EmbeddingConfig()
    device = select_device()
    seed_all(cfg.seed)
    use_amp = device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
    model = AutoModel.from_pretrained(cfg.text_model).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if Path(cfg.train_graph_path).exists():
        train_ds = GraphDescriptionDataset(cfg.train_graph_path)
        process_split("train", train_ds, tokenizer, model, cfg, device, cfg.train_out_csv, use_amp)
    else:
        print(f"[warn] train graph file not found: {cfg.train_graph_path}")

    if Path(cfg.val_graph_path).exists():
        val_ds = GraphDescriptionDataset(cfg.val_graph_path)
        process_split("val", val_ds, tokenizer, model, cfg, device, cfg.val_out_csv, use_amp)
    else:
        print(f"[warn] val graph file not found: {cfg.val_graph_path}")


if __name__ == "__main__":
    main()
