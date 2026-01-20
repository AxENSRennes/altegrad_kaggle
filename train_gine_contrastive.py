from __future__ import annotations

"""
Contrastive pretraining for the graph encoder using a frozen BERT text embedder.

Stage 1: encode molecule graphs with a small GINEConv encoder and align them to
frozen E5 sentence embeddings via a symmetric InfoNCE loss on cosine similarity.
This trains only the graph encoder (and small projection/temperature parameters).
"""

import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from data_utils import PreprocessedGraphDataset, batch_graphs_with_cache, load_id2emb
from models_gine import GINEConfig, GINEEncoder


# -----------------------
# Config
# -----------------------
@dataclass
class TrainingConfig:
    train_graph_path: str = "data/train_graphs.pkl"
    val_graph_path: str | None = "data/validation_graphs.pkl"
    train_text_emb_csv: str | None = "data/train_embeddings.csv"  # optional precomputed embeddings for train split
    val_text_emb_csv: str | None = "data/validation_embeddings.csv"    # optional precomputed embeddings for val split
    text_model: str = "intfloat/e5-large-v2"
    max_text_len: int = 200
    batch_size: int = 64
    epochs: int = 20
    lr: float = 2e-4
    min_lr: float = 5e-5
    weight_decay: float = 1e-4
    grad_accum: int = 1
    temperature: float = 0.07
    learn_temperature: bool = False
    logit_scale_max: float = 100.0
    proj_dim: int | None = None  # None -> match text hidden size
    use_amp: bool = False
    seed: int = 42
    save_dir: str = "ckpt_gine_contrastive"
    num_workers: int = 0
    resume_from_best: bool = True
    queue_size: int = 65536
    metric_for_best: str = "mrr"  # "loss" or "mrr"


# -----------------------
# Utilities
# -----------------------
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


def normalize_emb_keys(id2emb: dict | None):
    """
    Graph IDs are stored as ints in the cached pickles; make sure the embedding
    dict uses matching keys even if loaded from CSV as strings.
    """
    if id2emb is None:
        return None
    fixed = {}
    for k, v in id2emb.items():
        try:
            fixed[int(k)] = v
        except (ValueError, TypeError):
            fixed[k] = v
    return fixed


# -----------------------
# Data
# -----------------------
class ContrastiveDataset(Dataset):
    """
    Returns (graph, description_text) or (graph, precomputed_text_embedding).
    """

    def __init__(self, graph_path: str, text_emb_dict: dict | None = None):
        self.base = PreprocessedGraphDataset(graph_path, emb_dict=text_emb_dict)
        self.use_precomputed = text_emb_dict is not None
        self.ids = self.base.ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.use_precomputed:
            return self.base[idx]
        g = self.base.graphs[idx]
        desc = getattr(g, "description", "")
        return g, desc


def contrastive_collate(batch):
    graphs, texts_or_embs = zip(*batch)
    batch_graph = batch_graphs_with_cache(list(graphs))

    first = texts_or_embs[0]
    if isinstance(first, torch.Tensor):
        text_batch = torch.stack(texts_or_embs, dim=0)
    else:
        text_batch = list(texts_or_embs)
    return batch_graph, text_batch


# -----------------------
# Encoders and contrastive head
# -----------------------
class FrozenTextEncoder(nn.Module):
    """
    Frozen E5 encoder with mean pooling + L2 normalization.
    """

    def __init__(self, model_name: str, max_len: int, prefix: str = "passage: "):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.max_len = max_len
        self.hidden_size = int(self.model.config.hidden_size)
        self.prefix = prefix

    @torch.no_grad()
    def encode(self, texts: Sequence[str], device: str) -> torch.Tensor:
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        tok = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)
        self.model.to(device)
        out = self.model(**tok)
        emb = mean_pool(out.last_hidden_state, tok["attention_mask"])
        return F.normalize(emb, p=2, dim=-1)


class GraphTextContrastive(nn.Module):
    """
    Graph encoder + projection + contrastive loss against text embeddings.
    """

    def __init__(
        self,
        graph_encoder: GINEEncoder,
        text_dim: int,
        proj_dim: int,
        temperature: float = 0.07,
        learn_temperature: bool = False,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.graph_proj = nn.Linear(graph_encoder.output_dim, proj_dim)
        self.text_proj = nn.Identity() if text_dim == proj_dim else nn.Linear(text_dim, proj_dim, bias=False)

        logit_scale = math.log(1.0 / temperature)
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale), requires_grad=learn_temperature)

    def forward(self, graphs, text_emb: torch.Tensor, text_queue: torch.Tensor | None = None, max_logit_scale: float = 100.0):
        graph_emb, _ = self.graph_encoder(graphs)
        g = self.graph_proj(graph_emb)
        t = self.text_proj(text_emb)

        g = F.normalize(g, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=max_logit_scale)
        if text_queue is not None:
            all_t = torch.cat([t, text_queue], dim=0)
            logits = logit_scale * torch.matmul(g, all_t.t())
            targets = torch.arange(logits.size(0), device=logits.device)
            loss_i = F.cross_entropy(logits, targets)
            loss_t = F.cross_entropy(torch.matmul(t, g.t()) * logit_scale, targets)
            loss = 0.5 * (loss_i + loss_t)
        else:
            logits = logit_scale * torch.matmul(g, t.t())
            targets = torch.arange(logits.size(0), device=logits.device)
            loss_i = F.cross_entropy(logits, targets)
            loss_t = F.cross_entropy(logits.t(), targets)
            loss = 0.5 * (loss_i + loss_t)

        with torch.no_grad():
            pos_sim = (g * t).sum(dim=-1).mean().item()
            if text_queue is not None:
                neg_mask = torch.zeros_like(logits, dtype=torch.bool)
                diag = torch.arange(logits.size(0), device=logits.device)
                neg_mask[diag, diag] = True
            else:
                neg_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
            hardest_neg = logits.masked_fill(neg_mask, float("-inf")).max(dim=1).values.mean().item()

        stats = {
            "pos_sim": pos_sim,
            "hardest_neg_logit": hardest_neg,
            "logit_scale": logit_scale.item(),
        }
        return loss, stats


class MemoryQueue(nn.Module):
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.size = size
        self.register_buffer("queue", F.normalize(torch.randn(size, dim), p=2, dim=-1))
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


# -----------------------
# Training / evaluation
# -----------------------
def train_one_epoch(
    model: GraphTextContrastive,
    text_encoder: FrozenTextEncoder | None,
    dl,
    optim,
    device: str,
    grad_accum: int,
    use_amp: bool,
    text_queue: MemoryQueue | None,
    max_logit_scale: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pbar = tqdm(dl, desc="train", dynamic_ncols=True)
    optim.zero_grad(set_to_none=True)
    for step, (graphs, text_data) in enumerate(pbar, start=1):
        graphs = graphs.to(device)

        if isinstance(text_data, torch.Tensor):
            text_emb = text_data.to(device)
        else:
            assert text_encoder is not None, "Text encoder required when precomputed embeddings are not provided."
            text_emb = text_encoder.encode(text_data, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            queue_emb = text_queue.queue if text_queue is not None else None
            loss, stats = model(graphs, text_emb, queue_emb, max_logit_scale=max_logit_scale)
        loss = loss / grad_accum
        loss_val = loss.item() * grad_accum
        scaler.scale(loss).backward()
        if step % grad_accum == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        total_loss += loss_val
        total_steps += 1
        if text_queue is not None:
            with torch.no_grad():
                queued = model.text_proj(text_emb.detach())
                queued = F.normalize(queued, p=2, dim=-1)
            text_queue.enqueue(queued)
        pbar.set_postfix({
            "loss": f"{total_loss / total_steps:.4f}",
            "pos_sim": f"{stats['pos_sim']:.3f}",
            "scale": f"{stats['logit_scale']:.2f}",
        })

    if (len(dl) % grad_accum) != 0:
        # Flush any remaining grads
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

    return total_loss / max(1, total_steps)


@torch.no_grad()
def evaluate(
    model: GraphTextContrastive,
    text_encoder: FrozenTextEncoder | None,
    dl,
    device: str,
    use_amp: bool,
) -> float:
    model.eval()
    losses: List[float] = []

    for graphs, text_data in tqdm(dl, desc="val", dynamic_ncols=True):
        graphs = graphs.to(device)
        if isinstance(text_data, torch.Tensor):
            text_emb = text_data.to(device)
        else:
            assert text_encoder is not None, "Text encoder required when precomputed embeddings are not provided."
            text_emb = text_encoder.encode(text_data, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _ = model(graphs, text_emb)
        losses.append(loss.item())

    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def evaluate_retrieval(
    model: GraphTextContrastive,
    text_encoder: FrozenTextEncoder | None,
    dl,
    device: str,
    use_amp: bool,
) -> dict:
    model.eval()
    g_all: List[torch.Tensor] = []
    t_all: List[torch.Tensor] = []

    for graphs, text_data in tqdm(dl, desc="val_retrieval", dynamic_ncols=True):
        graphs = graphs.to(device)
        if isinstance(text_data, torch.Tensor):
            text_emb = text_data.to(device)
        else:
            assert text_encoder is not None, "Text encoder required when precomputed embeddings are not provided."
            text_emb = text_encoder.encode(text_data, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            graph_emb, _ = model.graph_encoder(graphs)
            g = model.graph_proj(graph_emb)
            t = model.text_proj(text_emb)

        g_all.append(F.normalize(g, p=2, dim=-1).cpu())
        t_all.append(F.normalize(t, p=2, dim=-1).cpu())

    G = torch.cat(g_all, dim=0).to(device)
    T = torch.cat(t_all, dim=0).to(device)

    sims = torch.matmul(G, T.t())
    ranks = sims.argsort(dim=-1, descending=True)
    target = torch.arange(G.size(0), device=device)
    pos = (ranks == target[:, None]).nonzero()[:, 1] + 1

    metrics = {
        "mrr": float((1.0 / pos.float()).mean().item()),
        "r1": float((pos <= 1).float().mean().item()),
        "r5": float((pos <= 5).float().mean().item()),
    }
    model.train()
    return metrics


def save_checkpoint(model: GraphTextContrastive, gine_cfg: GINEConfig, cfg: TrainingConfig, best_loss: float, path: Path):
    payload = {
        "model_state": model.state_dict(),
        "gine_cfg": asdict(gine_cfg),
        "train_cfg": asdict(cfg),
        "best_loss": best_loss,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[info] Saved checkpoint to {path}")


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    gine_cfg = GINEConfig(**ckpt["gine_cfg"]) if "gine_cfg" in ckpt else None
    train_cfg = ckpt.get("train_cfg", None)
    best_loss = ckpt.get("best_loss", float("inf"))
    return ckpt["model_state"], gine_cfg, train_cfg, best_loss


# -----------------------
# Main
# -----------------------
def main():
    cfg = TrainingConfig()
    device = select_device()
    seed_all(cfg.seed)

    text_emb_train = normalize_emb_keys(load_id2emb(cfg.train_text_emb_csv)) if cfg.train_text_emb_csv else None
    text_emb_val = normalize_emb_keys(load_id2emb(cfg.val_text_emb_csv)) if cfg.val_text_emb_csv else None

    train_ds = ContrastiveDataset(cfg.train_graph_path, text_emb_dict=text_emb_train)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=contrastive_collate,
    )

    val_dl = None
    if cfg.val_graph_path is not None and Path(cfg.val_graph_path).exists():
        val_ds = ContrastiveDataset(cfg.val_graph_path, text_emb_dict=text_emb_val)
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=contrastive_collate,
        )

    text_encoder = None
    if text_emb_train is None:
        text_encoder = FrozenTextEncoder(cfg.text_model, cfg.max_text_len)
        # Enable fast tokenizer pad token if missing
        if text_encoder.tokenizer.pad_token is None:
            text_encoder.tokenizer.pad_token = text_encoder.tokenizer.eos_token or text_encoder.tokenizer.cls_token
    text_dim = text_encoder.hidden_size if text_encoder is not None else len(next(iter(text_emb_train.values())))

    ckpt_dir = Path(cfg.save_dir)
    best_path = ckpt_dir / "best.pt"
    loaded_state = None
    loaded_train_cfg = None
    loaded_best = float("inf")
    loaded_gine_cfg = None
    if cfg.resume_from_best and best_path.exists():
        loaded_state, loaded_gine_cfg, loaded_train_cfg, loaded_best = load_checkpoint(best_path, device)
        print(f"[info] Loaded checkpoint from {best_path} with best_loss={loaded_best:.4f}")

    # Use checkpointed GINE config if present, else default
    default_gine_cfg = GINEConfig(hidden_dim=512, num_layers=4, dropout=0.1, readout="attn", residual=True, normalize=False)
    if loaded_state is not None:
        if loaded_gine_cfg is None:
            print("[warn] Checkpoint missing GINE config; skipping resume.")
            loaded_state = None
            loaded_best = float("inf")
            loaded_gine_cfg = None
            loaded_train_cfg = None
        elif loaded_gine_cfg != default_gine_cfg:
            print("[warn] GINE config mismatch with current defaults; skipping resume.")
            loaded_state = None
            loaded_best = float("inf")
            loaded_gine_cfg = None
            loaded_train_cfg = None
    gine_cfg = loaded_gine_cfg or default_gine_cfg
    graph_encoder = GINEEncoder(gine_cfg)

    saved_proj = None
    if loaded_train_cfg and isinstance(loaded_train_cfg, dict):
        saved_proj = loaded_train_cfg.get("proj_dim")
    proj_dim = cfg.proj_dim or saved_proj or text_dim
    model = GraphTextContrastive(
        graph_encoder=graph_encoder,
        text_dim=text_dim,
        proj_dim=proj_dim,
        temperature=cfg.temperature,
        learn_temperature=cfg.learn_temperature,
    ).to(device)

    if loaded_state is not None:
        model.load_state_dict(loaded_state, strict=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.epochs, eta_min=cfg.min_lr
    )
    text_queue = MemoryQueue(proj_dim, cfg.queue_size).to(device) if cfg.queue_size > 0 else None

    best_val = loaded_best if loaded_state is not None else float("inf")
    best_mrr = loaded_best if (loaded_state is not None and cfg.metric_for_best == "mrr") else -1.0

    for ep in range(1, cfg.epochs + 1):
        print(f"\n=== Epoch {ep}/{cfg.epochs} ===")
        train_loss = train_one_epoch(
            model,
            text_encoder,
            train_dl,
            optim,
            device,
            cfg.grad_accum,
            cfg.use_amp,
            text_queue,
            cfg.logit_scale_max,
        )
        scheduler.step()
        msg = f"train_loss={train_loss:.4f}"

        if val_dl is not None:
            val_loss = evaluate(model, text_encoder, val_dl, device, cfg.use_amp)
            val_metrics = evaluate_retrieval(model, text_encoder, val_dl, device, cfg.use_amp)
            msg += f", val_loss={val_loss:.4f}, val_mrr={val_metrics['mrr']:.4f}, val_r1={val_metrics['r1']:.4f}, val_r5={val_metrics['r5']:.4f}"
            if cfg.metric_for_best == "mrr":
                if val_metrics["mrr"] > best_mrr:
                    best_mrr = val_metrics["mrr"]
                    save_checkpoint(model, gine_cfg, cfg, best_mrr, best_path)
            else:
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, gine_cfg, cfg, best_val, best_path)
        else:
            # Still save the latest
            save_checkpoint(model, gine_cfg, cfg, best_val, best_path)

        print(msg)

    print("Done.")


if __name__ == "__main__":
    main()
