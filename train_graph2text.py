# train_graph2text.py
from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch import autocast
from torch.utils.data import Sampler

from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)

from data_utils import PreprocessedGraphDataset, collate_fn, batch_graphs_with_cache
from models_graphormer import GraphormerEncoder, GraphormerConfig


# ----------------------------
# CONFIG
# ----------------------------
TRAIN_GRAPHS = "data/train_graphs_cached.pkl"
VAL_GRAPHS   = "data/validation_graphs_cached.pkl"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # else "mps" if torch.backends.mps.is_available()
USE_AMP = (DEVICE == "mps")

MODEL_NAME = "gpt2-medium"      # as requested
SAVE_DIR = "ckpt_graph2text"

BATCH_SIZE = 16 if DEVICE == "mps" else 8                 # GPT2-medium is heavy; adjust
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_TEXT_LEN = 128              # aligns with baseline embedding script

GRAD_ACCUM = 2                  # simulate larger batch
CLIP_NORM = 1.0

SEED = 42


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from functools import partial

def make_collate(tokenizer, max_len):
    return partial(collate_graph_text, tokenizer=tokenizer, max_len=max_len)

# ----------------------------
# Collate for graph + text
# ----------------------------
class GraphTextDataset(torch.utils.data.Dataset):
    def __init__(self, graph_path: str):
        self.base = PreprocessedGraphDataset(graph_path, emb_dict=None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        g = self.base.graphs[idx]
        return g, g.description


def collate_graph_text(batch: List[Tuple[object, str]], tokenizer: GPT2TokenizerFast, max_len: int):
    graphs, texts = zip(*batch)
    batch_graph = batch_graphs_with_cache(list(graphs))

    tok = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    # For causal LM training: labels = input_ids with pads masked out
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return batch_graph, input_ids, attention_mask, labels



class BucketBatchSampler(Sampler[list[int]]):
    """
    Groups samples with similar sizes into the same mini-batch.
    - Sort indices by size
    - Split into 'chunks' (buckets)
    - Shuffle buckets each epoch
    - Yield batches from each bucket

    This keeps Nmax low within a batch (great for padded NxN tensors).
    """
    def __init__(
        self,
        sizes: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = 50,
        seed: int = 42,
    ):
        self.sizes = sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Each bucket holds multiple batches; larger = more randomness but looser size grouping
        self.bucket_size = batch_size * bucket_size_multiplier

        self.indices = list(range(len(sizes)))

    def __iter__(self):
        rng = random.Random(self.seed)

        # sort by size
        inds = sorted(self.indices, key=lambda i: self.sizes[i])

        # chunk into buckets
        buckets = [inds[i:i + self.bucket_size] for i in range(0, len(inds), self.bucket_size)]

        if self.shuffle:
            rng.shuffle(buckets)

        for b in buckets:
            if self.shuffle:
                rng.shuffle(b)

            # yield batches inside bucket
            for i in range(0, len(b), self.batch_size):
                batch = b[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

        # change seed each epoch-like iteration (so next __iter__ changes)
        self.seed += 1

    def __len__(self):
        if self.drop_last:
            return len(self.sizes) // self.batch_size
        return math.ceil(len(self.sizes) / self.batch_size)


# ----------------------------
# Model wrapper: Graph encoder + GPT2 decoder with cross-attention
# ----------------------------
class Graph2Text(nn.Module):
    def __init__(self, enc: GraphormerEncoder, dec: GPT2LMHeadModel):
        super().__init__()
        self.enc = enc
        self.dec = dec

        # Project encoder dim -> GPT2 hidden size if needed
        enc_dim = enc.cfg.dim
        dec_dim = dec.config.n_embd
        self.proj = nn.Identity() if enc_dim == dec_dim else nn.Linear(enc_dim, dec_dim)

    def forward(self, graphs: Batch, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        enc_states, enc_mask = self.enc(graphs)                      # [B,N,Denc], [B,N]
        enc_states = self.proj(enc_states)                           # [B,N,Ddec]

        out = self.dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=enc_states,
            encoder_attention_mask=enc_mask,
            use_cache=False,
        )
        return out


def main():
    seed_all(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    # GPT2 has no pad token by default; use eos as pad to allow batching.
    tokenizer.pad_token = tokenizer.eos_token

    # Decoder config with cross-attention enabled
    dec_cfg = GPT2Config.from_pretrained(MODEL_NAME)
    dec_cfg.add_cross_attention = True
    dec_cfg.pad_token_id = tokenizer.pad_token_id

    decoder = GPT2LMHeadModel.from_pretrained(MODEL_NAME, config=dec_cfg)

    # Encoder
    enc_cfg = GraphormerConfig(
        dim=1024,           # matches gpt2-medium n_embd=1024 => no proj needed
        layers=6,
        heads=8,
        dropout=0.1,
        max_dist=12,
    )
    encoder = GraphormerEncoder(enc_cfg)

    model = Graph2Text(encoder, decoder).to(DEVICE)

    # Data
    train_ds = GraphTextDataset(TRAIN_GRAPHS)
    train_sizes = [int(g.num_nodes) for g in train_ds.base.graphs]  # or train_ds.base.graphs depending on your dataset class

    batch_sampler = BucketBatchSampler(
        sizes=train_sizes,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        bucket_size_multiplier=20,  # try 20..100
        seed=SEED,
    )

    train_dl = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=0,           # macOS safe; try 1 later
        pin_memory=False,        # MPS
        collate_fn=make_collate(tokenizer, MAX_TEXT_LEN),
    )


    val_dl = None
    if os.path.exists(VAL_GRAPHS):
        val_ds = GraphTextDataset(VAL_GRAPHS)
        val_dl = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=make_collate(tokenizer, MAX_TEXT_LEN),
        )

    # Optim
    no_decay = ["bias", "ln", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(params, lr=LR)

    total_steps = math.ceil(len(train_dl) / GRAD_ACCUM) * EPOCHS
    warmup = int(total_steps * WARMUP_RATIO)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)

    def run_eval() -> float:
        if val_dl is None:
            return float("nan")
        model.eval()
        losses = []
        with torch.no_grad():
            for graphs, input_ids, attention_mask, labels in val_dl:
                graphs = graphs.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.to(DEVICE)
                with autocast(device_type="mps", dtype=torch.float16, enabled=USE_AMP):
                    out = model(graphs, input_ids, attention_mask, labels)
                losses.append(out.loss.detach().cpu().item())
        model.train()
        return sum(losses) / max(1, len(losses))

    # Train
    best_val = float("inf")
    global_step = 0
    model.train()


    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        optim.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_dl, start=1),
            total=len(train_dl),
            desc=f"Epoch {ep}/{EPOCHS}",
            leave=True,
            dynamic_ncols=True,
        )

        for it, (graphs, input_ids, attention_mask, labels) in pbar:
            graphs = graphs.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            with autocast(device_type="mps", dtype=torch.float16, enabled=USE_AMP):
                out = model(graphs, input_ids, attention_mask, labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            running += loss.item()

            if it % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                # Update tqdm bar every optimizer step
                pbar.set_postfix({
                    "loss": f"{running:.4f}",
                    "lr": f"{sched.get_last_lr()[0]:.2e}",
                    "step": f"{global_step}/{total_steps}",
                })
                running = 0.0


        val_loss = run_eval()
        print(f"Epoch {ep} done. val_loss={val_loss:.4f}")

        # Save best
        if val_dl is None:
            # still save each epoch
            ckpt = os.path.join(SAVE_DIR, f"epoch_{ep}")
            os.makedirs(ckpt, exist_ok=True)
            model.dec.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            torch.save(model.enc.state_dict(), os.path.join(ckpt, "graph_encoder.pt"))
        else:
            if val_loss < best_val:
                best_val = val_loss
                ckpt = os.path.join(SAVE_DIR, "best")
                os.makedirs(ckpt, exist_ok=True)
                model.dec.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                torch.save(model.enc.state_dict(), os.path.join(ckpt, "graph_encoder.pt"))
                print(f"Saved new best checkpoint to {ckpt}")

    print("Training finished.")


if __name__ == "__main__":
    main()
