#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text embeddings for Molecular Graph Captioning

- Model: intfloat/e5-base-v2 (contrastive, retrieval-optimized)
- Mean pooling + L2 normalization
- Batched inference + AMP
"""

version_embed = "v1"

import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "intfloat/e5-base-v2"
MAX_TOKEN_LENGTH = 256          # Longer = better for chemistry text
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# E5 requires a prefix for optimal performance
PREFIX = "passage: "

# =========================================================
# LOAD MODEL
# =========================================================
print(f"Loading text encoder: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(f"Model loaded on: {DEVICE}")

# =========================================================
# MEAN POOLING
# =========================================================
@torch.no_grad()
def mean_pooling(last_hidden, attention_mask):
    """
    last_hidden: [B, T, D]
    attention_mask: [B, T]
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

# =========================================================
# PROCESS SPLITS
# =========================================================
for split in ["train", "validation"]:
    print(f"\nProcessing {split} split")

    pkl_path = f"data/{split}_graphs.pkl"
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)

    print(f"Loaded {len(graphs)} graphs")

    descriptions = [PREFIX + g.description for g in graphs]
    graph_ids = [g.id for g in graphs]

    all_embeddings = []

    for i in tqdm(range(0, len(descriptions), BATCH_SIZE)):
        batch_texts = descriptions[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                outputs = model(**inputs)

        emb = mean_pooling(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        )
        emb = F.normalize(emb, dim=-1)

        all_embeddings.append(emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    df = pd.DataFrame({
        "ID": graph_ids,
        "embedding": [",".join(map(str, e)) for e in all_embeddings],
    })

    out_path = f"data/{split}_embeddings_{version_embed}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved embeddings to {out_path}")

print("\nText embeddings generated successfully")
