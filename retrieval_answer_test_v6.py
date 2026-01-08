#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST Retrieval-Augmented GENERATION (RAG)
"""

version_embed = "v1"
version_gnn   = "v4"

# =========================================================
# IMPORTS
# =========================================================
import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from train_gcn_v3 import MolGNN, infer_cardinalities_from_graphs
from data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
)

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

TRAIN_GRAPHS  = "data/train_graphs.pkl"
TEST_GRAPHS   = "data/test_graphs.pkl"
TRAIN_EMB_CSV = f"data/train_embeddings_{version_embed}.csv"

# GNN ckpt
GNN_CKPT = f"checkpoints/gnn_{version_gnn}_best.pt"
if not os.path.exists(GNN_CKPT):
    GNN_CKPT = f"checkpoints/gnn_{version_gnn}.pt"

# Retrieval (reduce work)
BATCH_SIZE_GNN = 256
TOP_K = 80
N_SHOTS = 3

CENTER_EMB = True

# Generator (fast)
LM_NAME = "google/flan-t5-small"

# Generation params (fast + stable)
MAX_NEW_TOKENS = 64
NUM_BEAMS = 1          # greedy (huge speedup vs beams)
NO_REPEAT_NGRAM = 0
LENGTH_PENALTY = 1.0

# Anti-copy / fallback (lightweight)
MIN_LEN_CHARS = 30
COPY_JACCARD_4GRAM = 0.30
FALLBACK_TO_BEST_RETRIEVAL = True

# Diversity selection (fast)
DIVERSITY_MAX_SIM = 0.92  # avoid picking near-duplicate examples

OUT_CSV = "outputs/submission_v63_rag_fast.csv"

# =========================================================
# UTILS
# =========================================================
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

@torch.no_grad()
def l2norm(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def maybe_center(x, mean=None):
    if mean is None:
        mean = x.mean(dim=0, keepdim=True)
    return x - mean, mean

def clean_caption(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.split(r"\b(Examples:|Task:|Description:)\b", text, maxsplit=1)[0].strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text

def normalize_for_overlap(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-(),.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ngrams(s: str, n: int):
    toks = normalize_for_overlap(s).split()
    if len(toks) < n:
        return set()
    return set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

def jaccard_ngrams(a: str, b: str, n: int = 4) -> float:
    A = ngrams(a, n)
    B = ngrams(b, n)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def build_prompt(examples):
    # Keep it short: less tokens => faster
    ex = "\n".join([f"- {e}" for e in examples])
    return (
        "Given similar molecule descriptions, write a factual description of the target molecule.\n"
        "Do NOT copy; paraphrase.\n"
        "Write 1–3 sentences.\n"
        "Examples:\n"
        f"{ex}\n"
        "Answer:"
    )

@torch.no_grad()
def select_diverse_shots(cand_idx: torch.Tensor, cand_txt_emb: torch.Tensor, n_shots: int, max_sim: float):
    """
    cand_idx: [K] indices into train set (already sorted by relevance)
    cand_txt_emb: [K,D] normalized embeddings (same order)
    Returns: list of selected indices (values are indices into train set)
    """
    # Pick first (most relevant)
    selected_pos = [0]
    selected_emb = cand_txt_emb[0:1]  # [1,D]

    # Then greedily pick next best that is not too similar to already selected
    # We keep this loop tiny (N_SHOTS <= 3), and the heavy part is vectorized.
    for _ in range(1, n_shots):
        sims = cand_txt_emb @ selected_emb.T  # [K, |S|]
        max_s = sims.max(dim=1).values        # [K]
        mask = max_s <= max_sim
        mask[torch.tensor(selected_pos, device=cand_txt_emb.device)] = False

        if mask.any():
            # Best remaining by relevance order = first True in mask since cand_idx is relevance-sorted
            next_pos = int(torch.nonzero(mask, as_tuple=False)[0].item())
        else:
            # fallback: just take the next one by relevance
            next_pos = min(len(selected_pos), cand_idx.numel() - 1)

        selected_pos.append(next_pos)
        selected_emb = cand_txt_emb[torch.tensor(selected_pos, device=cand_txt_emb.device)]

    return [int(cand_idx[p].item()) for p in selected_pos]

# =========================================================
# MAIN
# =========================================================
@torch.inference_mode()
def main():
    print(f"Device: {DEVICE}")

    # ---- Load train text embeddings ----
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    train_ids = list(train_emb.keys())

    train_txt = torch.stack([train_emb[i] for i in train_ids]).to(DEVICE)
    train_txt = l2norm(train_txt)

    if CENTER_EMB:
        train_txt, txt_mean = maybe_center(train_txt)
        train_txt = l2norm(train_txt)
    else:
        txt_mean = None

    train_id2desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    emb_dim = train_txt.size(1)
    print(f"Train captions: {len(train_ids)} | emb_dim={emb_dim}")

    # ---- Load GNN ----
    ckpt = torch.load(GNN_CKPT, map_location=DEVICE)
    h = ckpt.get("hparams", {})
    atom_card, edge_card = infer_cardinalities_from_graphs(TRAIN_GRAPHS)

    gnn = MolGNN(
        atom_card=atom_card,
        edge_card=edge_card,
        hidden=h.get("hidden", 512),
        out_dim=emb_dim,
        layers=h.get("layers", 5),
    ).to(DEVICE)

    gnn.load_state_dict(ckpt["state_dict"], strict=False)
    gnn.eval()
    print(f"✓ GNN loaded from {GNN_CKPT}")

    # ---- Load generator (fp16 on cuda) ----
    print(f"Loading generator: {LM_NAME}")
    tok = AutoTokenizer.from_pretrained(LM_NAME)

    if DEVICE == "cuda":
        lm = AutoModelForSeq2SeqLM.from_pretrained(LM_NAME, dtype=torch.float16).to(DEVICE)
    else:
        lm = AutoModelForSeq2SeqLM.from_pretrained(LM_NAME).to(DEVICE)

    lm.eval()
    print("✓ Generator loaded")

    # ---- Load test data ----
    test_ds = PreprocessedGraphDataset(TEST_GRAPHS)

    # Bigger batch = faster generation (until OOM)
    GEN_BATCH = 32 if DEVICE == "cuda" else 8

    test_dl = DataLoader(
        test_ds,
        batch_size=GEN_BATCH,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    results = []
    ptr = 0

    for graphs in test_dl:
        graphs = graphs.to(DEVICE, non_blocking=True)

        # Graph embeddings
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                q = gnn(graphs)
        else:
            q = gnn(graphs)

        if CENTER_EMB and txt_mean is not None:
            q = l2norm(q - txt_mean)
        else:
            q = l2norm(q)

        # Retrieval
        sims = q @ train_txt.T                      # [B,N]
        sim_vals, topk_idx = sims.topk(TOP_K, dim=-1)  # [B,K]

        prompts = []
        fallbacks = []

        for b in range(topk_idx.size(0)):
            cand = topk_idx[b]             # [K]
            cand_txt_emb = train_txt[cand] # [K,D] normalized

            # shots: diverse but fast
            selected = select_diverse_shots(
                cand_idx=cand,
                cand_txt_emb=cand_txt_emb,
                n_shots=N_SHOTS,
                max_sim=DIVERSITY_MAX_SIM,
            )

            examples = [train_id2desc[train_ids[i]] for i in selected]
            prompts.append(build_prompt(examples))

            # fallback: best retrieval
            fallbacks.append(train_id2desc[train_ids[int(cand[0])]])

        # Tokenize + generate (batch)
        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # shorter context => faster
        ).to(DEVICE)

        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                gen = lm.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM,
                    length_penalty=LENGTH_PENALTY,
                    early_stopping=True,
                )
        else:
            gen = lm.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=False,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
            )

        decoded = tok.batch_decode(gen, skip_special_tokens=True)

        # Post-process + light anti-copy
        for b, text in enumerate(decoded):
            cap = clean_caption(text)

            if len(cap) < MIN_LEN_CHARS:
                cap = fallbacks[b]

            if FALLBACK_TO_BEST_RETRIEVAL:
                if jaccard_ngrams(cap, fallbacks[b], n=4) > COPY_JACCARD_4GRAM:
                    cap = fallbacks[b]

            results.append({"ID": test_ds.ids[ptr], "description": cap})
            ptr += 1

    df = pd.DataFrame(results)
    ensure_dir(OUT_CSV)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Saved {len(df)} predictions to {OUT_CSV}")

if __name__ == "__main__":
    main()
