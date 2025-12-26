#!/usr/bin/env python3
from __future__ import annotations
import os
import math
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch import autocast

import sacrebleu
from bert_score import score as bertscore

from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

from data_utils import PreprocessedGraphDataset, collate_fn
from models_graphormer import GraphormerEncoder, GraphormerConfig
from tqdm.auto import tqdm


# -----------------------
# Paths / config
# -----------------------
VAL_GRAPHS = "data/validation_graphs_cached.pkl"
CKPT_DIR = "ckpt_graph2text/best"

NUM_EXAMPLES = 256
# Decoding hyperparams (tune these!)
BATCH_SIZE = 16
MAX_NEW_TOKENS = 96
NUM_BEAMS = 4
NO_REPEAT_NGRAM = 3
LENGTH_PENALTY = 1.1
EARLY_STOPPING = True

# Metrics config
# BERTScore defaults to roberta-large if model_type=None.
# For speed on a laptop, consider a smaller model like "roberta-base".
BERTSCORE_MODEL_TYPE = "roberta-base"
BERTSCORE_LANG = "en"

# Optional: save detailed outputs
SAVE_CSV = "val_generations_debug.csv"


# -----------------------
# Device selection
# -----------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
USE_AMP = DEVICE in ("cuda", "mps")
AMP_DTYPE = torch.float16  # try bfloat16 if your stack supports it


# -----------------------
# Bucketing (optional, recommended)
# -----------------------
class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sizes: list[int],
        batch_size: int,
        shuffle: bool = False,          # for eval keep False for determinism
        drop_last: bool = False,
        bucket_size_multiplier: int = 20,
        seed: int = 42,
    ):
        self.sizes = sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_size = batch_size * bucket_size_multiplier
        self.indices = list(range(len(sizes)))

    def __iter__(self):
        rng = random.Random(self.seed)

        inds = sorted(self.indices, key=lambda i: self.sizes[i])
        buckets = [inds[i:i + self.bucket_size] for i in range(0, len(inds), self.bucket_size)]
        if self.shuffle:
            rng.shuffle(buckets)

        for b in buckets:
            if self.shuffle:
                rng.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                batch = b[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

        self.seed += 1

    def __len__(self):
        if self.drop_last:
            return len(self.sizes) // self.batch_size
        return math.ceil(len(self.sizes) / self.batch_size)


# -----------------------
# Dataset (graph + reference text)
# -----------------------
class GraphTextEvalDataset(Dataset):
    def __init__(self, graph_path: str):
        self.base = PreprocessedGraphDataset(graph_path, emb_dict=None)
        self.ids = self.base.ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        g = self.base.graphs[idx]
        # ground truth is stored in g.description
        return g, g.description


def collate_graph_text_eval(batch):
    # graphs for generation, refs for metric
    graphs, refs = zip(*batch)
    batch_graph = collate_fn(list(graphs))
    return batch_graph, list(refs)


# -----------------------
# Generation
# -----------------------
@torch.inference_mode()
def generate_on_val(
    encoder: GraphormerEncoder,
    decoder: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    dl: DataLoader,
) -> tuple[list[str], list[str]]:
    all_preds: list[str] = []
    all_refs: list[str] = []

    pbar = tqdm(dl, total=len(dl), desc="Inference", unit="batch")
    for graphs, refs in pbar:
        graphs = graphs.to(DEVICE)
        B = graphs.num_graphs

        # Encode graph (AMP helps on MPS/CUDA)
        with autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            enc_states, enc_mask = encoder(graphs)

        # BOS token (GPT-2 uses eos as bos)
        input_ids = torch.full((B, 1), tokenizer.eos_token_id, device=DEVICE, dtype=torch.long)
        attn_mask = torch.ones_like(input_ids)

        # Generate (you can also wrap in autocast; if it errors, remove the autocast here)
        with autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            gen = decoder.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=False,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                length_penalty=LENGTH_PENALTY,
                early_stopping=EARLY_STOPPING,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                encoder_hidden_states=enc_states,
                encoder_attention_mask=enc_mask,
            )

        preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds = [" ".join(p.split()) for p in preds]  # light cleanup
        refs = [" ".join(r.split()) for r in refs]

        all_preds.extend(preds)
        all_refs.extend(refs)

        pbar.set_postfix({"total_generated": len(all_preds)})

    return all_preds, all_refs


# -----------------------
# Metrics
# -----------------------
def compute_bleu(preds: list[str], refs: list[str]) -> float:
    # sacrebleu expects refs as a list of reference lists
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return float(bleu.score)


def compute_bertscore(preds: list[str], refs: list[str]) -> dict:
    P, R, F1 = bertscore(
        cands=preds,
        refs=refs,
        lang=BERTSCORE_LANG,
        model_type=BERTSCORE_MODEL_TYPE,
        verbose=True,
        device=DEVICE,
        rescale_with_baseline=True,
    )
    return {
        "bertscore_precision": float(P.mean().item()),
        "bertscore_recall": float(R.mean().item()),
        "bertscore_f1": float(F1.mean().item()),
    }


def main():
    assert os.path.exists(VAL_GRAPHS), f"Missing validation graphs: {VAL_GRAPHS}"
    assert os.path.exists(CKPT_DIR), f"Checkpoint not found: {CKPT_DIR}"

    print("Device:", DEVICE, "| AMP:", USE_AMP)

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(CKPT_DIR)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Load decoder with cross-attention
    dec_cfg = GPT2Config.from_pretrained(CKPT_DIR)
    dec_cfg.add_cross_attention = True
    dec_cfg.pad_token_id = tokenizer.pad_token_id
    decoder = GPT2LMHeadModel.from_pretrained(CKPT_DIR, config=dec_cfg).to(DEVICE)
    decoder.eval()
    decoder.config.use_cache = True  # important for fast generation

    # Load encoder
    enc_cfg = GraphormerConfig(
        dim=dec_cfg.n_embd,
        layers=6,
        heads=8,
        dropout=0.1,
        max_dist=12,
    )
    encoder = GraphormerEncoder(enc_cfg).to(DEVICE)
    enc_path = os.path.join(CKPT_DIR, "graph_encoder.pt")
    encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    encoder.eval()

    # Data
    ds = GraphTextEvalDataset(VAL_GRAPHS)
    # Take a subset for faster eval (optional)
    if NUM_EXAMPLES is not None and NUM_EXAMPLES < len(ds):
        ds.base.graphs = ds.base.graphs[:NUM_EXAMPLES]
        ds.ids = ds.ids[:NUM_EXAMPLES]
        print(f"[info] Using only first {NUM_EXAMPLES} examples for eval")

    # Optional bucketing by num_nodes to reduce padding
    sizes = [int(g.num_nodes) for g in ds.base.graphs]
    batch_sampler = BucketBatchSampler(sizes=sizes, batch_size=BATCH_SIZE, shuffle=False)

    dl = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_graph_text_eval,
    )

    # Generate
    preds, refs = generate_on_val(encoder, decoder, tokenizer, dl)

    # Metrics
    bleu = compute_bleu(preds, refs)
    b = compute_bertscore(preds, refs)

    print("\n=== Validation metrics ===")
    print(f"BLEU (sacrebleu corpus BLEU): {bleu:.2f}")
    print(f"BERTScore P: {b['bertscore_precision']:.4f}")
    print(f"BERTScore R: {b['bertscore_recall']:.4f}")
    print(f"BERTScore F1: {b['bertscore_f1']:.4f}")

    # Save debug CSV
    if SAVE_CSV:
        df = pd.DataFrame({"ref": refs, "pred": preds})
        df.to_csv(SAVE_CSV, index=False)
        print(f"\nSaved per-example outputs to: {SAVE_CSV}")


if __name__ == "__main__":
    main()
