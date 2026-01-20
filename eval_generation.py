#!/usr/bin/env python3
from __future__ import annotations
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings

from transformers import logging as hf_logging

from model_config import ModelConfig, load_models_and_tokenizer, checkpoint_dirs
from graph2text_utils import (
    BucketBatchSampler,
    Graph2Text,
    GraphTextDataset,
    evaluate_model,
    make_text_collate,
    select_device,
)

# -----------------------
# Paths / config
# -----------------------
VAL_GRAPHS = "data/validation_graphs_cached.pkl"
MODEL_CFG = ModelConfig()
BATCH_SIZE = 32

NUM_EXAMPLES = 256
MAX_TEXT_LEN = 128

RUN_DIR, BEST_DIR = checkpoint_dirs(MODEL_CFG)
GEN_CFG = {
    "max_new_tokens": 96,
    "num_beams": 4,
    "no_repeat_ngram": 3,
    "length_penalty": 1.1,
    "early_stopping": True,
    "prefix": "The molecule is ",
}

# Metrics config
BERTSCORE_MODEL_TYPE = "roberta-base"
BERTSCORE_NUM_LAYERS = None
BERTSCORE_LANG = "en"
BERTSCORE_RESCALE = True

# Optional: save detailed outputs
SAVE_CSV = "val_generations_debug.csv"


# -----------------------
# Device selection
# -----------------------
DEVICE = select_device()
USE_AMP = False
# Silence HF padding warning for decoder-only models (we set left padding manually)
warnings.filterwarnings("ignore", message=".*decoder-only architecture.*right-padding.*")
hf_logging.set_verbosity_error()


def main():
    assert os.path.exists(VAL_GRAPHS), f"Missing validation graphs: {VAL_GRAPHS}"
    run_dir, best_dir = checkpoint_dirs(MODEL_CFG)
    assert os.path.exists(best_dir), f"Checkpoint not found: {best_dir}"

    print("Device:", DEVICE, "| AMP:", USE_AMP)

    tokenizer, decoder, encoder, load_dir = load_models_and_tokenizer(MODEL_CFG, DEVICE, prefer_best=True)
    assert load_dir is not None, f"Checkpoint not found: {best_dir}"
    tokenizer.padding_side = "left"
    decoder.generation_config.pad_token_id = tokenizer.pad_token_id
    decoder.generation_config.eos_token_id = tokenizer.eos_token_id
    decoder.generation_config.padding_side = "left"
    decoder.eval()
    encoder.eval()

    model = Graph2Text(encoder, decoder).to(DEVICE)

    # Data
    ds = GraphTextDataset(VAL_GRAPHS, include_text=True)
    if NUM_EXAMPLES is not None and NUM_EXAMPLES < len(ds):
        ds.base.graphs = ds.base.graphs[:NUM_EXAMPLES]
        ds.base.ids = ds.base.ids[:NUM_EXAMPLES]
        print(f"[info] Using only first {NUM_EXAMPLES} examples for eval")

    batch_sampler = BucketBatchSampler(
        sizes=ds.node_sizes(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    collate = make_text_collate(tokenizer, MAX_TEXT_LEN, return_text=True)
    dl = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )

    metrics, preds, refs = evaluate_model(
        model=model,
        dl=dl,
        tokenizer=tokenizer,
        device=DEVICE,
        use_amp=USE_AMP,
        gen_kwargs=GEN_CFG,
        compute_text_metrics=True,
        bertscore_model=BERTSCORE_MODEL_TYPE,
        bertscore_num_layers=BERTSCORE_NUM_LAYERS,
        bertscore_lang=BERTSCORE_LANG,
        bertscore_rescale=BERTSCORE_RESCALE,
        return_predictions=True,
    )

    print("\n=== Validation metrics ===")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"BLEU: {metrics['bleu']:.2f}")
    print(f"BERTScore P: {metrics['bertscore_precision']:.4f}")
    print(f"BERTScore R: {metrics['bertscore_recall']:.4f}")
    print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")

    if SAVE_CSV:
        df = pd.DataFrame({"ref": refs, "pred": preds})
        df.to_csv(SAVE_CSV, index=False)
        print(f"\nSaved per-example outputs to: {SAVE_CSV}")


if __name__ == "__main__":
    main()
