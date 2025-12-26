# generate_graph2text.py
from __future__ import annotations
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

from data_utils import PreprocessedGraphDataset, collate_fn
from models_graphormer import GraphormerEncoder, GraphormerConfig
from torch import autocast


TEST_GRAPHS  = "data/test_graphs_cached.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # else "mps" if torch.backends.mps.is_available()

USE_AMP = (DEVICE == "mps" or DEVICE == "cuda")
AMP_DTYPE = torch.float16

CKPT_DIR = "ckpt_graph2text/best"    # change if needed
OUT_CSV = "submission.csv"

BATCH_SIZE = 16 if DEVICE == "mps" or DEVICE == "cuda" else 8
MAX_NEW_TOKENS = 96
NUM_BEAMS = 4
NO_REPEAT_NGRAM = 3
EARLY_STOPPING = True
LENGTH_PENALTY = 1.1


class GraphOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, graph_path: str):
        self.base = PreprocessedGraphDataset(graph_path, emb_dict=None)
        self.ids = self.base.ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base.graphs[idx]


def main():
    assert os.path.exists(CKPT_DIR), f"Checkpoint not found: {CKPT_DIR}"
    assert os.path.exists(TEST_GRAPHS), f"Missing test graphs: {TEST_GRAPHS}"

    tokenizer = GPT2TokenizerFast.from_pretrained(CKPT_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    dec_cfg = GPT2Config.from_pretrained(CKPT_DIR)
    dec_cfg.add_cross_attention = True
    dec_cfg.pad_token_id = tokenizer.pad_token_id
    dec_cfg.use_cache = True # Enable KV caching for faster generation
    decoder = GPT2LMHeadModel.from_pretrained(CKPT_DIR, config=dec_cfg).to(DEVICE)
    decoder.eval()

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

    ds = GraphOnlyDataset(TEST_GRAPHS)
    # Sort dataset by num_nodes to minimize padding waste
    ds.graphs.sort(key=lambda g: g.num_nodes)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    results = []
    seen = 0

    with torch.no_grad():
        for graphs in dl:
            graphs = graphs.to(DEVICE)
            B = graphs.num_graphs

            enc_states, enc_mask = encoder(graphs)  # [B,N,D], [B,N]
            # Start generation with BOS (GPT2 uses EOS as BOS too)
            input_ids = torch.full((B, 1), tokenizer.eos_token_id, device=DEVICE, dtype=torch.long)

            with autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
                gen = decoder.generate(
                    input_ids=input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    early_stopping=EARLY_STOPPING,
                    length_penalty=LENGTH_PENALTY,
                    do_sample=False,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    encoder_hidden_states=enc_states,
                    encoder_attention_mask=enc_mask,
                )

            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            # Align IDs for this batch
            batch_ids = ds.ids[seen:seen + B]
            seen += B

            for _id, txt in zip(batch_ids, texts):
                # light cleanup
                txt = " ".join(txt.split())
                results.append({"ID": _id, "description": txt})

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"Saved {len(results)} predictions to {OUT_CSV}")


if __name__ == "__main__":
    main()
