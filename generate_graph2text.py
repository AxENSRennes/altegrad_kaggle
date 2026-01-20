# generate_graph2text.py
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import autocast
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from data_utils import batch_graphs_with_cache, load_id2emb
from graph2text_utils import BucketBatchSampler, GraphOnlyDataset, GraphTextDataset, clean_texts, select_device
from models_gine import GINEConfig, GINEEncoder


TEST_GRAPHS = "data/test_graphs_func_groups.pkl"
FALLBACK_TEST_GRAPHS = "data/test_graphs_cached.pkl"
TRAIN_GRAPHS = "data/train_graphs_func_groups.pkl"
TEST_GRAPH_EMB = ""
OUT_CSV = "submission.csv"

T5_MODEL_NAME = "t5-small"
SAVE_DIR = "ckpt_graph2text"
BEST_SUBDIR = "best"

GINE_CKPT_PATH = "ckpt_gine_contrastive/best.pt"
GRAPH_TOKEN_LEN = 8
ADAPTER_DROPOUT = 0.1
USE_NODE_TOKENS = True

FG_PROMPT_PREFIX = "Functional groups: "
FG_PROMPT_EMPTY = "none"
FG_PROMPT_SUFFIX = ". "
PREFIX = "The molecule is "
N_FUNC_GROUPS = 6
HEAVY_TAIL_QUANTILE = 0.0

BATCH_SIZE = 32
USE_AMP = False
AUTOCAST_DTYPE = torch.bfloat16 if USE_AMP else torch.float32
MAX_PROMPT_LEN = 100

GEN_CFG = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "no_repeat_ngram": 3,
    "length_penalty": 1.1,
    "early_stopping": True,
}

DEVICE = select_device()


def resolve_test_graphs() -> str:
    if os.path.exists(TEST_GRAPHS):
        return TEST_GRAPHS
    if os.path.exists(FALLBACK_TEST_GRAPHS):
        return FALLBACK_TEST_GRAPHS
    raise FileNotFoundError(f"Missing test graphs: {TEST_GRAPHS} or {FALLBACK_TEST_GRAPHS}")


def format_functional_groups_prompt(groups: List[str]) -> str:
    if not groups:
        groups_txt = FG_PROMPT_EMPTY
    else:
        groups_txt = ", ".join(groups)
    return f"{FG_PROMPT_PREFIX}{groups_txt}{FG_PROMPT_SUFFIX}"


def rate_functional_groups(graphs: List) -> dict:
    freq: dict[str, int] = {}
    for g in graphs:
        groups = getattr(g, "functional_groups", []) or []
        for fg in groups:
            freq[fg] = freq.get(fg, 0) + 1
    return freq


def select_rare_functional_groups(
    graphs: List,
    n_groups: int,
    heavy_tail_quantile: float,
) -> set[str]:
    freq = rate_functional_groups(graphs)
    if not freq or n_groups <= 0:
        return set()
    counts = sorted(freq.values())
    q = max(0.0, min(1.0, heavy_tail_quantile))
    cutoff_idx = int(math.floor(q * (len(counts) - 1)))
    cutoff = counts[cutoff_idx]
    filtered = {fg: c for fg, c in freq.items() if c >= cutoff}
    if not filtered:
        filtered = freq
    items = sorted(filtered.items(), key=lambda kv: (kv[1], kv[0]))
    return {fg for fg, _ in items[:n_groups]}


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


class GraphTokenAdapter(nn.Module):
    def __init__(self, in_dim: int, token_len: int, d_model: int, dropout: float):
        super().__init__()
        self.token_len = token_len
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(in_dim, token_len * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(graph_emb).view(-1, self.token_len, self.d_model)
        return self.ln(tokens)


class NodeTokenAdapter(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, node_emb: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        node_proj = self.ln(self.proj(node_emb))
        sizes = torch.bincount(batch_idx, minlength=batch_size).tolist()
        splits = torch.split(node_proj, sizes)
        tokens = nn.utils.rnn.pad_sequence(splits, batch_first=True)
        enc_mask = node_proj.new_zeros((batch_size, tokens.size(1)), dtype=torch.long)
        for i, n in enumerate(sizes):
            if n:
                enc_mask[i, :n] = 1
        return tokens, enc_mask


class Graph2TextT5(nn.Module):
    def __init__(
        self,
        encoder: GINEEncoder,
        adapter: GraphTokenAdapter | NodeTokenAdapter,
        decoder: T5ForConditionalGeneration,
        use_node_tokens: bool,
    ):
        super().__init__()
        self.enc = encoder
        self.adapter = adapter
        self.dec = decoder
        self.use_node_tokens = use_node_tokens

    def encode_graphs(self, graphs, graph_embs: torch.Tensor | None = None):
        if self.use_node_tokens:
            if graph_embs is not None:
                raise RuntimeError("Node-token adapter expects GINE node embeddings; disable precomputed embeddings.")
            _, node_embs = self.enc(graphs)
            return self.adapter(node_embs, graphs.batch, graphs.num_graphs)
        if graph_embs is None:
            graph_embs, _ = self.enc(graphs)
        tokens = self.adapter(graph_embs)
        enc_mask = torch.ones(tokens.size()[:2], device=tokens.device, dtype=torch.long)
        return tokens, enc_mask


class PrecomputedGraphEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, graphs):
        raise RuntimeError("Precomputed graph embeddings expected; encoder is disabled.")


def lookup_graph_embedding(emb_dict: dict, graph_id) -> torch.Tensor:
    candidates = [graph_id]
    try:
        candidates.append(int(graph_id))
    except (ValueError, TypeError):
        pass
    candidates.append(str(graph_id))
    for key in candidates:
        if key in emb_dict:
            return emb_dict[key]
    raise KeyError(f"Graph embedding not found for graph id={graph_id!r}")


def get_emb_dim(emb_dict: dict | None) -> int | None:
    if not emb_dict:
        return None
    sample = next(iter(emb_dict.values()))
    return int(sample.numel())


def make_t5_gen_collate(tokenizer, max_len: int, emb_dict: dict | None = None, fg_allowlist: set[str] | None = None):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _collate(batch):
        graphs = list(batch)
        batch_graph = batch_graphs_with_cache(graphs)
        graph_embs = None
        if emb_dict is not None:
            graph_embs = torch.stack([lookup_graph_embedding(emb_dict, g.id) for g in graphs], dim=0)
        if fg_allowlist:
            prompts = [
                format_functional_groups_prompt(
                    [fg for fg in (getattr(g, "functional_groups", []) or []) if fg in fg_allowlist]
                )
                for g in graphs
            ]
        else:
            prompts = [format_functional_groups_prompt(getattr(g, "functional_groups", [])) for g in graphs]
        prompt_tok = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_input_ids = prompt_tok["input_ids"]
        prompt_attention_mask = prompt_tok["attention_mask"]
        prompt_start = prompt_input_ids.new_full((prompt_input_ids.size(0), 1), decoder_start_token_id)
        prompt_input_ids = torch.cat([prompt_start, prompt_input_ids], dim=1)
        prompt_attention_mask = torch.cat(
            [prompt_attention_mask.new_ones((prompt_attention_mask.size(0), 1)), prompt_attention_mask], dim=1
        )
        if pad_token_id != tokenizer.pad_token_id:
            prompt_input_ids = prompt_input_ids.clone()
            prompt_input_ids[prompt_input_ids == tokenizer.pad_token_id] = pad_token_id
        return batch_graph, graph_embs, prompt_input_ids, prompt_attention_mask, prompts

    return _collate


@torch.inference_mode()
def generate_from_graph_batch(
    model: Graph2TextT5,
    tokenizer,
    graphs,
    device: str,
    use_amp: bool,
    gen_kwargs: dict,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_texts: List[str],
    graph_embs: torch.Tensor | None = None,
) -> List[str]:
    graphs = graphs.to(device)
    if graph_embs is not None:
        graph_embs = graph_embs.to(device)
    with autocast(device_type=device, dtype=AUTOCAST_DTYPE, enabled=use_amp):
        enc_states, enc_mask = model.encode_graphs(graphs, graph_embs=graph_embs)
    encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)
    gen = model.dec.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=enc_mask,
        decoder_input_ids=prompt_input_ids.to(device),
        decoder_attention_mask=prompt_attention_mask.to(device),
        max_new_tokens=gen_kwargs["max_new_tokens"],
        num_beams=gen_kwargs["num_beams"],
        no_repeat_ngram_size=gen_kwargs["no_repeat_ngram"],
        length_penalty=gen_kwargs["length_penalty"],
        early_stopping=gen_kwargs.get("early_stopping", True),
    )
    decoded = clean_texts(tokenizer.batch_decode(gen, skip_special_tokens=True))
    stripped = []
    for text, prompt in zip(decoded, prompt_texts):
        idx = text.find(PREFIX)
        if idx != -1:
            stripped.append(text[idx:].lstrip())
        elif text.startswith(prompt):
            stripped.append(text[len(prompt):].lstrip())
        else:
            stripped.append(text)
    return stripped


def main():
    test_graphs = resolve_test_graphs()
    run_dir = Path(SAVE_DIR) / T5_MODEL_NAME.replace("/", "_")
    best_dir = run_dir / BEST_SUBDIR
    if not best_dir.is_dir():
        raise FileNotFoundError(f"T5 checkpoint not found: {best_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(best_dir))
    decoder = T5ForConditionalGeneration.from_pretrained(str(best_dir)).to(DEVICE)
    decoder.eval()

    emb_dict = None
    if not USE_NODE_TOKENS:
        if Path(TEST_GRAPH_EMB).exists():
            emb_dict = load_id2emb(TEST_GRAPH_EMB)
            print(f"[info] Loaded test graph embeddings from {TEST_GRAPH_EMB}")
        else:
            print(f"[warn] Test graph embeddings not found at {TEST_GRAPH_EMB}; using GINE encoder.")
    elif Path(TEST_GRAPH_EMB).exists():
        print("[info] USE_NODE_TOKENS=True; ignoring precomputed graph embeddings.")

    emb_dim = get_emb_dim(emb_dict)
    if USE_NODE_TOKENS or emb_dict is None:
        gine_ckpt = Path(GINE_CKPT_PATH)
        if not gine_ckpt.exists():
            raise FileNotFoundError(f"GINE checkpoint not found: {gine_ckpt}")
        encoder = load_frozen_gine_encoder(str(gine_ckpt), DEVICE)
        encoder_output_dim = encoder.output_dim
    else:
        if emb_dim is None:
            raise ValueError("Graph embedding dimension could not be inferred from CSV file.")
        encoder = PrecomputedGraphEncoder(emb_dim).to(DEVICE)
        encoder_output_dim = emb_dim

    if USE_NODE_TOKENS:
        adapter = NodeTokenAdapter(encoder_output_dim, decoder.config.d_model, ADAPTER_DROPOUT).to(DEVICE)
    else:
        adapter = GraphTokenAdapter(encoder_output_dim, GRAPH_TOKEN_LEN, decoder.config.d_model, ADAPTER_DROPOUT).to(DEVICE)
    adapter_path = best_dir / "graph_adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")
    adapter.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
    adapter.eval()

    model = Graph2TextT5(encoder, adapter, decoder, use_node_tokens=USE_NODE_TOKENS).to(DEVICE)

    fg_allowlist = set()
    if os.path.exists(TRAIN_GRAPHS):
        train_ds = GraphTextDataset(TRAIN_GRAPHS, include_text=True)
        fg_allowlist = select_rare_functional_groups(
            train_ds.base.graphs,
            n_groups=N_FUNC_GROUPS,
            heavy_tail_quantile=HEAVY_TAIL_QUANTILE,
        )
        if fg_allowlist:
            print(
                f"[info] Using {len(fg_allowlist)} rare functional groups for prompts "
                f"(N_FUNC_GROUPS={N_FUNC_GROUPS}, HEAVY_TAIL_QUANTILE={HEAVY_TAIL_QUANTILE})"
            )
    else:
        print(f"[warn] Train graphs not found at {TRAIN_GRAPHS}; using all functional groups for prompts.")

    ds = GraphOnlyDataset(test_graphs)
    batch_sampler = BucketBatchSampler(
        sizes=ds.node_sizes(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    collate = make_t5_gen_collate(
        tokenizer,
        MAX_PROMPT_LEN,
        emb_dict=emb_dict,
        fg_allowlist=fg_allowlist if fg_allowlist else None,
    )
    dl = DataLoader(ds, batch_sampler=batch_sampler, shuffle=False, collate_fn=collate)

    results: List[Tuple[str, str]] = []
    seen = 0
    pbar = tqdm(dl, total=len(dl), desc="Generating", leave=False, dynamic_ncols=True)
    for batch_graph, graph_embs, prompt_input_ids, prompt_attention_mask, prompts in pbar:
        preds = generate_from_graph_batch(
            model=model,
            tokenizer=tokenizer,
            graphs=batch_graph,
            device=DEVICE,
            use_amp=USE_AMP,
            gen_kwargs=GEN_CFG,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_texts=prompts,
            graph_embs=graph_embs,
        )
        batch_ids = ds.ids[seen : seen + len(preds)]
        results.extend(zip(batch_ids, preds))
        seen += len(preds)

    submission = [{"ID": _id, "description": pred} for _id, pred in results]
    pd.DataFrame(submission).to_csv(OUT_CSV, index=False)
    print(f"Saved {len(submission)} predictions to {OUT_CSV}")


if __name__ == "__main__":
    main()
