# train_graph2text.py
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import autocast
from tqdm.auto import tqdm

from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from graph2text_utils import (
    DualBucketBatchSampler,
    GraphTextDataset,
    clean_texts,
    compute_bleu,
    compute_bertscore,
    select_device,
)
from data_utils import batch_graphs_with_cache, load_id2emb
from models_gine import GINEConfig, GINEEncoder


# ----------------------------
# CONFIG
# ----------------------------
TRAIN_GRAPHS = "data/train_graphs_func_groups.pkl"
VAL_GRAPHS = "data/validation_graphs_func_groups.pkl"
TRAIN_GRAPH_EMB = "data/train_graph_embeddings.csv"
VAL_GRAPH_EMB = "data/validation_graph_embeddings.csv"

DEVICE = select_device()
DEVICE = "cpu"
USE_AMP = False
AUTOCAST_DTYPE = torch.bfloat16 if USE_AMP else torch.float32

BATCH_SIZE = 42 if DEVICE in {"mps", "cuda"} else 16
EPOCHS = 30
LR_ADAPTER = 4e-4
MIN_LR_ADAPTER = 1e-4
LR_DECODER = 3e-5
MIN_LR_DECODER = 1e-5
WEIGHT_DECAY = 0.001
MAX_TEXT_LEN = 128
PROMPT_MAX_LEN = 100
N_FUNC_GROUPS = 6
HEAVY_TAIL_QUANTILE = 0.0

GRAD_ACCUM = 1
CLIP_NORM = 1.0

SEED = 42
BERTSCORE_MODEL = "roberta-base"
RESUME_FROM_BEST = True
FREEZE_T5_EPOCHS = 0

# GINE encoder (frozen)
GINE_CKPT_PATH = "ckpt_gine_contrastive/best.pt"

# T5 decoder
T5_MODEL_NAME = "t5-small"

# Graph-to-token adapter
USE_NODE_TOKENS = True
GRAPH_TOKEN_LEN = 8
ADAPTER_DROPOUT = 0.1

PREFIX = "The molecule is "
FG_PROMPT_PREFIX = "Functional groups: "
FG_PROMPT_EMPTY = "none"
FG_PROMPT_SUFFIX = ". "

GENERATION_CFG = {
    "max_new_tokens": 96,
    "num_beams": 4,
    "no_repeat_ngram": 3,
    "length_penalty": 1.1,
    "early_stopping": True,
}

SAVE_DIR = "ckpt_graph2text"
BEST_SUBDIR = "best"
VAL_OUTPUT_CSV = "val_generations.csv"


# ----------------------------
# Helpers
# ----------------------------
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_add_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    pref = prefix.strip()
    if text.lower().startswith(pref.lower()):
        return text
    return prefix + text.lstrip()


def format_functional_groups_prompt(groups: List[str]) -> str:
    if not groups:
        groups_txt = FG_PROMPT_EMPTY
    else:
        groups_txt = ", ".join(groups)
    return f"{FG_PROMPT_PREFIX}{groups_txt}{FG_PROMPT_SUFFIX}"


def shift_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    shifted = input_ids.new_full(input_ids.shape, pad_token_id)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    return shifted


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


def build_emb_tensor(graphs: List, emb_dict: dict) -> torch.Tensor:
    embs = [lookup_graph_embedding(emb_dict, g.id) for g in graphs]
    return torch.stack(embs, dim=0)


def rate_functional_groups(graphs: List) -> dict:
    freq: dict[str, int] = {}
    for g in graphs:
        groups = getattr(g, "functional_groups", []) or []
        for fg in groups:
            freq[fg] = freq.get(fg, 0) + 1
    return freq


def compute_desc_token_lengths(graphs: List, tokenizer, max_len: int, prefix: str) -> List[int]:
    texts = [maybe_add_prefix(getattr(g, "description", ""), prefix) for g in graphs]
    tok = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    return [len(ids) for ids in tok["input_ids"]]


def print_functional_group_distribution(freq: dict) -> None:
    if not freq:
        print("[info] Functional group frequency distribution is empty.")
        return
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    counts = sorted(freq.values())
    min_c = counts[0]
    max_c = counts[-1]
    mean_c = sum(counts) / len(counts)
    print(
        f"[info] Functional groups: {len(freq)} | "
        f"count min/mean/max={min_c}/{mean_c:.2f}/{max_c}"
    )
    for fg, c in items:
        print(f"[fg] {fg}\t{c}")


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
    # Drop the least frequent groups (rare tail), then select the rarest among remaining.
    filtered = {fg: c for fg, c in freq.items() if c >= cutoff}
    if not filtered:
        filtered = freq
    items = sorted(filtered.items(), key=lambda kv: (kv[1], kv[0]))
    selected = {fg for fg, _ in items[:n_groups]}
    return selected


def select_rarest_groups_for_graph(groups: List[str], freq: dict[str, int], n_groups: int) -> List[str]:
    if not groups or n_groups <= 0:
        return []
    ranked = sorted(groups, key=lambda fg: (freq.get(fg, 0), fg))
    return ranked[:n_groups]


class GraphTextWithEmbDataset(Dataset):
    def __init__(self, graphs: List, emb_tensor: torch.Tensor):
        self.graphs = graphs
        self.embs = emb_tensor

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g, g.description, self.embs[idx]

    def node_sizes(self) -> List[int]:
        return [int(g.num_nodes) for g in self.graphs]


def make_t5_collate(
    tokenizer,
    max_len: int,
    prefix: str,
    return_text: bool = False,
    emb_dict: dict | None = None,
    prompt_max_len: int | None = None,
    fg_freq: dict[str, int] | None = None,
    n_func_groups: int | None = None,
    return_fg_counts: bool = False,
):
    def _collate(batch):
        graph_embs = None
        if len(batch[0]) == 3:
            graphs, texts, emb_list = zip(*batch)
            graph_embs = torch.stack(list(emb_list), dim=0)
        else:
            graphs, texts = zip(*batch)
        batch_graph = batch_graphs_with_cache(list(graphs))
        if graph_embs is None and emb_dict is not None:
            graph_embs = torch.stack([lookup_graph_embedding(emb_dict, g.id) for g in graphs], dim=0)
        texts_proc = [maybe_add_prefix(t, prefix) for t in texts]
        if fg_freq is not None and n_func_groups is not None:
            kept_groups = [
                select_rarest_groups_for_graph(
                    list(getattr(g, "functional_groups", []) or []),
                    fg_freq,
                    n_func_groups,
                )
                for g in graphs
            ]
        else:
            kept_groups = [list(getattr(g, "functional_groups", []) or []) for g in graphs]
        prompts = [format_functional_groups_prompt(groups) for groups in kept_groups]
        fg_kept_count = sum(len(groups) for groups in kept_groups)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        decoder_start_token_id = pad_token_id
        prompt_tok = tokenizer(
            list(prompts),
            padding=False,
            truncation=prompt_max_len is not None,
            max_length=prompt_max_len,
            return_tensors=None,
            add_special_tokens=False,
        )
        desc_tok = tokenizer(
            list(texts_proc),
            padding=False,
            truncation=True,
            max_length=max_len,
            return_tensors=None,
            add_special_tokens=True,
        )
        prompt_lens = [len(p_ids) for p_ids in prompt_tok["input_ids"]]
        desc_lens = [len(d_ids) for d_ids in desc_tok["input_ids"]]
        if prompt_lens and desc_lens:
            prompt_mean = sum(prompt_lens) / len(prompt_lens)
            desc_mean = sum(desc_lens) / len(desc_lens)
            """
            print(
                f"[stats] prompt_len min/mean/max={min(prompt_lens)}/{prompt_mean:.1f}/{max(prompt_lens)} | "
                f"desc_len min/mean/max={min(desc_lens)}/{desc_mean:.1f}/{max(desc_lens)}"
            )"""
        full_input_ids = []
        for p_ids, d_ids in zip(prompt_tok["input_ids"], desc_tok["input_ids"]):
            full_input_ids.append(p_ids + d_ids)
        tok = tokenizer.pad(
            {"input_ids": full_input_ids},
            padding=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"]
        attention_mask = (input_ids != pad_token_id).long()
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            plen = min(plen, labels.size(1))
            labels[i, :plen] = -100
        labels[attention_mask == 0] = -100
        decoder_input_ids = shift_right(input_ids, pad_token_id, decoder_start_token_id)
        decoder_attention_mask = (decoder_input_ids != pad_token_id).long()
        prompt_input_ids = tokenizer.pad(
            {"input_ids": prompt_tok["input_ids"]},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        prompt_attention_mask = (prompt_input_ids != pad_token_id).long()
        prompt_start = prompt_input_ids.new_full((prompt_input_ids.size(0), 1), decoder_start_token_id)
        prompt_input_ids = torch.cat([prompt_start, prompt_input_ids], dim=1)
        prompt_attention_mask = torch.cat(
            [prompt_attention_mask.new_ones((prompt_attention_mask.size(0), 1)), prompt_attention_mask], dim=1
        )

        if return_text:
            return (
                batch_graph,
                graph_embs,
                decoder_input_ids,
                decoder_attention_mask,
                labels,
                list(texts_proc),
                list(prompts),
                prompt_input_ids,
                prompt_attention_mask,
            )
        if return_fg_counts:
            return batch_graph, graph_embs, decoder_input_ids, decoder_attention_mask, labels, fg_kept_count
        return batch_graph, graph_embs, decoder_input_ids, decoder_attention_mask, labels

    return _collate


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


def get_emb_dim(emb_dict: dict | None) -> int | None:
    if not emb_dict:
        return None
    sample = next(iter(emb_dict.values()))
    return int(sample.numel())


def set_t5_decoder_trainable(model: T5ForConditionalGeneration, trainable: bool):
    for p in model.decoder.parameters():
        p.requires_grad = trainable
    for p in model.lm_head.parameters():
        p.requires_grad = trainable


class PrecomputedGraphEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, graphs):
        raise RuntimeError("Precomputed graph embeddings expected; encoder is disabled.")


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

    def forward(self, graph_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.proj(graph_emb).view(-1, self.token_len, self.d_model)
        tokens = self.ln(tokens)
        enc_mask = torch.ones(tokens.size()[:2], device=tokens.device, dtype=torch.long)
        return tokens, enc_mask


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
            tokens, enc_mask = self.adapter(node_embs, graphs.batch, graphs.num_graphs)
            return tokens, enc_mask
        if graph_embs is None:
            graph_embs, _ = self.enc(graphs)
        tokens, enc_mask = self.adapter(graph_embs)
        return tokens, enc_mask

    def forward(self, graphs, decoder_input_ids, decoder_attention_mask, labels, graph_embs: torch.Tensor | None = None):
        enc_states, enc_mask = self.encode_graphs(graphs, graph_embs=graph_embs)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)
        return self.dec(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            use_cache=False,
        )


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


@torch.inference_mode()
def evaluate_model(
    model: Graph2TextT5,
    dl,
    tokenizer,
    device: str,
    use_amp: bool,
    gen_kwargs: dict | None = None,
    compute_text_metrics: bool = False,
    bertscore_model: str = "roberta-base",
):
    model.eval()
    losses: List[float] = []
    preds: List[str] = []
    refs: List[str] = []

    pbar = tqdm(dl, total=len(dl), desc="Eval", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if len(batch) == 9:
            (
                graphs,
                graph_embs,
                decoder_input_ids,
                decoder_attention_mask,
                labels,
                texts,
                prompts,
                prompt_ids,
                prompt_mask,
            ) = batch
        else:
            graphs, graph_embs, decoder_input_ids, decoder_attention_mask, labels = batch
            texts = None
            prompts = None
            prompt_ids = None
            prompt_mask = None

        graphs = graphs.to(device)
        if graph_embs is not None:
            graph_embs = graph_embs.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        decoder_attention_mask = decoder_attention_mask.to(device)
        labels = labels.to(device)
        with autocast(device_type=device, dtype=AUTOCAST_DTYPE, enabled=use_amp):
            out = model(graphs, decoder_input_ids, decoder_attention_mask, labels, graph_embs=graph_embs)
        losses.append(out.loss.detach().cpu().item())

        if compute_text_metrics:
            assert gen_kwargs is not None and texts is not None and prompts is not None
            assert prompt_ids is not None and prompt_mask is not None
            batch_preds = generate_from_graph_batch(
                model,
                tokenizer,
                graphs,
                device,
                use_amp,
                gen_kwargs,
                prompt_ids,
                prompt_mask,
                prompts,
                graph_embs,
            )
            preds.extend(batch_preds)
            refs.extend(clean_texts(texts))

        avg_loss = sum(losses) / len(losses)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    metrics = {"loss": sum(losses) / max(1, len(losses))}
    if compute_text_metrics:
        metrics["bleu"] = compute_bleu(preds, refs)
        metrics.update(
            compute_bertscore(
                preds,
                refs,
                model_type=bertscore_model,
                device=device,
            )
        )

    model.train()
    return metrics, preds, refs


# ----------------------------
# Main
# ----------------------------
def main():
    seed_all(SEED)
    run_dir = Path(SAVE_DIR) / T5_MODEL_NAME.replace("/", "_")
    best_dir = run_dir / BEST_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    decoder = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)

    # Data
    train_emb_dict = None
    val_emb_dict = None
    if not USE_NODE_TOKENS:
        if os.path.exists(TRAIN_GRAPH_EMB):
            train_emb_dict = load_id2emb(TRAIN_GRAPH_EMB)
            print(f"[info] Loaded train graph embeddings from {TRAIN_GRAPH_EMB}")
        else:
            print(f"[warn] Train graph embeddings not found at {TRAIN_GRAPH_EMB}; using GINE encoder.")
        if os.path.exists(VAL_GRAPH_EMB):
            val_emb_dict = load_id2emb(VAL_GRAPH_EMB)
            print(f"[info] Loaded val graph embeddings from {VAL_GRAPH_EMB}")
        else:
            print(f"[warn] Val graph embeddings not found at {VAL_GRAPH_EMB}; using GINE encoder.")
    elif os.path.exists(TRAIN_GRAPH_EMB) or os.path.exists(VAL_GRAPH_EMB):
        print("[info] USE_NODE_TOKENS=True; ignoring precomputed graph embeddings.")

    emb_dim = get_emb_dim(train_emb_dict) or get_emb_dim(val_emb_dict)
    if train_emb_dict is not None and val_emb_dict is not None:
        val_dim = get_emb_dim(val_emb_dict)
        if emb_dim != val_dim:
            raise ValueError(f"Train/val graph embedding dims mismatch: {emb_dim} vs {val_dim}")

    need_encoder = USE_NODE_TOKENS or train_emb_dict is None or (os.path.exists(VAL_GRAPHS) and val_emb_dict is None)
    if need_encoder:
        gine_ckpt = Path(GINE_CKPT_PATH)
        if not gine_ckpt.exists():
            raise FileNotFoundError(f"GINE checkpoint not found: {gine_ckpt}")
        encoder = load_frozen_gine_encoder(str(gine_ckpt), DEVICE)
        encoder_output_dim = encoder.output_dim
    else:
        if emb_dim is None:
            raise ValueError("Graph embedding dimension could not be inferred from CSV files.")
        encoder = PrecomputedGraphEncoder(emb_dim).to(DEVICE)
        encoder_output_dim = emb_dim

    if USE_NODE_TOKENS:
        adapter = NodeTokenAdapter(encoder_output_dim, decoder.config.d_model, ADAPTER_DROPOUT).to(DEVICE)
    else:
        adapter = GraphTokenAdapter(encoder_output_dim, GRAPH_TOKEN_LEN, decoder.config.d_model, ADAPTER_DROPOUT).to(DEVICE)
    model = Graph2TextT5(encoder, adapter, decoder, use_node_tokens=USE_NODE_TOKENS).to(DEVICE)

    if RESUME_FROM_BEST and best_dir.is_dir():
        decoder = T5ForConditionalGeneration.from_pretrained(str(best_dir)).to(DEVICE)
        model.dec = decoder
        adapter_path = best_dir / "graph_adapter.pt"
        if adapter_path.exists():
            model.adapter.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
        print(f"[info] Loaded decoder/adapter from {best_dir}")

    set_t5_decoder_trainable(model.dec, trainable=False)
    print(f"[info] Freezing T5 decoder for first {FREEZE_T5_EPOCHS} epoch(s)")

    train_ds = GraphTextDataset(TRAIN_GRAPHS, include_text=True)
    train_graphs = train_ds.base.graphs
    fg_freq = rate_functional_groups(train_graphs)
    print_functional_group_distribution(fg_freq)
    if N_FUNC_GROUPS > 0:
        print(f"[info] Using {N_FUNC_GROUPS} rarest groups per-graph for prompts")
    train_desc_sizes = compute_desc_token_lengths(train_graphs, tokenizer, MAX_TEXT_LEN, PREFIX)
    train_sizes = [int(g.num_nodes) for g in train_graphs]
    if train_emb_dict is not None:
        train_emb_tensor = build_emb_tensor(train_graphs, train_emb_dict)
        train_ds = GraphTextWithEmbDataset(train_graphs, train_emb_tensor)

    batch_sampler = DualBucketBatchSampler(
        primary_sizes=train_desc_sizes,
        secondary_sizes=train_sizes,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        bucket_size_multiplier=10,
        seed=SEED,
    )

    train_collate = make_t5_collate(
        tokenizer,
        MAX_TEXT_LEN,
        PREFIX,
        return_text=False,
        emb_dict=train_emb_dict,
        prompt_max_len=PROMPT_MAX_LEN,
        fg_freq=fg_freq,
        n_func_groups=N_FUNC_GROUPS,
        return_fg_counts=True,
    )
    train_dl = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True if DEVICE == "cpu" else False,
        collate_fn=train_collate,
    )

    val_dl = None
    if os.path.exists(VAL_GRAPHS):
        val_ds = GraphTextDataset(VAL_GRAPHS, include_text=True)
        if val_emb_dict is not None:
            val_emb_tensor = build_emb_tensor(val_ds.base.graphs, val_emb_dict)
            val_ds = GraphTextWithEmbDataset(val_ds.base.graphs, val_emb_tensor)
        val_collate = make_t5_collate(
            tokenizer,
            MAX_TEXT_LEN,
            PREFIX,
            return_text=True,
            emb_dict=val_emb_dict,
            prompt_max_len=PROMPT_MAX_LEN,
            fg_freq=fg_freq,
            n_func_groups=N_FUNC_GROUPS,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=val_collate,
        )

    optim = torch.optim.AdamW(
        [
            {"params": model.adapter.parameters(), "lr": LR_ADAPTER, "weight_decay": WEIGHT_DECAY},
            {"params": model.dec.parameters(), "lr": LR_DECODER, "weight_decay": WEIGHT_DECAY},
        ]
    )

    total_steps = math.ceil(len(train_dl) / GRAD_ACCUM) * EPOCHS
    sched = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_steps,
    )

    best_val = float("inf")
    if RESUME_FROM_BEST and best_dir.is_dir() and val_dl is not None:
        init_metrics, _, _ = evaluate_model(
            model=model,
            dl=val_dl,
            tokenizer=tokenizer,
            device=DEVICE,
            use_amp=USE_AMP,
            gen_kwargs=None,
            compute_text_metrics=False,
            bertscore_model=BERTSCORE_MODEL,
        )
        best_val = init_metrics["loss"]
        print(f"[info] Starting from resumed checkpoint with val_loss={best_val:.4f}")

    global_step = 0
    model.train()





    for ep in range(1, EPOCHS + 1):
        print("allocated:", torch.mps.current_allocated_memory() / 1024**3, "GiB")
        print("driver:", torch.mps.driver_allocated_memory() / 1024**3, "GiB")
        if ep == FREEZE_T5_EPOCHS + 1:
            set_t5_decoder_trainable(model.dec, trainable=True)
            print(f"[info] Unfroze T5 decoder at epoch {ep}")

        model.train()
        model.enc.eval()
        running = 0.0
        optim.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(train_dl, start=1),
            total=len(train_dl),
            desc=f"Epoch {ep}/{EPOCHS}",
            leave=True,
            dynamic_ncols=True,
        )

        for it, (graphs, graph_embs, decoder_input_ids, decoder_attention_mask, labels, fg_kept_count) in pbar:
            graphs = graphs.to(DEVICE)
            if graph_embs is not None:
                graph_embs = graph_embs.to(DEVICE)
            decoder_input_ids = decoder_input_ids.to(DEVICE)
            decoder_attention_mask = decoder_attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            with autocast(device_type=DEVICE, dtype=AUTOCAST_DTYPE, enabled=USE_AMP):
                out = model(graphs, decoder_input_ids, decoder_attention_mask, labels, graph_embs=graph_embs)

            loss = out.loss / GRAD_ACCUM
            loss.backward()

            running += loss.item()

            if it % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            
            pbar.set_postfix({
                "loss": f"{running:.4f}",
                "lr": f"{sched.get_last_lr()[0]:.2e}",
                "step": f"{global_step}/{total_steps}",
                "fg_kept": fg_kept_count,
            })
            running = 0.0

        if val_dl is None:
            val_loss = float("nan")
            val_metrics = {}
        else:
            want_preds = VAL_OUTPUT_CSV is not None
            eval_metrics, preds, refs = evaluate_model(
                model=model,
                dl=val_dl,
                tokenizer=tokenizer,
                device=DEVICE,
                use_amp=USE_AMP,
                gen_kwargs=GENERATION_CFG,
                compute_text_metrics=True,
                bertscore_model=BERTSCORE_MODEL,
            )
            val_metrics = eval_metrics
            val_loss = val_metrics["loss"]

            if want_preds and VAL_OUTPUT_CSV:
                import pandas as pd

                pd.DataFrame({"ref": refs, "pred": preds}).to_csv(VAL_OUTPUT_CSV, index=False)
                print(f"[info] Saved validation generations to {VAL_OUTPUT_CSV}")

            msg = {"val_loss": f"{val_loss:.4f}", "bleu": f"{val_metrics.get('bleu', float('nan')):.2f}"}
            if "bertscore_f1" in val_metrics:
                msg["bertscore_f1"] = f"{val_metrics['bertscore_f1']:.4f}"
            metric_str = ", ".join(f"{k}={v}" for k, v in msg.items())
            print(f"Epoch {ep} done. {metric_str}")

        if val_dl is None:
            ckpt = run_dir / f"epoch_{ep}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.dec.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            torch.save(model.adapter.state_dict(), ckpt / "graph_adapter.pt")
        else:
            if val_loss < best_val:
                best_val = val_loss
                best_dir.mkdir(parents=True, exist_ok=True)
                model.dec.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                torch.save(model.adapter.state_dict(), best_dir / "graph_adapter.pt")
                print(f"Saved new best checkpoint to {best_dir}")

    print("Training finished.")


if __name__ == "__main__":
    main()
