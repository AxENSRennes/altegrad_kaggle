#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from graph2text_utils import GraphTextDataset, select_device
from data_utils import load_id2emb
from train_graph2text import (
    ADAPTER_DROPOUT,
    FG_PROMPT_PREFIX,
    FG_PROMPT_SUFFIX,
    GINE_CKPT_PATH,
    GRAPH_TOKEN_LEN,
    MAX_TEXT_LEN,
    N_FUNC_GROUPS,
    PREFIX,
    PROMPT_MAX_LEN,
    T5_MODEL_NAME,
    TRAIN_GRAPHS,
    USE_NODE_TOKENS,
    VAL_GRAPH_EMB,
    VAL_GRAPHS,
    HEAVY_TAIL_QUANTILE,
    get_emb_dim,
    load_frozen_gine_encoder,
    make_t5_collate,
    rate_functional_groups,
)
from train_graph2text import Graph2TextT5, GraphTokenAdapter, NodeTokenAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize T5 cross-attention from graph encoder tokens to decoder tokens."
    )
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Path to T5 checkpoint dir.")
    parser.add_argument("--gine_ckpt", type=str, default=GINE_CKPT_PATH, help="Path to frozen GINE checkpoint.")
    parser.add_argument("--val_graphs", type=str, default=VAL_GRAPHS, help="Path to validation graphs pkl.")
    parser.add_argument("--val_graph_emb", type=str, default=VAL_GRAPH_EMB, help="Path to validation embeddings csv.")
    parser.add_argument("--index", type=int, default=0, help="Validation sample index.")
    parser.add_argument("--out_dir", type=str, default="plots/cross_attention", help="Output directory.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps.")
    parser.add_argument("--use_node_tokens", action="store_true", default=USE_NODE_TOKENS)
    parser.add_argument("--graph_token_len", type=int, default=GRAPH_TOKEN_LEN)
    parser.add_argument("--max_text_len", type=int, default=MAX_TEXT_LEN)
    parser.add_argument("--prompt_max_len", type=int, default=PROMPT_MAX_LEN)
    parser.add_argument("--prefix", type=str, default=PREFIX)
    parser.add_argument("--max_decoder_tokens", type=int, default=None)
    parser.add_argument("--max_encoder_tokens", type=int, default=None)
    parser.set_defaults(group_words=True)
    parser.add_argument(
        "--no_group_words",
        action="store_false",
        dest="group_words",
        help="Disable word-level grouping (use raw subword tokens).",
    )
    return parser.parse_args()


WORD_START_MARKERS = ("\u2581", "\u0120")  # sentencepiece, BPE


def group_tokens_by_word(tokens: list[str], tokenizer) -> tuple[list[str], list[list[int]]]:
    special = set(getattr(tokenizer, "all_special_tokens", []) or [])
    groups: list[list[int]] = []
    current: list[int] = []
    for i, tok in enumerate(tokens):
        if tok in special:
            if current:
                groups.append(current)
                current = []
            groups.append([i])
            continue
        is_new = tok.startswith(WORD_START_MARKERS) or not current
        if is_new and current:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    labels: list[str] = []
    for group in groups:
        pieces = [tokens[i] for i in group]
        if len(group) == 1 and pieces[0] in special:
            labels.append(pieces[0])
            continue
        word = tokenizer.convert_tokens_to_string(pieces).strip()
        if not word:
            word = "".join(p.replace("\u2581", " ").replace("\u0120", " ") for p in pieces).strip()
        labels.append(word if word else "<unk>")
    return labels, groups


def sum_attention_by_groups(
    attn: torch.Tensor,
    row_groups: list[list[int]],
    col_groups: list[list[int]] | None = None,
) -> torch.Tensor:
    if not row_groups:
        return attn.new_zeros((0, attn.size(1)))
    row_agg = torch.stack([attn[group].sum(dim=0) for group in row_groups], dim=0)
    if col_groups is None:
        return row_agg
    if not col_groups:
        return row_agg.new_zeros((row_agg.size(0), 0))
    col_agg = torch.stack([row_agg[:, group].sum(dim=1) for group in col_groups], dim=1)
    return col_agg


def set_ticks(ax, labels, axis: str, max_labels: int = 32):
    if not labels:
        return
    if len(labels) > max_labels:
        step = max(1, math.ceil(len(labels) / max_labels))
        ticks = list(range(0, len(labels), step))
        tick_labels = [labels[i] for i in ticks]
    else:
        ticks = list(range(len(labels)))
        tick_labels = labels
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=6)


def save_heatmap(attn, xlabels, ylabels, title: str, out_path: Path, vmax: float | None = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if vmax is None:
        vmax = float(attn.max()) if attn.numel() else 1.0
    im = ax.imshow(attn, aspect="auto", origin="upper", vmin=0.0, vmax=vmax, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Encoder tokens")
    ax.set_ylabel("Decoder tokens")
    set_ticks(ax, xlabels, axis="x")
    set_ticks(ax, ylabels, axis="y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    device = args.device or select_device()

    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = f"ckpt_graph2text/{T5_MODEL_NAME.replace('/', '_')}/best"
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"T5 checkpoint dir not found: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    decoder = T5ForConditionalGeneration.from_pretrained(str(ckpt_dir)).to(device)
    encoder = load_frozen_gine_encoder(args.gine_ckpt, device)

    emb_dict = None
    if not args.use_node_tokens:
        emb_dict = load_id2emb(args.val_graph_emb)
        emb_dim = get_emb_dim(emb_dict)
        if emb_dim is None:
            raise ValueError("Could not infer embedding dimension for graph tokens.")
        adapter = GraphTokenAdapter(emb_dim, args.graph_token_len, decoder.config.d_model, ADAPTER_DROPOUT)
    else:
        adapter = NodeTokenAdapter(encoder.output_dim, decoder.config.d_model, ADAPTER_DROPOUT)

    adapter_path = ckpt_dir / "graph_adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))

    model = Graph2TextT5(encoder, adapter, decoder, use_node_tokens=args.use_node_tokens).to(device)
    model.eval()

    train_ds = GraphTextDataset(TRAIN_GRAPHS, include_text=True)
    fg_freq = rate_functional_groups(train_ds.base.graphs)

    ds = GraphTextDataset(args.val_graphs, include_text=True)
    if args.index < 0 or args.index >= len(ds):
        raise IndexError(f"Index {args.index} out of range for validation set of size {len(ds)}.")

    collate = make_t5_collate(
        tokenizer=tokenizer,
        max_len=args.max_text_len,
        prefix=args.prefix,
        return_text=True,
        emb_dict=emb_dict,
        prompt_max_len=args.prompt_max_len,
        fg_freq=fg_freq,
        n_func_groups=N_FUNC_GROUPS,
    )
    batch = collate([ds[args.index]])
    (
        graphs,
        graph_embs,
        decoder_input_ids,
        decoder_attention_mask,
        _labels,
        texts,
        prompts,
        prompt_input_ids,
        prompt_attention_mask,
    ) = batch

    graphs = graphs.to(device)
    if graph_embs is not None:
        graph_embs = graph_embs.to(device)
    decoder_input_ids = decoder_input_ids.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)

    with torch.inference_mode():
        enc_states, enc_mask = model.encode_graphs(graphs, graph_embs=graph_embs)
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_states)
        out = model.dec(
            encoder_outputs=encoder_outputs,
            attention_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    cross_attn = torch.stack(out.cross_attentions, dim=0)  # [layers, batch, heads, tgt, src]
    cross_attn = cross_attn[:, 0]  # [layers, heads, tgt, src]
    decoder_attn = torch.stack(out.decoder_attentions, dim=0)  # [layers, batch, heads, tgt, tgt]
    decoder_attn = decoder_attn[:, 0]  # [layers, heads, tgt, tgt]

    dec_len = cross_attn.size(2)
    enc_len = int(enc_mask[0].sum().item())
    labels = _labels[0].to(device)
    desc_positions = (labels != -100).nonzero(as_tuple=False).squeeze(1).tolist()
    desc_positions = [pos for pos in desc_positions if pos < dec_len]
    if args.max_decoder_tokens:
        desc_positions = desc_positions[: args.max_decoder_tokens]
    if not desc_positions:
        raise ValueError("No description tokens found to visualize.")
    dec_len = max(desc_positions) + 1
    if args.max_encoder_tokens:
        enc_len = min(enc_len, args.max_encoder_tokens)

    desc_ids = labels[desc_positions].tolist()
    decoder_tokens = tokenizer.convert_ids_to_tokens(desc_ids)

    if args.use_node_tokens:
        encoder_tokens = [f"node_{i}" for i in range(enc_len)]
    else:
        encoder_tokens = [f"graph_tok_{i}" for i in range(enc_len)]

    prompt_len = int(prompt_attention_mask[0].sum().item()) - 1
    prompt_len = max(prompt_len, 0)
    prompt_len = min(prompt_len, dec_len - 1)
    prompt_ids = prompt_input_ids[0, 1 : 1 + prompt_len].tolist()
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)

    if args.group_words:
        decoder_tokens, decoder_groups = group_tokens_by_word(decoder_tokens, tokenizer)
        prompt_tokens, prompt_groups = group_tokens_by_word(prompt_tokens, tokenizer)
    else:
        decoder_groups = [[i] for i in range(len(decoder_tokens))]
        prompt_groups = [[i] for i in range(len(prompt_tokens))]

    out_dir = Path(args.out_dir)
    graph_dir = out_dir / "graph_tokens"
    prompt_dir = out_dir / "functional_groups"
    graph_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(cross_attn.size(0)):
        for head_idx in range(cross_attn.size(1)):
            graph_attn = cross_attn[layer_idx, head_idx, :dec_len, :enc_len].cpu()
            graph_attn = graph_attn[desc_positions, :]
            if args.group_words:
                graph_attn = sum_attention_by_groups(graph_attn, decoder_groups)

            prompt_attn = None
            if prompt_len > 0:
                prompt_cols = list(range(prompt_len))
                prompt_attn = decoder_attn[layer_idx, head_idx, :dec_len, :dec_len].cpu()
                prompt_attn = prompt_attn[desc_positions][:, prompt_cols]
                if args.group_words:
                    prompt_attn = sum_attention_by_groups(prompt_attn, decoder_groups, prompt_groups)

            vmax = float(graph_attn.max()) if graph_attn.numel() else 1.0
            if prompt_attn is not None and prompt_attn.numel():
                vmax = max(vmax, float(prompt_attn.max()))

            graph_title = f"Layer {layer_idx} Head {head_idx} | generated vs graph tokens"
            save_heatmap(
                graph_attn,
                xlabels=encoder_tokens,
                ylabels=decoder_tokens,
                title=graph_title,
                out_path=graph_dir / f"layer_{layer_idx}_head_{head_idx}.png",
                vmax=vmax,
            )

            if prompt_attn is not None:
                prompt_title = f"Layer {layer_idx} Head {head_idx} | generated vs functional groups"
                save_heatmap(
                    prompt_attn,
                    xlabels=prompt_tokens,
                    ylabels=decoder_tokens,
                    title=prompt_title,
                    out_path=prompt_dir / f"layer_{layer_idx}_head_{head_idx}.png",
                    vmax=vmax,
                )

    summary_path = out_dir / "sample_info.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"sample_index: {args.index}\n")
        f.write(f"prompt: {prompts[0]}\n")
        f.write(f"description: {texts[0]}\n")


if __name__ == "__main__":
    main()
