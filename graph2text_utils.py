"""
Shared utilities for Graph2Text training, evaluation, and inference.
Centralizes dataset helpers, batch bucketing, generation, and text metrics
to avoid duplication across scripts.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Iterable, List, Sequence, Tuple

import sacrebleu
import torch
import torch.nn as nn
from bert_score import score as bertscore
from tqdm.auto import tqdm
from torch import autocast
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import logging as hf_logging
import warnings

from data_utils import PreprocessedGraphDataset, batch_graphs_with_cache
from models_gine import GINEEncoder

AMP_DTYPE_F = torch.float16

warnings.filterwarnings("ignore", message=".*decoder-only architecture.*right-padding.*")
hf_logging.set_verbosity_error()

# -----------------------
# Device helpers
# -----------------------
def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_texts(texts: Sequence[str]) -> List[str]:
    return [" ".join(t.split()) for t in texts]


# -----------------------
# Model wrapper
# -----------------------
class Graph2Text(nn.Module):
    """
    Lightweight wrapper to pair a graph encoder with a GPT-2 decoder.
    """

    def __init__(self, enc: nn.Module, dec: GPT2LMHeadModel):
        super().__init__()
        self.enc = enc
        self.dec = dec

        if hasattr(enc, "cfg") and hasattr(enc.cfg, "dim"):
            enc_dim = enc.cfg.dim
        elif hasattr(enc, "output_dim"):
            enc_dim = enc.output_dim
        else:
            enc_dim = dec.config.n_embd
        dec_dim = dec.config.n_embd
        self.proj = nn.Identity() if enc_dim == dec_dim else nn.Linear(enc_dim, dec_dim)

    def forward(self, graphs: Batch, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        enc_states, enc_mask = self.enc(graphs)
        enc_states = self.proj(enc_states)
        out = self.dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=enc_states,
            encoder_attention_mask=enc_mask,
            use_cache=False,
        )
        return out


class GINEEncoderAdapter(nn.Module):
    """
    Adapts a GINE encoder to the (encoder_hidden_states, encoder_attention_mask) API.
    """

    def __init__(self, encoder: GINEEncoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = encoder.output_dim

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        _, node_emb = self.encoder(batch)
        batch_idx = batch.batch
        num_graphs = int(batch.num_graphs)
        counts = torch.bincount(batch_idx, minlength=num_graphs)
        max_nodes = int(counts.max().item()) if counts.numel() > 0 else 0

        enc_states = node_emb.new_zeros((num_graphs, max_nodes, node_emb.size(-1)))
        enc_mask = node_emb.new_zeros((num_graphs, max_nodes), dtype=torch.long)
        for g in range(num_graphs):
            idx = (batch_idx == g).nonzero(as_tuple=False).squeeze(1)
            n = int(idx.numel())
            if n == 0:
                continue
            enc_states[g, :n] = node_emb.index_select(0, idx)
            enc_mask[g, :n] = 1

        return enc_states, enc_mask


# -----------------------
# Datasets
# -----------------------
class GraphTextDataset(Dataset):
    """
    Wraps a pickled list of graphs, optionally returning the paired text.
    """

    def __init__(self, graph_path: str, include_text: bool = True):
        self.base = PreprocessedGraphDataset(graph_path, emb_dict=None)
        self.include_text = include_text

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        g = self.base.graphs[idx]
        if self.include_text:
            return g, g.description
        return g

    def node_sizes(self) -> List[int]:
        return [int(g.num_nodes) for g in self.base.graphs]


class GraphOnlyDataset(GraphTextDataset):
    """
    Variant that always returns graphs (no text).
    """

    def __init__(self, graph_path: str):
        super().__init__(graph_path, include_text=False)
        self.ids = self.base.ids

    def __getitem__(self, idx: int):
        return self.base.graphs[idx]


# -----------------------
# Bucketing sampler (shared)
# -----------------------
class BucketBatchSampler(Sampler[List[int]]):
    """
    Groups samples with similar sizes into the same mini-batch.
    Keeps max padding lower for graph batches.
    """

    def __init__(
        self,
        sizes: List[int],
        batch_size: int,
        shuffle: bool = True,
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
        buckets = [inds[i : i + self.bucket_size] for i in range(0, len(inds), self.bucket_size)]

        if self.shuffle:
            rng.shuffle(buckets)

        for b in buckets:
            if self.shuffle:
                rng.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                batch = b[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

        self.seed += 1

    def __len__(self):
        if self.drop_last:
            return len(self.sizes) // self.batch_size
        return math.ceil(len(self.sizes) / self.batch_size)


class DualBucketBatchSampler(Sampler[List[int]]):
    """
    Groups samples with similar primary and secondary sizes into the same mini-batch.
    Sorting is done by (primary_size, secondary_size).
    """

    def __init__(
        self,
        primary_sizes: List[int],
        secondary_sizes: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = 20,
        seed: int = 42,
    ):
        if len(primary_sizes) != len(secondary_sizes):
            raise ValueError("Primary and secondary sizes must have the same length.")
        self.primary_sizes = primary_sizes
        self.secondary_sizes = secondary_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_size = batch_size * bucket_size_multiplier
        self.indices = list(range(len(primary_sizes)))

    def __iter__(self):
        rng = random.Random(self.seed)
        inds = sorted(
            self.indices,
            key=lambda i: (self.primary_sizes[i], self.secondary_sizes[i]),
        )
        buckets = [inds[i : i + self.bucket_size] for i in range(0, len(inds), self.bucket_size)]

        if self.shuffle:
            rng.shuffle(buckets)

        for b in buckets:
            if self.shuffle:
                rng.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                batch = b[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

        self.seed += 1

    def __len__(self):
        if self.drop_last:
            return len(self.primary_sizes) // self.batch_size
        return math.ceil(len(self.primary_sizes) / self.batch_size)


def build_bucket_sampler(
    graphs: Sequence,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    bucket_size_multiplier: int = 20,
    seed: int = 42,
) -> BucketBatchSampler:
    sizes = [int(g.num_nodes) for g in graphs]
    return BucketBatchSampler(
        sizes=sizes,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        bucket_size_multiplier=bucket_size_multiplier,
        seed=seed,
    )


# -----------------------
# Collate for graph + text
# -----------------------
def make_text_collate(tokenizer: GPT2TokenizerFast, max_len: int, return_text: bool = False, prefix: str = "") -> Callable:
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []
    pref_len = len(prefix_ids)

    def _collate(batch: List[Tuple[object, str]]):
        graphs, texts = zip(*batch)
        batch_graph = batch_graphs_with_cache(list(graphs))

        if prefix:
            texts_proc = [prefix + t for t in texts]
        else:
            texts_proc = list(texts)

        tok = tokenizer(
            texts_proc,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if pref_len > 0:
            labels[:, :pref_len] = -100

        if return_text:
            return batch_graph, input_ids, attention_mask, labels, list(texts)
        return batch_graph, input_ids, attention_mask, labels

    return _collate


# -----------------------
# Generation helpers
# -----------------------
@torch.inference_mode()
def generate_from_graph_batch(
    encoder,
    decoder: GPT2LMHeadModel,
    proj: nn.Module | None,
    tokenizer: GPT2TokenizerFast,
    graphs,
    device: str,
    use_amp: bool,
    max_new_tokens: int,
    num_beams: int,
    no_repeat_ngram: int,
    length_penalty: float,
    early_stopping: bool = True,
    prefix: str = "",
) -> List[str]:
    graphs = graphs.to(device)
    batch_size = graphs.num_graphs


    with autocast(device_type=device, dtype=AMP_DTYPE_F, enabled=use_amp):
        enc_states, enc_mask = encoder(graphs)
        if proj is not None:
            enc_states = proj(enc_states)

    if prefix:
        tok = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(device).expand(batch_size, -1).contiguous()
        attn_mask = tok["attention_mask"].to(device).expand(batch_size, -1).contiguous()
    else:
        input_ids = torch.full((batch_size, 1), tokenizer.eos_token_id, device=device, dtype=torch.long)
        attn_mask = torch.ones_like(input_ids)

    with autocast(device_type=device, dtype=AMP_DTYPE_F, enabled=use_amp):
        gen = decoder.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=no_repeat_ngram,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            encoder_hidden_states=enc_states,
            encoder_attention_mask=enc_mask,
        )

    return clean_texts(tokenizer.batch_decode(gen, skip_special_tokens=True))


@torch.inference_mode()
def run_generation_loader(
    encoder,
    decoder: GPT2LMHeadModel,
    proj: nn.Module | None,
    tokenizer: GPT2TokenizerFast,
    dl,
    device: str,
    use_amp: bool,
    gen_kwargs: dict,
    ids: Sequence[str] | None = None,
) -> List[Tuple[str, str]] | List[str]:
    results = []
    seen = 0

    pbar = tqdm(dl, total=len(dl), desc="Generating", leave=False, dynamic_ncols=True)
    for graphs in pbar:
        preds = generate_from_graph_batch(
            encoder=encoder,
            decoder=decoder,
            proj=proj,
            tokenizer=tokenizer,
            graphs=graphs,
            device=device,
            use_amp=use_amp,
            max_new_tokens=gen_kwargs["max_new_tokens"],
            num_beams=gen_kwargs["num_beams"],
            no_repeat_ngram=gen_kwargs["no_repeat_ngram"],
            length_penalty=gen_kwargs["length_penalty"],
            early_stopping=gen_kwargs.get("early_stopping", True),
            prefix=gen_kwargs.get("prefix", ""),
        )

        if ids is None:
            results.extend(preds)
        else:
            batch_ids = ids[seen : seen + len(preds)]
            results.extend(zip(batch_ids, preds))
            seen += len(preds)
        pbar.update(1)

    return results


# -----------------------
# Metrics
# -----------------------
def compute_bleu(preds: List[str], refs: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return float(bleu.score)


def compute_bertscore(
    preds: List[str],
    refs: List[str],
    model_type: str,
    device: str,
    num_layers: int | None = None,
    lang: str | None = "en",
    rescale: bool = True,
) -> dict:
    P, R, F1 = bertscore(
        cands=preds,
        refs=refs,
        model_type=model_type,
        num_layers=num_layers,
        lang=lang,
        device=device,
        rescale_with_baseline=rescale,
        verbose=False,
    )
    return {
        "bertscore_precision": float(P.mean().item()),
        "bertscore_recall": float(R.mean().item()),
        "bertscore_f1": float(F1.mean().item()),
    }


# -----------------------
# Evaluation
# -----------------------
@torch.inference_mode()
def evaluate_model(
    model,
    dl,
    tokenizer: GPT2TokenizerFast,
    device: str,
    use_amp: bool,
    gen_kwargs: dict | None = None,
    compute_text_metrics: bool = False,
    bertscore_model: str = "roberta-base",
    bertscore_num_layers: int | None = None,
    bertscore_lang: str | None = "en",
    bertscore_rescale: bool = True,
    return_predictions: bool = False,
):
    model.eval()
    losses: List[float] = []
    preds: List[str] = []
    refs: List[str] = []

    pbar = tqdm(dl, total=len(dl), desc="Eval", leave=False, dynamic_ncols=True)

    for batch in pbar:
        if len(batch) == 5:
            graphs, input_ids, attention_mask, labels, texts = batch
        else:
            graphs, input_ids, attention_mask, labels = batch
            texts = None

        graphs = graphs.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with autocast(device_type=device, dtype=AMP_DTYPE_F, enabled=use_amp):
            out = model(graphs, input_ids, attention_mask, labels)

        losses.append(out.loss.detach().cpu().item())

        if compute_text_metrics:
            assert gen_kwargs is not None, "gen_kwargs required for text metrics"
            batch_preds = generate_from_graph_batch(
                encoder=model.enc,
                decoder=model.dec,
                proj=model.proj,
                tokenizer=tokenizer,
                graphs=graphs,
                device=device,
                use_amp=use_amp,
                max_new_tokens=gen_kwargs["max_new_tokens"],
                num_beams=gen_kwargs["num_beams"],
                no_repeat_ngram=gen_kwargs["no_repeat_ngram"],
                length_penalty=gen_kwargs["length_penalty"],
                early_stopping=gen_kwargs.get("early_stopping", True),
                prefix=gen_kwargs.get("prefix", ""),
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
                num_layers=bertscore_num_layers,
                lang=bertscore_lang,
                rescale=bertscore_rescale,
            )
        )

    model.train()

    if return_predictions:
        return metrics, preds, refs
    return metrics


def set_decoder_core_trainable(decoder: GPT2LMHeadModel, train_decoder_core: bool):
    """
    Enable/disable grads for GPT-2 self-attn and MLP blocks.
    Cross-attn blocks remain trainable to keep conditioning active.
    """
    for block in decoder.transformer.h:
        for mod in (block.attn, block.mlp, block.ln_1, block.ln_2):
            for p in mod.parameters():
                p.requires_grad = train_decoder_core


def set_decoder_dropout(decoder: GPT2LMHeadModel, self_attn_p: float | None, cross_attn_p: float | None):
    """
    Override attention dropout probabilities if provided.
    """
    for block in decoder.transformer.h:
        if self_attn_p is not None:
            block.attn.attn_dropout.p = self_attn_p
        if cross_attn_p is not None and getattr(block, "crossattention", None) is not None:
            block.crossattention.attn_dropout.p = cross_attn_p


def build_param_groups(model: Graph2Text, lr_settings: dict, freeze_decoder_epochs: int, weight_decay: float) -> List[dict]:
    """
    Create parameter groups with different LRs for encoder, cross-attn, and decoder core.
    """
    no_decay = ["bias", "ln", "LayerNorm.weight"]
    groups = {
        "enc_decay": [],
        "enc_nd": [],
        "cross_decay": [],
        "cross_nd": [],
        "dec_decay": [],
        "dec_nd": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad and freeze_decoder_epochs > 0:
            # Still register frozen params; they will get grads later when unfrozen.
            pass
        target = None
        if name.startswith("enc."):
            target = "enc"
        elif "crossattention" in name:
            target = "cross"
        else:
            target = "dec"

        nd = any(nd_tok in name for nd_tok in no_decay)
        key = f"{target}_nd" if nd else f"{target}_decay"
        if key not in groups:
            groups[key] = []
        groups[key].append(param)

    encoder_lr = lr_settings["encoder_lr"]
    cross_attn_lr = lr_settings["cross_attn_lr"]
    decoder_core_lr = lr_settings["decoder_core_lr"]
    pg = []
    if groups["enc_decay"]:
        pg.append({"params": groups["enc_decay"], "lr": encoder_lr, "weight_decay": weight_decay})
    if groups["enc_nd"]:
        pg.append({"params": groups["enc_nd"], "lr": encoder_lr, "weight_decay": 0.0})
    if groups["cross_decay"]:
        pg.append({"params": groups["cross_decay"], "lr": cross_attn_lr, "weight_decay": weight_decay})
    if groups["cross_nd"]:
        pg.append({"params": groups["cross_nd"], "lr": cross_attn_lr, "weight_decay": 0.0})
    if groups["dec_decay"]:
        pg.append({"params": groups["dec_decay"], "lr": decoder_core_lr, "weight_decay": weight_decay})
    if groups["dec_nd"]:
        pg.append({"params": groups["dec_nd"], "lr": decoder_core_lr, "weight_decay": 0.0})
    return pg
