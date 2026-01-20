#!/usr/bin/env python3
from __future__ import annotations

"""
Project GINE graph embeddings to 2D and color by functional groups.

Supports PCA / t-SNE / UMAP projections and binary, multiclass, or multilabel
coloring strategies for functional groups.
"""

import argparse
import contextlib
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import batch_graphs_with_cache, load_id2emb
from graph2text_utils import GraphOnlyDataset, select_device
from models_gine import GINEConfig, GINEEncoder

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Missing dependency: matplotlib. Install it before plotting.") from exc


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> str:
    if name == "auto":
        return select_device()
    return name


def collate_graphs(batch):
    return batch_graphs_with_cache(list(batch))


def load_encoder_and_proj(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    gine_cfg = GINEConfig(**ckpt["gine_cfg"]) if "gine_cfg" in ckpt else GINEConfig()

    encoder = GINEEncoder(gine_cfg).to(device)
    state = ckpt.get("model_state", ckpt)
    enc_state = {k.replace("graph_encoder.", "", 1): v for k, v in state.items() if k.startswith("graph_encoder.")}
    if not enc_state:
        raise ValueError(f"No graph_encoder weights found in {ckpt_path}")
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    proj = None
    if "graph_proj.weight" in state:
        g_w = state["graph_proj.weight"]
        g_b = state.get("graph_proj.bias", None)
        proj = torch.nn.Linear(g_w.size(1), g_w.size(0), bias=g_b is not None)
        proj.weight.data.copy_(g_w)
        if g_b is not None:
            proj.bias.data.copy_(g_b)
        proj = proj.to(device)
        proj.eval()
        for p in proj.parameters():
            p.requires_grad = False

    return encoder, proj


@torch.inference_mode()
def encode_embeddings(
    dataset: GraphOnlyDataset,
    encoder: GINEEncoder,
    proj: torch.nn.Module | None,
    device: str,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    use_proj: bool,
    normalize: bool,
) -> torch.Tensor:
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
    )
    embs: List[torch.Tensor] = []
    amp_ctx = torch.autocast(device_type=device, dtype=torch.float16) if use_amp else contextlib.nullcontext()
    for batch_graphs in dl:
        batch_graphs = batch_graphs.to(device)
        with amp_ctx:
            emb, _ = encoder(batch_graphs)
            if use_proj and proj is not None:
                emb = proj(emb)
            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


def load_embeddings_from_csv(graph_ids: Sequence, csv_path: str, normalize: bool) -> torch.Tensor:
    id2emb = load_id2emb(csv_path)
    rows = []
    for id_ in graph_ids:
        candidates = [id_]
        try:
            candidates.append(int(id_))
        except (ValueError, TypeError):
            pass
        candidates.append(str(id_))

        emb = None
        for key in candidates:
            if key in id2emb:
                emb = id2emb[key]
                break
        if emb is None:
            raise KeyError(f"Embedding not found for graph id={id_!r} in {csv_path}")
        rows.append(emb)

    embs = torch.stack(rows, dim=0)
    if normalize:
        embs = F.normalize(embs, p=2, dim=-1)
    return embs


def get_functional_groups(graphs: Iterable) -> List[List[str]]:
    groups: List[List[str]] = []
    missing = 0
    for g in graphs:
        g_groups = getattr(g, "functional_groups", None)
        if g_groups is None:
            missing += 1
            g_groups = []
        groups.append(list(g_groups))
    if missing:
        print(f"[warn] {missing} graphs missing functional_groups; treating as empty.")
        print("[hint] Run add_functional_groups.py to annotate graph files.")
    return groups


def sample_subset(
    embs: np.ndarray, ids: Sequence, groups: Sequence[List[str]], max_points: int, seed: int
) -> tuple[np.ndarray, List, List[List[str]]]:
    total = embs.shape[0]
    if max_points <= 0 or total <= max_points:
        return embs, list(ids), list(groups)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=max_points, replace=False))
    embs = embs[idx]
    ids = [ids[i] for i in idx]
    groups = [groups[i] for i in idx]
    print(f"[info] sampled {len(ids)} points out of {total}")
    return embs, ids, groups


def project_embeddings(
    embs: np.ndarray,
    method: str,
    seed: int,
    perplexity: float,
    pca_pre: int,
) -> np.ndarray:
    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise SystemExit("PCA projection requires scikit-learn.") from exc

        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(embs)

    if method == "tsne":
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise SystemExit("t-SNE projection requires scikit-learn.") from exc

        n_samples, n_dim = embs.shape
        if n_samples < 5:
            raise SystemExit("t-SNE needs at least 5 points; try --projection pca.")
        if n_dim > pca_pre:
            pca = PCA(n_components=min(pca_pre, n_dim), random_state=seed)
            embs = pca.fit_transform(embs)

        max_perp = max(5.0, (n_samples - 1) / 3.0)
        if perplexity >= max_perp:
            print(f"[warn] perplexity {perplexity:.1f} too high; using {max_perp:.1f}.")
            perplexity = max_perp

        tsne = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        return tsne.fit_transform(embs)

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise SystemExit("UMAP requires `umap-learn`. Install it to use --projection umap.") from exc

        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(embs)

    raise ValueError(f"Unknown projection method: {method}")


def pick_group_labels(
    groups: Sequence[List[str]],
    strategy: str,
    freq: Counter,
) -> List[str]:
    labels: List[str] = []
    for g in groups:
        if not g:
            labels.append("none")
            continue
        if strategy == "first":
            labels.append(g[0])
        else:
            labels.append(max(g, key=lambda name: (freq[name], name)))
    return labels


def select_multilabel_groups(
    groups: Sequence[List[str]],
    requested: Sequence[str] | None,
    top_k: int,
) -> List[str]:
    freq = Counter()
    for g in groups:
        freq.update(g)

    if requested:
        missing = [g for g in requested if g not in freq]
        if missing:
            print(f"[warn] Requested groups not found: {', '.join(missing)}")
        return [g for g in requested if g in freq]

    return [name for name, _ in freq.most_common(top_k)]


def apply_basic_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_binary(
    ax,
    coords: np.ndarray,
    present: np.ndarray,
    group_name: str,
    point_size: float,
    alpha: float,
):
    neg = ~present
    ax.scatter(coords[neg, 0], coords[neg, 1], c="#C7C7C7", s=point_size, alpha=alpha, edgecolors="none", label="absent")
    ax.scatter(coords[present, 0], coords[present, 1], c="#E24A33", s=point_size, alpha=alpha, edgecolors="none", label="present")
    ax.set_title(f"{group_name} ({present.sum()} / {len(present)})")
    apply_basic_axes(ax)


def plot_multiclass(
    coords: np.ndarray,
    labels: Sequence[str],
    out_path: Path,
    point_size: float,
    alpha: float,
    legend_max: int,
    title: str | None,
):
    label_counts = Counter(labels)
    ordered = [name for name, _ in label_counts.most_common()]
    cmap = plt.get_cmap("tab20") if len(ordered) <= 20 else plt.get_cmap("hsv")
    colors = {name: cmap(i / max(1, len(ordered) - 1)) for i, name in enumerate(ordered)}

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    labels_arr = np.asarray(labels, dtype=object)
    for name in ordered:
        mask = labels_arr == name
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            edgecolors="none",
            c=[colors[name]],
            label=f"{name} ({label_counts[name]})",
        )

    apply_basic_axes(ax)
    if title:
        ax.set_title(title)
    if len(ordered) <= legend_max:
        ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    else:
        print(f"[info] Skipping legend ({len(ordered)} labels > {legend_max}).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multilabel_grid(
    coords: np.ndarray,
    groups: Sequence[List[str]],
    names: Sequence[str],
    out_path: Path,
    point_size: float,
    alpha: float,
    subplot_size: float,
    title: str | None,
):
    if not names:
        raise SystemExit("No functional groups selected for multilabel plotting.")

    ncols = min(4, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_size * ncols, subplot_size * nrows), squeeze=False)

    groups_set = [set(g) for g in groups]
    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        present = np.array([name in g for g in groups_set])
        plot_binary(ax, coords, present, name, point_size, alpha)

    for idx in range(len(names), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_projection_csv(path: Path, ids: Sequence, coords: np.ndarray, groups: Sequence[List[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "x", "y", "functional_groups"])
        for id_, xy, g in zip(ids, coords, groups):
            writer.writerow([id_, f"{xy[0]:.6f}", f"{xy[1]:.6f}", ";".join(g)])
    print(f"[info] wrote projection CSV -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Project GINE embeddings and color by functional groups.")
    parser.add_argument("--graph-path", default="data/train_graphs_func_groups.pkl")
    parser.add_argument("--ckpt-path", default="ckpt_gine_contrastive/best.pt")
    parser.add_argument("--embeddings-csv", default=None, help="Optional precomputed embeddings CSV (ID, embedding).")
    parser.add_argument("--projection", choices=["tsne", "pca", "umap"], default="tsne")
    parser.add_argument("--mode", choices=["binary", "multiclass", "multilabel"], default="multiclass")
    parser.add_argument("--group", default=None, help="Functional group name for binary mode.")
    parser.add_argument("--groups", nargs="*", default=None, help="Explicit group names for multilabel mode.")
    parser.add_argument("--top-k", type=int, default=12, help="Top-K groups for multilabel mode.")
    parser.add_argument("--multiclass-strategy", choices=["first", "most_common"], default="most_common")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--pca-pre", type=int, default=50, help="PCA pre-reduction dim for t-SNE.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=0, help="Randomly subsample points for plotting.")
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--legend-max", type=int, default=20)
    parser.add_argument("--subplot-size", type=float, default=3.6)
    parser.add_argument("--out", default="plots/gine_projection.png")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-proj", action="store_true", help="Disable contrastive projection head if present.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization.")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    device = resolve_device(args.device)
    seed_all(args.seed)
    use_amp = device == "cuda"

    dataset = GraphOnlyDataset(str(graph_path))
    graphs = dataset.base.graphs
    ids = dataset.ids
    groups = get_functional_groups(graphs)
    freq = Counter()
    for g in groups:
        freq.update(g)

    if args.embeddings_csv:
        embs = load_embeddings_from_csv(ids, args.embeddings_csv, normalize=not args.no_normalize)
    else:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"GINE checkpoint not found: {ckpt_path}")
        encoder, proj = load_encoder_and_proj(ckpt_path, device)
        embs = encode_embeddings(
            dataset,
            encoder,
            proj,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
            use_proj=not args.no_proj,
            normalize=not args.no_normalize,
        )

    embs_np = embs.numpy()
    embs_np, ids, groups = sample_subset(embs_np, ids, groups, args.max_points, args.seed)

    coords = project_embeddings(
        embs_np,
        method=args.projection,
        seed=args.seed,
        perplexity=args.perplexity,
        pca_pre=args.pca_pre,
    )

    out_path = Path(args.out)
    title = args.title
    if title is None:
        title = f"{args.projection.upper()} | {args.mode}"

    if args.mode == "binary":
        if not freq:
            raise SystemExit("No functional groups found; run add_functional_groups.py first.")
        group_name = args.group or freq.most_common(1)[0][0]
        if args.group is None:
            print(f"[info] Using most common group for binary mode: {group_name}")
        present = np.array([group_name in g for g in groups])

        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        plot_binary(ax, coords, present, group_name, args.point_size, args.alpha)
        if args.title:
            fig.suptitle(args.title)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    elif args.mode == "multiclass":
        if not title:
            title = f"{args.projection.upper()} | {args.mode}"
        labels = pick_group_labels(groups, args.multiclass_strategy, freq)
        plot_multiclass(
            coords,
            labels,
            out_path,
            point_size=args.point_size,
            alpha=args.alpha,
            legend_max=args.legend_max,
            title=title,
        )
    else:
        if not title:
            title = f"{args.projection.upper()} | {args.mode}"
        names = select_multilabel_groups(groups, args.groups, args.top_k)
        plot_multilabel_grid(
            coords,
            groups,
            names,
            out_path,
            point_size=args.point_size,
            alpha=args.alpha,
            subplot_size=args.subplot_size,
            title=title,
        )

    print(f"[info] wrote plot -> {out_path}")
    if args.out_csv:
        write_projection_csv(Path(args.out_csv), ids, coords, groups)


if __name__ == "__main__":
    main()
