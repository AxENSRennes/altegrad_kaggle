#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference: Generate captions for test set and create submission.

Loads the best Stage 2 checkpoint and generates captions for all test molecules.
Outputs a CSV file in the format: ID,description
"""

import pickle
import csv
from typing import List, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from config import Config, get_config
from model_wrapper import MolCaptionModel, create_model
from utils import load_checkpoint, graph_to_smiles, ensure_dir


class TestDataset:
    """Simple dataset for test graphs."""

    def __init__(self, graph_path: str):
        print(f"Loading test graphs from: {graph_path}")
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        print(f"Loaded {len(self.graphs)} test graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph, graph.id


def collate_test_batch(batch):
    """Collate function for test data."""
    graphs, ids = zip(*batch)
    batched_graphs = Batch.from_data_list(list(graphs))
    smiles = [graph_to_smiles(g) for g in graphs]
    return batched_graphs, list(ids), smiles


@torch.no_grad()
def generate_submissions(
    model: MolCaptionModel,
    config: Config,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    temperature: float = 0.7,
    do_sample: bool = True,
    limit: Optional[int] = None,
) -> List[tuple]:
    """
    Generate captions for all test molecules.

    Args:
        model: Trained MolCaptionModel
        config: Configuration object
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate
        num_beams: Beam search width
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        limit: Optional limit on number of molecules to process

    Returns:
        List of (mol_id, caption) tuples
    """
    model.eval()
    device = model.device

    # Load test data
    test_dataset = TestDataset(config.test_graphs_path)
    if limit is not None:
        test_dataset.graphs = test_dataset.graphs[:limit]
        print(f"Limited to first {limit} molecules")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_test_batch,
        num_workers=0,  # Avoid multiprocessing issues
    )

    results = []

    print(f"\nGenerating captions for {len(test_dataset)} test molecules...")

    for graphs, mol_ids, smiles in tqdm(test_loader, desc="Generating"):
        try:
            # Generate captions
            captions = model.generate(
                graphs=graphs,
                smiles_list=smiles,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
            )

            # Clean up captions
            for mol_id, caption in zip(mol_ids, captions):
                # Remove any special tokens and clean whitespace
                caption = caption.strip()
                caption = caption.replace("<|endoftext|>", "").strip()
                caption = caption.replace("\n", " ").strip()

                # Ensure minimum length
                if len(caption) < 10:
                    caption = "This is a molecule with chemical properties."

                results.append((mol_id, caption))

        except Exception as e:
            print(f"Error generating batch: {e}")
            # Fallback for failed generations
            for mol_id in mol_ids:
                results.append((mol_id, "This molecule has various chemical properties."))

    return results


def save_submission(
    results: List[tuple],
    output_path: str,
):
    """
    Save results to submission CSV file.

    Args:
        results: List of (mol_id, caption) tuples
        output_path: Path to output CSV
    """
    ensure_dir(output_path)

    # Sort by ID if numeric
    try:
        results = sorted(results, key=lambda x: int(x[0].replace("test_", "")))
    except ValueError:
        results = sorted(results, key=lambda x: x[0])

    print(f"\nSaving {len(results)} captions to {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "description"])
        for mol_id, caption in results:
            writer.writerow([mol_id, caption])

    print(f"Submission saved!")


def run_inference(
    config: Optional[Config] = None,
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """
    Run full inference pipeline.

    Args:
        config: Optional config object (uses default if None)
        checkpoint_path: Optional checkpoint path (uses config default if None)
        output_path: Optional output path (uses config default if None)
        limit: Optional limit on number of molecules to process
        batch_size: Optional batch size for generation
    """
    # Setup
    if config is None:
        config = get_config(mode="full")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = create_model(config, device=device)

    # Load checkpoint
    ckpt_path = checkpoint_path or config.stage2_checkpoint_path
    print(f"\nLoading checkpoint from {ckpt_path}")

    try:
        checkpoint = load_checkpoint(ckpt_path, model, device=device)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if "metrics" in checkpoint:
            print(f"Checkpoint metrics: {checkpoint['metrics']}")
    except FileNotFoundError:
        print(f"WARNING: Checkpoint not found at {ckpt_path}")
        print("Running inference with untrained model...")

    # Generate
    results = generate_submissions(
        model,
        config,
        batch_size=batch_size or 8,
        max_new_tokens=128,
        num_beams=1,
        temperature=0.7,
        do_sample=True,
        limit=limit,
    )

    # Save
    out_path = output_path or config.submission_path
    save_submission(results, out_path)

    # Print sample outputs
    print("\nSample outputs:")
    print("-" * 60)
    for mol_id, caption in results[:5]:
        print(f"{mol_id}: {caption[:100]}...")
    print("-" * 60)

    return results


def main():
    """Main entry point for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate captions for test set")
    parser.add_argument("--mode", type=str, default="full", help="Experiment mode")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, help="Path to output CSV")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--limit", type=int, help="Limit number of molecules")
    args = parser.parse_args()

    config = get_config(mode=args.mode)
    run_inference(config, args.checkpoint, args.output, limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
