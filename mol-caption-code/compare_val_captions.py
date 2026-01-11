#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare generated captions vs ground truth for validation samples.
"""
import os
os.environ["NPY_DISABLE_ARRAY_API"] = "1"

import pickle
import torch
from torch_geometric.data import Batch

from config import get_config
from model_wrapper import create_model
from utils import load_checkpoint, graph_to_smiles


def main():
    # Configuration
    checkpoint_path = "/home/axel/Altegrad_kaggle/hf_checkpoints/stage2_full_best_ep2.pt"
    val_graphs_path = "/home/axel/Altegrad_kaggle/data/validation_graphs.pkl"
    num_samples = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load validation graphs
    print(f"\nLoading validation graphs from: {val_graphs_path}")
    with open(val_graphs_path, "rb") as f:
        val_graphs = pickle.load(f)
    print(f"Loaded {len(val_graphs)} validation graphs")

    # Take first N samples
    samples = val_graphs[:num_samples]

    # Create model
    print("\nCreating model...")
    config = get_config(mode="full")
    model = create_model(config, device=device)

    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    # Generate captions
    model.eval()
    print("\nGenerating captions...")

    # Prepare batch
    batched_graphs = Batch.from_data_list(samples)
    smiles_list = [graph_to_smiles(g) for g in samples]

    # Generate
    with torch.no_grad():
        generated_captions = model.generate(
            graphs=batched_graphs,
            smiles_list=smiles_list,
            max_new_tokens=128,
            num_beams=1,
            temperature=0.7,
            do_sample=True,
        )

    # Display comparisons
    print("\n" + "=" * 80)
    print("COMPARISON: Generated vs Ground Truth")
    print("=" * 80)

    for i, (graph, generated) in enumerate(zip(samples, generated_captions)):
        graph_id = graph.id
        ground_truth = graph.description
        smiles = smiles_list[i]

        # Clean up generated caption
        generated = generated.strip()
        generated = generated.replace("<|endoftext|>", "").strip()
        generated = generated.replace("\n", " ").strip()

        print(f"\n{'='*80}")
        print(f"Sample {i+1} (ID: {graph_id})")
        print(f"{'='*80}")
        print(f"\nSMILES: {smiles}")
        print(f"\n--- Ground Truth ---")
        print(ground_truth)
        print(f"\n--- Generated ---")
        print(generated)
        print()


if __name__ == "__main__":
    main()
