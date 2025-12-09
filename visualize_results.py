
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)
from train_gcn import (
    MolGNN, DEVICE, VAL_GRAPHS, VAL_EMB_CSV
)

def evaluate_and_plot(model, val_loader, device, output_dir="figures", output_prefix="evaluation"):
    """
    Computes embeddings, calculates retrieval metrics, and generates plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    model.eval()
    
    print("Computing embeddings for validation set...")
    all_mol_embs = []
    all_text_embs = []
    
    with torch.no_grad():
        for graphs, text_emb in tqdm(val_loader):
            graphs = graphs.to(device)
            text_emb = text_emb.to(device)
            
            mol_vec = model(graphs)
            txt_vec = F.normalize(text_emb, dim=-1)
            
            all_mol_embs.append(mol_vec.cpu())
            all_text_embs.append(txt_vec.cpu())
            
    all_mol_embs = torch.cat(all_mol_embs, dim=0)
    all_text_embs = torch.cat(all_text_embs, dim=0)
    
    print(f"Computed embeddings: Mol {all_mol_embs.shape}, Text {all_text_embs.shape}")

    # Similarity Matrix (Text x Mol)
    sims = all_text_embs @ all_mol_embs.t()
    
    # Calculate Ranks
    n_samples = sims.shape[0]
    correct_indices = torch.arange(n_samples)
    sorted_indices = sims.argsort(dim=1, descending=True)
    
    ranks = []
    for i in range(n_samples):
        rank = (sorted_indices[i] == correct_indices[i]).nonzero().item()
        ranks.append(rank + 1)
    
    ranks = np.array(ranks)
    
    # ==========================================
    # 1. Rank Distribution Histogram
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.hist(ranks, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of True Match Ranks (Lower is Better)")
    plt.xlabel("Rank of Correct Molecule")
    plt.ylabel("Frequency")
    plt.axvline(x=np.median(ranks), color='r', linestyle='--', label=f'Median Rank: {int(np.median(ranks))}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(output_dir, f"{output_prefix}_rank_histogram.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

    # ==========================================
    # 2. Recall@K Curve
    # ==========================================
    ks = range(1, 51)
    recalls = []
    for k in ks:
        recalls.append(np.mean(ranks <= k))
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, recalls, marker='o', markersize=4)
    plt.title("Recall@K (Hit Rate) Curve")
    plt.xlabel("K (Top-K Proposals)")
    plt.ylabel("Recall (Proportion of Correct Matches)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    save_path = os.path.join(output_dir, f"{output_prefix}_recall_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

    # ==========================================
    # 3. Similarity Distribution (Pos vs Neg)
    # ==========================================
    pos_sims = torch.diag(sims).numpy()
    mask = ~torch.eye(n_samples, dtype=torch.bool)
    neg_sims = sims[mask].flatten()
    if len(neg_sims) > 10000:
        neg_sims = np.random.choice(neg_sims, 10000, replace=False)
    
    plt.figure(figsize=(10, 6))
    plt.hist(neg_sims, bins=50, alpha=0.5, label='Negative Pairs (Mismatch)', density=True, color='gray')
    plt.hist(pos_sims, bins=50, alpha=0.5, label='Positive Pairs (Match)', density=True, color='green')
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(output_dir, f"{output_prefix}_similarity_dist.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")
    
    print("\nSummary Metrics:")
    print(f"Median Rank: {np.median(ranks)}")
    print(f"Mean Rank:   {np.mean(ranks):.1f}")
    print(f"R@1:  {np.mean(ranks <= 1):.4f}")
    print(f"R@5:  {np.mean(ranks <= 5):.4f}")
    print(f"R@10: {np.mean(ranks <= 10):.4f}")


def main():
    if not os.path.exists("model_checkpoint.pt"):
        print("Model checkpoint not found. Please train first.")
        return

    print(f"Device: {DEVICE}")
    
    # Load Embeddings
    print(f"Loading embeddings from {VAL_EMB_CSV}...")
    val_emb = load_id2emb(VAL_EMB_CSV)
    emb_dim = len(next(iter(val_emb.values())))
    
    # Load Model
    print("Loading model...")
    model = MolGNN(out_dim=emb_dim).to(DEVICE)
    model.load_state_dict(torch.load("model_checkpoint.pt", map_location=DEVICE))
    
    # Load Data
    print(f"Loading graphs from {VAL_GRAPHS}...")
    val_ds = PreprocessedGraphDataset(VAL_GRAPHS, val_emb)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Run Eval
    evaluate_and_plot(model, val_dl, DEVICE, output_dir="figures")

if __name__ == "__main__":
    main()
