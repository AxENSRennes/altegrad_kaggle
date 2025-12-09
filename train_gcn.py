import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_add_pool

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)

# =========================================================
# CONFIG
# =========================================================
# Data paths
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

TRAIN_EMB_CSV = "data/train_embeddings.csv"
VAL_EMB_CSV   = "data/validation_embeddings.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# MODEL: GNN
# =========================================================
class MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()

        # Use a single learnable embedding for all nodes (no node features)
        self.node_init = nn.Parameter(torch.randn(hidden))

        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GCNConv(hidden, hidden))

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch: Batch):
        num_nodes = batch.x.size(0)
        h = self.node_init.unsqueeze(0).expand(num_nodes, -1)
        
        for conv in self.convs:
            h = conv(h, batch.edge_index)
            h = F.relu(h)
        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g


# =========================================================
# LOSS FUNCTIONS
# =========================================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, mol_emb, text_emb):
        """
        mol_emb: [batch_size, dim]
        text_emb: [batch_size, dim] - corresponding correct descriptions
        
        We use in-batch negatives.
        For each molecule i:
          - Anchor: mol_emb[i]
          - Positive: text_emb[i]
          - Negatives: any text_emb[j] where j != i
        """
        scores = mol_emb @ text_emb.t()  # [B, B] cosine similarity (already normalized)
        diag = scores.diag()             # [B] - positive scores (Sim(A, P))

        # For each row i, we want Sim(A, P) > Sim(A, N) + margin
        # => Sim(A, N) - Sim(A, P) + margin < 0
        # We compute Loss = max(0, Sim(A, N) - Sim(A, P) + margin)
        
        # We can implement "hardest negative" or "mean negative". 
        # Standard approach: Sum over all negatives in the batch
        
        # Mask out the diagonal (positives)
        mask = torch.eye(scores.size(0), device=scores.device).bool()
        
        # scores without diagonal are tricky to reshape, so we can do:
        # Loss matrix: (Sim(A, N) - Sim(A, P) + margin)
        # We subtract col vector diag from matrix scores
        cost = scores - diag.view(-1, 1) + self.margin
        
        # Set diagonal to 0 (we don't want to penalize positive pair)
        cost = cost.masked_fill(mask, 0)
        
        # Keep only positive costs (ReLU)
        cost = F.relu(cost)
        
        # Mean over valid negatives
        # number of negatives per row is B-1
        return cost.sum() / (scores.size(0) * (scores.size(0) - 1))


# =========================================================
# Training and Evaluation (Modified)
# =========================================================
def train_epoch(mol_enc, loader, optimizer, device, loss_fn, loss_type='mse'):
    mol_enc.train()
    total_loss, total = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        if loss_type == 'mse':
            loss = loss_fn(mol_vec, txt_vec)
        else:
            loss = loss_fn(mol_vec, txt_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}
    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"Hit@{k}"] = hitk

    return results


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Train Graph Captioning Model")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "triplet"], help="Loss function to use")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for Triplet Loss")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Config: Loss={args.loss}, Margin={args.margin}, Epochs={args.epochs}, LR={args.lr}, BS={args.batch_size}")

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None
    
    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: {TRAIN_GRAPHS} not found")
        return
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    mol_enc = MolGNN(out_dim=emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(mol_enc.parameters(), lr=args.lr)

    # Select Loss
    if args.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss == 'triplet':
        loss_fn = TripletLoss(margin=args.margin)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # Training Loop
    for ep in range(args.epochs):
        train_loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE, loss_fn, args.loss)
        if val_emb is not None and os.path.exists(VAL_GRAPHS):
            val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
        else:
            val_scores = {}
        
        # Format scores for printing
        val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_scores.items()])
        print(f"Epoch {ep+1}/{args.epochs} | Loss: {train_loss:.5f} | Val: {val_str}")
    
    # Save Model
    if args.loss == 'triplet':
        model_name = f"model_triplet_m{args.margin}_lr{args.lr}_ep{args.epochs}.pt"
    else:
        model_name = f"model_mse_lr{args.lr}_ep{args.epochs}.pt"
        
    torch.save(mol_enc.state_dict(), model_name)
    print(f"\nModel saved to {model_name}")


if __name__ == "__main__":
    main()
