import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from data_utils import load_id2emb, load_descriptions_from_graphs

def compute_bleu(ref, cand):
    # Simple BLEU-4
    try:
        ref_tokens = nltk.word_tokenize(ref.lower())
        cand_tokens = nltk.word_tokenize(cand.lower())
        cc = SmoothingFunction()
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=cc.method1)
    except Exception:
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_graphs", default="data/train_graphs.pkl")
    parser.add_argument("--val_graphs", default="data/validation_graphs.pkl")
    parser.add_argument("--train_emb", default="data/train_embeddings.csv")
    parser.add_argument("--val_emb", default="data/validation_embeddings.csv")
    parser.add_argument("--top_k_candidates", type=int, default=100, help="Number of candidates to re-rank with actual BLEU")
    args = parser.parse_args()

    if not os.path.exists(args.train_emb) or not os.path.exists(args.val_emb):
        print("Embeddings not found. Please run generate_description_embeddings.py first.")
        return

    # Load descriptions
    print("Loading descriptions (this might take a moment)...")
    train_id2desc = load_descriptions_from_graphs(args.train_graphs)
    val_id2desc = load_descriptions_from_graphs(args.val_graphs)
    
    # Load embeddings for candidate filtering
    print("Loading embeddings...")
    train_id2emb = load_id2emb(args.train_emb)
    val_id2emb = load_id2emb(args.val_emb)
    
    # Align IDs and create Tensors
    # Ensure we only use IDs that exist in both map and descriptions
    train_ids = [mid for mid in train_id2emb.keys() if mid in train_id2desc]
    val_ids = [mid for mid in val_id2emb.keys() if mid in val_id2desc]
    
    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples: {len(val_ids)}")

    train_embs = torch.stack([train_id2emb[mid] for mid in train_ids])
    val_embs = torch.stack([val_id2emb[mid] for mid in val_ids])
    
    if torch.cuda.is_available():
        train_embs = train_embs.cuda()
        val_embs = val_embs.cuda()

    train_embs = F.normalize(train_embs, dim=-1)
    val_embs = F.normalize(val_embs, dim=-1)
    
    # Compute similarity for top-k
    print("Computing embedding similarities to find candidates...")
    # Chunking to avoid OOM if needed, but 30k x 1k is small enough
    sims = val_embs @ train_embs.t()
    
    # Strategy: For each val molecule, we look at the top-K neighbors in embedding space
    # We assume 'Oracle' is likely within these semantically similar descriptions.
    # Searching ALL 30k for every val is slow (30k * 1k = 30M comparisons), 
    # but top-100 is fast.
    
    print(f"Computing Oracle BLEU scores on validation set with Top-{args.top_k_candidates} candidates...")
    
    _, top_indices = sims.topk(args.top_k_candidates, dim=1)
    top_indices = top_indices.cpu()
    
    scores_oracle = []
    scores_baseline = []
    
    for i in tqdm(range(len(val_ids))):
        val_id = val_ids[i]
        val_desc = val_id2desc[val_id]
        
        cands_idx = top_indices[i].tolist()
        
        # Candidates (Text)
        candidates_text = [train_id2desc[train_ids[idx]] for idx in cands_idx]
        
        # Compute BLEU for each candidate
        bleus = [compute_bleu(val_desc, c_text) for c_text in candidates_text]
        
        # The baseline is what our model would return (the Top-1 embedding match)
        baseline_bleu = bleus[0]
        
        # The Oracle is the best possible match among these candidates
        # (Assuming the best match is within the top-100 semantic matches)
        oracle_bleu = max(bleus)
        
        scores_baseline.append(baseline_bleu)
        scores_oracle.append(oracle_bleu)

    mean_baseline = np.mean(scores_baseline)
    mean_oracle = np.mean(scores_oracle)

    print("\n" + "="*60)
    print("ORACLE ANALYSIS RESULTS")
    print("="*60)
    print(f"Baseline Retrieval (Top-1 Embedding):  BLEU-4 = {mean_baseline:.4f}")
    print(f"Oracle Retrieval (Best of Top-{args.top_k_candidates}):     BLEU-4 = {mean_oracle:.4f}")
    print("-" * 60)
    print(f"Gap (Potential Improvement):           {mean_oracle - mean_baseline:.4f}")
    print("="*60)
    print("Note: 'Baseline' assumes we use the generic BERT embeddings for retrieval.")
    print("If your trained GCN is good, it should approach the 'Baseline' score.")
    print("The 'Oracle' is the theoretical limit if your retrieval was perfect among the top candidates.")

if __name__ == "__main__":
    main()
