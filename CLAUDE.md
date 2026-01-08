# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Molecule-Text Retrieval system for the ALTEGRAD Kaggle Data Challenge on Molecular Graph Captioning. Uses a Graph Convolutional Network (GCN) to encode molecular graphs and match them with text descriptions via embedding-based retrieval.

**Reference Document**: Always read `ALTEGRAD_Data_Challenge__Molecular_Graph_Captioning-3.pdf` for challenge context before making significant changes.

## Environment

Use the WSL virtual environment for all Python commands:
```bash
source /home/axel/wsl_venv/bin/activate
```

## Commands

```bash
# Activate venv first
source /home/axel/wsl_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Full pipeline (run in order)
python inspect_graph_data.py           # Validate graph structure
python generate_description_embeddings.py  # Generate BERT embeddings
python train_gcn.py                    # Train GCN model
python retrieval_answer.py             # Generate submission CSV

# Individual commands
python train_gcn.py          # Train with existing embeddings (creates model_checkpoint.pt)
python retrieval_answer.py   # Generate test_retrieved_descriptions.csv
```

## Architecture

### Data Flow
1. **Graphs** (`data/*.pkl`): PyTorch Geometric Data objects with molecular topology and text descriptions
2. **Embeddings** (`data/*_embeddings.csv`): BERT embeddings (768-dim) for text descriptions
3. **Output**: CSV with ID and retrieved description

### Key Components

| File | Purpose |
|------|---------|
| `data_utils.py` | Data loading utilities: `PreprocessedGraphDataset`, `collate_fn`, feature maps (x_map, e_map) |
| `train_gcn.py` | GCN model (`MolGNN`) and training loop |
| `generate_description_embeddings.py` | BERT embedding generation using `bert-base-uncased` |
| `retrieval_answer.py` | Inference and retrieval for submission |

### Model Architecture (MolGNN)
- 3 GCNConv layers (hidden_dim=128)
- Global add pooling → Linear projection to 768-dim
- L2 normalized embeddings
- MSE loss between molecule and text embeddings

### Graph Features
- **Node**: atomic_num, chirality, degree, formal_charge, num_hs, hybridization, is_aromatic, is_in_ring
- **Edge**: bond_type, stereo, is_conjugated

## Key Hyperparameters (train_gcn.py)

- BATCH_SIZE: 32
- EPOCHS: 5
- LEARNING_RATE: 1e-3
- EMBEDDING_DIM: 768
- GNN_HIDDEN: 128
- GNN_LAYERS: 3
