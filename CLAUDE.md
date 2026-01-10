# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Molecule-Text system for the ALTEGRAD Kaggle Data Challenge on Molecular Graph Captioning.

**Kaggle Competition**: `molecular-graph-captioning`

Two approaches:
1. **Retrieval-based**: GNN encodes molecular graphs, matches with text descriptions via contrastive embedding
2. **Generative**: GNN + Qwen3-0.6B LLM with LoRA for caption generation

**Reference Document**: Read `ALTEGRAD_Data_Challenge__Molecular_Graph_Captioning-3.pdf` for challenge context.

## Environment

```bash
source /home/axel/wsl_venv/bin/activate
```

## Commands

### Retrieval Pipeline (current best)

```bash
# Generate BERT embeddings for text descriptions
python generate_description_embeddings_v1.py

# Train contrastive GNN (CLIP-style with memory queue)
python train_gcn_v5.py

# Generate submission via retrieval or RAG
python retrieval_answer_test_v5.py  # Pure retrieval
python retrieval_answer_test_v6.py  # RAG with Flan-T5
```

### Generative Pipeline (experimental)

```bash
cd mol-caption-code

# Quick test (~5 min) / Medium (~1 hour) / Full (~9 hours)
python run.py --mode quick
python run.py --mode medium
python run.py --mode full

# Inference only
python run.py --inference --checkpoint outputs/stage2_best.pt
```

### Utilities

```bash
python inspect_graph_data.py  # Validate graph structure
python plot_logs.py           # Visualize training logs
```

## Architecture

### Data Flow

1. **Graphs** (`data/*.pkl`): PyTorch Geometric Data objects with molecular topology
2. **Embeddings** (`data/*_embeddings_v1.csv`): BERT embeddings (768-dim) for descriptions
3. **Checkpoints** (`checkpoints/gnn_v*.pt`): Trained GNN models
4. **Outputs** (`outputs/*.csv`): Submission files

### Retrieval Model (train_gcn_v5.py)

- **Architecture**: GINEConv layers with residual connections + Attentional Aggregation
- **Loss**: CLIP-style contrastive with 64K memory queue
- **Training**: Cosine LR scheduler, early stopping on validation MRR
- **Saves**: `checkpoints/gnn_v5.pt` (last), `checkpoints/gnn_v5_best.pt` (best)

Key hyperparameters:
- HIDDEN=512, LAYERS=5, BATCH_SIZE=256
- LR=2e-4, EPOCHS=80, PATIENCE=6
- QUEUE_SIZE=65536

### Generative Model (mol-caption-code/)

Two-stage training:
1. **Stage 1 (Alignment)**: Train projector to map GNN→LLM space (MSE loss), freeze GNN+LLM
2. **Stage 2 (SFT)**: Train projector + LoRA jointly (Cross-Entropy), generate captions

Components:
- `model_gnn.py`: MolGNN encoder (matches train_gcn architecture)
- `model_projector.py`: 3-layer MLP (768→1024→1024) "Solid Bridge"
- `model_wrapper.py`: MolCaptionModel combining GNN + Projector + Qwen3-0.6B
- `config.py`: Experiment modes (quick/medium/full) and hyperparameters

### Data Utilities (data_utils.py)

- `PreprocessedGraphDataset`: Loads graphs + optional text embeddings
- `collate_fn`: Batches graphs with PyG's `Batch.from_data_list`
- `load_id2emb`: Loads CSV embeddings to dict
- `x_map`, `e_map`: Node/edge feature vocabularies

### Graph Features

- **Node (9)**: atomic_num, chirality, degree, formal_charge, num_hs, num_radical_electrons, hybridization, is_aromatic, is_in_ring
- **Edge (3)**: bond_type, stereo, is_conjugated

## Versioning Convention

Files use version suffixes (e.g., `train_gcn_v5.py`, `retrieval_answer_test_v6.py`). Use the highest version number for the latest approach.

## Notebooks

- `gen_caption_notebook.ipynb`: Generative captioning experiments (for Kaggle)
- `notebook.ipynb`: General experimentation
