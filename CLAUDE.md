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
python retrieval_answer_test_v5.py  # Pure retrieval (nearest neighbor)
python retrieval_answer_test_v6.py  # RAG with Flan-T5 (retrieves diverse shots, generates paraphrased caption)
```

### Generative Pipeline

```bash
cd mol-caption-code

# Training modes
python run.py --mode quick     # 500 samples, 1 epoch each stage (testing)
python run.py --mode medium    # 5000 samples, 2 epochs each stage
python run.py --mode full      # All data, 3+5 epochs

# Hardware options
python run.py --mode quick --hardware cpu   # Local testing without GPU
python run.py --mode quick --hardware gpu   # CUDA with 4-bit quantization
accelerate launch --config_file accelerate_config_tpu.yaml run.py --mode quick --hardware tpu

# Resume training (skip Stage 1 if checkpoint exists)
python run.py --mode full --skip-stage1

# Inference only
python run.py --inference --checkpoint outputs/stage2_full_best.pt
python run.py --inference --checkpoint outputs/stage2_full_best.pt --limit 100  # Test on first 100

# With W&B logging
python run.py --mode full --wandb
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
- **Logs**: `losses/log_v5.csv`

Key hyperparameters:
- HIDDEN=512, LAYERS=5, BATCH_SIZE=256
- LR=2e-4, EPOCHS=80, PATIENCE=6
- QUEUE_SIZE=65536

### Generative Model (mol-caption-code/)

Two-stage training:
1. **Stage 1 (Alignment)**: Train projector only (MSE loss), GNN+LLM frozen. LR=1e-3.
2. **Stage 2 (SFT)**: Train projector (LR=1e-4) + LoRA (LR=1e-5) jointly (Cross-Entropy).

The model injects graph embeddings as a soft token at `<|graph|>` position using `model_wrapper.inject_graph_tokens()`.

Key files:
- `model_gnn.py`: MolGNN encoder (matches train_gcn architecture)
- `model_projector.py`: 3-layer MLP (768→1024→1024) "SolidBridgeProjector"
- `model_wrapper.py`: MolCaptionModel combining GNN + Projector + Qwen3-0.6B
- `config.py`: Experiment modes and all hyperparameters
- `train_stage1.py` / `train_stage2.py`: Stage-specific training loops
- `dataset_caption.py`: Graph+text dataset with SMILES and chat template formatting

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

## Checkpoints

Local checkpoints are stored in `checkpoints/` and `outputs/`. Shared/versioned checkpoints (for Kaggle or cross-machine use) are in `hf_checkpoints/`. The config automatically falls back to `hf_checkpoints/` if local paths don't exist.

Checkpoint naming:
- `gnn_v{N}.pt` / `gnn_v{N}_best.pt`: Retrieval GNN (last / best by val MRR)
- `stage1_{mode}_best.pt`: Stage 1 alignment checkpoint
- `stage2_{mode}_best.pt`: Stage 2 SFT checkpoint

## Notebooks

- `gen_caption_notebook.ipynb`: Generative captioning experiments (for Kaggle)
- `notebook.ipynb`: General experimentation

## Historical Notes

### Retrieval Pipeline Evolution

The retrieval pipeline evolved through several versions:
- **v4**: Simple retrieval with light reranking - good balance of performance vs complexity
- **v5**: More sophisticated neural reranking - best score but marginal gain for significantly higher complexity

Both v4 and v5 use `version_embed="v1"` (BERT embeddings) and `version_gnn="v4"` GNN checkpoint.

### Submission Notes (from README_gabriel.md)

**Submission 4** (`retrieval_answer_v4.py`): Simple retrieval with light reranking. Good tradeoff between performance and complexity.

**Submission 5** (`retrieval_answer_v5.py`): Complex neural retrieval pipeline. Best score but only marginal improvement over v4.

Future improvement paths considered:
- Systematic hyperparameter optimization (grid/random/Bayesian search)
- More faithful implementation of challenge spec recommendations (would require major pipeline refactoring)

### Generative Pipeline Design

The generative pipeline (`mol-caption-code/`) was designed based on a two-stage approach:
1. **Stage 1 (Alignment)**: Train MLP projector to align GNN graph embeddings with LLM hidden space
2. **Stage 2 (SFT)**: Fine-tune projector + LoRA adapters for caption generation

Architecture: MolGNN encoder → "Solid Bridge" 3-layer MLP projector → Qwen3-0.6B (4-bit LoRA)

The original design document specified memory estimates (~4-5GB on T4), training time (~9h for full), and target metrics (BLEU-4 > 30, BERTScore > 0.8).
