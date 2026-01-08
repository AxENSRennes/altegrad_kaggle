# Implementation Plan: Generative Molecular Captioning with Qwen3-0.6B

## Overview
Two-stage generative captioning system using MolGNN encoder + "Solid Bridge" projector + Qwen3-0.6B (4-bit LoRA).

## File Structure (Upload as Kaggle Dataset `mol-caption-code`)

```
mol-caption-code/
├── config.py              # Hyperparameters, paths, experiment modes
├── model_gnn.py           # MolGNN encoder (from train_gcn_v5.py)
├── model_projector.py     # 3-layer MLP "Solid Bridge"
├── model_wrapper.py       # MolCaptionModel (GNN + Projector + LLM)
├── dataset_caption.py     # Dataset + collate function
├── train_stage1.py        # Stage 1: Alignment training (with tqdm)
├── train_stage2.py        # Stage 2: SFT training (with tqdm)
├── inference.py           # Generation for submission
├── metrics.py             # BLEU-4, METEOR, BERTScore computation
├── report.py              # Pretty training reports (rich library)
└── utils.py               # W&B helpers, SMILES reconstruction, checkpointing
```

## Files to Create

### 1. `config.py`
Includes **experiment modes** for quick testing:
```python
@dataclass
class Config:
    # Experiment mode: "quick" (5min test), "medium" (1h), "full" (9h)
    experiment_mode: str = "quick"

    # Auto-set based on mode:
    # quick:  stage1_epochs=1, stage2_epochs=1, train_subset=500
    # medium: stage1_epochs=2, stage2_epochs=2, train_subset=5000
    # full:   stage1_epochs=3, stage2_epochs=5, train_subset=None (all)

    train_subset: int = None  # Limit training samples for quick experiments
    eval_every_n_steps: int = 100  # Evaluate metrics periodically

    # ... other params
```
- Dataclass with all hyperparameters
- Kaggle paths: `/kaggle/input/altegrad-2024/`, `/kaggle/input/mol-caption-code/`
- Model config: `Qwen/Qwen3-0.6B`, LoRA r=64, alpha=128
- Stage 1: batch=32, grad_accum=2, lr=1e-3, epochs=3 (full mode)
- Stage 2: batch=8, grad_accum=8, lr_proj=5e-4, lr_lora=2e-4, epochs=5 (full mode)

### 2. `model_gnn.py`
- Copy MolGNN, AtomEncoder, EdgeEncoder from `train_gcn_v5.py:144-232`
- Copy `infer_cardinalities_from_graphs()` from `train_gcn_v5.py:133-141`
- Keep GINEConv + AttentionalAggregation architecture

### 3. `model_projector.py`
- `SolidBridgeProjector` class: 768 → 1024 → 1024 → 896 (Qwen hidden size)
- Each layer: Linear + GELU + LayerNorm + Dropout
- Output: `[batch, num_graph_tokens, llm_hidden]`

### 4. `model_wrapper.py`
- `MolCaptionModel` class combining:
  - Frozen MolGNN
  - Trainable SolidBridgeProjector
  - Qwen3-0.6B with 4-bit NF4 + LoRA
- Add special token `<|graph|>` to tokenizer
- `forward()`: inject soft token at `<|graph|>` position in `inputs_embeds`
- `generate()`: batch inference for test set
- `get_llm_text_embedding()`: mean-pooled hidden states for alignment

### 5. `dataset_caption.py`
- `MolCaptionDataset`: loads graphs from pkl, returns (graph, description, smiles)
- `collate_caption_batch()`: batch graphs + tokenize prompts + create labels

### 6. `train_stage1.py`
- Freeze GNN and LLM, train only Projector
- `alignment_loss()`: cosine distance between projected embedding and LLM text embedding
- W&B logging: loss, lr, val_loss
- **tqdm progress bars** with loss/lr display
- Returns metrics dict + calls `print_training_report()` at end

### 7. `train_stage2.py`
- Train Projector + LoRA adapters
- Cross-entropy loss on generated tokens
- Separate LR for projector (5e-4) and LoRA (2e-4)
- **Periodic evaluation**: every `eval_every_n_steps`, compute BLEU/METEOR on val subset
- W&B logging: loss, perplexity, bleu4, meteor, sample outputs
- **tqdm progress bars** with metrics display
- Returns metrics dict + calls `print_training_report()` at end

### 8. `inference.py`
- Load best checkpoint
- Generate captions for test set
- Save `submission.csv`

### 9. `metrics.py` (NEW)
Compute high-level evaluation metrics:
```python
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
# Optional: from bert_score import score as bert_score

def compute_metrics(predictions: list, references: list) -> dict:
    """Compute BLEU-4, METEOR, and optionally BERTScore."""
    # BLEU-4
    refs = [[r.split()] for r in references]
    hyps = [p.split() for p in predictions]
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25),
                        smoothing_function=SmoothingFunction().method1)

    # METEOR (average)
    meteor_scores = [meteor_score([r], p) for r, p in zip(references, predictions)]
    meteor = sum(meteor_scores) / len(meteor_scores)

    return {"bleu4": bleu4 * 100, "meteor": meteor * 100}
```

### 10. `report.py` (NEW)
Pretty report generation after training:
```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def print_training_report(stage: str, metrics: dict, config, samples: list = None):
    """Print a formatted training report with metrics and sample outputs."""
    console = Console()

    # Header
    console.print(Panel(f"[bold]{stage} Training Complete[/bold]", style="green"))

    # Metrics table
    table = Table(title="Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    # Sample outputs (if provided)
    if samples:
        console.print("\n[bold]Sample Outputs:[/bold]")
        for i, (pred, ref) in enumerate(samples[:3]):
            console.print(f"[dim]#{i+1}[/dim]")
            console.print(f"  [green]Pred:[/green] {pred[:100]}...")
            console.print(f"  [blue]Ref:[/blue]  {ref[:100]}...")

    # W&B link
    if config.use_wandb:
        console.print(f"\n[link]View full metrics on W&B[/link]")
```

## Prompt Format
```
<|user|>
Molecule Structure: <|graph|>
SMILES: {smiles}
Task: Describe the molecule's chemical properties and functional groups.
<|assistant|>
{description}
```

## SMILES Reconstruction (in `utils.py`)
Graphs don't have SMILES stored, but we can reconstruct from features using RDKit:
```python
def graph_to_smiles(graph) -> str:
    """Reconstruct SMILES from PyG graph using RDKit."""
    from rdkit import Chem
    mol = Chem.RWMol()

    # Bond type mapping (from data_utils.py e_map)
    BOND_TYPES = {
        1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE, 12: Chem.BondType.AROMATIC,
    }

    # Add atoms (x[:, 0]=atomic_num, x[:, 3]=formal_charge with offset -5)
    for i in range(graph.x.size(0)):
        atom = Chem.Atom(int(graph.x[i, 0].item()))
        atom.SetFormalCharge(int(graph.x[i, 3].item()) - 5)
        mol.AddAtom(atom)

    # Add bonds (deduplicate bidirectional edges)
    added = set()
    for j in range(graph.edge_index.size(1)):
        src, dst = int(graph.edge_index[0, j]), int(graph.edge_index[1, j])
        if (min(src, dst), max(src, dst)) in added:
            continue
        added.add((min(src, dst), max(src, dst)))
        bt = BOND_TYPES.get(int(graph.edge_attr[j, 0]), Chem.BondType.SINGLE)
        mol.AddBond(src, dst, bt)

    return Chem.MolToSmiles(mol.GetMol())
```

## Memory Estimate (T4 16GB)
| Component | Memory |
|-----------|--------|
| Qwen3-0.6B 4-bit | ~400 MB |
| LoRA adapters | ~50 MB |
| MolGNN (frozen) | ~20 MB |
| Projector | ~10 MB |
| Activations (batch=8) | ~3 GB |
| **Total** | **~4-5 GB** |

## Training Time Estimate
- Stage 1: ~1.5 hours
- Stage 2: ~7.5 hours
- **Total: ~9 hours** (fits Kaggle session)

## Notebook Execution Flow

```python
# Cell 1: Install
!pip install -q transformers>=4.36 peft bitsandbytes accelerate wandb rich nltk

# Cell 2: Import modules
import sys
sys.path.insert(0, "/kaggle/input/mol-caption-code")
from config import Config
from model_wrapper import MolCaptionModel
from train_stage1 import train_stage1
from train_stage2 import train_stage2
from metrics import compute_metrics
from report import print_training_report

# Cell 3: Configure experiment mode
config = Config(
    experiment_mode="quick",  # "quick" (5min), "medium" (1h), or "full" (9h)
    use_wandb=True,
)
config.apply_mode()  # Auto-adjusts epochs, subset size, etc.

# Cell 4: W&B init
import wandb
wandb.login()
wandb.init(project="mol-caption-gen", config=vars(config), tags=[config.experiment_mode])

# Cell 5: Load GNN checkpoint & build model
gnn = load_gnn(config)
model = MolCaptionModel(config, gnn)

# Cell 6: Prepare data (respects train_subset)
train_loader, val_loader = prepare_dataloaders(config, model.tokenizer)

# Cell 7: Stage 1 training (alignment)
# - tqdm progress bar
# - Prints report at end with alignment loss
stage1_metrics = train_stage1(model, train_loader, val_loader, config)
# Output: ╭─ Stage 1 Training Complete ─╮
#         │ Metric       │ Value       │
#         │ train_loss   │ 0.1234      │
#         │ val_loss     │ 0.1456      │
#         ╰──────────────┴─────────────╯

# Cell 8: Stage 2 training (SFT)
# - tqdm progress bar with BLEU/METEOR updates
# - Evaluates metrics every eval_every_n_steps
stage2_metrics = train_stage2(model, train_loader, val_loader, config)
# Output: ╭─ Stage 2 Training Complete ─╮
#         │ Metric       │ Value       │
#         │ train_loss   │ 2.3456      │
#         │ val_loss     │ 2.5678      │
#         │ bleu4        │ 32.45       │
#         │ meteor       │ 28.67       │
#         ╰──────────────┴─────────────╯
#         Sample Outputs:
#         #1 Pred: The molecule contains a hydroxyl group...
#            Ref:  The molecule is a primary alcohol with...

# Cell 9: Generate submission (full mode only)
if config.experiment_mode == "full":
    generate_submission(model, config)
```

## W&B Metrics to Track
- **Stage 1**: `stage1/loss`, `stage1/val_loss`, `stage1/lr`, `stage1/cosine_sim`
- **Stage 2**: `stage2/loss`, `stage2/val_loss`, `stage2/perplexity`
- **Eval** (periodic): `eval/bleu4`, `eval/meteor`, sample outputs table
- **Final**: Full validation metrics, model artifacts

## Critical Source Files
1. `train_gcn_v5.py:144-232` - MolGNN architecture
2. `data_utils.py:101-146` - Dataset and collate functions
3. `retrieval_answer_test_v6.py` - Reference for LLM generation

## Kaggle Setup Steps
1. Create private Dataset "mol-caption-code" with all .py files
2. Create private Dataset "gnn-checkpoints" with `gnn_v5_best.pt`
3. Ensure "altegrad-2024" competition data is attached
4. Enable GPU accelerator (T4 x2 or P100)
5. Add W&B API key to Kaggle secrets

## Verification Plan
1. **Stage 1**: Alignment loss should decrease steadily; target < 0.3 cosine distance
2. **Stage 2**: Perplexity should decrease; target < 10
3. **Manual inspection**: Generate 5-10 samples, check chemical validity
4. **Metrics**: BLEU-4 > 30, BERTScore > 0.8 on validation set
