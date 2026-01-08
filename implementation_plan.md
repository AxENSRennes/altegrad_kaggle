# Generative Molecular Captioning (2026 SOTA Edition)

Transition to a generative system using the ultra-efficient **Qwen3-0.6B** fine-tuned via Q-LoRA, with a robust two-stage "solid bridge" alignment.

## Proposed Changes

### 1. Model Architecture: The "Solid Bridge"
- **Encoder**: [MolGNN](file:///home/axel/Altegrad_kaggle/retrieval_answer_test_v5.py#145-199) (v4) extracting structural features (dim=768).
- **Projector (The Bridge)**: 
    - **Architecture**: A 3-layer MLP (768 -> 1024 -> 1024) with **GELU** and **LayerNorm**. 
    - **Rationale**: A deep projector is better at mapping the abstract structural geometry of a GNN into the semantic token space of an LLM.
- **LLM Backbone**: **Qwen3-0.6B** (non-thinking).
    - **Quantization**: 4-bit NF4.
    - **LoRA**: Targeting all linear layers (`q`, `k`, `v`, `o`, `gate`, `up`, `down`) for maximum adaptability.

### 2. Two-Stage Training Strategy
To ensure the bridge is "solid", we don't train everything at once :
- **Stage 1 : Modality Alignment (Alignment)**
    - **Goal**: Teach the Projector to map GNN embeddings into the LLM's semantic space.
    - **Method**: Freeze the LLM and GNN. Train **only the Projector** to minimize the distance (MSE or Cosine) between the projected graph embedding and the LLM's own embedding of the ground-truth description.
- **Stage 2 : Generative Fine-Tuning (SFT)**
    - **Goal**: End-to-end caption generation.
    - **Method**: Train the Projector and the **LoRA adapters** jointly using Cross-Entropy loss. This refines the bridge while teaching the LLM the "chemical language" of our dataset.

### 3. Final Pipeline
- **Input**: Graph + SMILES.
- **Prompt**:
  ```text
  <|user|>Molecule Structure: [GRAPH_TOKEN] | SMILES: [SMILES]
  Task: Describe the molecule's chemical properties and groups.<|assistant|>
  ```

## Verification Plan

### Metrics
- **BLEU-4 / METEOR**: Comparative analysis against the retrieval baseline.
- **Alignment Loss**: Monitor the "Stage 1" loss to ensure the bridge has successfully learned the mapping before starting the generation phase.

## Verification Plan

### Metrics
- **BLEU-4 / METEOR**: Target surpassing the ~38 score of SOTA models (LLaMo).
- **Chemical Validity**: Use a parser to check if functional groups mentioned in text match the graph structure.
