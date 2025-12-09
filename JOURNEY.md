# Scientific Journey: optimizing Graph-Text Retrieval

## 1. Initial Observation
**Context**: We started with a baseline GCN model trained to align molecule graph embeddings with BERT text embeddings of their descriptions.
**Observation**: The model failed to learn effectively.
*   **Loss Curve**: Flat or stagnant.
*   **Metrics**: MRR (Mean Reciprocal Rank) stuck at ~0.014 (random guessing territory).
*   **Visuals**: Rank histograms were flat/uniform, showing the model could not rank the correct description higher than random ones.

## 2. Hypothesis 1: Loss Function Rigidity
**Analysis**: The initial training used **MSE Loss** (`Mean Squared Error`) to force the graph vector to be *identical* to the text vector.
**Hypothesis**: MSE is too restrictive for retrieval tasks. We don't need the vectors to be identical, we only need the *correct* pair to be closer than *incorrect* pairs.
**Solution**:
*   Implemented **Triplet Loss** (Contrastive Learning).
*   Objective: Maximize $Sim(Anchor, Positive) - Sim(Anchor, Negative) + Margin$.
*   Outcome: Training became closer to the true retrieval objective.

## 3. Hypothesis 2: Feature Blindness (The Critical Flaw)
**Observation**: Even with Triplet Loss, convergence was extremely slow and metrics remained low (MRR ~0.014).
**Investigation**: We inspected the data using `inspect_graph_data.py`.
*   **Data**: `graph.x` contained indices for 9 categorical features (Atomic Number, Chirality, etc.).
*   **Model**: The `MolGNN` class ignored `graph.x` entirely. It initialized a single shared learnable vector `self.node_init` for *all* nodes in *all* graphs.
**Hypothesis**: The model was trying to distinguish molecules based solely on their topology (graph shape) without knowing the atom types (Carbon vs Oxygen vs Nitrogen). It was effectively "chemically blind".
**Solution**:
*   Implemented an **`AtomEncoder`**.
*   This module contains 9 separate Embedding layers (one for each feature in `graph.x`).
*   It sums these embeddings to create a rich node representation that captures chemical identity.

## 4. Final Architecture
*   **Encoder**: GCN with `AtomEncoder` (3 layers).
*   **Loss**: Triplet Loss (Margin=0.2).
*   **Training**: Batch size 32, Learning Rate 0.001.

**Conclusion**: By fixing the feature encoding, we transformed the problem from "guessing shapes" to "learning chemistry", enabling the model to actually align molecular structures with textual descriptions.
