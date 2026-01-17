#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for molecular captioning tests.

Provides:
- tokenizer_with_graph_token: Real Qwen3 tokenizer with <|graph|> token
- mock_graph / mock_graphs: PyG Data objects mimicking molecular graphs
- sample_features: Pre-built dataset items for collator tests
- build_prompt helper function
"""

import pytest
import torch
from torch_geometric.data import Data

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SYSTEM_PROMPT, USER_PROMPT_FORMAT


# ============================================================================
# Tokenizer Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def tokenizer_with_graph_token():
    """
    Load Qwen3 tokenizer with <|graph|> special token added.

    Session-scoped for efficiency - loads once per test session.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        trust_remote_code=True,
    )

    # Add graph token
    special_tokens = {"additional_special_tokens": ["<|graph|>"]}
    tokenizer.add_special_tokens(special_tokens)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use left padding for decoder-only models
    tokenizer.padding_side = "left"

    return tokenizer


@pytest.fixture
def graph_token_id(tokenizer_with_graph_token):
    """Get the token ID for <|graph|>."""
    return tokenizer_with_graph_token.convert_tokens_to_ids("<|graph|>")


# ============================================================================
# Mock Graph Fixtures
# ============================================================================

@pytest.fixture
def mock_graph():
    """
    Create a simple mock molecular graph (water: H-O-H).

    Returns a PyG Data object with:
    - 3 nodes (O, H, H)
    - 2 edges (O-H bonds)
    - Node features: 9-dimensional
    - Edge features: 3-dimensional
    """
    # Node features: [atomic_num, chirality, degree, formal_charge, num_hs,
    #                 num_radical_electrons, hybridization, is_aromatic, is_in_ring]
    x = torch.tensor([
        [8, 0, 2, 0, 0, 0, 3, 0, 0],  # Oxygen
        [1, 0, 1, 0, 0, 0, 0, 0, 0],  # Hydrogen
        [1, 0, 1, 0, 0, 0, 0, 0, 0],  # Hydrogen
    ], dtype=torch.long)

    # Edge index: O-H and H-O bonds (bidirectional)
    edge_index = torch.tensor([
        [0, 1, 0, 2],
        [1, 0, 2, 0],
    ], dtype=torch.long)

    # Edge features: [bond_type, stereo, is_conjugated]
    edge_attr = torch.tensor([
        [1, 0, 0],  # O-H single bond
        [1, 0, 0],  # H-O single bond
        [1, 0, 0],  # O-H single bond
        [1, 0, 0],  # H-O single bond
    ], dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    graph.id = "test_mol_001"
    graph.description = "Water molecule with two hydrogen atoms bonded to oxygen."
    graph.smiles = "O"

    return graph


@pytest.fixture
def mock_graph_benzene():
    """
    Create a mock benzene ring graph (C6H6).
    More complex than water for testing batching.
    """
    # 6 carbon atoms in a ring
    x = torch.tensor([
        [6, 0, 2, 0, 1, 0, 2, 1, 1],  # C (aromatic, in ring)
        [6, 0, 2, 0, 1, 0, 2, 1, 1],
        [6, 0, 2, 0, 1, 0, 2, 1, 1],
        [6, 0, 2, 0, 1, 0, 2, 1, 1],
        [6, 0, 2, 0, 1, 0, 2, 1, 1],
        [6, 0, 2, 0, 1, 0, 2, 1, 1],
    ], dtype=torch.long)

    # Ring edges (simplified - bidirectional)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
    ], dtype=torch.long)

    # Aromatic bonds
    edge_attr = torch.tensor([
        [4, 0, 1],  # aromatic bond
    ] * 12, dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    graph.id = "test_mol_002"
    graph.description = "Benzene is an aromatic hydrocarbon with a six-membered carbon ring."
    graph.smiles = "c1ccccc1"

    return graph


@pytest.fixture
def mock_graphs(mock_graph, mock_graph_benzene):
    """Return a list of mock graphs for batch testing."""
    return [mock_graph, mock_graph_benzene]


# ============================================================================
# Prompt Building Helper
# ============================================================================

def build_prompt(tokenizer, smiles: str) -> str:
    """
    Build the chat prompt for a molecule using the tokenizer's chat template.

    This mirrors the logic in MolCaptionDatasetTRL._build_prompt().
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_FORMAT.format(smiles=smiles)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt


@pytest.fixture
def build_prompt_fn(tokenizer_with_graph_token):
    """Return a prompt builder function bound to the fixture tokenizer."""
    def _build(smiles: str) -> str:
        return build_prompt(tokenizer_with_graph_token, smiles)
    return _build


# ============================================================================
# Sample Feature Fixtures (for collator tests)
# ============================================================================

@pytest.fixture
def sample_feature(mock_graph, tokenizer_with_graph_token):
    """
    Create a sample feature dict as returned by MolCaptionDatasetTRL.__getitem__.
    """
    smiles = mock_graph.smiles
    description = mock_graph.description

    prompt = build_prompt(tokenizer_with_graph_token, smiles)
    full_text = prompt + description + tokenizer_with_graph_token.eos_token

    prompt_tokens = tokenizer_with_graph_token.encode(prompt, add_special_tokens=False)
    prompt_length = len(prompt_tokens)

    return {
        'graph': mock_graph,
        'text': full_text,
        'prompt_length': prompt_length,
        'description': description,
        'smiles': smiles,
    }


@pytest.fixture
def sample_features(mock_graphs, tokenizer_with_graph_token):
    """
    Create a list of sample features for batch collator tests.
    """
    features = []
    for graph in mock_graphs:
        smiles = graph.smiles
        description = graph.description

        prompt = build_prompt(tokenizer_with_graph_token, smiles)
        full_text = prompt + description + tokenizer_with_graph_token.eos_token

        prompt_tokens = tokenizer_with_graph_token.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens)

        features.append({
            'graph': graph,
            'text': full_text,
            'prompt_length': prompt_length,
            'description': description,
            'smiles': smiles,
        })

    return features


# ============================================================================
# Mock Embedding Layer Fixtures
# ============================================================================

@pytest.fixture
def mock_embedding_layer():
    """
    Create a mock embedding layer for testing injection.

    Returns an nn.Embedding that maps token IDs to 1024-dim embeddings.
    """
    vocab_size = 152064 + 1  # Qwen3 vocab + <|graph|>
    hidden_dim = 1024

    embedding = torch.nn.Embedding(vocab_size, hidden_dim)
    return embedding


@pytest.fixture
def llm_hidden_dim():
    """Return the LLM hidden dimension (Qwen3-0.6B uses 1024)."""
    return 1024
