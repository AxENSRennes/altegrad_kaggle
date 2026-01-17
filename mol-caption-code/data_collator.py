#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom DataCollator for TRL SFTTrainer with PyG molecular graphs.

This module provides a collator that:
1. Batches PyG graphs using Batch.from_data_list
2. Computes graph embeddings using frozen GNN
3. Tokenizes text with proper label masking
4. Returns dict format compatible with TRL SFTTrainer
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch_geometric.data import Batch
from transformers import PreTrainedTokenizerBase


@dataclass
class MolCaptionDataCollator:
    """
    DataCollator for molecular captioning with TRL SFTTrainer.

    Handles batching of PyG graphs and tokenization of text sequences.
    Graph embeddings are computed at collation time using a frozen GNN.

    Attributes:
        tokenizer: HuggingFace tokenizer
        gnn: Frozen GNN module for graph encoding
        projector: Projector module to map GNN embeddings to LLM space
        graph_token_id: Token ID for <|graph|> special token
        max_length: Maximum sequence length for tokenization
        device: Device to run GNN forward pass on
    """
    tokenizer: PreTrainedTokenizerBase
    gnn: torch.nn.Module
    projector: torch.nn.Module
    graph_token_id: int
    max_length: int = 256
    device: str = "cuda"
    txt_mean: Optional[torch.Tensor] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features for TRL SFTTrainer.

        Args:
            features: List of dicts with keys:
                - 'graph': PyG Data object
                - 'text': Full text sequence (prompt + description + eos)
                - 'prompt_length': Number of tokens in prompt (for label masking)

        Returns:
            Dict with keys:
                - 'input_ids': Token IDs [batch_size, seq_len]
                - 'attention_mask': Attention mask [batch_size, seq_len]
                - 'labels': Target labels with prompt masked [batch_size, seq_len]
                - 'graph_embeddings': Projected graph embeddings [batch_size, num_tokens, hidden]
        """
        graphs = [f['graph'] for f in features]
        texts = [f['text'] for f in features]
        prompt_lengths = [f['prompt_length'] for f in features]

        # Batch graphs and compute embeddings
        batched_graphs = Batch.from_data_list(graphs)
        batched_graphs = batched_graphs.to(self.device)

        with torch.no_grad():
            self.gnn.eval()
            graph_emb = self.gnn(batched_graphs)  # [B, gnn_out_dim]

            # Apply centering and normalization if txt_mean provided
            if self.txt_mean is not None and graph_emb.size(-1) == self.txt_mean.size(-1):
                graph_emb = graph_emb - self.txt_mean
            graph_emb = torch.nn.functional.normalize(graph_emb, p=2, dim=-1)

            # Project to LLM space
            graph_embeddings = self.projector(graph_emb)  # [B, num_tokens, llm_hidden]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (mask prompt tokens with -100)
        labels = encodings['input_ids'].clone()

        for i, plen in enumerate(prompt_lengths):
            # Find where content starts (after left padding)
            content_positions = (encodings['attention_mask'][i] == 1).nonzero(as_tuple=True)[0]
            if len(content_positions) > 0:
                content_start = content_positions[0].item()
                labels[i, content_start:content_start + plen] = -100

        # Mask padding tokens
        labels[encodings['attention_mask'] == 0] = -100

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels,
            'graph_embeddings': graph_embeddings,
        }


@dataclass
class MolCaptionDataCollatorSimple:
    """
    Simpler DataCollator that doesn't compute graph embeddings.

    Use this when graph embeddings are pre-computed or when the model
    handles graph encoding internally during forward pass.

    This collator only handles tokenization and label masking.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 256

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: List of dicts with keys:
                - 'graph': PyG Data object
                - 'text': Full text sequence
                - 'prompt_length': Number of tokens in prompt
                - 'description': Original description text
                - 'smiles': SMILES string

        Returns:
            Dict with batched tensors and metadata.
        """
        graphs = [f['graph'] for f in features]
        texts = [f['text'] for f in features]
        prompt_lengths = [f['prompt_length'] for f in features]
        descriptions = [f.get('description', '') for f in features]
        smiles_list = [f.get('smiles', '') for f in features]

        # Batch graphs
        batched_graphs = Batch.from_data_list(graphs)

        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (mask prompt tokens with -100)
        labels = encodings['input_ids'].clone()

        for i, plen in enumerate(prompt_lengths):
            # Find where content starts (after left padding)
            content_positions = (encodings['attention_mask'][i] == 1).nonzero(as_tuple=True)[0]
            if len(content_positions) > 0:
                content_start = content_positions[0].item()
                labels[i, content_start:content_start + plen] = -100

        # Mask padding tokens
        labels[encodings['attention_mask'] == 0] = -100

        return {
            'graphs': batched_graphs,
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels,
            'descriptions': descriptions,
            'smiles': smiles_list,
        }
