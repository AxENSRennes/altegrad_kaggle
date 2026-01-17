#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset and collation utilities for molecular captioning.

Provides:
- MolCaptionDataset: Dataset for loading molecular graphs with descriptions
- collate_caption_batch: Collate function for batching with tokenization
- Utilities for loading descriptions from graph files
"""

import pickle
from typing import Dict, List, Optional, Tuple, Any, Callable
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Batch

from config import SYSTEM_PROMPT, USER_PROMPT_FORMAT
from utils import graph_to_smiles


def load_descriptions_from_graphs(graph_path: str) -> Dict[str, str]:
    """
    Load ID to description mapping from preprocessed graph file.

    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs

    Returns:
        Dictionary mapping ID (str) to description (str)
    """
    with open(graph_path, "rb") as f:
        graphs = pickle.load(f)

    id2desc = {}
    for graph in graphs:
        if hasattr(graph, "description") and graph.description:
            id2desc[graph.id] = graph.description

    return id2desc


class MolCaptionDataset(Dataset):
    """
    Dataset for molecular captioning.

    Loads molecular graphs from pickle files and pairs them with descriptions.
    Returns (graph, description, mol_id) tuples.
    """

    def __init__(
        self,
        graph_path: str,
        descriptions: Optional[Dict[str, str]] = None,
        load_descriptions_from_file: bool = True,
    ):
        """
        Args:
            graph_path: Path to .pkl file with molecular graphs
            descriptions: Optional dict mapping mol_id to description
            load_descriptions_from_file: If True and descriptions is None,
                                        load descriptions from graph file
        """
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        print(f"Loaded {len(self.graphs)} graphs")

        # Load descriptions
        if descriptions is not None:
            self.descriptions = descriptions
        elif load_descriptions_from_file:
            self.descriptions = {}
            for g in self.graphs:
                if hasattr(g, "description") and g.description:
                    self.descriptions[g.id] = g.description
        else:
            self.descriptions = {}

        # Store IDs
        self.ids = [g.id for g in self.graphs]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[Any, str, str]:
        """
        Get a single sample.

        Returns:
            Tuple of (graph, description, mol_id)
        """
        graph = self.graphs[idx]
        mol_id = graph.id
        description = self.descriptions.get(mol_id, "")
        return graph, description, mol_id


def collate_caption_batch(
    batch: List[Tuple[Any, str, str]],
    tokenizer,
    max_length: int = 256,
    include_labels: bool = True,
) -> Dict[str, Any]:
    """
    Collate function for caption training.

    Batches graphs using PyG's Batch.from_data_list and tokenizes prompts.

    Args:
        batch: List of (graph, description, mol_id) tuples
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        include_labels: Whether to create labels for training

    Returns:
        Dictionary with:
        - 'graphs': Batched PyG graphs
        - 'input_ids': Tokenized input IDs
        - 'attention_mask': Attention mask
        - 'labels': Target labels (if include_labels=True)
        - 'smiles': List of SMILES strings
        - 'mol_ids': List of molecule IDs
        - 'descriptions': List of descriptions
    """
    graphs, descriptions, mol_ids = zip(*batch)
    graphs = list(graphs)

    # Batch graphs
    batched_graphs = Batch.from_data_list(graphs)

    # Get SMILES for each graph
    smiles_list = [graph_to_smiles(g) for g in graphs]

    # Build prompts using tokenizer.apply_chat_template
    prompts = []
    for s in smiles_list:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_FORMAT.format(smiles=s)}
        ]
        # Use enable_thinking=False to strictly disable reasoning and add empty <think> tags
        p = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(p)

    # Build full sequences (prompt + description + eos)
    if include_labels:
        full_sequences = [
            p + d + tokenizer.eos_token
            for p, d in zip(prompts, descriptions)
        ]
    else:
        full_sequences = prompts

    # Tokenize
    encodings = tokenizer(
        full_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    result = {
        "graphs": batched_graphs,
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "smiles": smiles_list,
        "mol_ids": list(mol_ids),
        "descriptions": list(descriptions),
    }

    # Create labels (mask prompt tokens with -100)
    if include_labels:
        labels = encodings["input_ids"].clone()

        for i, prompt in enumerate(prompts):
            # Find prompt length (tokens before the description)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)

            # Mask prompt tokens
            labels[i, :prompt_len] = -100

            # Also mask padding tokens
            pad_mask = encodings["attention_mask"][i] == 0
            labels[i, pad_mask] = -100

        result["labels"] = labels

    return result


def create_collate_fn(tokenizer, max_length: int = 256, include_labels: bool = True) -> Callable:
    """
    Create a collate function with bound tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        include_labels: Whether to include labels

    Returns:
        Collate function
    """
    return partial(
        collate_caption_batch,
        tokenizer=tokenizer,
        max_length=max_length,
        include_labels=include_labels,
    )


def prepare_dataloaders(
    config,
    tokenizer,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration object
        tokenizer: HuggingFace tokenizer
        train_subset: Optional limit on training samples
        val_subset: Optional limit on validation samples

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = MolCaptionDataset(config.train_graphs_path)
    val_dataset = MolCaptionDataset(config.val_graphs_path)

    # Apply subset if specified
    train_subset = train_subset or config.train_subset
    val_subset = val_subset or config.val_subset

    if train_subset is not None and train_subset < len(train_dataset):
        indices = list(range(train_subset))
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {train_subset} training samples")

    if val_subset is not None and val_subset < len(val_dataset):
        indices = list(range(val_subset))
        val_dataset = Subset(val_dataset, indices)
        print(f"Using {val_subset} validation samples")

    # Create collate functions
    train_collate = create_collate_fn(tokenizer, config.max_seq_length, include_labels=True)
    val_collate = create_collate_fn(tokenizer, config.max_seq_length, include_labels=True)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.stage2_batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.stage2_batch_size,
        shuffle=False,
        collate_fn=val_collate,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class AlignmentDataset(Dataset):
    """
    Dataset for Stage 1 alignment training.

    Returns (graph, description_text) pairs for computing alignment loss
    between projected graph embeddings and LLM text embeddings.
    """

    def __init__(self, graph_path: str):
        """
        Args:
            graph_path: Path to .pkl file with molecular graphs
        """
        print(f"Loading graphs for alignment from: {graph_path}")
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        """
        Returns:
            Tuple of (graph, description_text)
        """
        graph = self.graphs[idx]
        description = getattr(graph, "description", "")
        return graph, description


def collate_alignment_batch(batch: List[Tuple[Any, str]]) -> Tuple[Batch, List[str]]:
    """
    Collate function for alignment training.

    Args:
        batch: List of (graph, description) tuples

    Returns:
        Tuple of (batched_graphs, descriptions_list)
    """
    graphs, descriptions = zip(*batch)
    batched_graphs = Batch.from_data_list(list(graphs))
    return batched_graphs, list(descriptions)


def prepare_alignment_dataloaders(
    config,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for Stage 1 alignment training.

    Args:
        config: Configuration object
        train_subset: Optional limit on training samples
        val_subset: Optional limit on validation samples

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = AlignmentDataset(config.train_graphs_path)
    val_dataset = AlignmentDataset(config.val_graphs_path)

    # Apply subset
    train_subset = train_subset or config.train_subset
    val_subset = val_subset or config.val_subset

    if train_subset is not None and train_subset < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(train_subset)))
        print(f"Using {train_subset} training samples for alignment")

    if val_subset is not None and val_subset < len(val_dataset):
        val_dataset = Subset(val_dataset, list(range(val_subset)))
        print(f"Using {val_subset} validation samples for alignment")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.stage1_batch_size,
        shuffle=True,
        collate_fn=collate_alignment_batch,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.stage1_batch_size,
        shuffle=False,
        collate_fn=collate_alignment_batch,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class MolCaptionDatasetTRL(Dataset):
    """
    TRL-compatible dataset for molecular captioning.

    Returns dict format expected by TRL SFTTrainer:
    - 'graph': PyG Data object
    - 'text': Full text sequence (prompt + description + eos)
    - 'prompt_length': Number of tokens in prompt (for label masking)
    - 'description': Original description text
    - 'smiles': SMILES string
    """

    def __init__(
        self,
        graph_path: str,
        tokenizer,
        descriptions: Optional[Dict[str, str]] = None,
        load_descriptions_from_file: bool = True,
    ):
        """
        Args:
            graph_path: Path to .pkl file with molecular graphs
            tokenizer: HuggingFace tokenizer for encoding prompts
            descriptions: Optional dict mapping mol_id to description
            load_descriptions_from_file: If True and descriptions is None,
                                        load descriptions from graph file
        """
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, "rb") as f:
            self.graphs = pickle.load(f)
        print(f"Loaded {len(self.graphs)} graphs")

        self.tokenizer = tokenizer

        # Load descriptions
        if descriptions is not None:
            self.descriptions = descriptions
        elif load_descriptions_from_file:
            self.descriptions = {}
            for g in self.graphs:
                if hasattr(g, "description") and g.description:
                    self.descriptions[g.id] = g.description
        else:
            self.descriptions = {}

        # Store IDs
        self.ids = [g.id for g in self.graphs]

    def __len__(self) -> int:
        return len(self.graphs)

    def _build_prompt(self, smiles: str) -> str:
        """Build the chat prompt for a molecule."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_FORMAT.format(smiles=smiles)}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return prompt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample in TRL-compatible dict format.

        Returns:
            Dict with keys: graph, text, prompt_length, description, smiles
        """
        graph = self.graphs[idx]
        mol_id = graph.id
        description = self.descriptions.get(mol_id, "")
        smiles = graph_to_smiles(graph)

        # Build prompt and full text
        prompt = self._build_prompt(smiles)
        full_text = prompt + description + self.tokenizer.eos_token

        # Calculate prompt length in tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens)

        return {
            'graph': graph,
            'text': full_text,
            'prompt_length': prompt_length,
            'description': description,
            'smiles': smiles,
        }


def prepare_trl_dataloaders(
    config,
    tokenizer,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Create TRL-compatible datasets for Stage 2 training.

    Args:
        config: Configuration object
        tokenizer: HuggingFace tokenizer
        train_subset: Optional limit on training samples
        val_subset: Optional limit on validation samples

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = MolCaptionDatasetTRL(config.train_graphs_path, tokenizer)
    val_dataset = MolCaptionDatasetTRL(config.val_graphs_path, tokenizer)

    # Apply subset if specified
    train_subset = train_subset or config.train_subset
    val_subset = val_subset or config.val_subset

    if train_subset is not None and train_subset < len(train_dataset):
        indices = list(range(train_subset))
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {train_subset} training samples")

    if val_subset is not None and val_subset < len(val_dataset):
        indices = list(range(val_subset))
        val_dataset = Subset(val_dataset, indices)
        print(f"Using {val_subset} validation samples")

    return train_dataset, val_dataset
