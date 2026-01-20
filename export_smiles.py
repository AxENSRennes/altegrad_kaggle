#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export SMILES for the test split to a CSV file.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import List

import torch
from rdkit import Chem

from data_utils import x_map, e_map

DATA_DIR = Path("data")
SPLIT = "validation"


def resolve_graph_path(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / f"{split}_graphs.pkl",
        data_dir / f"{split}_graphs_func_groups.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No graph file found for split '{split}' in {data_dir}")


def _bond_type_from_index(bond_idx: int) -> Chem.BondType:
    bond_name = e_map["bond_type"][bond_idx] if 0 <= bond_idx < len(e_map["bond_type"]) else "UNSPECIFIED"
    if bond_name == "DOUBLE":
        return Chem.BondType.DOUBLE
    if bond_name == "TRIPLE":
        return Chem.BondType.TRIPLE
    if bond_name == "AROMATIC":
        return Chem.BondType.AROMATIC
    return Chem.BondType.SINGLE


def get_mol(graph) -> Chem.Mol:
    if hasattr(graph, "rdkit_mol"):
        return graph.rdkit_mol
    if hasattr(graph, "mol"):
        return graph.mol
    if hasattr(graph, "smiles"):
        mol = Chem.MolFromSmiles(graph.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for graph id={getattr(graph, 'id', 'N/A')}")
        return mol

    if not hasattr(graph, "x") or not hasattr(graph, "edge_index") or not hasattr(graph, "edge_attr"):
        raise ValueError(
            f"Graph id={getattr(graph, 'id', 'N/A')} missing x/edge_index/edge_attr to build RDKit mol."
        )

    x = graph.x
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()
    if isinstance(edge_attr, torch.Tensor):
        edge_attr = edge_attr.cpu()

    mol = Chem.RWMol()
    for i in range(x.size(0)):
        atomic_num_idx = int(x[i, 0].item())
        atomic_num = x_map["atomic_num"][atomic_num_idx]
        atom = Chem.Atom(int(atomic_num))
        mol.AddAtom(atom)

    seen = set()
    for k in range(edge_index.size(1)):
        u = int(edge_index[0, k].item())
        v = int(edge_index[1, k].item())
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))

        bond_idx = int(edge_attr[k, 0].item()) if edge_attr.numel() > 0 else 0
        bond_type = _bond_type_from_index(bond_idx)
        mol.AddBond(a, b, bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            mol.GetAtomWithIdx(a).SetIsAromatic(True)
            mol.GetAtomWithIdx(b).SetIsAromatic(True)

    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def load_graphs(path: Path) -> List:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:

    data_dir = DATA_DIR
    graph_path = resolve_graph_path(data_dir, SPLIT)
    graphs = load_graphs(graph_path)

    out_path = Path(f"{SPLIT}_smiles.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "smiles"])
        for g in graphs:
            mol = get_mol(g)
            smiles = Chem.MolToSmiles(mol)
            graph_id = getattr(g, "id", None)
            writer.writerow([graph_id, smiles])

    print(f"[info] wrote {len(graphs)} SMILES rows -> {out_path}")


if __name__ == "__main__":
    main()
