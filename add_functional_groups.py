#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate molecule graphs with functional groups using RDKit/SMARTS.

Reads train/validation/test graphs and writes new .pkl files with a
`functional_groups` attribute (list of group names) on each graph.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from rdkit import Chem

from data_utils import x_map, e_map


SMARTS_PATTERNS: List[Tuple[str, str]] = [
    ("alcohol", "[OX2H][CX4;!$(C=O)]"),
    ("phenol", "c[OX2H]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("hemiacetal", "[CX4]([OX2H])[OX2][#6]"),
    ("acetal", "[CX4]([OX2][#6])[OX2][#6]"),
    ("anomeric_carbon", "[CX4]([OX2])[OX2]"),
    ("glycosidic_bond", "[OX2][CX4]([OX2])[OX2]"),
    ("amine_primary", "[NX3;H2][#6]"),
    ("amine_secondary", "[NX3;H1]([#6])[#6]"),
    ("amine_tertiary", "[NX3;H0]([#6])([#6])[#6]"),
    ("aniline", "c[NH2]"),
    ("amine_aliphatic", "[NX3;!$([N]C=O)]"),
    ("quaternary_ammonium", "[NX4+]"),
    ("zwitterion_amino_acid", "[NX3+][CX3](=O)[O-]"),
    ("amide", "[NX3][CX3](=O)[#6]"),
    ("carbamate", "[NX3,NX4+][CX3](=O)[OX2,OX1-]"),
    ("urea", "[NX3][CX3](=O)[NX3]"),
    ("amidine", "[NX3][C]=[NX2]"),
    ("guanidine", "[NX3][CX3](=N)[NX3]"),
    ("peptide_bond", "N[C](=O)C"),
    ("glycine_motif", "NCC(=O)O"),
    ("lysine_motif", "NCCCCCN"),
    ("carboxylic_acid", "C(=O)[OX2H1]"),
    ("carboxylate", "[CX3](=O)[O-]"),
    ("ester", "C(=O)O[#6]"),
    ("thioester", "C(=O)S"),
    ("acyl_coa_thioester", "[CX3](=O)SCCNC(=O)C(C)C(=O)NCCOP(=O)(O)O"),
    ("lactone", "O=C1OCCCC1"),
    ("lactam", "C(=O)N1CCCC1"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("ketone", "[CX3](=O)[#6]"),
    ("ketone_generic", "[#6][CX3](=O)[#6]"),
    ("enone", "[CX3]=[CX3][CX3](=O)[#6]"),
    ("thiocarbonyl", "C(=S)"),
    ("nitrile", "[CX2]#N"),
    ("azide", "[NX1]=[NX2+]=[NX1-]"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]"),
    ("nitroso", "[NX2]=[OX1]"),
    ("azo", "[#6]-[NX2]=[NX2]-[#6]"),
    ("halogen", "[F,Cl,Br,I]"),
    ("sulfone", "S(=O)(=O)[#6]"),
    ("sulfoxide", "S(=O)[#6]"),
    ("sulfonamide", "S(=O)(=O)N"),
    ("sulfonate", "S(=O)(=O)[O-]"),
    ("sulfate_ester", "OS(=O)(=O)O"),
    ("sulfonic_acid", "[SX4](=O)(=O)[OX2H,OX1-]"),
    ("thiol", "[SX2H]"),
    ("thioether", "[#16X2H0]"),
    ("disulfide", "[#16X2H0][#16X2H0]"),
    ("peroxide", "[OX2,OX1-][OX2,OX1-]"),
    ("phosphate", "P(=O)(O)(O)"),
    ("pyrophosphate", "P(=O)(O)OP(=O)(O)O"),
    ("triphosphate", "P(=O)(O)OP(=O)(O)OP(=O)(O)O"),
    ("phosphonium", "[PX4+]"),
    ("aromatic_ring", "a1aaaaa1"),
    ("benzene", "c1ccccc1"),
    ("pyridine", "n1ccccc1"),
    ("pyridine_ring", "c1ccncc1"),
    ("pyrrole", "c1cncc1"),
    ("indole", "c1cc2ccccc2[nH]1"),
    ("imidazole", "n1cc[nH]c1"),
    ("pyrazole", "c1nncc1"),
    ("phenothiazine", "c1ccc2sc3ccccn3c2c1"),
    ("furan", "o1cccc1"),
    ("benzofuran", "c1oc2ccccc2c1"),
    ("furanose", "[CX4]1[CX4][CX4][CX4][OX2]1"),
    ("pyranose_ring", "O1CCCCC1"),
    ("furanose_ring", "O1CCCC1"),
    ("purine", "n1cnc2ncnc12"),
    ("pyrimidine", "n1ccnc(=O)n1"),
    ("isoprene_unit", "C=C(C)C"),
    ("steroid_core", "C1CCC2C3CCC4C(C3)CCC4C2C1"),
    ("alkyl_chain_c3", "[#6X4]-[#6X4]-[#6X4]"),
    ("alkene", "[CX3]=[CX3]"),
    ("alkyne", "[CX2]#[CX2]"),
    ("allene", "[CX3]=[CX2]=[CX3]"),
    ("macrocycle_12", "C1CCCCCCCCCCC1"),
    ("epoxide", "[OX2]1[CX4][CX4]1"),
    ("long_alkyl_chain", "[CX4][CX4][CX4][CX4][CX4][CX4][CX4][CX4]"),
    ("polyunsaturated", "[CX3]=[CX3]-[CX3]=[CX3]"),
    ("omega_hydroxy_acid", "O[CX4][CX4][CX4][CX4][CX4][CX3](=O)O"),
    ("o_acetyl", "OC(=O)C"),
    ("n_acetyl", "NC(=O)C"),
    ("pantetheine_chain", "NCC(=O)NCCS"),
    ("beta_lactam", "N1C(=O)CC1"),
    ("macrolide_core", "C1CCCCCCCCCCCC(=O)O1"),
    ("thienyl/thiophene", "s1cccc1"),
    ("thiazole", "c1ncsc1"),
    ("oxazole", "c1ocnc1"),
    ("piperidine", "N1CCCCC1"),
    ("morpholine", "O1CCNCC1"),
]


def compile_patterns() -> List[Tuple[str, Chem.Mol]]:
    compiled = []
    for name, smarts in SMARTS_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS for {name}: {smarts}")
        compiled.append((name, patt))
    return compiled


def resolve_graph_path(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / f"{split}_graphs_cached.pkl",
        data_dir / f"{split}_graphs.pkl",
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


def annotate_graphs(graphs: Iterable, patterns: List[Tuple[str, Chem.Mol]]) -> None:
    for g in graphs:
        mol = get_mol(g)
        groups = []
        for name, patt in patterns:
            try :
                if mol.HasSubstructMatch(patt):
                    groups.append(name)
            except Exception as e:
                print(f"[warning] error matching pattern {name} on graph id={getattr(g, 'id', 'N/A')}: {e}")
        g.functional_groups = groups


def load_graphs(path: Path) -> List:
    with path.open("rb") as f:
        return pickle.load(f)


def save_graphs(graphs: List, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(graphs, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate graphs with functional groups.")
    parser.add_argument("--data-dir", default="data", help="Directory containing graph .pkl files.")
    parser.add_argument("--out-suffix", default="_graphs_func_groups.pkl", help="Suffix for output files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    patterns = compile_patterns()

    for split in ("train", "validation", "test"):
        in_path = resolve_graph_path(data_dir, split)
        graphs = load_graphs(in_path)
        annotate_graphs(graphs, patterns)

        out_path = data_dir / f"{split}{args.out_suffix}"
        save_graphs(graphs, out_path)
        print(f"[info] wrote {len(graphs)} graphs -> {out_path}")


if __name__ == "__main__":
    main()
