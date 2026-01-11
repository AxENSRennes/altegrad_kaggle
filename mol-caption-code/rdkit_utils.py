#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized RDKit utilities and compatibility fixes.
Uses an isolated subprocess check to safely detect RDKit/NumPy conflicts.
"""

import os
import sys
import subprocess

# Fix for RDKit/NumPy compatibility issue: AttributeError: _ARRAY_API not found
# This MUST be set before any RDKit or NumPy imports in the process
os.environ["NPY_DISABLE_ARRAY_API"] = "1"

# Global flags for RDKit availability
_RDKIT_AVAILABLE = None
_Chem = None
_RWMol = None

def _is_rdkit_viable():
    """Check if RDKit is importable and functional using an isolated subprocess."""
    check_code = (
        "import os; "
        "os.environ['NPY_DISABLE_ARRAY_API'] = '1'; "
        "try: "
        "  from rdkit import Chem; "
        "  mol = Chem.RWMol(); "
        "  print('SUCCESS'); "
        "except: "
        "  print('FAILURE')"
    )
    try:
        # Run in the same interpreter as current process
        result = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy()
        )
        return "SUCCESS" in result.stdout
    except Exception:
        return False

def _get_rdkit():
    """Lazily load RDKit, skipping subprocess check since it gives false negatives."""
    global _RDKIT_AVAILABLE, _Chem, _RWMol

    if _RDKIT_AVAILABLE is not None:
        return _RDKIT_AVAILABLE, _Chem, _RWMol

    if os.environ.get("DISABLE_RDKIT", "0") == "1":
        _RDKIT_AVAILABLE = False
        return False, None, None

    # Direct import - the env var NPY_DISABLE_ARRAY_API is already set at module level
    # Skip subprocess check as it gives false negatives due to NumPy warnings on stderr
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
        _Chem = Chem
        _RWMol = RWMol
        _RDKIT_AVAILABLE = True
    except Exception:
        _RDKIT_AVAILABLE = False

    return _RDKIT_AVAILABLE, _Chem, _RWMol

def graph_to_smiles(graph) -> str:
    """
    Reconstruct SMILES from PyG graph using RDKit.
    Safe version that avoids crashing if RDKit is broken.
    """
    try:
        available, Chem, RWMol_func = _get_rdkit()
        
        if not available or Chem is None:
            return "SMILES_UNAVAILABLE (RDKit Load Failed)"

        mol = RWMol_func()

        # Bond type mapping
        BOND_TYPES = {
            0: Chem.BondType.UNSPECIFIED,
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.QUADRUPLE,
            5: Chem.BondType.QUINTUPLE,
            6: Chem.BondType.HEXTUPLE,
            7: Chem.BondType.ONEANDAHALF,
            8: Chem.BondType.TWOANDAHALF,
            9: Chem.BondType.THREEANDAHALF,
            10: Chem.BondType.FOURANDAHALF,
            11: Chem.BondType.FIVEANDAHALF,
            12: Chem.BondType.AROMATIC,
            13: Chem.BondType.IONIC,
            14: Chem.BondType.HYDROGEN,
            15: Chem.BondType.THREECENTER,
            16: Chem.BondType.DATIVEONE,
            17: Chem.BondType.DATIVE,
            18: Chem.BondType.DATIVEL,
            19: Chem.BondType.DATIVER,
            20: Chem.BondType.OTHER,
            21: Chem.BondType.ZERO,
        }

        # Add atoms
        for i in range(graph.x.size(0)):
            atomic_num = int(graph.x[i, 0].item())
            if atomic_num == 0:
                atomic_num = 6  # Default to carbon
            atom = Chem.Atom(atomic_num)

            # Formal charge: index 3, stored as value + 5 (range -5 to +6)
            if graph.x.size(1) > 3:
                formal_charge = int(graph.x[i, 3].item()) - 5
                atom.SetFormalCharge(formal_charge)

            mol.AddAtom(atom)

        # Add bonds
        added_bonds = set()
        if graph.edge_index is not None and graph.edge_index.numel() > 0:
            for j in range(graph.edge_index.size(1)):
                src = int(graph.edge_index[0, j].item())
                dst = int(graph.edge_index[1, j].item())

                if src == dst: continue
                bond_key = (min(src, dst), max(src, dst))
                if bond_key in added_bonds: continue
                added_bonds.add(bond_key)

                if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
                    bond_type_idx = int(graph.edge_attr[j, 0].item())
                    bond_type = BOND_TYPES.get(bond_type_idx, Chem.BondType.SINGLE)
                else:
                    bond_type = Chem.BondType.SINGLE

                mol.AddBond(src, dst, bond_type)

        molecule = mol.GetMol()
        smiles = Chem.MolToSmiles(molecule)
        return smiles if smiles else ""

    except Exception:
        return "SMILES_CONVERSION_ERROR"
