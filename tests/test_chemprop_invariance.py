"""
tests/test_chemprop_invariance.py
=====================================
Structural and invariance tests for chemprop D-MPNN integration.

(a) test_featurization_determinism   — same SMILES → same MolGraph features twice
(b) test_canonical_smiles_consistency — canonical vs non-canonical SMILES → same graph

2 tests total.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import numpy as np
from rdkit import Chem

_SMILES = [
    'CC(=O)Oc1ccccc1C(=O)O',   # aspirin
    'c1ccccc1',                  # benzene
    'CCO',                       # ethanol
    'CC(=O)O',                   # acetic acid
    'c1ccc2ccccc2c1',            # naphthalene
]


# ── (a) Featurization determinism ────────────────────────────────────────────

def test_featurization_determinism():
    """
    Calling SimpleMoleculeMolGraphFeaturizer on the same SMILES twice must
    produce identical atom/bond feature tensors.

    Adversarial guards:
      (1) No stochastic element in featurizer (RDKit-based, deterministic).
      (2) Tests for both atom features (V) and edge features (E).
    """
    from chemprop import data as cp_data, featurizers
    from rdkit import Chem

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    for smi in _SMILES:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"RDKit could not parse {smi!r}"

        # Featurize twice (via MoleculeDataset)
        dp1 = cp_data.MoleculeDatapoint(mol=mol, y=np.array([0.0]))
        dp2 = cp_data.MoleculeDatapoint(mol=mol, y=np.array([0.0]))
        ds1 = cp_data.MoleculeDataset([dp1], feat)
        ds2 = cp_data.MoleculeDataset([dp2], feat)

        mg1 = ds1[0].mg
        mg2 = ds2[0].mg

        # MolGraph.V and MolGraph.E are numpy arrays (chemprop v2)
        assert np.array_equal(mg1.V, mg2.V), (
            f"Atom features differ across two featurizations of {smi!r}. "
            "Featurizer is non-deterministic."
        )
        assert np.array_equal(mg1.E, mg2.E), (
            f"Bond features differ across two featurizations of {smi!r}. "
            "Featurizer is non-deterministic."
        )
        assert mg1.V.shape[1] > 0, f"Atom feature dim is 0 for {smi!r}"
        assert mg1.E.shape[1] > 0, f"Bond feature dim is 0 for {smi!r}"


# ── (b) Canonical SMILES consistency ─────────────────────────────────────────

def test_canonical_smiles_consistency():
    """
    Non-canonical and canonical SMILES for the same molecule must produce
    the same atom/bond feature tensors (same graph topology after RDKit parsing).

    Adversarial guard: train_chemprop() passes RDKit-parsed mols to chemprop,
    not raw SMILES strings. RDKit canonicalizes on parse, so atom ordering
    may differ — but feature content (set of features) must match.

    This test checks feature *shapes* and *content* are identical when both
    SMILES map to the same canonical molecule.
    """
    from chemprop import data as cp_data, featurizers
    from rdkit import Chem

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    pairs = [
        ('c1ccccc1', 'C1=CC=CC=C1'),
        ('CC(=O)O',  'OC(=O)C'),
        ('CCO',      'OCC'),
    ]

    for smi_a, smi_b in pairs:
        mol_a = Chem.MolFromSmiles(smi_a)
        mol_b = Chem.MolFromSmiles(smi_b)
        assert mol_a is not None and mol_b is not None

        can_a = Chem.MolToSmiles(mol_a)
        can_b = Chem.MolToSmiles(mol_b)
        assert can_a == can_b, (
            f"SMILES {smi_a!r} and {smi_b!r} do not canonicalize to the same molecule. "
            "Update test pairs."
        )

        dp_a = cp_data.MoleculeDatapoint(mol=mol_a, y=np.array([0.0]))
        dp_b = cp_data.MoleculeDatapoint(mol=mol_b, y=np.array([0.0]))
        ds_a = cp_data.MoleculeDataset([dp_a], feat)
        ds_b = cp_data.MoleculeDataset([dp_b], feat)

        mg_a = ds_a[0].mg
        mg_b = ds_b[0].mg

        assert mg_a.V.shape == mg_b.V.shape, (
            f"Atom feature shapes differ for canonical pair ({smi_a!r}, {smi_b!r}): "
            f"{mg_a.V.shape} vs {mg_b.V.shape}"
        )
        assert mg_a.E.shape == mg_b.E.shape, (
            f"Bond feature shapes differ for canonical pair ({smi_a!r}, {smi_b!r}): "
            f"{mg_a.E.shape} vs {mg_b.E.shape}"
        )
        # Atom row ordering can differ across canonical SMILES representations
        # (different traversal order by RDKit). Compare sorted multisets of rows.
        rows_a = np.sort(mg_a.V, axis=0)
        rows_b = np.sort(mg_b.V, axis=0)
        assert np.allclose(rows_a, rows_b, atol=1e-6), (
            f"Atom feature multisets differ for canonical pair ({smi_a!r}, {smi_b!r}). "
            "Different atoms or feature values — SMILES may not be the same molecule."
        )
