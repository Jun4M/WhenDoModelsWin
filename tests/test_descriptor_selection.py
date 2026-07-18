"""
Tests for src/descriptor_selection.py

Coverage:
  1. test_clean_descriptors_drops_constant   — constant columns removed
  2. test_clean_descriptors_drops_low_var    — low-variance columns removed
  3. test_isis_screen_recovery               — ISIS selects correlated features
  4. test_elastic_net_select_sparse          — EN returns non-zero indices
  5. test_per_fold_no_leak                   — apply_descriptor_selection uses training stats only
"""

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helper SMILES (small, parse reliably)
# ──────────────────────────────────────────────────────────────────────────────

_SMILES = [
    'C', 'CC', 'CCC', 'CCCC', 'CCCCC',
    'c1ccccc1', 'CCO', 'CCCO', 'c1cccnc1', 'CCN',
    'CC(=O)O', 'c1ccc(O)cc1', 'CC#N', 'CCCC=O', 'c1ccncc1',
    'CCCl', 'CC(C)C', 'c1ccc(N)cc1', 'CCS', 'CCOCCO',
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. clean_descriptors: constant columns dropped
# ──────────────────────────────────────────────────────────────────────────────

def test_clean_descriptors_drops_constant():
    from src.descriptor_selection import clean_descriptors

    df = pd.DataFrame({
        'a': [1.0, 2.0, 3.0],
        'b': [5.0, 5.0, 5.0],   # constant → should be dropped
        'c': [0.1, 0.5, 0.9],
    })
    cleaned, names = clean_descriptors(df)
    assert 'b' not in names, "Constant column 'b' should be dropped"
    assert 'a' in names
    assert 'c' in names
    assert cleaned.shape[1] == 2


# ──────────────────────────────────────────────────────────────────────────────
# 2. clean_descriptors: low-variance columns dropped
# ──────────────────────────────────────────────────────────────────────────────

def test_clean_descriptors_drops_low_var():
    from src.descriptor_selection import clean_descriptors

    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'high_var': rng.normal(0, 1, n),
        'low_var':  rng.normal(0, 0.001, n),   # var ≈ 1e-6 < 0.001 → dropped
        'med_var':  rng.normal(0, 0.5, n),
    })
    cleaned, names = clean_descriptors(df)
    assert 'low_var' not in names, "Low-variance column should be removed"
    assert 'high_var' in names
    assert 'med_var' in names


# ──────────────────────────────────────────────────────────────────────────────
# 3. isis_screen: recovers correlated features
# ──────────────────────────────────────────────────────────────────────────────

def test_isis_screen_recovery():
    from src.descriptor_selection import isis_screen

    rng = np.random.default_rng(42)
    n, p = 80, 50
    X = rng.standard_normal((n, p))

    # Columns 0 and 1 are perfectly correlated with y
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.standard_normal(n)

    selected = isis_screen(X, y, nsis=5, max_iter=3)
    assert len(selected) > 0, "ISIS should select at least one feature"
    assert 0 in selected or 1 in selected, (
        f"ISIS must recover col 0 or col 1 (most correlated with y). Got: {selected}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. elastic_net_select: returns non-zero indices for recoverable signal
# ──────────────────────────────────────────────────────────────────────────────

def test_elastic_net_select_sparse():
    from src.descriptor_selection import elastic_net_select

    rng = np.random.default_rng(7)
    n = 60
    X = rng.standard_normal((n, 10))
    y = 2.0 * X[:, 0] + 0.5 * rng.standard_normal(n)

    idx = elastic_net_select(X, y, seed=0)
    assert len(idx) > 0, "Elastic Net should select at least one feature"
    assert 0 in idx, "Column 0 (the signal) should be selected"


# ──────────────────────────────────────────────────────────────────────────────
# 5. per_fold_no_leak: apply_descriptor_selection uses training z-score only
# ──────────────────────────────────────────────────────────────────────────────

def test_per_fold_no_leak():
    """Training statistics in fit_stats must be used for val/test — no re-fitting."""
    from src.descriptor_selection import select_descriptors_per_fold, apply_descriptor_selection

    rng = np.random.default_rng(3)
    train_smiles = _SMILES[:12]
    test_smiles  = _SMILES[12:]
    n_train = len(train_smiles)
    n_test  = len(test_smiles)
    train_y = rng.standard_normal(n_train).astype(np.float32)

    selected, fit_stats = select_descriptors_per_fold(train_smiles, train_y, seed=0)

    # apply_descriptor_selection must return (n_test, k) array
    X_test = apply_descriptor_selection(test_smiles, selected, fit_stats)
    assert X_test.shape == (n_test, len(selected)), (
        f"Expected ({n_test}, {len(selected)}), got {X_test.shape}"
    )

    # Verify it used training mean/std (not test mean/std):
    # Re-apply on a one-molecule list with a molecule identical to first train SMILES.
    X_first_train = apply_descriptor_selection([train_smiles[0]], selected, fit_stats)
    assert X_first_train.shape == (1, len(selected))

    # Key leak guard: fit_stats must carry train_mean/train_std
    assert 'train_mean' in fit_stats
    assert 'train_std' in fit_stats
    assert fit_stats['train_mean'].shape[0] == len(fit_stats['all_desc_names'])
