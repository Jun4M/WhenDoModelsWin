"""
Determinism test for KROVEX.

Test:
  1. test_seed_determinism — same seed → identical test_preds across two runs
"""

import numpy as np
import pytest
import torch


_SMILES = [
    'C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'c1ccccc1', 'CCO', 'CCCO',
    'c1cccnc1', 'CCN', 'CC(=O)O', 'c1ccc(O)cc1', 'CCCl', 'CC(C)C',
    'c1ccc(N)cc1', 'CCS', 'CC#N', 'CCCC=O', 'c1ccncc1', 'CCOCCO',
]


def _run(smiles_list, seed, epochs=3):
    from src.train import train_krovex
    n = len(smiles_list)
    n_train = int(n * 0.6)
    n_val   = int(n * 0.2)

    rng = np.random.default_rng(seed)
    y_all = rng.standard_normal(n).astype(np.float32)

    train_smi = smiles_list[:n_train]
    val_smi   = smiles_list[n_train:n_train + n_val]
    test_smi  = smiles_list[n_train + n_val:]
    train_y   = y_all[:n_train]
    val_y     = y_all[n_train:n_train + n_val]
    test_y    = y_all[n_train + n_val:]

    res = train_krovex(
        train_smi, train_y, val_smi, val_y, test_smi, test_y,
        target_name='test_target',
        epochs=epochs,
        batch_size=8,
        lr=1e-3,
        patience=epochs + 1,
        device='cpu',
        seed=seed,
    )
    return res['test_preds']


def test_seed_determinism():
    """Two runs with the same seed must produce identical test predictions (ATOL=1e-5)."""
    preds_a = _run(_SMILES, seed=7, epochs=3)
    preds_b = _run(_SMILES, seed=7, epochs=3)

    assert preds_a.shape == preds_b.shape, (
        f"Shape mismatch: {preds_a.shape} vs {preds_b.shape}"
    )
    assert np.allclose(preds_a, preds_b, atol=1e-5), (
        f"Non-deterministic: max diff = {np.abs(preds_a - preds_b).max():.2e}"
    )
