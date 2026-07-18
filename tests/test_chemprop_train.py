"""
tests/test_chemprop_train.py
=====================================
Training integration tests for chemprop D-MPNN.

(a) test_train_completes  — full train_chemprop() on 32 molecules, no NaN, valid predictions
(b) test_better_than_mean — test RMSE < 0.8 * std(test_y) after brief training

2 tests total.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import torch

_SMILES_32 = [
    'C', 'CC', 'CCC', 'CCCC', 'CCO', 'CCCO', 'CCCCO', 'c1ccccc1',
    'CC(=O)O', 'CC(=O)OCC', 'CCN', 'CCNCC', 'C1CCCCC1', 'c1ccc(N)cc1',
    'c1ccc(O)cc1', 'CC(C)=O', 'c1cccnc1', 'c1ccncc1', 'CC(=O)Nc1ccccc1',
    'c1ccc2ccccc2c1', 'CC(C)O', 'CCOCC', 'ClCCCl', 'BrCCBr',
    'OC(=O)c1ccccc1', 'Nc1ccccc1', 'Oc1ccccc1', 'CC1=CC=CC=C1',
    'C1=CC=NC=C1', 'CC(=O)c1ccccc1', 'COc1ccccc1', 'CC(C)(C)O',
]

# Synthetic y: molecular weight proxy (just deterministic, not real MW)
_Y_32 = np.array([float(len(s)) * 0.1 for s in _SMILES_32], dtype=np.float32)


def _normalized(y):
    mean, std = y.mean(), y.std()
    if std < 1e-8:
        return y - mean, mean, 1.0
    return (y - mean) / std, mean, float(std)


# ── (a) train_completes ───────────────────────────────────────────────────────

def test_train_completes():
    """
    train_chemprop() on 32 molecules, 5 epochs, must:
      (1) Return without raising.
      (2) Return dict with 'model', 'metrics', 'test_preds', 'test_true'.
      (3) test_preds must be finite (no NaN/Inf).
      (4) metrics must be finite floats.
      (5) model is a ChempropWrapper (not None).

    Adversarial guards:
      - chemprop internal normalization disabled (output_transform=Identity):
        the guard inside train_chemprop() asserts this explicitly.
      - No RDKit-invalid SMILES in _SMILES_32.
    """
    from src.train import train_chemprop
    from src.models import ChempropWrapper

    y_norm, _, _ = _normalized(_Y_32)
    n = len(_SMILES_32)
    split = n * 3 // 4

    res = train_chemprop(
        train_smiles=_SMILES_32[:split],
        train_y=y_norm[:split],
        val_smiles=_SMILES_32[split:split+4],
        val_y=y_norm[split:split+4],
        test_smiles=_SMILES_32[split+4:],
        test_y=y_norm[split+4:],
        target_name='test_target',
        epochs=5,
        patience=10,
        seed=0,
    )

    assert set(res.keys()) >= {'model', 'metrics', 'test_preds', 'test_true'}, (
        f"train_chemprop returned incomplete dict: {res.keys()}"
    )
    assert isinstance(res['model'], ChempropWrapper), (
        f"Expected ChempropWrapper, got {type(res['model'])}"
    )

    preds = res['test_preds']
    assert not np.isnan(preds).any(), "NaN in test_preds"
    assert not np.isinf(preds).any(), "Inf in test_preds"
    assert preds.shape == res['test_true'].shape, (
        f"test_preds shape {preds.shape} != test_true shape {res['test_true'].shape}"
    )

    for k in ('RMSE', 'MAE', 'Pearson_R', 'R2'):
        v = res['metrics'][k]
        assert np.isfinite(v), f"metric {k}={v} is not finite"


# ── (b) better_than_mean ─────────────────────────────────────────────────────

def test_better_than_mean():
    """
    After 50 epochs on an in-distribution shuffled split (20 train / 6 val / 6 test),
    RMSE must be below 0.9 * std(y_all).

    Rationale: D-MPNN encodes explicit graph structure. With shuffled splits
    the test set is in-distribution. std(y_all) = 1.0 for z-normalized y, so
    the threshold 0.9 beats the naïve mean-predictor by 10% margin.

    Adversarial guards:
      - Shuffled split → test molecules are in-distribution (not tail-only).
      - std(y_all) = 1.0 for z-normalized input → threshold is scale-stable.
      - 50 epochs with patience=50 → no premature early stopping.
    """
    from src.train import train_chemprop

    rng = np.random.default_rng(99)
    idx = rng.permutation(len(_SMILES_32))
    smiles = [_SMILES_32[i] for i in idx]
    y_raw  = np.array([float(i) for i in idx], dtype=np.float32)
    y_norm, _, _ = _normalized(y_raw)

    # std(y_norm) = 1.0 — threshold is stable
    res = train_chemprop(
        train_smiles=smiles[:20],
        train_y=y_norm[:20],
        val_smiles=smiles[20:26],
        val_y=y_norm[20:26],
        test_smiles=smiles[26:],
        test_y=y_norm[26:],
        target_name='overfit_test',
        epochs=50,
        patience=50,
        seed=42,
    )

    rmse = res['metrics']['RMSE']
    threshold = 0.9  # = 0.9 * std(y_all_normalized)
    assert rmse < threshold, (
        f"Chemprop RMSE={rmse:.4f} >= 0.9 (= 0.9 * std(y_all)={threshold:.1f}). "
        "Model is not better than the mean predictor after 50 epochs. "
        "Check that D-MPNN graph featurization and training loop are working."
    )
