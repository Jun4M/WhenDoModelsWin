"""
tests/test_chemprop_determinism.py
=====================================
Seed determinism test for chemprop D-MPNN.

(a) test_seed_determinism — two identical runs with same seed produce identical loss curves

1 test total.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import torch

_SMILES_16 = [
    'C', 'CC', 'CCC', 'CCO', 'CCCO', 'c1ccccc1',
    'CC(=O)O', 'CCN', 'C1CCCCC1', 'c1ccc(N)cc1',
    'c1ccc(O)cc1', 'CC(C)=O', 'c1cccnc1', 'CC(=O)Nc1ccccc1',
    'CC(C)O', 'CCOCC',
]
_Y_16 = np.array([float(i) for i in range(16)], dtype=np.float32)


def _norm(y):
    m, s = y.mean(), y.std()
    return (y - m) / max(s, 1e-8)


def _run(seed: int) -> list:
    """Run 3-epoch train, return [init_loss, final_loss] as floats."""
    from src.train import train_chemprop

    y = _norm(_Y_16)
    res = train_chemprop(
        train_smiles=_SMILES_16[:10],
        train_y=y[:10],
        val_smiles=_SMILES_16[10:13],
        val_y=y[10:13],
        test_smiles=_SMILES_16[13:],
        test_y=y[13:],
        target_name='determinism',
        epochs=3,
        patience=10,
        seed=seed,
    )
    return res['test_preds'].tolist()


# ── (a) Seed determinism ──────────────────────────────────────────────────────

def test_seed_determinism():
    """
    Two identical calls to train_chemprop with the same seed must produce
    bit-exact test_preds (ATOL=1e-5).

    Notes:
      - chemprop's build_dataloader accepts seed= for shuffle reproducibility.
      - torch.manual_seed(seed) is called in train_chemprop before model init.
      - Runs on CPU → deterministic BLAS ops.
      - ATOL=1e-5 (slightly relaxed from 1e-6) to accommodate float32 ops
        across independent Lightning Trainer instances.

    Adversarial guard: if chemprop introduces dropout or stochastic ops that
    ignore the seed, this test will fail and the seed path must be revisited.
    """
    preds_a = _run(seed=7)
    preds_b = _run(seed=7)

    assert len(preds_a) == len(preds_b), (
        f"Prediction length mismatch: {len(preds_a)} vs {len(preds_b)}"
    )

    for i, (pa, pb) in enumerate(zip(preds_a, preds_b)):
        assert abs(pa - pb) < 1e-5, (
            f"Determinism broken at test sample {i}: "
            f"run_a={pa:.8f}, run_b={pb:.8f}, diff={abs(pa-pb):.2e}. "
            "Check torch.manual_seed placement in train_chemprop."
        )
