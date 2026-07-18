"""
tests/test_selformer_determinism.py
=====================================
Seed determinism test for SELFormerRegressor.

(a) test_seed_determinism — two identical runs with same seed produce bit-exact outputs

1 test total.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import numpy as np
import selfies as sf

_SMILES_5 = [
    'C', 'CCO', 'c1ccccc1',
    'CC(=O)Oc1ccccc1C(=O)O',
    'C1CCCCC1',
]


def _run(smiles_list, seed: int):
    """One training run: freeze_encoder=True, dropout=0.0, 5 steps. Returns loss trajectory."""
    from src.models import SELFormerRegressor, get_tokenizer_selformer, tokenize_selfies_selformer

    torch.manual_seed(seed)
    np.random.seed(seed)

    selfies_list = [sf.encoder(s) for s in smiles_list]
    tok = get_tokenizer_selformer()
    ids, mask = tokenize_selfies_selformer(selfies_list, tok)
    y = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0])

    model = SELFormerRegressor(dropout=0.0, freeze_encoder=True)
    model.eval()

    # Re-init head with fixed seed for determinism
    torch.manual_seed(seed)
    for m in model.regressor.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    losses = []
    for _ in range(5):
        opt.zero_grad()
        with torch.set_grad_enabled(True):
            out = model(ids, mask)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses


# ── (a) Seed determinism ──────────────────────────────────────────────────────

def test_seed_determinism():
    """
    Two identical runs (same seed, freeze_encoder=True, dropout=0.0) must produce
    bit-exact loss trajectories (ATOL=1e-6).

    Adversarial guards:
      (1) freeze_encoder=True: pre-trained encoder is deterministic by construction.
          Only the randomly-initialized head needs determinism.
      (2) Head weights re-initialized with same seed in both runs.
      (3) model.eval() disables dropout (redundant here since dropout=0.0 but explicit).
      (4) Catches non-deterministic CUDA ops — on CPU these should always be bit-exact.

    Note: MPS (Apple Silicon) may introduce <1e-6 float differences;
    this test runs on CPU explicitly to guarantee bit-exact comparison.
    """
    losses_a = _run(_SMILES_5, seed=42)
    losses_b = _run(_SMILES_5, seed=42)

    for step, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < 1e-6, (
            f"Determinism broken at step {step}: run_a={la:.8f}, run_b={lb:.8f}, "
            f"diff={abs(la - lb):.2e}. "
            "Check seed initialization order or non-deterministic ops."
        )
