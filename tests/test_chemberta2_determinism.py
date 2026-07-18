"""
tests/test_chemberta2_determinism.py
=====================================
Seed determinism for ChemBERTa2Regressor.

(g) Two training runs with the same torch.manual_seed must produce bitwise-
    identical loss trajectories over 5 steps.

Strategy identical to test_chemberta_determinism.py:
  - model.eval() during training loop (disables all nn.Dropout)
  - freeze_encoder=True (no encoder backward, fast convergence)
  - dropout=0.0 in the head (explicit safety net)

CB-2-specific note: CB-2 has 3 encoder layers (not 6) and hidden=384.
LayerNorm-based architecture (same as CB-1), so eval() only gates dropout.

1 test.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F

ATOL_DET = 1e-6

_DET_SMILES = [
    'c1ccccc1',
    'CCO',
    'CC(=O)O',
    'c1cccnc1',
    'CC(=O)Nc1ccc(O)cc1',
    'c1ccc2ccccc2c1',
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'OC(=O)c1ccccc1',
]


def _build_batch(smiles, seed=0):
    from src.models import get_tokenizer_v2, tokenize_smiles_v2
    torch.manual_seed(seed)
    y   = torch.randn(len(smiles))
    tok = get_tokenizer_v2()
    ids, mask = tokenize_smiles_v2(smiles, tok)
    return ids, mask, y


def _run_steps(ids, mask, y, n, seed):
    from src.models import ChemBERTa2Regressor
    torch.manual_seed(seed)
    model = ChemBERTa2Regressor(dropout=0.0, freeze_encoder=True)
    model.eval()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    losses = []
    for _ in range(n):
        opt.zero_grad()
        out  = model(ids, mask)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ── (g) Seed determinism ─────────────────────────────────────────────────────

def test_seed_determinism():
    """
    Two runs with seed=123 and identical data produce the same loss at every
    step (within 1e-6).

    Integrity guards:
      (1) initial loss > 1e-6 — non-trivial target values
      (2) loss must change over 5 steps — head is actually training
    """
    ids, mask, y = _build_batch(_DET_SMILES, seed=77)

    losses_a = _run_steps(ids, mask, y, n=5, seed=123)
    losses_b = _run_steps(ids, mask, y, n=5, seed=123)

    assert losses_a[0] > 1e-6, (
        f"Initial loss={losses_a[0]:.2e} is near zero."
    )
    assert losses_a[0] != losses_a[-1], (
        f"Loss unchanged over 5 steps — regressor head may not be training."
    )

    for step, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < ATOL_DET, (
            f"Non-determinism at step {step}: "
            f"run_a={la:.10f}, run_b={lb:.10f}  (Δ={abs(la-lb):.2e})\n"
            "Cause: dropout not gated by self.training, weight init not seeded, "
            "or Adam moments diverging."
        )
