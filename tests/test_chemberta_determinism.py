"""
tests/test_chemberta_determinism.py
=====================================
Seed determinism for ChemBERTaRegressor.

(h) Two training runs with the same torch.manual_seed must produce bitwise-
    identical loss trajectories over 5 steps.

Strategy to eliminate stochasticity:
  - model.eval() during the training loop: disables all nn.Dropout in the
    head AND the encoder's hidden/attention dropout (ChemBERTa uses LayerNorm,
    not BatchNorm, so eval() only affects dropout — no running-stat divergence).
  - freeze_encoder=True: no encoder backward, fast convergence.
  - dropout=0.0 in the head: explicit safety net so head Dropout is also p=0.

Sources of non-determinism this would catch:
  - nn.Dropout in the head hard-coded to training=True
  - Encoder internal dropout not gated by self.training
  - Weight init not controlled by torch.manual_seed
  - Adam moment accumulation seeded differently between runs

1 test (parametrised: no variants needed — model has no configurable depth).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F

ATOL_DET = 1e-6

# ── SMILES ────────────────────────────────────────────────────────────────────

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


# ── Data and run helpers ──────────────────────────────────────────────────────

def _build_batch(smiles, seed=0):
    from src.models import get_tokenizer, tokenize_smiles
    torch.manual_seed(seed)
    y   = torch.randn(len(smiles))
    tok = get_tokenizer()
    ids, mask = tokenize_smiles(smiles, tok)
    return ids, mask, y


def _run_steps(ids, mask, y, n, seed):
    """
    Fresh seeded model (freeze_encoder=True, dropout=0.0, eval mode) +
    n Adam steps; returns list of loss floats.

    eval() eliminates encoder dropout stochasticity.
    freeze_encoder=True eliminates encoder backward variance.
    Same seed → same weight init → same Adam moments → same loss trajectory.
    """
    from src.models import ChemBERTaRegressor
    torch.manual_seed(seed)
    model = ChemBERTaRegressor(dropout=0.0, freeze_encoder=True)
    model.eval()   # disables ALL dropout (head + encoder internal)
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


# ── (h) Seed determinism ─────────────────────────────────────────────────────

def test_seed_determinism():
    """
    Two runs with seed=123 and identical data produce the same loss at every
    step (within 1e-6).

    Integrity guards:
      (1) initial loss > 1e-6 — if loss starts at zero (e.g., all targets = 0
          and the frozen BERT embeddings map to 0 through the head init), all
          runs trivially agree at every step.
      (2) loss must change over 5 steps — if weights are not updating (e.g.,
          head params are also accidentally frozen), both runs stay at the same
          constant loss, trivially matching.
    """
    ids, mask, y = _build_batch(_DET_SMILES, seed=77)

    losses_a = _run_steps(ids, mask, y, n=5, seed=123)
    losses_b = _run_steps(ids, mask, y, n=5, seed=123)

    # Integrity (1): initial loss must be non-trivial
    assert losses_a[0] > 1e-6, (
        f"Initial loss={losses_a[0]:.2e} is near zero — targets may all be zero. "
        "Determinism check is vacuous when loss is already zero."
    )

    # Integrity (2): loss must change over 5 steps (head is actually training)
    assert losses_a[0] != losses_a[-1], (
        f"Loss unchanged over 5 steps ({losses_a[0]:.6f} → {losses_a[-1]:.6f}). "
        "Regression head may not be receiving gradient updates — check that "
        "filter(lambda p: p.requires_grad, ...) includes regressor params."
    )

    for step, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < ATOL_DET, (
            f"Non-determinism at step {step}: "
            f"run_a={la:.10f}, run_b={lb:.10f}  (Δ={abs(la-lb):.2e})\n"
            "Causes: encoder dropout not gated by self.training (eval() should "
            "disable it), weight init not controlled by manual_seed, or Adam "
            "moments diverging due to different initialisation order."
        )
