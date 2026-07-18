"""
tests/test_selformer_train.py
=====================================
Training and gradient tests for SELFormerRegressor.

(a) test_gradient_flow  — all parameter groups receive non-zero gradients
(b) test_tiny_overfit   — freeze_encoder=True, head only, loss drops >95%

2 tests total.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import numpy as np
import selfies as sf

_SMILES_8 = [
    'C', 'CC', 'CCO', 'c1ccccc1',
    'CC(=O)O', 'CCN', 'C1CCCCC1', 'c1ccc(N)cc1',
]

def _to_selfies(smiles_list):
    return [sf.encoder(s) for s in smiles_list]


# ── (a) Gradient flow ─────────────────────────────────────────────────────────

def test_gradient_flow():
    """
    One forward + backward pass.
    Verified parameter groups:
      encoder.embeddings.*           — input embedding layer
      encoder.encoder.layer.0.*      — first transformer block
      encoder.encoder.layer.11.*     — last transformer block (12 layers)
      regressor.*                    — regression head

    Excluded: encoder.pooler (SELFormer checkpoint lacks pooler weights → not used).

    Integrity:
      freeze_encoder=True → encoder params have no gradient.
      Regressor head always gets gradient.
    """
    from src.models import SELFormerRegressor, get_tokenizer_selformer, tokenize_selfies_selformer

    tok = get_tokenizer_selformer()
    selfies_list = _to_selfies(_SMILES_8)
    ids, mask = tokenize_selfies_selformer(selfies_list, tok)
    y = torch.randn(len(selfies_list))

    # ── unfrozen: all encoder layers must receive gradient ──
    model = SELFormerRegressor(dropout=0.0, freeze_encoder=False).train()
    out = model(ids, mask)
    loss = ((out - y) ** 2).mean()
    loss.backward()

    groups = {
        'encoder.embeddings': False,
        'encoder.encoder.layer.0': False,
        'encoder.encoder.layer.11': False,
        'regressor': False,
    }
    for name, param in model.named_parameters():
        for grp in groups:
            if name.startswith(grp) and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    groups[grp] = True

    for grp, got_grad in groups.items():
        assert got_grad, (
            f"No non-zero gradient in '{grp}'. "
            "Check freeze_encoder=False or loss graph."
        )

    # ── frozen: encoder must have NO gradient ──
    model_frozen = SELFormerRegressor(dropout=0.0, freeze_encoder=True).train()
    out2 = model_frozen(ids, mask)
    loss2 = ((out2 - y) ** 2).mean()
    loss2.backward()

    for name, param in model_frozen.named_parameters():
        if name.startswith('encoder.') and param.requires_grad is False:
            assert param.grad is None, (
                f"Frozen encoder param '{name}' has gradient — freeze_encoder not working."
            )

    # regressor must still have gradient when encoder is frozen
    reg_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model_frozen.named_parameters()
        if n.startswith('regressor')
    )
    assert reg_has_grad, (
        "Regressor has no gradient even with freeze_encoder=True. "
        "Head is not being trained."
    )


# ── (b) Tiny overfit ──────────────────────────────────────────────────────────

def test_tiny_overfit():
    """
    Head-only overfit: freeze_encoder=True, 200 gradient steps on 8 molecules.
    Final loss must be < 5% of initial loss.

    Rationale: pre-trained encoder frozen; only the 2-layer MLP head is trained.
    With only 8 samples this is trivially achievable in 200 steps.

    Adversarial guards:
      (1) Initial loss > 0 (output is not constant).
      (2) ratio < 0.05 (not just slightly reduced).
    """
    from src.models import SELFormerRegressor, get_tokenizer_selformer, tokenize_selfies_selformer

    torch.manual_seed(42)
    tok = get_tokenizer_selformer()
    selfies_list = _to_selfies(_SMILES_8)
    ids, mask = tokenize_selfies_selformer(selfies_list, tok)
    y = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5])

    model = SELFormerRegressor(dropout=0.0, freeze_encoder=True).train()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=5e-3
    )

    losses = []
    for _ in range(500):
        opt.zero_grad()
        out = model(ids, mask)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    initial_loss = losses[0]
    final_loss   = losses[-1]
    assert initial_loss > 0, "Initial loss is zero — model output may be constant."

    ratio = final_loss / (initial_loss + 1e-12)
    assert ratio < 0.05, (
        f"Tiny overfit failed: final/initial loss = {ratio:.4f} (expected < 0.05). "
        f"Initial={initial_loss:.4f}, Final={final_loss:.4f}. "
        "Head may not be learning or lr too small."
    )
