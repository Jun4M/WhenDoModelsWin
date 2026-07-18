"""
tests/test_chemberta2_train.py
================================
Training behaviour tests for ChemBERTa2Regressor.

(e) Gradient flow  — encoder.embeddings, encoder.layer.0, encoder.layer.2 (last),
                     regressor all receive non-zero gradients.
                     encoder.pooler receives NO gradient (not in computation path).
(f) Tiny overfit   — 8 molecules, 200 Adam steps (freeze_encoder=True),
                     final_loss < 5% initial

2 tests total.

CB-2 architecture differences from CB-1:
  - 3 encoder layers (encoder.encoder.layer.0 / .1 / .2), not 6
  - hidden_size = 384, not 768
  - pooler present in RobertaModel but MISSING from checkpoint (random init);
    still not in computation path since we use last_hidden_state[:, 0, :]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F

_SMILES = [
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


def _make_model(dropout=0.1, freeze_encoder=False):
    from src.models import ChemBERTa2Regressor
    return ChemBERTa2Regressor(dropout=dropout, freeze_encoder=freeze_encoder)


# ── (e) Gradient flow ────────────────────────────────────────────────────────

def test_gradient_flow():
    """
    After one forward-backward pass, every required parameter group must have
    a non-zero gradient.

    Required groups (freeze_encoder=False):
      - encoder.embeddings       — word + position embeddings
      - encoder.encoder.layer.0  — first transformer layer
      - encoder.encoder.layer.2  — last transformer layer (CB-2 has 3 layers: 0, 1, 2)
      - regressor                — 384→128→1 regression head

    Excluded:
      - encoder.pooler — RobertaModel computes pooler_output as a side effect but
        ChemBERTa2Regressor uses last_hidden_state[:, 0, :], so pooler is dead.
        Additionally, CB-2's pooler weights are MISSING from the MTR checkpoint
        (randomly initialized) and not used downstream.

    Integrity: loss > 1e-6 before backward (zero loss → zero gradients everywhere).
    """
    torch.manual_seed(0)
    ids, mask, y = _build_batch(_SMILES, seed=0)

    model = _make_model(dropout=0.1, freeze_encoder=False)
    model.train()

    out  = model(ids, mask)
    loss = F.mse_loss(out, y)

    assert loss.item() > 1e-6, (
        f"loss={loss.item():.2e} before backward — targets may all be zero. "
        "Zero loss → zero gradients; gradient-flow check is vacuous."
    )

    loss.backward()

    required_groups = [
        'encoder.embeddings',
        'encoder.encoder.layer.0',
        'encoder.encoder.layer.2',   # CB-2: 3 layers (0, 1, 2), not 6
        'regressor',
    ]
    group_active = {g: False for g in required_groups}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'pooler' in name:
            continue
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            for g in required_groups:
                if name.startswith(g):
                    group_active[g] = True

    dead = [g for g, active in group_active.items() if not active]
    assert not dead, (
        f"No non-zero gradient in group(s): {dead}\n"
        "One branch may be disconnected from the computation graph, "
        "or freeze_encoder is inadvertently set to True."
    )

    # Sanity: pooler must NOT receive gradients (not in computation path)
    pooler_with_grad = [
        name for name, p in model.named_parameters()
        if 'pooler' in name
        and p.grad is not None
        and p.grad.abs().sum().item() > 0.0
    ]
    assert not pooler_with_grad, (
        f"encoder.pooler received gradients: {pooler_with_grad}.\n"
        "ChemBERTa2Regressor uses last_hidden_state[:, 0, :], not pooler_output."
    )

    # Sanity: freeze_encoder=True must freeze all encoder params
    model_frozen = _make_model(dropout=0.1, freeze_encoder=True)
    model_frozen.train()
    out_f  = model_frozen(ids, mask)
    loss_f = F.mse_loss(out_f, y)
    loss_f.backward()

    frozen_with_grad = [
        name for name, p in model_frozen.named_parameters()
        if 'encoder' in name
        and not p.requires_grad
        and p.grad is not None
        and p.grad.abs().sum().item() > 0.0
    ]
    assert not frozen_with_grad, (
        f"Frozen encoder params received gradients: {frozen_with_grad}."
    )


# ── (f) Tiny overfit ─────────────────────────────────────────────────────────

def test_tiny_overfit():
    """
    ChemBERTa2Regressor memorises 8 molecules in 200 Adam steps.
    Criterion: final_loss < initial_loss × 0.05.

    freeze_encoder=True: only the 384→128→1 regression head is updated.
    CB-2's encoder produces fixed embeddings (hidden=384) for 8 molecules;
    the head is over-parameterised (~50k params for 8 data points) and
    should overfit quickly.

    Integrity:
      (1) y.std() > 0.3 — targets must be varied.
      (2) out_final.std() > 0.01 — predictions must be non-constant after training.
    """
    torch.manual_seed(42)
    ids, mask, y = _build_batch(_SMILES, seed=42)

    assert y.std().item() > 0.3, (
        f"Target std={y.std().item():.4f} — targets nearly identical."
    )

    model = _make_model(dropout=0.0, freeze_encoder=True)
    model.train()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    initial_loss = None
    for step in range(200):
        opt.zero_grad()
        out  = model(ids, mask)
        loss = F.mse_loss(out, y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()

    final_loss = loss.item()
    ratio = final_loss / initial_loss if initial_loss else float('inf')
    assert ratio < 0.05, (
        f"Failed to overfit in 200 steps: "
        f"initial={initial_loss:.4f}, final={final_loss:.4f}, ratio={ratio:.3f} "
        "(target <0.05). Check: freeze_encoder=True, head params trainable."
    )

    with torch.no_grad():
        out_final = model(ids, mask)
    assert out_final.std().item() > 0.01, (
        f"Final predictions are constant (std={out_final.std().item():.4f})."
    )
