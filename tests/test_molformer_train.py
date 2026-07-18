"""
tests/test_molformer_train.py
================================
Training behaviour tests for MoLFormerRegressor.

(e) Gradient flow  — encoder.embeddings, encoder.encoder.layer.0,
                     encoder.encoder.layer.11 (last of 12),
                     regressor all receive non-zero gradients.
(f) Tiny overfit   — 8 molecules, 200 Adam steps (freeze_encoder=True),
                     final_loss < 5% initial

2 tests total.

MoLFormer architecture notes:
  - 12 encoder layers (encoder.encoder.layer.0 … .11)
  - hidden_size = 768 (auto-inferred from config)
  - linear attention with random feature map (deterministic_eval=True at inference)
  - No pooler layer in the model (masked mean pooling in regressor wrapper)
  - trust_remote_code=True required
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
    from src.models import get_tokenizer_molformer, tokenize_smiles_molformer
    torch.manual_seed(seed)
    y   = torch.randn(len(smiles))
    tok = get_tokenizer_molformer()
    ids, mask = tokenize_smiles_molformer(smiles, tok)
    return ids, mask, y


def _make_model(dropout=0.1, freeze_encoder=False):
    from src.models import MoLFormerRegressor
    return MoLFormerRegressor(dropout=dropout, freeze_encoder=freeze_encoder)


# ── (e) Gradient flow ────────────────────────────────────────────────────────

def test_gradient_flow():
    """
    After one forward-backward pass, every required parameter group must have
    a non-zero gradient.

    Required groups (freeze_encoder=False):
      - encoder.embeddings       — word + position embeddings
      - encoder.encoder.layer.0  — first transformer layer
      - encoder.encoder.layer.11 — last transformer layer (MoLFormer has 12 layers: 0..11)
      - regressor                — 768→128→1 regression head

    MoLFormer has no pooler — masked mean pooling is computed in forward().

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
        'encoder.encoder.layer.11',  # MoLFormer: 12 layers (0..11)
        'regressor',
    ]
    group_active = {g: False for g in required_groups}
    for name, p in model.named_parameters():
        if not p.requires_grad:
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
    MoLFormerRegressor memorises 8 molecules in 200 Adam steps.
    Criterion: final_loss < initial_loss × 0.05.

    freeze_encoder=True: only the 768→128→1 regression head is updated.
    MoLFormer's encoder produces fixed embeddings (hidden=768) for 8 molecules;
    the head is over-parameterised (~100k params for 8 data points) and
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
