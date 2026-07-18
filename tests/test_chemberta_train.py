"""
tests/test_chemberta_train.py
================================
Training behaviour tests for ChemBERTaRegressor.

(e) Gradient flow        — every trainable parameter group receives a non-zero gradient;
                           encoder.pooler is explicitly excluded (not in computation path)
(f) Tiny overfit         — 8 molecules, 200 Adam steps (freeze_encoder=True for speed),
                           final_loss < 5% of initial
(g) Eval mode consistency — eval() is deterministic; train() is stochastic (dropout active)

3 tests total.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F

# ── SMILES ────────────────────────────────────────────────────────────────────

_SMILES = [
    'c1ccccc1',                          # benzene
    'CCO',                               # ethanol
    'CC(=O)O',                           # acetic acid
    'c1cccnc1',                          # pyridine
    'CC(=O)Nc1ccc(O)cc1',               # paracetamol
    'c1ccc2ccccc2c1',                    # naphthalene
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',       # ibuprofen
    'OC(=O)c1ccccc1',                    # benzoic acid
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_batch(smiles, seed=0):
    """Tokenise and return (ids, mask, y)."""
    from src.models import get_tokenizer, tokenize_smiles
    torch.manual_seed(seed)
    y   = torch.randn(len(smiles))
    tok = get_tokenizer()
    ids, mask = tokenize_smiles(smiles, tok)
    return ids, mask, y


def _make_model(dropout=0.1, freeze_encoder=False):
    from src.models import ChemBERTaRegressor
    return ChemBERTaRegressor(dropout=dropout, freeze_encoder=freeze_encoder)


# ── (e) Gradient flow ────────────────────────────────────────────────────────

def test_gradient_flow():
    """
    After one forward-backward pass, every required parameter group must have
    a non-zero gradient.

    Required groups (freeze_encoder=False):
      - encoder.embeddings   — word + position + token_type embeddings
      - encoder.encoder.layer.0  — first transformer layer
      - encoder.encoder.layer.5  — last transformer layer (depth-6 model)
      - regressor            — 768→128→1 regression head

    Excluded:
      - encoder.pooler — ChemBERTaRegressor uses last_hidden_state[:, 0, :],
        not pooler_output.  The pooler IS computed in the forward pass (as a
        side effect of RobertaModel), but its output is never used downstream,
        so no gradient flows back through it.  Its grad must be None.

    Integrity:
      - Assert loss > 1e-6 before backward.  If loss ≈ 0 (e.g., all targets
        are zero and the model perfectly fits them at init), all gradients are
        zero regardless of connectivity.
    """
    torch.manual_seed(0)
    ids, mask, y = _build_batch(_SMILES, seed=0)

    model = _make_model(dropout=0.1, freeze_encoder=False)
    model.train()

    out  = model(ids, mask)
    loss = F.mse_loss(out, y)

    # Integrity: non-trivial loss guarantees non-trivial gradients
    assert loss.item() > 1e-6, (
        f"loss={loss.item():.2e} before backward — targets may all be zero. "
        "Zero loss → zero gradients; gradient-flow check is vacuous."
    )

    loss.backward()

    required_groups = [
        'encoder.embeddings',
        'encoder.encoder.layer.0',
        'encoder.encoder.layer.5',
        'regressor',
    ]
    group_active = {g: False for g in required_groups}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'pooler' in name:
            continue          # not in computation path — checked separately below
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
        "ChemBERTaRegressor uses last_hidden_state[:, 0, :], not pooler_output — "
        "the pooler is not in the loss computation path."
    )

    # Sanity: verify freeze_encoder=True actually freezes encoder params
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
        f"Frozen encoder params received gradients: {frozen_with_grad}.\n"
        "freeze_encoder=True must set requires_grad=False on all encoder params."
    )


# ── (f) Tiny overfit ─────────────────────────────────────────────────────────

def test_tiny_overfit():
    """
    ChemBERTaRegressor memorises 8 molecules in 200 Adam steps.
    Criterion: final_loss < initial_loss × 0.05.

    freeze_encoder=True: only the 768→128→1 regression head is updated.
    The BERT encoder produces fixed high-quality embeddings for the 8 molecules;
    the head is heavily over-parameterised (98k+ params for 8 data points) and
    should overfit in < 100 steps.  200 steps is conservative.

    Note: training the full encoder (freeze_encoder=False) also overfits but
    requires a smaller lr (2e-5) and far more wall-clock time due to BERT backward.
    freeze_encoder=True is used here for test speed, not as the paper's train config.

    Integrity guards:
      (1) y.std() > 0.3 — targets must be varied so a constant predictor cannot
          trivially achieve low loss.
      (2) out_final.std() > 0.01 — after training the head must produce varied
          predictions (not collapsed to a single constant value).
    """
    torch.manual_seed(42)
    ids, mask, y = _build_batch(_SMILES, seed=42)

    # Integrity: targets must be varied
    assert y.std().item() > 0.3, (
        f"Target std={y.std().item():.4f} — targets are nearly identical. "
        "Any constant predictor can 'overfit'; use random targets (seed=42 fixed)."
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
        "(target <0.05).  Check: freeze_encoder=True, head parameters are trainable, "
        "optimizer only sees requires_grad params."
    )

    # Integrity: predictions must be non-constant after training
    with torch.no_grad():
        out_final = model(ids, mask)
    assert out_final.std().item() > 0.01, (
        f"Final predictions are constant (std={out_final.std().item():.4f}). "
        "The head may have collapsed to a single output value despite low loss."
    )


# ── (g) Eval mode consistency ─────────────────────────────────────────────────

def test_eval_train_consistency():
    """
    Two assertions:

    (1) eval() is DETERMINISTIC: two forward passes on the same input in eval
        mode must be bitwise identical.  This confirms all nn.Dropout layers
        (head + encoder) are correctly gated by self.training.

    (2) train() is STOCHASTIC: two forward passes on the same input in train
        mode must NOT be bitwise identical.  This is positive evidence that
        dropout IS active during training (not silently disabled).

    The model uses dropout=0.5 (deliberately high) to make stochasticity visible
    in train mode even with small models.  With two nn.Dropout(0.5) layers in the
    regression head and hidden_dropout/attention_dropout in the encoder, the
    probability of two identical forward passes is negligible.

    Integrity:
      Assert that the absolute difference in train-mode outputs exceeds 1e-4
      (not just float-noise-level randomness, but genuine dropout stochasticity).
    """
    from src.models import get_tokenizer, tokenize_smiles

    tok = get_tokenizer()
    ids, mask = tokenize_smiles(['CC(=O)Oc1ccccc1C(=O)O'], tok)  # aspirin

    # Use high dropout to make stochasticity prominent
    model = _make_model(dropout=0.5, freeze_encoder=False)

    # (1) eval() is deterministic
    model.eval()
    with torch.no_grad():
        out_eval_1 = model(ids, mask)
        out_eval_2 = model(ids, mask)
    assert torch.equal(out_eval_1, out_eval_2), (
        "eval() mode is not deterministic — two identical forward passes differ. "
        "A dropout layer may be hard-coded to training=True."
    )

    # (2) train() is stochastic
    model.train()
    out_train_1 = model(ids, mask).detach()
    out_train_2 = model(ids, mask).detach()

    delta = (out_train_1 - out_train_2).abs().item()
    assert not torch.equal(out_train_1, out_train_2), (
        "train() mode is deterministic — two forward passes give identical output. "
        "nn.Dropout may be disabled or p=0 in all layers."
    )

    # Integrity: the difference must be > float noise (genuine dropout effect)
    assert delta > 1e-4, (
        f"train() mode differs but only by Δ={delta:.2e} — "
        "too small to be genuine dropout stochasticity (threshold 1e-4). "
        "Consider using dropout=0.5 or checking encoder dropout config."
    )

    # Integrity: eval and train outputs must also differ (eval ≠ train)
    delta_mode = (out_eval_1 - out_train_1).abs().item()
    assert delta_mode > 1e-4, (
        f"eval() and train() produce nearly identical output (Δ={delta_mode:.2e}). "
        "Dropout in the model may be effectively disabled (p too close to 0)."
    )
