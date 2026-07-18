"""
tests/test_painn_train.py
==========================
Trainability tests for PaiNNRegressor:

  (1) gradient_flow  — single forward/backward on 8 samples,
                        no NaN/Inf, every parameter grad nonzero
  (2) tiny_overfit   — 32-sample memorization, 300-step Adam,
                        final loss < 5 % of initial loss

All data is synthetic (no disk I/O).

Run:
    pytest tests/test_painn_train.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import torch
import torch.nn as nn
import pytest

from src.models import PaiNNRegressor

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic molecule builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_batch(
    n_mols: int,
    atoms_per_mol: int = 5,
    radius: float = 2.0,
    seed: int = 0,
    dtype=torch.float32,
    device: str = "cpu",
):
    """
    Build a synthetic batch of n_mols molecules.

    Each molecule has `atoms_per_mol` atoms placed uniformly inside a sphere
    of `radius` Å (well within the 5 Å cutoff so edges always form).

    Returns
    -------
    x      : (N, 30)   atom features;  x[:,0] = atom-type index in [1, 8]
    pos    : (N, 3)    3-D coordinates
    batch  : (N,)      molecule index  0 … n_mols-1
    target : (n_mols,) random scalar targets ∈ [-1, 1]
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    N = n_mols * atoms_per_mol
    # Positions: uniform in [-radius, radius]³, then clamp inside sphere
    raw_pos = torch.empty(N, 3, dtype=dtype, device=device).uniform_(-radius, radius, generator=rng)
    norms = raw_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    # Keep points inside the sphere (rescale those outside)
    scale = torch.where(norms > radius, radius / norms, torch.ones_like(norms))
    pos = raw_pos * scale

    x = torch.zeros(N, 30, dtype=dtype, device=device)
    atom_types = torch.randint(1, 9, (N,), generator=rng, dtype=dtype, device=device)
    x[:, 0] = atom_types

    batch = torch.arange(n_mols, device=device).repeat_interleave(atoms_per_mol)

    target = torch.empty(n_mols, dtype=dtype, device=device).uniform_(-1.0, 1.0, generator=rng)

    return x, pos, batch, target


def _make_model(hidden: int = 64, layers: int = 3, seed: int = 42):
    torch.manual_seed(seed)
    return PaiNNRegressor(hidden_channels=hidden, num_layers=layers,
                          cutoff=5.0, num_rbf=20)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — gradient flow
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow():
    """
    Single forward → MSE loss → backward on 8 samples.

    Checks:
    - Loss is finite (not NaN, not Inf)
    - Every leaf parameter has a nonzero gradient
    """
    model = _make_model()
    model.train()

    x, pos, batch, target = _make_batch(n_mols=8, atoms_per_mol=5, seed=1)

    out  = model(x, pos, batch)
    loss = nn.functional.mse_loss(out, target)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()

    print("\n--- per-parameter gradient norms ---")
    zero_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            gnorm = param.grad.norm().item()
            print(f"  {name:50s}  |g|={gnorm:.4e}")
            if gnorm == 0.0:
                zero_grad_params.append(name)

    assert len(zero_grad_params) == 0, (
        f"Zero-gradient parameters: {zero_grad_params}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — tiny overfit
# ─────────────────────────────────────────────────────────────────────────────

OVERFIT_STEPS = 300
OVERFIT_RATIO = 0.05   # final loss must be < 5 % of initial loss


def test_tiny_overfit():
    """
    Memorise 32 synthetic molecules in 300 Adam steps.

    Checks:
    - final_loss < initial_loss * 0.05  (95 % reduction → model can learn)
    """
    model = _make_model(seed=0)
    model.train()

    x, pos, batch, target = _make_batch(n_mols=32, atoms_per_mol=5, seed=7)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(OVERFIT_STEPS):
        optimizer.zero_grad()
        out  = model(x, pos, batch)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
            print(f"\n  step {step:4d}  loss={loss.item():.6f}  (initial)")
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}  loss={loss.item():.6f}")

    final_loss = loss.item()
    threshold  = initial_loss * OVERFIT_RATIO
    print(f"\n  initial_loss = {initial_loss:.6f}")
    print(f"  final_loss   = {final_loss:.6f}")
    print(f"  threshold    = {threshold:.6f}  ({OVERFIT_RATIO*100:.0f}% of initial)")
    print(f"  reduction    = {initial_loss / final_loss:.1f}x")

    assert torch.isfinite(torch.tensor(final_loss)), "Final loss is not finite"
    assert final_loss < threshold, (
        f"Overfit failed: final_loss={final_loss:.6f} >= threshold={threshold:.6f} "
        f"(only {initial_loss/final_loss:.1f}x reduction, need 20x)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial review — 5 overfit failure causes (text)
# ─────────────────────────────────────────────────────────────────────────────

"""
ADVERSARIAL REVIEW: why might the tiny-overfit test falsely pass or fail?

[1] Wrong output normalisation hides learning
    Cause : model output is divided by a dataset-level std that is computed on
            the real data but not applied here, so the predicted scale is ~1
            while targets are in [−1, 1] — MSE can appear "already low."
    Detection : print initial and final mean absolute value of `out` and
                `target`.  They should converge; if out ≈ 0 throughout, the
                head is dead (bias saturated or weight initialised to zero).

[2] Target variance too small (trivial task)
    Cause : if all 32 targets are nearly identical (e.g. seed produces
            uniform(−0.01, 0.01)), the initial loss is already < 1e-4 and
            the 5 % threshold (~5e-6) may never be reached despite learning.
    Detection : assert target.std() > 0.1 before training; also print
                target.min(), target.max() in the fixture.

[3] Frozen / disconnected parameters (dead gradients)
    Cause : a parameter that is always multiplied by zero (e.g. v-channel
            gating if all unit vectors cancel) never receives a gradient even
            though requires_grad=True.  The gradient-flow test catches this
            for one forward pass but the overfit test could still pass via
            the scalar path alone.
    Detection : in test_gradient_flow, assert every weight matrix has
                |grad| > 1e-10, not just > 0, and check both msg and upd
                layer parameters separately.

[4] In-place operations block autograd
    Cause : `scatter_add_` (in-place) on a leaf or a view can silently
            corrupt the gradient tape if the tensor version counter
            mismatches.  PyTorch raises a RuntimeError in strict mode but
            not always in eval mode or after .detach() views.
    Detection : run with `torch.autograd.set_detect_anomaly(True)` and
                ensure no "RuntimeError: one of the variables needed for
                gradient computation has been modified by an inplace
                operation" is raised.

[5] Learning rate / scale mismatch collapses loss immediately
    Cause : with lr=1e-3 and hidden=64 the Adam step can overshoot if the
            initial gradient norm is very large (e.g. poor weight-init or
            high-magnitude random targets).  Loss drops to NaN after step 1
            — technically final < initial but the assertion for finiteness
            catches this only if we check every step, not just the last.
    Detection : record loss at every step; assert no NaN/Inf mid-training
                (`assert all(math.isfinite(l) for l in loss_history)`).
"""
