"""
tests/test_painn_equivariance.py
=================================
SE(3) equivariance tests for PaiNNRegressor.

(a) Rotation invariance  — scalar output unchanged under R ∈ SO(3)
(b) Translation invariance — scalar output unchanged under t ∈ R³
(c) Vector equivariance   — internal v transforms as R·v under rotation

Run:
    pytest tests/test_painn_equivariance.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest
from scipy.spatial.transform import Rotation

from src.models import PaiNNRegressor, _PaiNNUpdate

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

CUTOFF  = 5.0
ATOL    = 1e-5


def _methane(dtype=torch.float64, device="cpu"):
    """
    Methane (CH4): C at origin + 4 H at tetrahedral vertices, ~1.09 Å C-H.
    Returns (x, pos, batch).
    """
    d = 1.09 / np.sqrt(3)
    pos = torch.tensor([
        [ 0.0,  0.0,  0.0],   # C
        [ d,    d,    d],      # H
        [ d,   -d,   -d],      # H
        [-d,    d,   -d],      # H
        [-d,   -d,    d],      # H
    ], dtype=dtype, device=device)

    # x[:,0] = atom index (6=C, 1=H); remaining dims zero
    x = torch.zeros(5, 30, dtype=dtype, device=device)
    x[0, 0] = 6.0
    x[1:, 0] = 1.0

    batch = torch.zeros(5, dtype=torch.long, device=device)
    return x, pos, batch


def _water(dtype=torch.float64, device="cpu"):
    """Water (H2O): O at origin + 2 H at 104.5° angle, ~0.96 Å O-H."""
    r = 0.96
    a = np.radians(104.5 / 2)
    pos = torch.tensor([
        [0.0,            0.0, 0.0],
        [r * np.sin(a),  r * np.cos(a), 0.0],
        [-r * np.sin(a), r * np.cos(a), 0.0],
    ], dtype=dtype, device=device)

    x = torch.zeros(3, 30, dtype=dtype, device=device)
    x[0, 0] = 8.0   # O
    x[1:, 0] = 1.0  # H

    batch = torch.zeros(3, dtype=torch.long, device=device)
    return x, pos, batch


def _make_model(seed=42):
    torch.manual_seed(seed)
    model = PaiNNRegressor(hidden_channels=32, num_layers=3,
                           cutoff=CUTOFF, num_rbf=20)
    model = model.double()   # float64 for numerical precision
    model.eval()
    return model


def _rotation(seed=0):
    """Random rotation matrix as float64 tensor."""
    R = Rotation.random(random_state=seed).as_matrix()
    return torch.tensor(R, dtype=torch.float64)


def _rotate_pos(R, pos):
    """Apply (3,3) rotation R to (N,3) positions."""
    return (R @ pos.T).T


def _rotate_v(R, v):
    """
    Apply R to vector features v of shape (N, F, 3).
    Equivariance requires: v_rot[n,f] = R @ v_orig[n,f]
    Implemented as: v_rot = (R @ v.reshape(-1,3).T).T.reshape(N,F,3)
    """
    N, F, _ = v.shape
    flat = v.reshape(-1, 3)          # (N*F, 3)
    rotated = (R @ flat.T).T         # (N*F, 3)
    return rotated.reshape(N, F, 3)


class _VHook:
    """Forward hook to capture v output of _PaiNNUpdate."""
    def __init__(self):
        self.v = None

    def __call__(self, module, inp, output):
        # _PaiNNUpdate.forward returns (s_new, v_new)
        self.v = output[1].detach().clone()


def _capture_v(model, x, pos, batch):
    """Run forward, return (scalar_output, v_after_last_update_layer)."""
    hook = _VHook()
    handle = model.upd_layers[-1].register_forward_hook(hook)
    with torch.no_grad():
        out = model(x, pos, batch)
    handle.remove()
    return out, hook.v


# ─────────────────────────────────────────────────────────────────────────────
# (a) Rotation invariance
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mol_fn,label", [
    (_methane, "methane"),
    (_water,   "water"),
])
@pytest.mark.parametrize("rot_seed", [0, 1, 7])
def test_rotation_invariance(mol_fn, label, rot_seed):
    model = _make_model()
    x, pos, batch = mol_fn()
    R = _rotation(rot_seed)
    pos_rot = _rotate_pos(R, pos)

    with torch.no_grad():
        out_orig = model(x, pos, batch)
        out_rot  = model(x, pos_rot, batch)

    diff = (out_orig - out_rot).abs().item()
    if diff > ATOL:
        print(f"\n[{label}/rot_seed={rot_seed}] "
              f"orig={out_orig.item():.8f}  rot={out_rot.item():.8f}  diff={diff:.3e}")
    assert diff < ATOL, (
        f"Rotation invariance violated ({label}, seed={rot_seed}): diff={diff:.3e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (b) Translation invariance
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mol_fn,label", [
    (_methane, "methane"),
    (_water,   "water"),
])
@pytest.mark.parametrize("t", [
    [3.14, -2.71,  1.41],
    [0.0,   0.0,  10.0],
    [-5.5,  5.5,  -5.5],
])
def test_translation_invariance(mol_fn, label, t):
    model = _make_model()
    x, pos, batch = mol_fn()
    shift = torch.tensor(t, dtype=torch.float64).unsqueeze(0)
    pos_shifted = pos + shift

    with torch.no_grad():
        out_orig    = model(x, pos, batch)
        out_shifted = model(x, pos_shifted, batch)

    diff = (out_orig - out_shifted).abs().item()
    if diff > ATOL:
        print(f"\n[{label}/t={t}] "
              f"orig={out_orig.item():.8f}  shifted={out_shifted.item():.8f}  diff={diff:.3e}")
    assert diff < ATOL, (
        f"Translation invariance violated ({label}, t={t}): diff={diff:.3e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (c) Vector equivariance
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rot_seed", [0, 1, 7])
def test_vector_equivariance(rot_seed):
    model = _make_model()
    x, pos, batch = _methane()
    R = _rotation(rot_seed)
    pos_rot = _rotate_pos(R, pos)

    _, v_orig = _capture_v(model, x, pos, batch)
    _, v_rot  = _capture_v(model, x, pos_rot, batch)

    # expected: applying R to each 3D vector in v_orig
    v_expected = _rotate_v(R, v_orig)

    diff = (v_rot - v_expected).abs().max().item()
    if diff > ATOL:
        print(f"\n[vector_equivariance/rot_seed={rot_seed}] max diff={diff:.3e}")
        print(f"  v_rot[0,0]      = {v_rot[0, 0].tolist()}")
        print(f"  v_expected[0,0] = {v_expected[0, 0].tolist()}")
    assert diff < ATOL, (
        f"Vector equivariance violated (seed={rot_seed}): max diff={diff:.3e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sanity: model is not degenerate (silent-failure guards)
# ─────────────────────────────────────────────────────────────────────────────

def test_edges_are_nonzero():
    """Guard 1: radius graph must have at least one edge (pos not ignored)."""
    model = _make_model()
    x, pos, batch = _methane()
    edge_index = PaiNNRegressor._radius_graph(pos, CUTOFF, batch)
    n_edges = edge_index.shape[1]
    assert n_edges > 0, (
        f"No edges within cutoff={CUTOFF} Å — model ignores pos entirely. "
        f"Check atom positions or cutoff."
    )


def test_vector_features_nonzero():
    """Guard 2: v must be non-zero after forward (vector path must fire)."""
    model = _make_model()
    x, pos, batch = _methane()
    _, v = _capture_v(model, x, pos, batch)
    v_norm = v.norm().item()
    assert v_norm > 1e-6, (
        f"Vector features v are all zero (norm={v_norm:.2e}). "
        f"The vector message path is broken — equivariance is trivially satisfied."
    )


def test_output_varies_with_geometry():
    """Guard 3: output must change when geometry changes (not a constant function)."""
    model = _make_model()
    x, pos, batch = _methane()

    # stretch C-H bonds by 2x
    pos_stretched = pos.clone()
    pos_stretched[1:] *= 2.0

    with torch.no_grad():
        out_orig     = model(x, pos, batch)
        out_stretched = model(x, pos_stretched, batch)

    diff = (out_orig - out_stretched).abs().item()
    assert diff > 1e-6, (
        f"Output is identical for different geometries (diff={diff:.2e}). "
        f"Model may be ignoring 3D coordinates entirely."
    )


def test_output_varies_across_molecules():
    """Guard 4: methane and water must give different outputs (not constant model)."""
    model = _make_model()
    x_ch4, pos_ch4, batch_ch4 = _methane()
    x_h2o, pos_h2o, batch_h2o = _water()

    with torch.no_grad():
        out_ch4 = model(x_ch4, pos_ch4, batch_ch4)
        out_h2o = model(x_h2o, pos_h2o, batch_h2o)

    diff = (out_ch4 - out_h2o).abs().item()
    assert diff > 1e-6, (
        f"Methane and water give identical output (diff={diff:.2e}). "
        f"Model may have collapsed to a constant."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Guard #1 (enhanced) — geometry sensitivity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("delta,tag", [
    (0.05, "small_0.05A"),
    (0.30, "medium_0.30A"),
    (1.00, "large_1.00A"),
])
def test_geometry_sensitivity_progressive(delta, tag):
    """
    #1 adversarial: scalar output must change for small, medium, large atom displacements.
    Scenario caught: model that ignores pos and uses only atom-type embeddings.
    A pos-blind model would return identical output for any displacement.
    """
    model = _make_model()
    x, pos, batch = _methane()
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    pos_pert = pos.clone()
    pos_pert[1] = pos[1] + delta * direction  # displace one H along x

    with torch.no_grad():
        out_orig = model(x, pos, batch)
        out_pert = model(x, pos_pert, batch)

    diff = (out_orig - out_pert).abs().item()
    if diff <= 1e-8:
        print(f"\n[geometry_sensitivity/{tag}]"
              f"  orig={out_orig.item():.8f}  pert={out_pert.item():.8f}  diff={diff:.3e}")
        print(f"  BROKEN: output does not respond to {delta}Å displacement.")
        print(f"  Model likely ignores pos — only atom embeddings are used.")
    assert diff > 1e-8, (
        f"[#1/{tag}] output unchanged after {delta}Å displacement (diff={diff:.3e}). "
        f"pos appears to be ignored."
    )


def test_geometry_sensitivity_multiscale():
    """
    #1 adversarial: output must respond to perturbations at small, medium, and large scales.
    All three deltas (0.05, 0.50, 2.00 Å) stay within the 5 Å cutoff so graph topology
    is preserved. A pos-blind model would return 0 diff for every delta.
    Note: strict monotonicity is NOT asserted — a randomly initialized nonlinear network
    has no reason to respond monotonically; we only require nonzero sensitivity at each scale.
    """
    model = _make_model()
    x, pos, batch = _methane()
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    deltas = [0.05, 0.50, 2.00]
    diffs = {}
    with torch.no_grad():
        out_orig = model(x, pos, batch)
        for delta in deltas:
            pos_p = pos.clone()
            pos_p[1] = pos[1] + delta * direction
            out_p = model(x, pos_p, batch)
            diffs[delta] = (out_orig - out_p).abs().item()

    failures = [d for d, v in diffs.items() if v <= 1e-8]
    if failures:
        print(f"\n[geometry_multiscale] diffs={diffs}")
        for d in failures:
            print(f"  BROKEN: no response at Δ={d}Å (diff={diffs[d]:.3e}).")
            print(f"  pos may be ignored for this perturbation scale.")
    for d in failures:
        assert False, (
            f"[#1] Output unchanged at Δ={d}Å displacement (diff={diffs[d]:.3e}). "
            f"Model appears insensitive to this geometry change."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Guard #2 (enhanced) — vector path active
# ─────────────────────────────────────────────────────────────────────────────

def _capture_v_all_layers(model, x, pos, batch):
    """Capture v output from every _PaiNNUpdate layer via hooks."""
    vs = {}
    handles = []
    for i, layer in enumerate(model.upd_layers):
        def _make_hook(idx):
            def _hook(module, inp, out):
                vs[idx] = out[1].detach().clone()
            return _hook
        handles.append(layer.register_forward_hook(_make_hook(i)))
    with torch.no_grad():
        model(x, pos, batch)
    for h in handles:
        h.remove()
    return vs


def test_vector_path_fires_per_layer():
    """
    #2 adversarial: v must be nonzero at EVERY update layer, not just the last.
    A broken path that produces v=0 after layer 0 still satisfies R@0=0 equivariance
    trivially for all subsequent layers — the equivariance test cannot catch this.
    """
    model = _make_model()
    x, pos, batch = _methane()
    vs = _capture_v_all_layers(model, x, pos, batch)

    for layer_idx, v in sorted(vs.items()):
        v_norm = v.norm().item()
        if v_norm <= 1e-8:
            print(f"\n[vector_path_per_layer] layer {layer_idx}: v.norm()={v_norm:.3e}")
            print(f"  BROKEN: vector features collapsed to zero at layer {layer_idx}.")
            print(f"  R@0=0 is trivially true — equivariance test gives false green.")
        assert v_norm > 1e-8, (
            f"[#2] v=0 at layer {layer_idx} (norm={v_norm:.3e}). "
            f"Vector path dead — equivariance trivially satisfied via R@0=0."
        )


def test_vector_direction_contributes_to_scalar():
    """
    #2 adversarial: zeroing unit vectors in every message layer must change scalar output.
    Proves that the directional channel (msg_vr = x_vr * unit_ij) actually feeds into
    scalars via the inner product in _PaiNNUpdate.
    Scenario caught: x_vr slice dead (zero weights), so direction never reaches output.
    """
    from src.models import _PaiNNMessage
    import unittest.mock as mock

    model = _make_model()
    x, pos, batch = _methane()

    with torch.no_grad():
        out_normal = model(x, pos, batch)

    _orig_msg_forward = _PaiNNMessage.forward

    def _forward_no_direction(self, s, v, edge_index, rbf, unit):
        return _orig_msg_forward(self, s, v, edge_index, rbf, torch.zeros_like(unit))

    with mock.patch.object(_PaiNNMessage, 'forward', _forward_no_direction):
        with torch.no_grad():
            out_no_dir = model(x, pos, batch)

    diff = (out_normal - out_no_dir).abs().item()
    if diff <= 1e-6:
        print(f"\n[vector_direction_contributes]")
        print(f"  out_normal={out_normal.item():.8f}  out_no_dir={out_no_dir.item():.8f}")
        print(f"  diff={diff:.3e}")
        print(f"  BROKEN: zeroing unit vectors has no effect on scalar output.")
        print(f"  x_vr branch is dead-weight — direction never flows to scalars.")
    assert diff > 1e-6, (
        f"[#2] Direction path contributes nothing: diff={diff:.3e} when unit=0. "
        f"x_vr weights may be zero or disconnected from scalar head."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Guard #4 (enhanced) — output not constant
# ─────────────────────────────────────────────────────────────────────────────

def _ammonia(dtype=torch.float64, device="cpu"):
    """Ammonia (NH3): N + 3H, pyramidal, ~1.01 Å N-H, ~107° H-N-H angle."""
    r, angle = 1.01, np.radians(112.0)
    h  = r * np.cos(angle / 2)
    xy = r * np.sin(angle / 2)
    pos = torch.tensor([
        [ 0.0,                    0.0,  0.0],
        [ xy,                     0.0,  h  ],
        [-xy / 2,  xy * np.sqrt(3) / 2, h  ],
        [-xy / 2, -xy * np.sqrt(3) / 2, h  ],
    ], dtype=dtype, device=device)
    x = torch.zeros(4, 30, dtype=dtype, device=device)
    x[0, 0] = 7.0
    x[1:, 0] = 1.0
    batch = torch.zeros(4, dtype=torch.long, device=device)
    return x, pos, batch


def test_output_not_constant_multiple_pairs():
    """
    #4 adversarial: outputs must differ for CH4, H2O, and NH3 — all three pairs.
    Scenario caught: model collapsed to a constant (all outputs identical regardless of input).
    A single pair test (e.g., CH4 vs H2O) could pass by coincidence if only those two
    happen to differ, while the model is near-constant everywhere else.
    """
    model = _make_model()
    molecules = {
        "CH4": _methane(),
        "H2O": _water(),
        "NH3": _ammonia(),
    }
    outputs = {}
    with torch.no_grad():
        for name, (x, pos, batch) in molecules.items():
            outputs[name] = model(x, pos, batch).item()

    pairs = [("CH4", "H2O"), ("CH4", "NH3"), ("H2O", "NH3")]
    failures = [(a, b, abs(outputs[a] - outputs[b])) for a, b in pairs
                if abs(outputs[a] - outputs[b]) <= 1e-6]

    if failures:
        print(f"\n[output_not_constant] outputs={outputs}")
        for a, b, diff in failures:
            print(f"  {a}={outputs[a]:.8f}  {b}={outputs[b]:.8f}  diff={diff:.3e}")
            print(f"  BROKEN: {a} and {b} are indistinguishable → model may be constant.")
    for a, b, diff in failures:
        assert False, (
            f"[#4] {a} and {b} outputs identical (diff={diff:.3e}). "
            f"Model appears to be a constant function."
        )


def test_output_determinism():
    """
    #4 adversarial: three forward passes on identical input must give identical output.
    Scenario caught: dropout or batchnorm accidentally left in train mode,
    or any stochastic operation inside forward.
    """
    model = _make_model()   # _make_model calls model.eval()
    x, pos, batch = _methane()

    with torch.no_grad():
        outs = [model(x, pos, batch).item() for _ in range(3)]

    spread = max(outs) - min(outs)
    if spread > 1e-10:
        print(f"\n[output_determinism] runs={outs}  spread={spread:.3e}")
        print(f"  BROKEN: model is not deterministic in eval mode.")
        print(f"  Likely cause: dropout/batchnorm still in train mode.")
    assert spread < 1e-10, (
        f"[#4] Non-deterministic: spread={spread:.3e} over 3 identical runs. "
        f"Check model.eval() and stochastic ops."
    )


def test_output_variance_across_init_seeds():
    """
    #4 adversarial: outputs across 5 random init seeds must span a meaningful range.
    Scenario caught: weight init scheme that always collapses to the same network
    (e.g., all-zeros or all-ones init), causing every random seed to give the same output.
    """
    x, pos, batch = _methane()
    outputs = {}
    for seed in range(5):
        m = _make_model(seed=seed)
        with torch.no_grad():
            outputs[seed] = m(x, pos, batch).item()

    spread = max(outputs.values()) - min(outputs.values())
    if spread <= 1e-6:
        print(f"\n[variance_across_seeds] outputs={outputs}  spread={spread:.3e}")
        print(f"  BROKEN: all init seeds produce nearly identical output.")
        print(f"  Init scheme may collapse weights to same configuration.")
    assert spread > 1e-6, (
        f"[#4] No variance across 5 init seeds (spread={spread:.3e}). "
        f"Outputs: {outputs}"
    )


def test_rotation_changes_unit_vectors():
    """Guard 5: rotating pos must actually change unit vectors (not aliased)."""
    x, pos, batch = _methane()
    R = _rotation(seed=0)
    pos_rot = _rotate_pos(R, pos)

    # build edge_index for both
    edge_index = PaiNNRegressor._radius_graph(pos, CUTOFF, batch)
    src, dst = edge_index
    unit_orig = pos[src] - pos[dst]
    unit_rot  = pos_rot[src] - pos_rot[dst]

    diff = (unit_orig - unit_rot).abs().max().item()
    assert diff > 1e-6, (
        f"Rotating pos did not change unit vectors (max diff={diff:.2e}). "
        f"Rotation matrix may be identity or pos may be invariant to R."
    )
