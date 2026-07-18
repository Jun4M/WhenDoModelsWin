"""
tests/test_gnn_determinism.py
==============================
Seed determinism: two training runs with the same torch.manual_seed must
produce loss trajectories that are bitwise identical (or within 1e-6).

dropout=0.0 eliminates all stochasticity, so the only variable is the
random seed controlling weight initialisation — which we control.

Each test runs 5 gradient steps (enough to exercise weight updates,
BatchNorm running-stat evolution, and Adam moment accumulation).

If non-determinism is detected, probable sources:
  - Dropout left active in train mode with stochastic draw
  - DataLoader shuffle changing batch composition
  - Non-deterministic CUDA kernels (not relevant on CPU, but noted)
  - BN running stats out of sync between the two runs

3 models = 3 test cases.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

ATOL_DETERMINISM = 1e-6


# ── SMILES ───────────────────────────────────────────────────────────────────

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


# ── Data helper ───────────────────────────────────────────────────────────────

def _build_batch(smiles, seed=0):
    from src.featurizer import featurize_smiles_to_graphs, dcgraph_to_pyg
    torch.manual_seed(seed)
    y_vals    = torch.randn(len(smiles))
    graphs, valid = featurize_smiles_to_graphs(smiles)
    data_list = [dcgraph_to_pyg(g, float(y_vals[i]))
                 for i, g in zip(valid, graphs)]
    y = torch.tensor([d.y.item() for d in data_list])
    return Batch.from_data_list(data_list), y


def _forward(model, batched, batch):
    from src.models import GCNRegressor
    if isinstance(model, GCNRegressor):
        return model(batched.x, batched.edge_index, batch)
    return model(batched.x, batched.edge_index, batched.edge_attr, batch)


def _run_n_steps(make_model, batched, y, n_steps, seed):
    """
    Fresh model (seeded), n_steps of Adam, return list of loss floats.
    dropout=0.0 — eliminates stochasticity so losses must be bitwise equal.
    """
    torch.manual_seed(seed)
    model = make_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        out  = _forward(model, batched, batched.batch)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ── Model factories (dropout=0 — pure determinism check) ────────────────────

def _gcn():
    from src.models import GCNRegressor
    return GCNRegressor(node_feat_dim=30, hidden_dim=64, num_layers=2, dropout=0.0)


def _afp():
    from src.models import AttentiveFPRegressor
    return AttentiveFPRegressor(
        in_channels=30, edge_dim=11, hidden_channels=64,
        num_layers=2, num_timesteps=2, dropout=0.0,
    )


def _gps():
    from src.models import GPSRegressor
    return GPSRegressor(
        in_channels=30, hidden_channels=64, num_layers=2,
        dropout=0.0, attn_dropout=0.0, walk_length=5,
    )


_MODELS = [
    pytest.param(_gcn, id='GCN'),
    pytest.param(_afp, id='AFP'),
    pytest.param(_gps, id='GPS'),
]


# ── (f) Seed determinism ──────────────────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_seed_determinism(make_model):
    """
    Two training runs with identical seed and data produce the same
    loss trajectory (all 5 steps) within 1e-6.

    Non-determinism sources this test would catch:
      • Dropout sampled independently per run (→ different stochastic masks)
      • Weight init seeded by global state rather than torch.manual_seed
      • BatchNorm running stats updating with different accumulation order
      • Adam moments diverging due to different gradient rounding order

    All are eliminated here by dropout=0.0 and CPU execution.
    If this test fails: audit F.dropout calls for training=True hard-coding,
    check nn.Module init uses seeded RNG, verify BatchNorm is BN1d (not IN).
    """
    batched, y = _build_batch(_DET_SMILES, seed=99)

    losses_a = _run_n_steps(make_model, batched, y, n_steps=5, seed=123)
    losses_b = _run_n_steps(make_model, batched, y, n_steps=5, seed=123)

    # Integrity: initial loss must be non-trivial (zero loss → all runs trivially equal)
    assert losses_a[0] > 1e-6, (
        f"[{make_model.__name__}] initial loss={losses_a[0]:.2e} is near zero "
        "— targets may all be zero; determinism check is vacuous."
    )

    # Integrity: loss must change over training steps (model is actually learning)
    assert losses_a[0] != losses_a[-1], (
        f"[{make_model.__name__}] loss unchanged over 5 steps "
        f"({losses_a[0]:.6f} → {losses_a[-1]:.6f}) — gradients may not reach weights."
    )

    # Integrity: eval mode must be deterministic (no stochastic ops in eval)
    torch.manual_seed(123)
    model_eval = make_model()
    model_eval.eval()
    with torch.no_grad():
        out1 = _forward(model_eval, batched, batched.batch)
        out2 = _forward(model_eval, batched, batched.batch)
    assert torch.equal(out1, out2), (
        f"[{make_model.__name__}] eval() not deterministic — "
        "dropout may be hard-coded to training=True."
    )

    for step, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < ATOL_DETERMINISM, (
            f"[{make_model.__name__}] non-determinism at step {step}: "
            f"run_a={la:.10f}, run_b={lb:.10f}  (Δ={abs(la-lb):.2e})\n"
            "Possible causes: hard-coded training=True in F.dropout, "
            "non-seeded weight init, BatchNorm stat divergence."
        )
