"""
tests/test_gnn_train.py
========================
Gradient flow and tiny-overfit tests for GCNRegressor,
AttentiveFPRegressor, and GPSRegressor.

(d) Gradient flow    — every trainable param gets a non-zero gradient
(e) Tiny overfit     — final loss < 5% of initial after 400 steps

3 models × 2 test types = 6 test cases.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


# ── SMILES ───────────────────────────────────────────────────────────────────

_GRAD_SMILES = [
    'c1ccccc1',                          # benzene
    'CCO',                               # ethanol
    'CC(=O)O',                           # acetic acid
    'c1cccnc1',                          # pyridine
    'CC(=O)Nc1ccc(O)cc1',               # paracetamol
    'c1ccc2ccccc2c1',                    # naphthalene
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',       # ibuprofen
    'OC(=O)c1ccccc1',                    # benzoic acid
]

_OVERFIT_SMILES = _GRAD_SMILES + [
    'c1ccsc1',                           # thiophene
    'Cc1ccccc1',                         # toluene
    'CC(C)O',                            # isopropanol
    'CCCC',                              # butane
    'CC1CCCCC1',                         # methylcyclohexane
    'c1ccc(O)cc1',                       # phenol
    'CC(=O)c1ccccc1',                    # acetophenone
    'c1cncnc1',                          # imidazole
]


# ── Data helpers ─────────────────────────────────────────────────────────────

def _build_batch(smiles, seed=0):
    """
    Featurize smiles list with MolGraphConv → PyG Batch + y tensor.
    y values are random (fixed by seed) — only used to define a training target.
    """
    from src.featurizer import featurize_smiles_to_graphs, dcgraph_to_pyg
    torch.manual_seed(seed)
    y_vals  = torch.randn(len(smiles))
    graphs, valid = featurize_smiles_to_graphs(smiles)
    data_list = [
        dcgraph_to_pyg(g, float(y_vals[i]))
        for i, g in zip(valid, graphs)
    ]
    y = torch.tensor([d.y.item() for d in data_list])
    return Batch.from_data_list(data_list), y


def _forward(model, batched, batch):
    from src.models import GCNRegressor
    if isinstance(model, GCNRegressor):
        return model(batched.x, batched.edge_index, batch)
    return model(batched.x, batched.edge_index, batched.edge_attr, batch)


# ── Model factories (realistic dropout for training tests) ──────────────────

def _gcn():
    from src.models import GCNRegressor
    return GCNRegressor(node_feat_dim=30, hidden_dim=64, num_layers=2, dropout=0.1)


def _afp():
    from src.models import AttentiveFPRegressor
    return AttentiveFPRegressor(
        in_channels=30, edge_dim=11, hidden_channels=64,
        num_layers=2, num_timesteps=2, dropout=0.1,
    )


def _gps():
    from src.models import GPSRegressor
    return GPSRegressor(
        in_channels=30, hidden_channels=64, num_layers=2,
        dropout=0.1, attn_dropout=0.1, walk_length=5,
    )


_MODELS = [
    pytest.param(_gcn, id='GCN'),
    pytest.param(_afp, id='AFP'),
    pytest.param(_gps, id='GPS'),
]


# ── (d) Gradient flow ────────────────────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_gradient_flow(make_model):
    """
    Every trainable parameter receives a non-zero gradient after one
    forward-backward pass on an 8-molecule batch.

    Dead gradients indicate: disconnected layers, wrong forward path,
    or parameters that are effectively pruned from the computation graph.
    """
    torch.manual_seed(0)
    batched, y = _build_batch(_GRAD_SMILES, seed=0)

    model = make_model()
    model.train()

    out  = _forward(model, batched, batched.batch)
    loss = F.mse_loss(out, y)

    # Integrity: loss must be non-trivial; zero loss → zero gradients everywhere
    assert loss.item() > 1e-6, (
        f"[{make_model.__name__}] loss={loss.item():.2e} is effectively zero "
        "before training — targets may all be zero or model perfectly fits init."
    )

    loss.backward()

    dead = [
        name for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)
    ]
    assert not dead, (
        f"[{make_model.__name__}] {len(dead)} dead-gradient parameter(s):\n"
        + "\n".join(f"  {n}" for n in dead)
    )


# ── (e) Tiny overfit ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_tiny_overfit(make_model):
    """
    Model memorises 16 molecules in ≤400 gradient steps (Adam, lr=1e-3).
    Criterion: final_loss < initial_loss × 0.05  (≥95% reduction).

    A model that cannot overfit a tiny fixed dataset has insufficient
    capacity, an incorrect forward pass, or a broken gradient path.
    """
    torch.manual_seed(42)
    batched, y = _build_batch(_OVERFIT_SMILES, seed=42)

    # Integrity: targets must be varied (std > 0.3) so overfit is meaningful
    assert y.std().item() > 0.3, (
        f"Target std={y.std().item():.4f} is too low — all targets are nearly "
        "identical and any constant predictor would 'overfit'."
    )

    model = make_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(400):
        opt.zero_grad()
        out  = _forward(model, batched, batched.batch)
        loss = F.mse_loss(out, y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()

    final_loss = loss.item()
    ratio = final_loss / initial_loss if initial_loss else float('inf')
    assert ratio < 0.05, (
        f"[{make_model.__name__}] failed to overfit in 400 steps: "
        f"initial={initial_loss:.4f}, final={final_loss:.4f}, "
        f"ratio={ratio:.3f} (target < 0.05)"
    )

    # Integrity: final predictions must be non-constant (model learned variation)
    with torch.no_grad():
        out_final = _forward(model, batched, batched.batch)
    assert out_final.std().item() > 0.01, (
        f"[{make_model.__name__}] final predictions are constant (std={out_final.std().item():.4f}) "
        "— model collapsed to a single value despite low loss."
    )
