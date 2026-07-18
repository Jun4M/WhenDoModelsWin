"""
Training tests for KROVEX.

Tests:
  1. test_gradient_flow  — parameters receive gradients after one batch
  2. test_tiny_overfit   — model can overfit a 4-molecule dataset (freeze desc selection)
"""

import numpy as np
import pytest
import torch


_SMILES_TRAIN = [
    'C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'c1ccccc1', 'CCO', 'CCCO',
    'c1cccnc1', 'CCN', 'CC(=O)O', 'c1ccc(O)cc1', 'CCCl', 'CC(C)C',
    'c1ccc(N)cc1', 'CCS',
]
_SMILES_VAL  = ['CC#N', 'CCCC=O', 'c1ccncc1', 'CCOCCO']
_SMILES_TEST = ['c1ccncc1', 'CCCBr', 'CC(F)(F)F', 'Cc1ccccc1']


def _make_y(smiles, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(len(smiles)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Gradient flow
# ──────────────────────────────────────────────────────────────────────────────

def test_gradient_flow():
    """All trainable parameters must receive non-zero gradients after one update."""
    from src.models import KROVEXNet
    from src.featurizer import featurize_smiles_to_krovex_graph
    from torch_geometric.data import Batch, Data

    rng = np.random.default_rng(0)
    num_desc = 5

    graphs, valid = featurize_smiles_to_krovex_graph(_SMILES_TRAIN[:8])
    assert valid, "No valid graphs from training SMILES"

    data_list = []
    y_vals = []
    for i, g in enumerate(graphs):
        desc = torch.tensor(rng.standard_normal((1, num_desc)).astype(np.float32))
        y_val = float(rng.standard_normal())
        data_list.append(Data(x=g.x, edge_index=g.edge_index, desc=desc,
                               y=torch.tensor([y_val])))
        y_vals.append(y_val)

    batch = Batch.from_data_list(data_list)

    torch.manual_seed(0)
    model = KROVEXNet(num_desc=num_desc, dim_in=8)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred = model(batch.x, batch.edge_index, batch.batch, batch.desc)
    loss = ((pred - batch.y.squeeze(-1)) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()

    no_grad_params = [
        name for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)
    ]
    assert not no_grad_params, (
        f"Parameters with zero or missing gradients: {no_grad_params}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tiny overfit
# ──────────────────────────────────────────────────────────────────────────────

def test_tiny_overfit():
    """KROVEXNet must overfit 4 molecules within 500 steps (frozen desc, fixed features)."""
    from src.models import KROVEXNet
    from src.featurizer import featurize_smiles_to_krovex_graph
    from torch_geometric.data import Batch, Data

    rng = np.random.default_rng(1)
    num_desc = 3
    smiles_4 = ['C', 'CC', 'CCC', 'c1ccccc1']
    y_target = torch.tensor([1.0, -1.0, 2.0, -2.0])

    graphs, valid = featurize_smiles_to_krovex_graph(smiles_4)
    assert len(graphs) == 4, f"Expected 4 graphs, got {len(graphs)}"

    data_list = []
    for i, g in enumerate(graphs):
        desc = torch.tensor(rng.standard_normal((1, num_desc)).astype(np.float32))
        data_list.append(Data(x=g.x, edge_index=g.edge_index, desc=desc,
                               y=y_target[i:i+1]))

    batch = Batch.from_data_list(data_list)

    torch.manual_seed(0)
    model = KROVEXNet(num_desc=num_desc, dim_in=8)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    initial_loss = None
    for step in range(500):
        pred = model(batch.x, batch.edge_index, batch.batch, batch.desc)
        loss = ((pred - batch.y.squeeze(-1)) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    ratio = final_loss / (initial_loss + 1e-12)
    assert ratio < 0.05, (
        f"Overfit failed: initial_loss={initial_loss:.4f}, "
        f"final_loss={final_loss:.4f}, ratio={ratio:.4f} (expected <0.05)"
    )
