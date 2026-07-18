"""
test_attentivefp_mtl_invariance.py
Two invariance tests for AttentiveFPMTLRegressor:
  (a) output_shape: forward produces (B, n_tasks)
  (b) per_task_independence: outputs differ across tasks (not collapsed)
"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from src.models import AttentiveFPMTLRegressor


def _make_batch(n_mols: int = 4, n_atoms: int = 5, n_tasks: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        n = n_atoms
        x = torch.randn(n, 30)
        # Random connected graph (ring) — guarantees no isolated nodes
        src = torch.arange(n)
        dst = torch.roll(src, 1)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_attr  = torch.randn(edge_index.shape[1], 11)
        y = torch.randn(1, n_tasks)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(data_list)


def test_output_shape():
    """forward returns (B, n_tasks) — no squeeze."""
    n_tasks = 12
    model   = AttentiveFPMTLRegressor(in_channels=30, edge_dim=11, n_tasks=n_tasks)
    model.eval()
    batch   = _make_batch(n_mols=4, n_tasks=n_tasks)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert out.shape == (4, n_tasks), f"expected (4, {n_tasks}), got {out.shape}"


def test_per_task_independence():
    """12 output columns have distinct values — not all collapsed to one scalar."""
    n_tasks = 12
    model   = AttentiveFPMTLRegressor(in_channels=30, edge_dim=11, n_tasks=n_tasks)
    model.eval()
    batch   = _make_batch(n_mols=8, n_tasks=n_tasks)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    # Std across tasks (dim=1) should be nonzero for at least most molecules
    task_stds = out.std(dim=1)  # (B,) — std of 12 task values per molecule
    assert (task_stds > 1e-6).sum() >= 6, (
        f"Less than 6/8 molecules have non-trivial task diversity. "
        f"All outputs may be collapsed. task_stds={task_stds}"
    )
