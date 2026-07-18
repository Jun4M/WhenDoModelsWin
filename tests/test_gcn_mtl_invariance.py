"""
test_gcn_mtl_invariance.py
Three invariance tests for GCNMTLRegressor:
  (a) test_output_shape: forward → (B, n_tasks)
  (b) test_permutation_invariance_with_mtl: atom shuffle → all 12 task outputs identical
  (c) test_per_task_independence: task outputs differ across tasks (not collapsed)
"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from src.models import GCNMTLRegressor


def _make_batch(n_mols: int = 4, n_atoms: int = 8, n_tasks: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        n = n_atoms
        x = torch.randn(n, 30)
        # Ring graph: fully connected enough to propagate all atom features
        src = torch.arange(n)
        dst = torch.roll(src, 1)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        y = torch.randn(1, n_tasks)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return Batch.from_data_list(data_list)


def test_output_shape():
    """forward returns (B, n_tasks) — not squeezed."""
    n_tasks = 12
    model   = GCNMTLRegressor(node_feat_dim=30, n_tasks=n_tasks)
    model.eval()
    batch   = _make_batch(n_mols=4, n_tasks=n_tasks)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)
    assert out.shape == (4, n_tasks), f"expected (4, {n_tasks}), got {out.shape}"


def test_permutation_invariance_with_mtl():
    """Permuting atom order does not change any of the 12 task predictions."""
    n_tasks = 12
    torch.manual_seed(7)
    n = 10
    x          = torch.randn(n, 30)
    src        = torch.arange(n)
    dst        = torch.roll(src, 1)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    batch_idx  = torch.zeros(n, dtype=torch.long)
    y          = torch.randn(1, n_tasks)
    d_orig     = Data(x=x, edge_index=edge_index, batch=batch_idx, y=y)

    # Permute atoms
    perm       = torch.randperm(n)
    x_perm     = x[perm]
    # remap edge_index under permutation
    inv_perm   = torch.argsort(perm)
    ei_perm    = inv_perm[edge_index]
    d_perm     = Data(x=x_perm, edge_index=ei_perm, batch=batch_idx, y=y)

    model = GCNMTLRegressor(node_feat_dim=30, n_tasks=n_tasks)
    model.eval()
    with torch.no_grad():
        out_orig = model(d_orig.x, d_orig.edge_index, d_orig.batch)
        out_perm = model(d_perm.x, d_perm.edge_index, d_perm.batch)

    torch.testing.assert_close(out_orig, out_perm, atol=1e-5, rtol=1e-4,
        msg="GCN-MTL output changed under atom permutation")


def test_per_task_independence():
    """12 output columns are not all identical — heads are independent."""
    n_tasks = 12
    model   = GCNMTLRegressor(node_feat_dim=30, n_tasks=n_tasks)
    model.eval()
    batch   = _make_batch(n_mols=8, n_tasks=n_tasks, seed=42)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)
    # Std across tasks per molecule should be nonzero for most molecules
    task_stds = out.std(dim=1)  # (B,)
    assert (task_stds > 1e-6).sum() >= 6, (
        f"Fewer than 6/8 molecules show task diversity. Outputs may be collapsed. "
        f"task_stds={task_stds}"
    )
