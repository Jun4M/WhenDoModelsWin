"""
test_attentivefp_mtl_train.py
Three training tests for train_attentivefp_mtl:
  (a) test_gradient_flow: all parameters receive gradients
  (b) test_tiny_overfit: joint MSE loss drops by ≥90% on 16 molecules
  (c) test_per_task_metric_extraction: metrics_per_task is populated, all finite, no NaN
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from src.train import train_attentivefp_mtl


N_TASKS = 12
_TASK_NAMES = ['homo', 'lumo', 'gap', 'mu', 'alpha', 'ZPVE',
               'U0', 'U', 'H', 'G', 'Cv', 'R2']
_STATS = [(0.0, 1.0)] * N_TASKS   # dummy stats; denorm done externally


def _make_pyg_list(n_mols: int, n_atoms: int = 5, n_tasks: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        n = n_atoms
        x = torch.randn(n, 30)
        src = torch.arange(n)
        dst = torch.roll(src, 1)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_attr  = torch.randn(edge_index.shape[1], 11)
        y = torch.randn(1, n_tasks)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data_list


def _make_y(n_mols: int, n_tasks: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_mols, n_tasks)).astype(np.float32)


def test_gradient_flow():
    """All named parameters should receive nonzero gradients after one backward."""
    n_mols  = 16
    pyg     = _make_pyg_list(n_mols, n_tasks=N_TASKS, seed=42)
    y_np    = _make_y(n_mols, seed=42)
    # Sync .y in pyg to y_np for consistency
    for i, d in enumerate(pyg):
        d.y = torch.tensor(y_np[i], dtype=torch.float32).unsqueeze(0)

    result = train_attentivefp_mtl(
        train_pyg=pyg, val_pyg=pyg, test_pyg=pyg,
        train_y_normalized=y_np, val_y_normalized=y_np, test_y_normalized=y_np,
        stats=_STATS, target_names=_TASK_NAMES, n_tasks=N_TASKS,
        epochs=1, batch_size=16, lr=1e-3, patience=5, device='cpu', seed=0,
    )
    model = result['model']
    # Manually run one backward pass to check gradients
    from torch_geometric.loader import DataLoader as PyGDataLoader
    loader = PyGDataLoader(pyg, batch_size=16)
    batch  = next(iter(loader))
    out    = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    import torch.nn.functional as F
    loss   = F.mse_loss(out, batch.y.view(-1, N_TASKS))
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert len(no_grad) == 0, f"Params without gradient: {no_grad}"


def test_tiny_overfit():
    """Joint MSE loss on 16 molecules should drop by ≥90% after 100 epochs."""
    n_mols  = 16
    pyg     = _make_pyg_list(n_mols, n_tasks=N_TASKS, seed=7)
    y_np    = _make_y(n_mols, seed=7)
    for i, d in enumerate(pyg):
        d.y = torch.tensor(y_np[i], dtype=torch.float32).unsqueeze(0)

    import torch.nn.functional as F
    from src.models import AttentiveFPMTLRegressor
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch.optim as optim_mod

    model     = AttentiveFPMTLRegressor(in_channels=30, edge_dim=11, n_tasks=N_TASKS)
    optimizer = optim_mod.Adam(model.parameters(), lr=1e-3)
    loader    = PyGDataLoader(pyg, batch_size=16)
    batch     = next(iter(loader))

    model.train()
    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(out, batch.y.view(-1, N_TASKS))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    drop = (losses[0] - losses[-1]) / (losses[0] + 1e-9)
    assert drop >= 0.90, f"Expected ≥90% loss drop but got {drop:.2%} (first={losses[0]:.4f} last={losses[-1]:.4f})"


def test_per_task_metric_extraction():
    """metrics_per_task has all 12 task names, all metrics are finite, none NaN."""
    n_mols = 20
    pyg    = _make_pyg_list(n_mols, n_tasks=N_TASKS, seed=3)
    y_np   = _make_y(n_mols, seed=3)
    for i, d in enumerate(pyg):
        d.y = torch.tensor(y_np[i], dtype=torch.float32).unsqueeze(0)

    result = train_attentivefp_mtl(
        train_pyg=pyg, val_pyg=pyg, test_pyg=pyg,
        train_y_normalized=y_np, val_y_normalized=y_np, test_y_normalized=y_np,
        stats=_STATS, target_names=_TASK_NAMES, n_tasks=N_TASKS,
        epochs=2, batch_size=20, lr=1e-3, patience=5, device='cpu', seed=0,
    )
    mpt = result['metrics_per_task']
    assert set(mpt.keys()) == set(_TASK_NAMES), (
        f"Missing tasks: {set(_TASK_NAMES) - set(mpt.keys())}"
    )
    for tname, m in mpt.items():
        for k, v in m.items():
            assert np.isfinite(v), f"Non-finite metric {k}={v} for task {tname}"
    # test_preds / test_true shapes
    assert result['test_preds'].shape == (n_mols, N_TASKS)
    assert result['test_true'].shape  == (n_mols, N_TASKS)
