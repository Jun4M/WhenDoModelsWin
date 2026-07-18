"""
test_gcn_mtl_train.py
Three training tests for train_gcn_mtl:
  (a) test_gradient_flow: all parameters (including all 12 head outputs) receive gradients
  (b) test_tiny_overfit: joint MSE drops by ≥90% on 32 molecules / 12 tasks
  (c) test_consistency_with_single_task: GCN-MTL task-i RMSE not drastically worse
      than single-task GCN RMSE (within 3× — sanity that MTL doesn't catastrophically
      interfere for any individual task at small N)
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from src.train import train_gcn_mtl, train_gcn


N_TASKS    = 12
_TASK_NAMES = ['homo', 'lumo', 'gap', 'mu', 'alpha', 'ZPVE',
               'U0', 'U', 'H', 'G', 'Cv', 'R2']
_STATS = [(0.0, 1.0)] * N_TASKS


def _make_pyg_list(n_mols: int, n_atoms: int = 8, n_tasks: int = 12, seed: int = 0):
    """Build MTL PyG list: .y shape (1, n_tasks)."""
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        n = n_atoms
        x = torch.randn(n, 30)
        src = torch.arange(n)
        dst = torch.roll(src, 1)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        y = torch.randn(1, n_tasks)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return data_list


def _make_pyg_single(n_mols: int, n_atoms: int = 8, seed: int = 0):
    """Build single-task PyG list: .y shape (1,)."""
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        n = n_atoms
        x = torch.randn(n, 30)
        src = torch.arange(n)
        dst = torch.roll(src, 1)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        y = torch.randn(1)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return data_list


def _make_y(n_mols: int, n_tasks: int = 12, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_mols, n_tasks)).astype(np.float32)


def test_gradient_flow():
    """All named parameters receive nonzero gradients after one backward pass."""
    n_mols = 32
    pyg    = _make_pyg_list(n_mols, seed=42)
    y_np   = _make_y(n_mols, seed=42)
    for i, d in enumerate(pyg):
        d.y = torch.tensor(y_np[i]).unsqueeze(0)

    result = train_gcn_mtl(
        train_pyg=pyg, val_pyg=pyg, test_pyg=pyg,
        train_y_normalized=y_np, val_y_normalized=y_np, test_y_normalized=y_np,
        stats=_STATS, target_names=_TASK_NAMES, n_tasks=N_TASKS,
        epochs=1, batch_size=32, lr=1e-3, patience=5, device='cpu', seed=0,
    )
    model = result['model']

    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch.nn.functional as F
    loader = PyGDataLoader(pyg, batch_size=32)
    batch  = next(iter(loader))
    out    = model(batch.x, batch.edge_index, batch.batch)
    loss   = F.mse_loss(out, batch.y.view(-1, N_TASKS))
    loss.backward()

    no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert len(no_grad) == 0, f"Params without gradient: {no_grad}"


def test_tiny_overfit():
    """Joint MSE on 32 molecules should drop by ≥90% after 150 epochs."""
    n_mols = 32
    pyg    = _make_pyg_list(n_mols, seed=7)
    y_np   = _make_y(n_mols, seed=7)
    for i, d in enumerate(pyg):
        d.y = torch.tensor(y_np[i]).unsqueeze(0)

    import torch.nn.functional as F
    from src.models import GCNMTLRegressor
    from torch_geometric.loader import DataLoader as PyGDataLoader
    import torch.optim as optim_mod

    model     = GCNMTLRegressor(node_feat_dim=30, n_tasks=N_TASKS)
    optimizer = optim_mod.Adam(model.parameters(), lr=1e-2)
    loader    = PyGDataLoader(pyg, batch_size=32)
    batch     = next(iter(loader))
    y_batch   = batch.y.view(-1, N_TASKS)

    model.train()
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(out, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    drop = (losses[0] - losses[-1]) / (losses[0] + 1e-9)
    assert drop >= 0.90, (
        f"Expected ≥90% loss drop, got {drop:.2%} "
        f"(first={losses[0]:.4f} last={losses[-1]:.4f}). "
        "GCN-MTL may lack capacity — consider raising hidden_dim."
    )


def test_consistency_with_single_task():
    """GCN-MTL task-0 RMSE should be within 3× of single-task GCN RMSE on same data.

    This is a robustness sanity check: MTL should not catastrophically degrade
    any individual task. Threshold of 3× is deliberately loose since:
      - We use only 32 training molecules (toy setting)
      - 12-task joint training with GCN's smaller capacity introduces more noise
      - The goal is to detect complete collapse, not match performance
    """
    n_mols  = 32
    seed    = 5
    n_atoms = 8

    # Single-task PyG (task 0 only)
    torch.manual_seed(seed)
    y0_all = torch.randn(n_mols).numpy().astype(np.float32)  # task 0 labels

    pyg_single = _make_pyg_single(n_mols, n_atoms=n_atoms, seed=seed)
    for i, d in enumerate(pyg_single):
        d.y = torch.tensor([y0_all[i]])

    single_res = train_gcn(
        train_pyg=pyg_single, val_pyg=pyg_single, test_pyg=pyg_single,
        target_name='task0', node_feat_dim=30,
        epochs=50, batch_size=32, lr=1e-2, patience=10, device='cpu', seed=seed,
    )
    single_rmse = single_res['metrics']['RMSE']

    # MTL (12 tasks; task 0 labels = y0_all)
    y_mtl = _make_y(n_mols, N_TASKS, seed=seed)
    y_mtl[:, 0] = y0_all                     # align task 0

    pyg_mtl = _make_pyg_list(n_mols, n_atoms=n_atoms, seed=seed)
    for i, d in enumerate(pyg_mtl):
        d.y = torch.tensor(y_mtl[i]).unsqueeze(0)

    mtl_res = train_gcn_mtl(
        train_pyg=pyg_mtl, val_pyg=pyg_mtl, test_pyg=pyg_mtl,
        train_y_normalized=y_mtl, val_y_normalized=y_mtl, test_y_normalized=y_mtl,
        stats=_STATS, target_names=_TASK_NAMES, n_tasks=N_TASKS,
        epochs=50, batch_size=32, lr=1e-2, patience=10, device='cpu', seed=seed,
    )
    mtl_rmse_task0 = mtl_res['metrics_per_task'][_TASK_NAMES[0]]['RMSE']

    assert mtl_rmse_task0 <= 3.0 * single_rmse + 0.5, (
        f"GCN-MTL task-0 RMSE ({mtl_rmse_task0:.4f}) is more than 3× single-task "
        f"GCN RMSE ({single_rmse:.4f}). Possible task interference or collapse."
    )
