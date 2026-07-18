"""
test_unimol_train.py
Training sanity checks for UniMolRegressor:
  (a) test_gradient_flow: all parameters receive non-zero gradients after backward
  (b) test_tiny_overfit:  ≥ 90 % MSE drop after 200 epochs on 16 molecules
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim_mod
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.models import UniMolRegressor
from src.train import train_unimol


def _make_pyg_list(n_mols: int, n_atoms: int = 8, seed: int = 0):
    """Build list of PyG Data with .z + .pos + .y for UniMol."""
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        z   = torch.randint(1, 9, (n_atoms,), dtype=torch.long)
        pos = torch.randn(n_atoms, 3)
        y   = torch.randn(1)
        data_list.append(Data(z=z, pos=pos, y=y))
    return data_list


def test_gradient_flow():
    """All named parameters receive nonzero gradients after one backward."""
    n_mols = 16
    pyg = _make_pyg_list(n_mols, seed=42)
    y_np = np.random.default_rng(42).standard_normal(n_mols).astype(np.float32)

    result = train_unimol(
        train_3d=pyg, val_3d=pyg, test_3d=pyg,
        train_y=y_np, val_y=y_np, test_y=y_np,
        target_name='test', epochs=1, batch_size=16,
        lr=1e-4, patience=5, device='cpu', seed=0,
    )
    model = result['model']

    loader = PyGDataLoader(pyg, batch_size=16)
    batch  = next(iter(loader))
    out    = model(batch.z, batch.pos, batch.batch)
    loss   = F.mse_loss(out, batch.y.squeeze(-1))
    loss.backward()

    no_grad = [
        n for n, p in model.named_parameters()
        if p.grad is None or p.grad.abs().sum() == 0
    ]
    assert len(no_grad) == 0, f"Params without gradient: {no_grad}"


def test_tiny_overfit():
    """MSE should drop by ≥ 90% after 200 epochs of overfit on 16 molecules."""
    n_mols  = 16
    n_atoms = 6
    seed    = 7

    torch.manual_seed(seed)
    data_list = _make_pyg_list(n_mols, n_atoms=n_atoms, seed=seed)
    y_target  = torch.stack([d.y.squeeze() for d in data_list])  # (n_mols,)

    loader = PyGDataLoader(data_list, batch_size=n_mols)
    batch  = next(iter(loader))
    y_b    = batch.y.squeeze(-1)

    model     = UniMolRegressor()
    optimizer = optim_mod.Adam(model.parameters(), lr=1e-2)

    model.train()
    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        out  = model(batch.z, batch.pos, batch.batch)
        loss = F.mse_loss(out, y_b)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    drop = (losses[0] - losses[-1]) / (losses[0] + 1e-9)
    assert drop >= 0.90, (
        f"Expected ≥90% loss drop, got {drop:.2%} "
        f"(first={losses[0]:.4f} last={losses[-1]:.4f}). "
        "UniMol may lack capacity or LR is too low."
    )
