"""
test_attentivefp_mtl_determinism.py
Seed determinism: same seed → identical test_preds (ATOL=1e-5).
"""

import numpy as np
import torch
from torch_geometric.data import Data

from src.train import train_attentivefp_mtl


N_TASKS    = 12
_TASK_NAMES = ['homo', 'lumo', 'gap', 'mu', 'alpha', 'ZPVE',
               'U0', 'U', 'H', 'G', 'Cv', 'R2']
_STATS = [(0.0, 1.0)] * N_TASKS


def _make_pyg_list(n_mols: int, n_tasks: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        n = 5
        x = torch.randn(n, 30)
        src = torch.arange(n)
        dst = torch.roll(src, 1)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_attr  = torch.randn(edge_index.shape[1], 11)
        y = torch.randn(1, n_tasks)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data_list


def _make_y(n_mols: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_mols, N_TASKS)).astype(np.float32)


def test_seed_determinism():
    n_mols = 16
    pyg    = _make_pyg_list(n_mols, seed=42)
    y_np   = _make_y(n_mols, seed=42)
    for i, d in enumerate(pyg):
        d.y = torch.tensor(y_np[i], dtype=torch.float32).unsqueeze(0)

    kwargs = dict(
        train_pyg=pyg, val_pyg=pyg, test_pyg=pyg,
        train_y_normalized=y_np, val_y_normalized=y_np, test_y_normalized=y_np,
        stats=_STATS, target_names=_TASK_NAMES, n_tasks=N_TASKS,
        epochs=3, batch_size=16, lr=1e-3, patience=5, device='cpu',
    )
    r1 = train_attentivefp_mtl(**kwargs, seed=99)
    r2 = train_attentivefp_mtl(**kwargs, seed=99)
    np.testing.assert_allclose(
        r1['test_preds'], r2['test_preds'], atol=1e-5,
        err_msg="MTL test_preds differ between runs with the same seed",
    )
