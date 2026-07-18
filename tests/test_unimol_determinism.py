"""
test_unimol_determinism.py
Seed determinism: same seed → identical test_preds (ATOL=1e-5).
"""

import numpy as np
import torch
from torch_geometric.data import Data

from src.train import train_unimol


def _make_pyg_list(n_mols: int, seed: int = 0):
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        z   = torch.randint(1, 9, (8,), dtype=torch.long)
        pos = torch.randn(8, 3)
        y   = torch.randn(1)
        data_list.append(Data(z=z, pos=pos, y=y))
    return data_list


def test_seed_determinism():
    """Two runs with the same seed produce identical test_preds."""
    n_mols = 16
    pyg    = _make_pyg_list(n_mols, seed=42)
    y_np   = np.random.default_rng(42).standard_normal(n_mols).astype(np.float32)

    kwargs = dict(
        train_3d=pyg, val_3d=pyg, test_3d=pyg,
        train_y=y_np, val_y=y_np, test_y=y_np,
        target_name='test', epochs=3, batch_size=16,
        lr=1e-4, patience=5, device='cpu',
    )

    r1 = train_unimol(**kwargs, seed=77)
    r2 = train_unimol(**kwargs, seed=77)

    np.testing.assert_allclose(
        r1['test_preds'], r2['test_preds'], atol=1e-5,
        err_msg="UniMol test_preds differ between runs with the same seed",
    )
