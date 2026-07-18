"""
test_unimol_invariance.py
SE(3)-invariance checks for UniMolRegressor:
  (a) test_output_shape: forward produces scalar per molecule
  (b) test_rotation_invariance: 3D rotation → identical output (atol=1e-5)
  (c) test_translation_invariance: global translation → identical output (atol=1e-5)
"""

import torch
import numpy as np
from torch_geometric.data import Data, Batch

from src.models import UniMolRegressor


def _make_batch(n_mols: int = 4, n_atoms: int = 8, seed: int = 0) -> Batch:
    """Build a random PyG Batch with .z and .pos attributes."""
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_mols):
        # Atomic numbers 1..8 (H to O range), shape (n_atoms,)
        z = torch.randint(1, 9, (n_atoms,), dtype=torch.long)
        pos = torch.randn(n_atoms, 3)
        data_list.append(Data(z=z, pos=pos, y=torch.tensor([0.0])))
    return Batch.from_data_list(data_list)


def _random_rotation(seed: int = 42) -> torch.Tensor:
    """Return a random 3×3 rotation matrix (QR decomposition)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return torch.tensor(Q, dtype=torch.float32)


def test_output_shape():
    """UniMolRegressor produces one scalar per molecule."""
    model = UniMolRegressor()
    model.eval()
    batch = _make_batch(n_mols=4)
    with torch.no_grad():
        out = model(batch.z, batch.pos, batch.batch)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_rotation_invariance():
    """Rotating all atom positions by a random 3D rotation → same predictions."""
    torch.manual_seed(7)
    model = UniMolRegressor()
    model.eval()

    batch = _make_batch(n_mols=3, n_atoms=6, seed=99)
    R = _random_rotation(seed=42)  # (3, 3)

    with torch.no_grad():
        out_orig = model(batch.z, batch.pos, batch.batch)
        batch_rot = Batch.from_data_list([
            Data(z=d.z, pos=d.pos @ R.T, y=d.y)
            for d in batch.to_data_list()
        ])
        out_rot = model(batch_rot.z, batch_rot.pos, batch_rot.batch)

    np.testing.assert_allclose(
        out_orig.numpy(), out_rot.numpy(), atol=1e-4,
        err_msg="UniMol predictions changed under 3D rotation — not SE(3)-invariant",
    )


def test_translation_invariance():
    """Translating all atom positions by a constant vector → same predictions."""
    torch.manual_seed(13)
    model = UniMolRegressor()
    model.eval()

    batch = _make_batch(n_mols=3, n_atoms=6, seed=77)
    shift = torch.tensor([5.0, -3.0, 2.0])   # Å displacement

    batch_shifted = Batch.from_data_list([
        Data(z=d.z, pos=d.pos + shift, y=d.y)
        for d in batch.to_data_list()
    ])

    with torch.no_grad():
        out_orig  = model(batch.z, batch.pos, batch.batch)
        out_shift = model(batch_shifted.z, batch_shifted.pos, batch_shifted.batch)

    np.testing.assert_allclose(
        out_orig.numpy(), out_shift.numpy(), atol=1e-4,
        err_msg="UniMol predictions changed under global translation — not translation-invariant",
    )
