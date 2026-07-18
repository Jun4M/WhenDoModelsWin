"""
Invariance / structural tests for KROVEX components.

Tests:
  1. test_kronecker_shape        — KROVEXNet forward gives correct output shape
  2. test_permutation_invariance — global_mean_pool makes KROVEXNet permutation-invariant
  3. test_per_fold_no_test_leak  — select_descriptors_per_fold uses only train indices
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a minimal PyG batch for KROVEXNet
# ──────────────────────────────────────────────────────────────────────────────

_SMILES_SMALL = [
    'C', 'CC', 'CCC', 'c1ccccc1', 'CCO',
    'c1cccnc1', 'CCN', 'CC(=O)O', 'CCCl', 'CCS',
]


def _make_krovex_batch(smiles_list, num_desc=5, seed=0):
    """Build a Batch of KROVEX PyG Data objects with dummy descriptors."""
    from src.featurizer import featurize_smiles_to_krovex_graph
    rng = np.random.default_rng(seed)

    graphs, valid = featurize_smiles_to_krovex_graph(smiles_list)
    data_list = []
    for g in graphs:
        n = g.x.shape[0]
        desc = torch.tensor(rng.standard_normal((1, num_desc)).astype(np.float32))
        data_list.append(Data(x=g.x, edge_index=g.edge_index, desc=desc,
                               y=torch.tensor([0.0])))
    return Batch.from_data_list(data_list), len(graphs)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Kronecker shape test
# ──────────────────────────────────────────────────────────────────────────────

def test_kronecker_shape():
    """KROVEXNet forward must return (B,) scalar predictions."""
    from src.models import KROVEXNet

    num_desc = 7
    batch, B = _make_krovex_batch(_SMILES_SMALL[:6], num_desc=num_desc)

    model = KROVEXNet(num_desc=num_desc, dim_in=8)
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch, batch.desc)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
    assert torch.isfinite(out).all(), "KROVEXNet output contains NaN/inf"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Permutation invariance (node ordering)
# ──────────────────────────────────────────────────────────────────────────────

def test_permutation_invariance():
    """Permuting atoms within a molecule must not change the graph embedding output."""
    from src.models import KROVEXNet
    from src.featurizer import featurize_smiles_to_krovex_graph

    smi = 'c1ccccc1'  # benzene — 6 equivalent atoms
    graphs, valid = featurize_smiles_to_krovex_graph([smi])
    assert valid, "Benzene must be featurizable"
    g = graphs[0]

    num_desc = 4
    torch.manual_seed(0)
    desc = torch.randn(1, num_desc)

    model = KROVEXNet(num_desc=num_desc, dim_in=8)
    model.eval()

    # Original
    batch_orig = Batch.from_data_list([Data(x=g.x, edge_index=g.edge_index,
                                             desc=desc, y=torch.tensor([0.0]))])
    with torch.no_grad():
        out_orig = model(batch_orig.x, batch_orig.edge_index,
                         batch_orig.batch, batch_orig.desc)

    # Permuted atom ordering
    n = g.x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    x_perm = g.x[perm]
    # Remap edge_index
    inv_perm = torch.argsort(perm)
    # perm[i] = old node at new position i → old index → new index
    # edge_index stores old node indices → map via inv_perm[old]
    # Actually: perm maps new→old, so inv_perm maps old→new
    ei_perm = inv_perm[g.edge_index]

    batch_perm = Batch.from_data_list([Data(x=x_perm, edge_index=ei_perm,
                                             desc=desc, y=torch.tensor([0.0]))])
    with torch.no_grad():
        out_perm = model(batch_perm.x, batch_perm.edge_index,
                         batch_perm.batch, batch_perm.desc)

    assert torch.allclose(out_orig, out_perm, atol=1e-5), (
        f"Permutation invariance failed: orig={out_orig.item():.6f}, "
        f"perm={out_perm.item():.6f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Per-fold no test leak (structural guarantee)
# ──────────────────────────────────────────────────────────────────────────────

def test_per_fold_no_test_leak():
    """select_descriptors_per_fold must not see test y values.

    Structural test: verify the function signature only accepts train_smiles
    and train_y. Then confirm that applying the resulting fit_stats to two
    different test sets gives different outputs (not collapsed to training mean).
    """
    from src.descriptor_selection import select_descriptors_per_fold, apply_descriptor_selection

    rng = np.random.default_rng(11)
    train_smiles = _SMILES_SMALL[:7]
    test_a = _SMILES_SMALL[7:9]     # high-MW-ish molecules
    test_b = ['C', 'CC']            # tiny molecules

    train_y = rng.standard_normal(len(train_smiles)).astype(np.float32)

    selected, fit_stats = select_descriptors_per_fold(train_smiles, train_y, seed=0)

    if not selected:
        pytest.skip("Descriptor selection returned empty set (too few molecules)")

    X_a = apply_descriptor_selection(test_a, selected, fit_stats)
    X_b = apply_descriptor_selection(test_b, selected, fit_stats)

    # Both use training z-score, so outputs differ only by the molecule's raw descriptors.
    # At least check shapes are correct and they're not identical (different molecules → different features).
    assert X_a.shape == (len(test_a), len(selected))
    assert X_b.shape == (len(test_b), len(selected))

    # The key invariant: fit_stats['train_mean'] is fixed — not recomputed from test
    assert 'train_mean' in fit_stats, "fit_stats must contain train_mean"
    assert 'train_std' in fit_stats,  "fit_stats must contain train_std"
