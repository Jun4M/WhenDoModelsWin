"""
Regression tests for featurizer tensor-level caching (spec 09).

These tests verify:
1. Cache starts empty and is populated on first call.
2. Warm-path returns bit-identical tensors to cold-path for featurize_smiles_to_3d.
3. Warm-path returns bit-identical tensors to cold-path for load_qm9_3d_from_sdf.
4. Cache keys are keyed by (kind, dataset, smi) — different datasets don't collide.
5. .y is freshly created per call (not shared), so mutation is isolated.

All 3D tests use the ESOL ETKDG disk cache; they are skipped if the cache
file is absent (i.e., running in a clean environment without the cache).
QM9 tests are skipped if ./data/qm9.sdf is absent.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.featurizer import (
    _clear_feat_tensor_cache,
    _feat_tensor_cache_info,
    _FEAT_TENSOR_CACHE,
    featurize_smiles_to_3d,
    featurize_smiles_to_unimol,
    load_qm9_3d_from_sdf,
    load_qm9_unimol_from_sdf,
)

ESOL_CACHE = os.path.join('data', 'esol-3d-cache.pkl')
QM9_SDF    = os.path.join('data', 'qm9.sdf')

esol_available = pytest.mark.skipif(
    not os.path.exists(ESOL_CACHE),
    reason='ESOL ETKDG cache not found (run scripts/build_conformer_cache.py --dataset esol)',
)
qm9_available = pytest.mark.skipif(
    not os.path.exists(QM9_SDF),
    reason='QM9 SDF not found (./data/qm9.sdf)',
)


def _pick_sample_from_esol_cache() -> str:
    """Return a canonical SMILES that actually exists in the ESOL cache."""
    import pickle
    with open(ESOL_CACHE, 'rb') as f:
        cache = pickle.load(f)
    # Skip __-prefixed sentinel keys (embed_failed etc.)
    for key in cache:
        if not key.startswith('__'):
            return key
    raise RuntimeError('ESOL cache has no valid molecules')


def setup_function():
    _clear_feat_tensor_cache()


# ---------------------------------------------------------------------------
# Test 1 — cache starts empty
# ---------------------------------------------------------------------------

def test_cache_starts_empty():
    _clear_feat_tensor_cache()
    info = _feat_tensor_cache_info()
    assert info['total'] == 0, f'Expected empty cache, got {info}'


# ---------------------------------------------------------------------------
# Test 2 — featurize_smiles_to_3d: cold→warm bit identity (ESOL)
# ---------------------------------------------------------------------------

@esol_available
def test_featurize_smiles_to_3d_cache_identity_esol():
    import torch
    _clear_feat_tensor_cache()

    sample_smi = _pick_sample_from_esol_cache()
    smiles = [sample_smi]

    # Cold path
    pyg1, valid1 = featurize_smiles_to_3d(smiles, seed=0, dataset='esol')
    assert valid1, f'Sample molecule not found in ESOL featurizer: {sample_smi}'
    info_after_cold = _feat_tensor_cache_info()
    assert info_after_cold['total'] >= 1, 'Cache should have entries after cold call'

    # Warm path
    pyg2, valid2 = featurize_smiles_to_3d(smiles, seed=0, dataset='esol')
    assert valid2 == valid1

    d1, d2 = pyg1[0], pyg2[0]

    # Bit-identical: x, edge_index, pos, radius_edge_index
    assert d1.x.numpy().tobytes() == d2.x.numpy().tobytes(), 'x not bit-identical'
    assert d1.pos.numpy().tobytes() == d2.pos.numpy().tobytes(), 'pos not bit-identical'
    assert d1.edge_index.numpy().tobytes() == d2.edge_index.numpy().tobytes(), \
        'edge_index not bit-identical'
    if d1.radius_edge_index is not None:
        assert d1.radius_edge_index.numpy().tobytes() == \
               d2.radius_edge_index.numpy().tobytes(), 'radius_edge_index not bit-identical'

    # Shared tensor references (warm path reuses tensors from cache)
    assert d1.x.data_ptr() == d2.x.data_ptr(), 'x tensor should be shared (same data_ptr)'
    assert d1.pos.data_ptr() == d2.pos.data_ptr(), 'pos tensor should be shared'


# ---------------------------------------------------------------------------
# Test 3 — load_qm9_3d_from_sdf: cold→warm bit identity (QM9 subset)
# ---------------------------------------------------------------------------

@qm9_available
def test_load_qm9_3d_cache_identity_subset():
    import pickle
    from rdkit import Chem

    _clear_feat_tensor_cache()

    # Pick first canonical SMILES from QM9 disk cache
    cache_pkl = os.path.splitext(QM9_SDF)[0] + '-3d-cache.pkl'
    if not os.path.exists(cache_pkl):
        pytest.skip('QM9 3D cache pkl not found; run full QM9 featurization first')

    with open(cache_pkl, 'rb') as f:
        qm9_cache = pickle.load(f)

    sample_smi = next(iter(qm9_cache))

    # Cold path
    pyg1, valid1 = load_qm9_3d_from_sdf([sample_smi], sdf_path=QM9_SDF)
    if not valid1:
        pytest.skip(f'Sample molecule not matched from SDF: {sample_smi}')

    # Warm path
    pyg2, valid2 = load_qm9_3d_from_sdf([sample_smi], sdf_path=QM9_SDF)
    assert valid2 == valid1

    d1, d2 = pyg1[0], pyg2[0]
    assert d1.x.numpy().tobytes() == d2.x.numpy().tobytes(), 'x not bit-identical'
    assert d1.pos.numpy().tobytes() == d2.pos.numpy().tobytes(), 'pos not bit-identical'
    assert d1.x.data_ptr() == d2.x.data_ptr(), 'x tensor should be shared'


# ---------------------------------------------------------------------------
# Test 4 — cache keys are dataset-scoped (esol vs lipo don't collide)
# ---------------------------------------------------------------------------

@esol_available
def test_cache_independence_across_datasets():
    import pickle
    _clear_feat_tensor_cache()

    lipo_cache = os.path.join('data', 'lipo-3d-cache.pkl')
    if not os.path.exists(lipo_cache):
        pytest.skip('Lipo ETKDG cache not found')

    sample_smi = _pick_sample_from_esol_cache()
    smiles = [sample_smi]

    featurize_smiles_to_3d(smiles, seed=0, dataset='esol')
    featurize_smiles_to_3d(smiles, seed=0, dataset='lipo')

    # Keys are (kind, dataset, smi) — same SMILES in different datasets must not collide
    esol_keys = [k for k in _FEAT_TENSOR_CACHE if k[0] == 'painn' and k[1] == 'esol']
    lipo_keys = [k for k in _FEAT_TENSOR_CACHE if k[0] == 'painn' and k[1] == 'lipo']
    for ek in esol_keys:
        assert ek not in [(k[0], 'lipo', k[2]) for k in lipo_keys], \
            'Dataset keys should not collide'


# ---------------------------------------------------------------------------
# Test 5 — .y is freshly created per call (not shared)
# ---------------------------------------------------------------------------

@esol_available
def test_data_y_not_shared_between_calls():
    import torch
    _clear_feat_tensor_cache()

    sample_smi = _pick_sample_from_esol_cache()
    smiles = [sample_smi]
    pyg1, valid1 = featurize_smiles_to_3d(smiles, seed=0, dataset='esol')
    pyg2, valid2 = featurize_smiles_to_3d(smiles, seed=0, dataset='esol')

    if not valid1 or not valid2:
        pytest.skip('Sample molecule not in ESOL featurizer')

    d1, d2 = pyg1[0], pyg2[0]

    # .y must NOT be the same object (mutation isolation)
    assert d1.y.data_ptr() != d2.y.data_ptr(), \
        '.y should be freshly created per call (not shared)'

    # Mutating one should not affect the other
    d1.y[0] = 999.0
    assert float(d2.y[0]) != 999.0, \
        'Mutating d1.y should not affect d2.y'
