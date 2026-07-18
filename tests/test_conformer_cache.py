"""
Tests for scripts/build_conformer_cache.py and the updated featurize_smiles_to_3d().

Design notes:
  - All tests use build_cache_from_smiles() directly (not CLI main()) for isolation.
  - Timeout / NaN / unreasonable-coord tests use FakePool or unittest.mock.patch
    to avoid spawning slow/unreliable subprocesses.
  - featurize_smiles_to_3d tests inject the cache via _ETKDG_DISK_CACHE to
    avoid file I/O and dependency on data/ directory.
  - build_cache_from_smiles now returns (cache, fail_log, opt_log, stats).
"""

import multiprocessing
import os
import pickle
import time
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_SMILES = ['C', 'CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'Cn1cnc2c1c(=O)n(c(=O)n2C)C']
# methane, ethanol, benzene, aspirin, caffeine

_OPT_STATUSES = frozenset(
    {'mmff_converged', 'mmff_not_converged', 'uff_converged',
     'uff_not_converged', 'no_optimization',
     'restored_etkdg_after_optimization_corruption'}
)


@pytest.fixture(autouse=True)
def reset_etkdg_disk_cache():
    """Reset _ETKDG_DISK_CACHE before/after each test."""
    import src.featurizer as feat_mod
    feat_mod._ETKDG_DISK_CACHE.clear()
    yield
    feat_mod._ETKDG_DISK_CACHE.clear()


# ---------------------------------------------------------------------------
# (a) Build: 5 SMILES → all succeed, cache file created
# ---------------------------------------------------------------------------

def test_build_all_succeed():
    """build_cache_from_smiles returns entries for all 5 test molecules."""
    from scripts.build_conformer_cache import build_cache_from_smiles

    cache, fail_log, opt_log, stats = build_cache_from_smiles(
        TEST_SMILES, seed=42, workers=1, timeout=60,
    )

    mol_entries = {k: v for k, v in cache.items() if not k.startswith('__')}
    assert len(mol_entries) == len(TEST_SMILES), (
        f"Expected {len(TEST_SMILES)} entries, got {len(mol_entries)}. "
        f"fail_log={fail_log}"
    )
    assert stats['n_ok'] == len(TEST_SMILES)
    assert stats['n_fail'] == 0
    assert stats['n_timeout'] == 0
    assert not fail_log

    for can_smi, pos in mol_entries.items():
        assert isinstance(pos, np.ndarray), f"Expected ndarray for {can_smi}"
        assert pos.dtype == np.float32,     f"Expected float32 for {can_smi}"
        assert pos.ndim == 2 and pos.shape[1] == 3
        assert np.isfinite(pos).all(),      f"NaN/Inf in coords for {can_smi}"

    # Every cached molecule has an opt_log entry
    for can_smi in mol_entries:
        assert can_smi in opt_log, f"Missing opt_log for {can_smi}"
        assert opt_log[can_smi] in _OPT_STATUSES, (
            f"Unknown opt_status '{opt_log[can_smi]}' for {can_smi}"
        )

    assert '__rdkit_version__' in cache


# ---------------------------------------------------------------------------
# (b) Incremental: second run with same input → 0 new molecules processed
# ---------------------------------------------------------------------------

def test_incremental_skips_existing():
    """Re-running with the same SMILES list processes 0 new molecules."""
    from scripts.build_conformer_cache import build_cache_from_smiles

    cache1, _, opt_log1, stats1 = build_cache_from_smiles(
        TEST_SMILES, seed=42, workers=1, timeout=60,
    )
    assert stats1['n_ok'] == len(TEST_SMILES)

    cache2, fail2, opt_log2, stats2 = build_cache_from_smiles(
        TEST_SMILES, seed=42, workers=1, timeout=60,
        existing_cache=cache1, existing_opt_log=opt_log1,
    )

    assert stats2['n_skip'] == len(TEST_SMILES)
    assert stats2['n_ok'] == 0
    assert not fail2

    mol1 = {k: v for k, v in cache1.items() if not k.startswith('__')}
    mol2 = {k: v for k, v in cache2.items() if not k.startswith('__')}
    assert set(mol1.keys()) == set(mol2.keys())


# ---------------------------------------------------------------------------
# (c) Invalid SMILES → fail log records 'parse_failed'
# ---------------------------------------------------------------------------

def test_invalid_smiles_logged():
    """'not_a_smiles' → fail_log entry with reason 'parse_failed'."""
    from scripts.build_conformer_cache import build_cache_from_smiles

    bad_smi = 'not_a_smiles'
    cache, fail_log, opt_log, stats = build_cache_from_smiles(
        [bad_smi], seed=42, workers=1, timeout=30,
    )

    mol_entries = {k: v for k, v in cache.items() if not k.startswith('__')}
    assert len(mol_entries) == 0
    assert bad_smi in fail_log
    assert fail_log[bad_smi] == 'parse_failed'
    assert stats['n_fail'] == 1
    assert bad_smi not in opt_log


# ---------------------------------------------------------------------------
# (d) Timeout simulation: one molecule hangs → logged, pool reset, others OK
# ---------------------------------------------------------------------------

def test_timeout_logged_and_pool_reset():
    """
    FakePool makes one molecule time out. Verify:
      - that molecule is in fail_log as 'timeout'
      - the other molecule is still processed (n_ok == 1)
      - pool was reset (two FakePool instances created)
    """
    hang_smi = 'CC'
    ok_smi   = 'C'
    pools_created = []

    class FakeAsyncResult:
        def __init__(self, input_smi):
            self._smi = input_smi

        def get(self, timeout):
            if self._smi == hang_smi:
                raise multiprocessing.TimeoutError()
            mol_noh = Chem.RemoveHs(Chem.MolFromSmiles(self._smi))
            pos = np.zeros((mol_noh.GetNumAtoms(), 3), dtype=np.float32)
            return 'mmff_converged', pos

    class FakePool:
        def __init__(self, workers):
            pools_created.append(self)
            self.terminated = self.joined = False

        def apply_async(self, fn, args):
            return FakeAsyncResult(args[0])

        def terminate(self): self.terminated = True
        def join(self):      self.joined     = True

    from scripts.build_conformer_cache import build_cache_from_smiles

    cache, fail_log, opt_log, stats = build_cache_from_smiles(
        [hang_smi, ok_smi],
        seed=42, workers=1, timeout=5,
        pool_factory=FakePool,
    )

    assert fail_log.get(hang_smi) == 'timeout'
    assert stats['n_timeout'] == 1
    assert stats['n_ok'] == 1
    assert len(pools_created) == 2, "Pool must be reset after timeout"
    assert pools_created[0].terminated and pools_created[0].joined


# ---------------------------------------------------------------------------
# (e) Atom order consistency: same SMILES + seed → identical coords
# ---------------------------------------------------------------------------

def test_atom_order_consistency():
    """Two independent build_cache_from_smiles calls produce bit-identical coords."""
    from scripts.build_conformer_cache import build_cache_from_smiles

    smiles = ['c1ccccc1', 'CCO']
    cache1, _, _, _ = build_cache_from_smiles(smiles, seed=42, workers=1, timeout=60)
    cache2, _, _, _ = build_cache_from_smiles(smiles, seed=42, workers=1, timeout=60)

    m1 = {k: v for k, v in cache1.items() if not k.startswith('__')}
    m2 = {k: v for k, v in cache2.items() if not k.startswith('__')}

    assert set(m1) == set(m2), "Key sets differ"
    for key in m1:
        assert np.array_equal(m1[key], m2[key]), (
            f"Coordinates differ for {key}"
        )


# ---------------------------------------------------------------------------
# (f) featurize_smiles_to_3d: cache lookup is correct
# ---------------------------------------------------------------------------

def test_featurize_uses_cache():
    """Injected cache → featurize_smiles_to_3d produces PaiNNData with correct shapes."""
    import src.featurizer as feat_mod
    from src.featurizer import featurize_smiles_to_3d
    from scripts.build_conformer_cache import build_cache_from_smiles

    smiles = ['CCO', 'c1ccccc1']
    cache, _, _, _ = build_cache_from_smiles(smiles, seed=42, workers=1, timeout=60)
    feat_mod._ETKDG_DISK_CACHE['esol'] = cache

    pyg_list, valid = featurize_smiles_to_3d(smiles, dataset='esol')

    assert len(valid) == len(smiles)
    assert len(pyg_list) == len(smiles)
    for data in pyg_list:
        assert data.pos is not None
        assert data.pos.shape[1] == 3
        assert data.x.shape[0] == data.pos.shape[0]


# ---------------------------------------------------------------------------
# (g) Cache miss → molecule excluded from output
# ---------------------------------------------------------------------------

def test_cache_miss_excluded():
    """Cache has only ethanol; propanol is absent → excluded from featurize output."""
    import src.featurizer as feat_mod
    from src.featurizer import featurize_smiles_to_3d
    from scripts.build_conformer_cache import build_cache_from_smiles

    cache, _, _, _ = build_cache_from_smiles(['CCO'], seed=42, workers=1, timeout=60)
    feat_mod._ETKDG_DISK_CACHE['esol'] = cache

    pyg_list, valid = featurize_smiles_to_3d(['CCO', 'CCCO'], dataset='esol')

    assert len(valid) == 1
    assert 0 in valid
    assert 1 not in valid


# ---------------------------------------------------------------------------
# (h) MMFF forced failure → UFF fallback → molecule cached
# ---------------------------------------------------------------------------

def test_mmff_failure_falls_back_to_uff():
    """When MMFF raises, embed_one falls back to UFF and still returns coords."""
    from scripts.build_conformer_cache import embed_one

    smi = 'CCO'
    with patch.object(AllChem, 'MMFFOptimizeMolecule',
                      side_effect=Exception('MMFF mock failure')):
        status, coords = embed_one(smi, 42)

    assert coords is not None, f"Expected success despite MMFF failure, got status={status}"
    assert isinstance(coords, np.ndarray)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()
    assert 'uff' in status or status == 'no_optimization', (
        f"Expected UFF or no_optimization fallback, got '{status}'"
    )


# ---------------------------------------------------------------------------
# (i) MMFF + UFF both fail → molecule cached with 'no_optimization'
# ---------------------------------------------------------------------------

def test_mmff_and_uff_both_fail():
    """When both MMFF and UFF raise, molecule is still cached (raw ETKDG coords)."""
    from scripts.build_conformer_cache import embed_one

    smi = 'CCO'
    exc = Exception('optimization forced failure')
    with patch.object(AllChem, 'MMFFOptimizeMolecule', side_effect=exc), \
         patch.object(AllChem, 'UFFOptimizeMolecule',  side_effect=exc):
        status, coords = embed_one(smi, 42)

    assert coords is not None, "Should succeed with raw ETKDG coords"
    assert status == 'no_optimization', f"Expected no_optimization, got '{status}'"
    assert np.isfinite(coords).all()


# ---------------------------------------------------------------------------
# (j) NaN coordinate simulation → cache exclusion, fail_log 'nan_coords'
# ---------------------------------------------------------------------------

def test_nan_coords_excluded():
    """
    embed_one returning ('nan_coords', None) → excluded from cache, logged.
    Uses FakePool to inject the nan_coords scenario cleanly.
    """
    from scripts.build_conformer_cache import build_cache_from_smiles

    smi = 'CCO'

    class FakeAsyncResult:
        def get(self, timeout):
            return 'nan_coords', None

    class FakePool:
        def __init__(self, w): pass
        def apply_async(self, fn, args): return FakeAsyncResult()
        def terminate(self): pass
        def join(self):      pass

    cache, fail_log, opt_log, stats = build_cache_from_smiles(
        [smi], seed=42, workers=1, timeout=10, pool_factory=FakePool,
    )

    mol_entries = {k: v for k, v in cache.items() if not k.startswith('__')}
    assert len(mol_entries) == 0
    assert fail_log.get(smi) == 'nan_coords'
    assert stats['n_fail'] == 1
    assert smi not in opt_log


def test_nan_in_embed_one_directly():
    """
    When optimization produces NaN coordinates, embed_one restores from the
    pre-optimization ETKDG snapshot and returns
    ('restored_etkdg_after_optimization_corruption', valid_coords).
    """
    from scripts.build_conformer_cache import embed_one

    smi = 'CCO'
    original_array = np.array

    def inject_nan(data, dtype=None, **kwargs):
        arr = original_array(data, dtype=dtype, **kwargs)
        if dtype == np.float32 and getattr(arr, 'ndim', 0) == 2 and arr.shape[1:] == (3,):
            arr[:] = float('nan')
        return arr

    with patch('scripts.build_conformer_cache.np.array', side_effect=inject_nan):
        status, coords = embed_one(smi, 42)

    assert status == 'restored_etkdg_after_optimization_corruption', \
        f"Expected restored_etkdg_after_optimization_corruption, got '{status}'"
    assert coords is not None
    assert coords.dtype == np.float32
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert np.isfinite(coords).all()
    assert np.abs(coords).max() <= 100.0


# ---------------------------------------------------------------------------
# (k) Coordinate > 100 Å → 'unreasonable_coords'
# ---------------------------------------------------------------------------

def test_unreasonable_coords_rejected():
    """
    Coordinates where max(|x|) > 100 Å are rejected as 'unreasonable_coords'.
    Uses FakePool to inject the scenario via build_cache_from_smiles.
    """
    from scripts.build_conformer_cache import build_cache_from_smiles

    smi = 'CCO'

    class FakeAsyncResult:
        def get(self, timeout):
            return 'unreasonable_coords', None

    class FakePool:
        def __init__(self, w): pass
        def apply_async(self, fn, args): return FakeAsyncResult()
        def terminate(self): pass
        def join(self):      pass

    cache, fail_log, opt_log, stats = build_cache_from_smiles(
        [smi], seed=42, workers=1, timeout=10, pool_factory=FakePool,
    )

    mol_entries = {k: v for k, v in cache.items() if not k.startswith('__')}
    assert len(mol_entries) == 0
    assert fail_log.get(smi) == 'unreasonable_coords'
    assert stats['n_fail'] == 1


def test_unreasonable_coords_in_embed_one():
    """
    When optimization produces coordinates > 100 Å, embed_one restores from
    the pre-optimization ETKDG snapshot and returns
    ('restored_etkdg_after_optimization_corruption', valid_coords).
    """
    from scripts.build_conformer_cache import embed_one

    smi = 'CCO'
    original_array = np.array

    def inject_large(data, dtype=None, **kwargs):
        arr = original_array(data, dtype=dtype, **kwargs)
        if dtype == np.float32 and getattr(arr, 'ndim', 0) == 2 and arr.shape[1:] == (3,):
            arr[:] = 200.0   # > 100 Å threshold
        return arr

    with patch('scripts.build_conformer_cache.np.array', side_effect=inject_large):
        status, coords = embed_one(smi, 42)

    assert status == 'restored_etkdg_after_optimization_corruption', \
        f"Expected restored_etkdg_after_optimization_corruption, got '{status}'"
    assert coords is not None
    assert coords.dtype == np.float32
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert np.isfinite(coords).all()
    assert np.abs(coords).max() <= 100.0


# ---------------------------------------------------------------------------
# (l) MMFF corrupts conformer → ETKDG snapshot restored
# ---------------------------------------------------------------------------

def test_snapshot_restored_after_optimization_corruption():
    """
    When optimization produces NaN/unreasonable coords, embed_one falls back
    to the pre-optimization ETKDG snapshot and returns
    ('restored_etkdg_after_optimization_corruption', finite_coords).

    Mechanism: np.array is patched to inject NaN into the (N,3) float32
    extraction call that happens AFTER optimization.  The snapshot is taken
    via GetPositions().copy() (no np.array call), so it is not affected.
    Restoration via etkdg_snapshot[heavy_indices].astype(float32) also
    bypasses the patch, so the returned coords are finite.
    """
    from scripts.build_conformer_cache import embed_one

    smi = 'CCO'
    original_array = np.array

    def inject_nan(data, dtype=None, **kwargs):
        arr = original_array(data, dtype=dtype, **kwargs)
        if dtype == np.float32 and getattr(arr, 'ndim', 0) == 2 and arr.shape[1:] == (3,):
            arr[:] = float('nan')
        return arr

    with patch('scripts.build_conformer_cache.np.array', side_effect=inject_nan):
        status, coords = embed_one(smi, 42)

    assert status == 'restored_etkdg_after_optimization_corruption', (
        f"Expected restored status, got '{status}'"
    )
    assert coords is not None,            "Restored coords should not be None"
    assert coords.dtype == np.float32,    "Restored coords must be float32"
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert np.isfinite(coords).all(),     "Restored ETKDG snapshot must be finite"
    assert np.abs(coords).max() <= 100.0, "Restored snapshot should be within 100 Å"


# ---------------------------------------------------------------------------
# Adversarial Review
# ---------------------------------------------------------------------------

class AdversarialReview:
    """
    Silent failure modes for the fallback-chain conformer cache.

    (1) STERIC CLASHES IN UNOPTIMIZED CACHE ENTRIES
        --------------------------------------------
        Failure: A molecule where both MMFF and UFF fail (status='no_optimization')
        enters the cache with raw ETKDG coordinates.  ETKDG does not guarantee
        clash-free geometries for all molecules — in edge cases, atom distances
        may be unrealistically short (< 0.8 Å).  PaiNN uses Euclidean distances
        as input features; steric clashes produce very small distances and large
        interaction energies that act as noisy outliers in the training set.

        Defense (partial, implemented): the 100 Å upper-bound sanity check rejects
        physically unreasonable coordinates but does NOT catch atom clashes (which
        are too-small, not too-large).  To add a lower-bound check:

            from scipy.spatial.distance import pdist
            if len(coords) > 1 and pdist(coords).min() < 0.5:
                return 'steric_clash', None

        This is not implemented to avoid importing scipy in the worker; recommend
        adding if no_optimization entries exceed ~2% of the dataset.

    (2) RDKIT VERSION COORDINATE DRIFT
        --------------------------------
        Failure: ETKDGv3 coordinates are not bit-identical across RDKit versions.
        Cache built with RDKit 2023.9 and used by a process running 2024.3 produces
        identical keys but subtly different coordinates, silently breaking
        reproducibility. No crash; the model trains normally but results change.

        Defense (implemented): cache['__rdkit_version__'] stores the build-time
        version.  Recommended: add a version mismatch warning in _load_etkdg_cache:

            stored = data.get('__rdkit_version__', 'unknown')
            if stored != current_version:
                warnings.warn(f"Cache built with RDKit {stored}, ...")

        This is documented but not yet enforced — intentional, to avoid blocking
        runs when minor versions differ on the same major release.

    (3) OPTIMIZATION STATUS MISMATCH AND PaiNN LEARNING
        --------------------------------------------------
        Failure: training set contains molecules with wildly different geometry
        quality — some mmff_converged (high quality), some no_optimization (raw
        ETKDG).  One might worry PaiNN will learn spurious geometry artifacts.

        Why this is acceptable: PaiNN uses conformer coordinates solely to compute
        inter-atomic distances (via radius graph).  It does not receive the
        opt_status as input.  Raw ETKDG conformers are sufficient to establish
        connectivity patterns — they are not accurate enough for energy prediction
        but are adequate for property-prediction tasks (ESOL, Lipo, BACE) where
        the 3D input encodes molecular shape rather than precise geometries.

        Monitoring recommendation: include the opt_log file in experiment tracking.
        If no_optimization fraction exceeds ~5%, investigate whether those molecules
        are consistent outliers in prediction error.

    (4) LARGE MOLECULES (> 100 HEAVY ATOMS) WITH MEANINGLESS ETKDG COORDS
        --------------------------------------------------------------------
        Failure: For very large flexible molecules (e.g., macrocycles, lipids
        with long alkyl chains), ETKDG can produce extended or folded conformers
        that are far from the equilibrium geometry.  With no_optimization, these
        pass the 100 Å guard but encode the wrong shape.

        Defense: the 100 Å guard catches gross failures.  For more precision,
        add an optional --max_atoms CLI flag that skips molecules exceeding a
        heavy-atom count threshold:

            if mol.GetNumAtoms() > args.max_atoms:
                fail_log[smi] = 'too_large'
                continue

        ESOL (~20 atoms median), Lipo, and BACE are dominated by drug-like
        molecules (Lipinski-compliant, < 50 heavy atoms), so this is low priority.

    (5) MMFF→UFF COORD MUTATION AND NaN GUARD ORDERING
        --------------------------------------------------
        Failure: MMFF runs first and may produce near-NaN coordinates (very large
        but not exactly ±inf).  UFF then runs on the corrupted geometry and may
        produce actual ±inf.  The NaN guard `np.isfinite(coords).all()` catches
        ±inf as well as NaN (since np.isfinite returns False for both), so the
        molecule is correctly rejected.

        However, there is a subtle ordering issue: RemoveHs() is called AFTER
        both optimization steps.  If the optimized H-containing mol has NaN
        atoms, RemoveHs() might raise an exception before GetConformer() is
        called.  The current code is protected because embed_one's outer try/except
        in build_cache_from_smiles catches all exceptions from get(), but
        embed_one itself has no global try/except.  If RemoveHs() raises, the
        worker subprocess crashes, the async_result.get() raises a generic
        RemoteTraceback exception, and the molecule is logged as 'exception:...'
        in fail_log (not in cache).

        Defense (implemented indirectly): the 'exception:...' fail_log entry
        prevents the molecule from being silently included.  Recommendation:
        wrap the coord extraction block in embed_one with:

            try:
                mol_noh = Chem.RemoveHs(mol)
                coords  = np.array(mol_noh.GetConformer().GetPositions(), ...)
            except Exception:
                return 'extract_failed', None
    """
