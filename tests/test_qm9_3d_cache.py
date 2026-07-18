"""
Unit tests for _load_or_build_qm9_3d_cache() in src/featurizer.py.

Uses a tiny synthetic SDF (3 molecules, no real QM9 data needed) so tests
run in < 1 s without network access or the 500 MB QM9 file.

Adversarial scenarios covered in docstrings below (see AdversarialReview).
"""

import os
import pickle
import tempfile
import time
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Helpers: build minimal valid SDF bytes
# ---------------------------------------------------------------------------

def _make_sdf_bytes(smiles_list: list[str]) -> bytes:
    """
    Build a minimal SDF string for a list of SMILES.
    Each molecule gets a single ETKDGv3 conformer (deterministic, seed=42).
    Returns bytes so we can write to a temp file.
    """
    lines = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Bad SMILES: {smi}"
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        block = Chem.MolToMolBlock(mol)
        lines.append(block)
        lines.append("$$$$")
    return "\n".join(lines).encode()


# Three distinct small molecules for testing
_TEST_SMILES = ["C", "CC", "CCO"]


@pytest.fixture()
def sdf_and_cache_paths(tmp_path):
    """
    Yields (sdf_path, cache_path) pointing to a temp dir.
    SDF is written; cache does NOT exist yet (each test starts clean).
    """
    sdf_path   = str(tmp_path / "test_qm9.sdf")
    cache_path = str(tmp_path / "test_qm9-3d-cache.pkl")
    sdf_bytes  = _make_sdf_bytes(_TEST_SMILES)
    with open(sdf_path, "wb") as fh:
        fh.write(sdf_bytes)
    yield sdf_path, cache_path


# ---------------------------------------------------------------------------
# Reset the module-level in-process cache between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_inprocess_cache():
    """
    _QM9_3D_DISK_CACHE is a module-level global. Reset it before every test
    so that one test's warm cache does not bleed into the next.
    """
    import src.featurizer as feat_mod
    feat_mod._QM9_3D_DISK_CACHE = None
    yield
    feat_mod._QM9_3D_DISK_CACHE = None


# ---------------------------------------------------------------------------
# Test (a): First call — SDF parsed, cache file created
# ---------------------------------------------------------------------------

def test_first_call_builds_cache_file(sdf_and_cache_paths):
    """
    On first call the cache file must not exist beforehand.
    After the call it must exist and contain the correct number of entries.
    """
    from src.featurizer import _load_or_build_qm9_3d_cache

    sdf_path, cache_path = sdf_and_cache_paths
    assert not os.path.exists(cache_path), "Cache should not exist before first call"

    result = _load_or_build_qm9_3d_cache(sdf_path, cache_path)

    # Cache file was created
    assert os.path.exists(cache_path), "Cache file should be created after first call"

    # Returned dict has entries for all test molecules
    assert len(result) == len(_TEST_SMILES)

    # Each value is a float32 numpy array with shape (N_heavy, 3)
    for smi in _TEST_SMILES:
        can = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi)), canonical=True)
        assert can in result, f"Missing canonical SMILES: {can}"
        arr = result[can]
        assert isinstance(arr, np.ndarray), "Values must be np.ndarray"
        assert arr.dtype == np.float32, "Dtype must be float32"
        assert arr.ndim == 2 and arr.shape[1] == 3, "Shape must be (N_heavy, 3)"

    # .tmp file must not linger after atomic write
    assert not os.path.exists(cache_path + ".tmp"), "Temp file should be cleaned up"


# ---------------------------------------------------------------------------
# Test (b): Second call — loads from disk, < 5 s, identical coordinates
# ---------------------------------------------------------------------------

def test_second_call_is_fast_and_identical(sdf_and_cache_paths):
    """
    Second call (cache on disk, newer than SDF) must:
      1. Complete in < 5 s (effectively < 0.5 s for a tiny cache).
      2. Return coordinates bit-for-bit identical to the first call.
    """
    from src.featurizer import _load_or_build_qm9_3d_cache
    import src.featurizer as feat_mod

    sdf_path, cache_path = sdf_and_cache_paths

    # First call (builds cache)
    result_first = _load_or_build_qm9_3d_cache(sdf_path, cache_path)

    # Deep-copy canonical coords before resetting the in-process cache
    saved_coords = {k: v.copy() for k, v in result_first.items()}

    # Reset in-process cache to force a disk read on second call
    feat_mod._QM9_3D_DISK_CACHE = None

    t0 = time.perf_counter()
    result_second = _load_or_build_qm9_3d_cache(sdf_path, cache_path)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"Second call took {elapsed:.2f} s — expected < 5 s"

    # Coordinates must be bit-for-bit identical
    assert set(result_second.keys()) == set(saved_coords.keys()), "Key sets differ"
    for key, arr_orig in saved_coords.items():
        arr_new = result_second[key]
        assert np.array_equal(arr_orig, arr_new), (
            f"Coordinates differ for {key}: max diff = {np.abs(arr_orig - arr_new).max()}"
        )


# ---------------------------------------------------------------------------
# Test (c): SDF mtime touched to future → rebuild triggered
# ---------------------------------------------------------------------------

def test_stale_cache_triggers_rebuild(sdf_and_cache_paths):
    """
    If the SDF is newer than the cache file, the cache must be rebuilt from
    scratch (not served from disk).  We simulate this by touching the SDF
    after writing the cache.
    """
    from src.featurizer import _load_or_build_qm9_3d_cache
    import src.featurizer as feat_mod

    sdf_path, cache_path = sdf_and_cache_paths

    # First call: build cache
    _load_or_build_qm9_3d_cache(sdf_path, cache_path)
    feat_mod._QM9_3D_DISK_CACHE = None

    # Touch SDF to a future mtime (cache is now stale)
    future_mtime = time.time() + 10
    os.utime(sdf_path, (future_mtime, future_mtime))

    # Second call must detect staleness and rebuild
    # We verify indirectly: rebuild path re-writes the cache (new mtime)
    cache_mtime_before = os.path.getmtime(cache_path)

    result = _load_or_build_qm9_3d_cache(sdf_path, cache_path)

    cache_mtime_after = os.path.getmtime(cache_path)
    assert cache_mtime_after > cache_mtime_before, (
        "Cache mtime should be newer after a forced rebuild"
    )

    # Rebuilt result still has all entries
    assert len(result) == len(_TEST_SMILES)


# ---------------------------------------------------------------------------
# Test (d): Only .tmp file present (crashed mid-write) → normal rebuild
# ---------------------------------------------------------------------------

def test_tmp_only_triggers_rebuild(sdf_and_cache_paths):
    """
    If a previous run crashed after writing .tmp but before os.replace(),
    only the .tmp file exists (not the final cache file).
    The function must ignore the orphaned .tmp and do a full rebuild.

    Adversarial note: the code never reads *.tmp — os.path.exists(cache_path)
    checks the *final* path, so an orphaned .tmp is simply invisible to the
    cache-hit branch.  This test verifies that assumption holds.
    """
    from src.featurizer import _load_or_build_qm9_3d_cache

    sdf_path, cache_path = sdf_and_cache_paths
    tmp_path = cache_path + ".tmp"

    # Write a corrupt/partial payload into the .tmp file only
    with open(tmp_path, "wb") as fh:
        fh.write(b"CORRUPT_PARTIAL_WRITE")

    # Final cache file does NOT exist
    assert not os.path.exists(cache_path)
    assert os.path.exists(tmp_path)

    # Must succeed: build from SDF, overwrite tmp with real data
    result = _load_or_build_qm9_3d_cache(sdf_path, cache_path)

    assert len(result) == len(_TEST_SMILES), "Should build a full valid cache"
    assert os.path.exists(cache_path), "Final cache file should be created"

    # Verify the written cache is loadable and correct
    with open(cache_path, "rb") as fh:
        on_disk = pickle.load(fh)
    assert isinstance(on_disk, dict) and len(on_disk) == len(_TEST_SMILES)


# ---------------------------------------------------------------------------
# AdversarialReview
# ---------------------------------------------------------------------------

class AdversarialReview:
    """
    Five silent failure modes documented for human review.
    These are *not* automated tests because they require either the real SDF,
    a different RDKit version, or OS-level race conditions.

    (1) ATOM ORDER MISMATCH
        Risk: The SDF conformer has all-H atom ordering, but the graph
        featurizer (MolGraphConvFeaturizer) operates on the heavy-atom-only
        mol (RemoveHs).  If atom index ordering differs between the SDF mol
        and the featurized mol, pos rows would be misaligned with node features.
        Mitigation in code: `heavy_pos.shape[0] != x.shape[0]` guard in
        load_qm9_3d_from_sdf (line ~356) rejects mismatches, and
        canonical SMILES lookup ensures the same molecule.
        Residual risk: two non-isomorphic heavy-atom orderings with the same
        count would pass the guard but silently misalign features↔pos.
        Recommendation: for production, add an explicit atom-by-atom symbol
        cross-check between the SDF heavy-atom sequence and the featurized mol.

    (2) DTYPE INCONSISTENCY
        Risk: Code stores float32 in cache but PaiNN's .pos tensor is created
        via `torch.tensor(heavy_pos)` without explicit dtype.  PyTorch infers
        float32 from np.float32, so currently safe.  If cache dtype changes
        to float64 in a future edit, .pos becomes float64 while model weights
        remain float32, causing a runtime dtype error deep in PaiNN forward.
        Recommendation: add `dtype=torch.float32` explicitly in
        `pos_tensor = torch.tensor(heavy_pos, dtype=torch.float32)`.

    (3) RDKIT VERSION DIFFERENCES
        Risk: `Chem.MolToSmiles` canonical output is not guaranteed across
        RDKit versions.  A cache built with RDKit 2023 may use different
        canonical SMILES than a lookup in RDKit 2024, causing cache misses
        for every molecule (zero hits, silently falls back to slow SDF).
        Mitigation: None in current code.
        Recommendation: store the rdkit.__version__ in the cache dict under
        a reserved key (e.g. "__rdkit_version__"), and invalidate if it changes.

    (4) PARTIAL WRITE + CONCURRENT READ (race condition)
        Risk: Two processes start simultaneously (e.g., two seeds launched in
        parallel on a cluster).  Both check the cache, find it missing, both
        start building.  os.replace() is atomic on POSIX but both processes
        still spend 90 s each building the same cache in parallel.
        Mitigation in code: os.replace() is atomic, so the final file is always
        valid (last writer wins).  No corruption risk.
        Residual risk: doubled wall-clock time on first run if parallelized.
        Recommendation: use a file-based lock (fcntl.flock) before building,
        and have the second process wait and then load from disk.

    (5) ESOL/LIPO CACHE CONFUSION
        Risk: cache_path is derived as `os.path.splitext(sdf_path)[0] + '-3d-cache.pkl'`.
        If someone calls load_qm9_3d_from_sdf with a path like
        `./data/qm9_subset.sdf` and another call uses `./data/qm9.sdf`,
        the module-level _QM9_3D_DISK_CACHE singleton would return the first
        call's dict for the second call, silently looking up wrong coordinates.
        The _QM9_3D_DISK_CACHE global does NOT check which SDF it was built from.
        Recommendation: key the in-process cache by (sdf_path, cache_path)
        tuple instead of using a bare global, or document that
        load_qm9_3d_from_sdf must only be called with one SDF path per process.
    """
