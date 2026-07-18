"""
Offline ETKDG conformer builder for ESOL / Lipophilicity / BACE.

Output files (all in data/):
  {dataset}-3d-cache.pkl        dict[canonical_smiles] → np.ndarray(N_heavy,3) float32
  {dataset}-3d-fail.json        hard failures only (parse/embed/nan/unreasonable)
  {dataset}-3d-optimization.json  per-molecule optimization status (informational)

Run once before training PaiNN on non-QM9 datasets:
    python scripts/build_conformer_cache.py --dataset esol
    python scripts/build_conformer_cache.py --dataset lipo
    python scripts/build_conformer_cache.py --dataset bace

Key design decisions:
  - All molecules through subprocess pool — no atom-count branching.
  - embed_one uses MMFF → UFF fallback: ETKDG success already gives valid
    coordinates; optimization is best-effort and never discards the molecule.
  - Hard failures: only parse_failed / embed_failed / nan_coords /
    unreasonable_coords.  mmff_not_converged / no_optimization are soft states
    that still enter the cache.
  - Timeout → pool.terminate() + join() + new Pool().
  - Atomic write (*.tmp → os.replace).
  - Incremental: already-cached canonical SMILES are skipped on re-run.
"""

import argparse
import json
import multiprocessing
import os
import pickle
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Worker (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def embed_one(smi: str, seed: int):
    """
    ETKDG + MMFF→UFF fallback conformer generation.

    Step 1 — parse SMILES; Step 2 — ETKDG embed (hard failure on -1);
    Step 3 — best-effort MMFF→UFF optimization (never discards);
    Step 4 — coord extraction + sanity checks.

    Returns:
        (opt_status, coords)  — success; opt_status ∈ {mmff_converged,
                                  mmff_not_converged, uff_converged,
                                  uff_not_converged, no_optimization,
                                  restored_etkdg_after_optimization_corruption};
                                  coords: np.ndarray (N_heavy, 3) float32
        (fail_reason, None)   — hard failure; fail_reason ∈ {parse_failed,
                                  embed_failed}
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Step 1: parse
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 'parse_failed', None
    mol = Chem.AddHs(mol)

    # Step 2: ETKDG embed
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = 1000
    if AllChem.EmbedMolecule(mol, params) == -1:
        return 'embed_failed', None
    # From here: a conformer always exists.
    # Snapshot the ETKDG coords now — MMFF/UFF modify the conformer in-place,
    # so if optimization corrupts them we can fall back to this known-good state.
    etkdg_snapshot = mol.GetConformer().GetPositions().copy()  # (N_all, 3) float64

    # Step 3: best-effort optimization — MMFF first, UFF as fallback
    opt_status = None
    try:
        r = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        opt_status = 'mmff_converged' if r == 0 else 'mmff_not_converged'
    except Exception:
        pass  # opt_status stays None → try UFF

    if opt_status in (None, 'mmff_not_converged'):
        try:
            r = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            opt_status = ('uff_converged'     if r == 0
                          else 'uff_not_converged' if opt_status is None
                          else opt_status)   # mmff_not_converged is more informative
        except Exception:
            if opt_status is None:
                opt_status = 'no_optimization'

    if opt_status is None:
        opt_status = 'no_optimization'

    # Step 4: extract coords + sanity checks
    mol_noh = Chem.RemoveHs(mol)
    coords = np.array(mol_noh.GetConformer().GetPositions(), dtype=np.float32)

    if not np.isfinite(coords).all() or np.abs(coords).max() > 100.0:
        # Optimization corrupted the conformer — restore from ETKDG snapshot.
        # Use explicit heavy-atom indices into mol (the AddHs version) so the
        # mapping to etkdg_snapshot rows is unambiguous regardless of atom ordering.
        heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        coords = etkdg_snapshot[heavy_indices].astype(np.float32)
        opt_status = 'restored_etkdg_after_optimization_corruption'

    return opt_status, coords


# ---------------------------------------------------------------------------
# Core builder (injectable pool_factory for testing)
# ---------------------------------------------------------------------------

_HARD_FAILURES = frozenset(
    {'parse_failed', 'embed_failed', 'nan_coords', 'unreasonable_coords', 'timeout'}
)


def build_cache_from_smiles(
    smiles_list: list,
    seed: int = 42,
    workers: int = 4,
    timeout: int = 10,
    existing_cache: dict = None,
    existing_fail_log: dict = None,
    existing_opt_log: dict = None,
    pool_factory=None,
) -> tuple:
    """
    Process smiles_list and return (cache, fail_log, opt_log, stats).

    cache:    canonical_smi → np.ndarray(N_heavy, 3) float32
              plus '__rdkit_version__' metadata key
    fail_log: input_smi → hard-failure reason (parse/embed/nan/unreasonable/timeout)
    opt_log:  canonical_smi → opt_status string (informational; not a failure)
    stats:    {n_ok, n_timeout, n_fail, n_skip, n_mmff, n_uff, n_unopt}

    pool_factory: callable(workers) → pool-like object.
                  Override in tests to avoid real subprocess overhead.
    """
    from rdkit import Chem
    try:
        from rdkit import __version__ as rdkit_version
    except ImportError:
        rdkit_version = 'unknown'

    if pool_factory is None:
        pool_factory = multiprocessing.Pool

    cache   = dict(existing_cache)   if existing_cache   else {}
    fail_log = dict(existing_fail_log) if existing_fail_log else {}
    opt_log  = dict(existing_opt_log)  if existing_opt_log  else {}

    already_cached = {k for k in cache if not k.startswith('__')}

    # Build todo list (smi, can) — skip already-cached or failed molecules
    todo = []
    n_skip = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        can = None
        if mol is not None:
            try:
                can = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
            except Exception:
                pass

        if (can and can in already_cached) or smi in fail_log:
            n_skip += 1
        else:
            todo.append((smi, can))

    n_ok = n_timeout = n_fail = n_mmff = n_uff = n_unopt = 0
    pool = pool_factory(workers)

    for smi, can in todo:
        async_result = pool.apply_async(embed_one, (smi, seed))
        try:
            status, coords = async_result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            fail_log[smi] = 'timeout'
            n_timeout += 1
            pool.terminate()
            pool.join()
            pool = pool_factory(workers)
            continue
        except Exception as exc:
            fail_log[smi] = f'exception:{exc}'
            n_fail += 1
            continue

        if coords is None:
            # Hard failure — log under the original input SMILES
            fail_log[smi] = status
            n_fail += 1
        else:
            # Success — use pre-computed canonical key when available
            cache_key = can if can is not None else smi
            cache[cache_key] = coords
            opt_log[cache_key] = status
            n_ok += 1
            if 'mmff' in status:
                n_mmff += 1
            elif 'uff' in status:
                n_uff += 1
            else:
                n_unopt += 1

    pool.terminate()
    pool.join()

    cache['__rdkit_version__'] = rdkit_version

    stats = {
        'n_ok': n_ok, 'n_timeout': n_timeout, 'n_fail': n_fail, 'n_skip': n_skip,
        'n_mmff': n_mmff, 'n_uff': n_uff, 'n_unopt': n_unopt,
    }
    return cache, fail_log, opt_log, stats


# ---------------------------------------------------------------------------
# Dataset SMILES loader
# ---------------------------------------------------------------------------

def _load_dataset_smiles(dataset: str, data_dir: str) -> list:
    """Return all SMILES for the dataset via data_loader.load_raw_data."""
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.data_loader import load_raw_data
    smiles, _, _ = load_raw_data(dataset, data_dir)
    return list(smiles)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build offline ETKDG conformer cache for PaiNN (non-QM9 datasets).'
    )
    parser.add_argument('--dataset',  required=True, choices=['esol', 'lipo', 'bace'])
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--timeout',  type=int, default=10)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--data_dir', default='./data')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    cache_path   = os.path.join(args.data_dir, f'{args.dataset}-3d-cache.pkl')
    fail_path    = os.path.join(args.data_dir, f'{args.dataset}-3d-fail.json')
    opt_log_path = os.path.join(args.data_dir, f'{args.dataset}-3d-optimization.json')

    def _load_json(path):
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _load_pickle(path):
        if os.path.exists(path):
            try:
                with open(path, 'rb') as fh:
                    return pickle.load(fh)
            except Exception as exc:
                print(f"  Warning: could not load {path} ({exc}) — starting fresh")
        return {}

    existing_cache = _load_pickle(cache_path)
    n_prev = sum(1 for k in existing_cache if not k.startswith('__'))
    if n_prev:
        print(f"  Loaded existing cache: {n_prev} entries")

    existing_fail    = _load_json(fail_path)
    existing_opt_log = _load_json(opt_log_path)
    if existing_fail:
        print(f"  Loaded existing fail log: {len(existing_fail)} entries")

    print(f"Loading {args.dataset.upper()} SMILES ...")
    smiles_list = _load_dataset_smiles(args.dataset, args.data_dir)
    print(f"  {len(smiles_list)} total molecules")

    try:
        from tqdm import tqdm
        smiles_list = tqdm(smiles_list, desc=f'Building {args.dataset} cache')
    except ImportError:
        pass

    cache, fail_log, opt_log, stats = build_cache_from_smiles(
        smiles_list,
        seed=args.seed,
        workers=args.workers,
        timeout=args.timeout,
        existing_cache=existing_cache,
        existing_fail_log=existing_fail,
        existing_opt_log=existing_opt_log,
    )

    def _atomic_write_pickle(obj, path):
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _atomic_write_json(obj, path):
        tmp = path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)

    _atomic_write_pickle(cache,   cache_path)
    _atomic_write_json(fail_log,  fail_path)
    _atomic_write_json(opt_log,   opt_log_path)

    n_mols = sum(1 for k in cache if not k.startswith('__'))
    print(
        f"\nDone. cached {n_mols} total — "
        f"mmff {stats['n_mmff']}, uff {stats['n_uff']}, "
        f"unoptimized {stats['n_unopt']}; "
        f"failed {stats['n_fail'] + stats['n_timeout']} "
        f"(timeout={stats['n_timeout']}, other={stats['n_fail']})"
    )
    if fail_log:
        from collections import Counter
        print(f"Failure breakdown: {dict(Counter(fail_log.values()))}")


if __name__ == '__main__':
    main()
