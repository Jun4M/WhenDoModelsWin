"""
data_loader.py
Unified dataset loader for QM9, ESOL, Lipophilicity, BACE.
Supports graph, ECFP4, and 3D featurization.
"""

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import pickle
import numpy as np
import deepchem as dc

from src.featurizer import (
    canonicalize_and_filter,
    featurize_smiles_to_graphs,
    featurize_smiles_to_ecfp,
    featurize_smiles_to_3d,
    featurize_smiles_to_unimol,
    load_qm9_3d_from_sdf,
    load_qm9_unimol_from_sdf,
    build_pyg_list,
    build_pyg_list_mtl,
)

# ---------------------------------------------------------------------------
# QM9 task definitions (kept for backward compat)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dataset train_size caps (avail // 3 constraint from scaffold split)
# ---------------------------------------------------------------------------

DATASET_MAX_TRAIN = {
    'qm9':  3000,
    'esol':  375,
    'lipo': 1000,
    'bace':  500,
}

def filter_valid_train_sizes(df, dataset: str):
    """Drop rows where train_size exceeds the effective cap for the dataset."""
    max_ts = DATASET_MAX_TRAIN.get(dataset, 3000)
    return df[df['train_size'] <= max_ts].copy()


QM9_TASKS = [
    'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap',
    'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom',
    'u298_atom', 'h298_atom', 'g298_atom', 'cv_atom',
]
TARGET_TASKS = ['homo', 'lumo', 'gap']

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    'qm9':  {'name': 'QM9',           'target_tasks': ['homo', 'lumo', 'gap']},
    'esol': {'name': 'ESOL',          'target_tasks': ['measured log solubility in mols per litre']},
    'lipo': {'name': 'Lipophilicity', 'target_tasks': ['exp']},
    'bace': {'name': 'BACE',          'target_tasks': ['pIC50']},
}

# ---------------------------------------------------------------------------
# Internal: raw loaders
# ---------------------------------------------------------------------------

def _load_qm9_raw(data_dir: str):
    """
    Returns (smiles_list, y_matrix [N, n_tasks], task_names).
    Uses DiskDataset cache if available to avoid re-reading the SDF file.
    """
    # DeepChem saves to: {save_dir}/qm9-featurized/CircularFingerprint/None/
    cache_path = os.path.join(data_dir, 'qm9-featurized', 'CircularFingerprint', 'None')

    if os.path.exists(os.path.join(cache_path, 'metadata.csv.gzip')):
        print(f"  [data_loader] Loading QM9 from cache: {cache_path}")
        ds = dc.data.DiskDataset(cache_path)
    else:
        print(f"  [data_loader] Featurizing QM9 from SDF (first time, ~5 min) ...")
        tasks, datasets, _ = dc.molnet.load_qm9(
            featurizer=dc.feat.CircularFingerprint(size=2048, radius=2),
            splitter=None,
            data_dir=data_dir,
            save_dir=data_dir,
            transformers=[],
        )
        ds = datasets[0]

    smiles = list(ds.ids)
    y = ds.y
    # Task names from metadata or fallback
    try:
        task_names = list(ds.tasks)
    except Exception:
        task_names = list(dc.molnet.load_qm9.__doc__ and QM9_TASKS or QM9_TASKS)
    return smiles, y, task_names


def _load_molnet_raw(loader_fn, data_dir: str, target: str):
    """Generic MoleculeNet loader. Returns (smiles, y [N,1], task_names)."""
    tasks, datasets, _ = loader_fn(
        featurizer=dc.feat.CircularFingerprint(size=2048, radius=2),
        splitter=None,
        data_dir=data_dir,
        save_dir=data_dir,
        transformers=[],
    )
    ds = datasets[0]
    smiles = list(ds.ids)
    task_list = list(tasks)
    print(f"  [data_loader] Available tasks: {task_list}")
    col = task_list.index(target) if target in task_list else 0
    if target not in task_list:
        print(f"  [warn] target '{target}' not found, using col 0 ({task_list[0]})")
    y = ds.y[:, col:col+1]
    return smiles, y, task_list


# ---------------------------------------------------------------------------
# Core: scaffold split
# ---------------------------------------------------------------------------

# In-process caches for scaffold grouping (Spec: scaffold-groups-inmem-cache,
# 2026-06-12). `_get_scaffold_groups` was previously called fresh on every
# (train_size, seed, target) cell, each time re-reading the on-disk pkl
# (~130k-entry dict for QM9) and rebuilding + sorting the groups list.
# These two caches make repeated calls within the same process near-free:
#   _SCAFFOLD_MAP_CACHE    : cache_path -> smi_to_scaf dict (skips disk I/O)
#   _SCAFFOLD_GROUPS_CACHE : (cache_path, id(smiles), len(smiles)) -> groups
# `id(smiles)` is stable because `run_learning_curve.py` loads `smiles` once
# per (dataset, target) via `preloaded_raw` and keeps it alive for the whole
# train_size x seed loop; `len(smiles)` is an extra cheap guard against the
# unlikely event of id() reuse after garbage collection.
_SCAFFOLD_MAP_CACHE: dict = {}
_SCAFFOLD_GROUPS_CACHE: dict = {}


def _get_scaffold_groups(smiles: list, cache_path: str) -> list:
    """
    Compute or load cached Murcko scaffold groups.
    Returns list of scaffold groups (each group is a list of indices).
    Groups are sorted descending by size (largest scaffold group first).

    Cache format v2: dict {smiles_str -> scaffold_smiles_str}.
    Portable across machines with different molecule orderings (e.g. Mac vs Colab
    Drive DiskDataset). The caller's `smiles` list is used to build indices on load,
    so the same dict gives the same *molecules* in each group regardless of order.

    Legacy v1 format (list[list[int]]) is index-based and NOT portable; if detected,
    it is silently discarded and recomputed in v2 format.

    In-process results are cached (see _SCAFFOLD_MAP_CACHE / _SCAFFOLD_GROUPS_CACHE)
    to avoid redundant disk reads + rebuilds across cells within one run.
    """
    def _build_groups_from_map(smiles, smi_to_scaf):
        scaf_to_indices = {}
        for i, smi in enumerate(smiles):
            scaf = smi_to_scaf.get(smi, smi)
            scaf_to_indices.setdefault(scaf, []).append(i)
        return sorted(scaf_to_indices.values(), key=len, reverse=True)

    groups_key = (cache_path, id(smiles), len(smiles))
    if groups_key in _SCAFFOLD_GROUPS_CACHE:
        return _SCAFFOLD_GROUPS_CACHE[groups_key]

    smi_to_scaf = _SCAFFOLD_MAP_CACHE.get(cache_path)

    if smi_to_scaf is None and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if isinstance(cached, dict):
            # v2: SMILES-keyed, portable
            smi_to_scaf = cached
        else:
            # v1: index-based — not portable across different SMILES orderings.
            # Discard and recompute as v2.
            print(f"  [data_loader] Scaffold pkl v1 (index-based) detected, "
                  f"recomputing as portable v2 ...")

    if smi_to_scaf is None:
        print(f"  [data_loader] Computing scaffold groups for {len(smiles)} molecules "
              f"(one-time, will cache to {os.path.basename(cache_path)}) ...")

        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError:
            raise ImportError("rdkit required for scaffold splitting. pip install rdkit-pypi")

        smi_to_scaf = {}
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaf = ''
            else:
                try:
                    scaf = MurckoScaffold.MurckoScaffoldSmiles(
                        mol=mol, includeChirality=False
                    )
                except Exception:
                    scaf = ''
            smi_to_scaf[smi] = scaf

        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(smi_to_scaf, f)
        n_unique = len(set(smi_to_scaf.values()))
        print(f"  [data_loader] Scaffold groups cached (v2): {n_unique} unique scaffolds.")

    _SCAFFOLD_MAP_CACHE[cache_path] = smi_to_scaf

    groups = _build_groups_from_map(smiles, smi_to_scaf)
    _SCAFFOLD_GROUPS_CACHE[groups_key] = groups
    return groups


def _clear_scaffold_groups_cache() -> None:
    """Clear in-process scaffold caches (used by tests / long-lived sessions)."""
    _SCAFFOLD_MAP_CACHE.clear()
    _SCAFFOLD_GROUPS_CACHE.clear()


def _scaffold_split(smiles: list, y: np.ndarray, train_size: int, val_size: int,
                    test_size: int, seed: int, cache_path: str = None):
    """
    Scaffold split using cached scaffold groups.
    Groups large scaffolds → test first, then val, then train (standard approach).
    Different seeds shuffle groups of equal size differently.
    Returns (train_idx, val_idx, test_idx) as lists of indices into smiles.
    """
    groups = _get_scaffold_groups(smiles, cache_path)
    total  = len(smiles)

    rng = np.random.default_rng(seed)

    # Shuffle groups of the same size for seed diversity
    # (sort by size desc, shuffle ties)
    from itertools import groupby
    shuffled = []
    for _, g in groupby(groups, key=len):
        chunk = list(g)
        perm  = rng.permutation(len(chunk))
        shuffled.extend([chunk[i] for i in perm])

    # Assign: fill test first (large scaffolds), then val, then train
    # This mirrors DeepChem's approach and ensures scaffold leakage is minimized
    test_idx, val_idx, train_idx = [], [], []

    for group in shuffled:
        if len(test_idx) < test_size:
            test_idx.extend(group)
        elif len(val_idx) < val_size:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    # Truncate to requested sizes
    train_idx = train_idx[:train_size]
    val_idx   = val_idx[:val_size]
    test_idx  = test_idx[:test_size]

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Core: featurize a split
# ---------------------------------------------------------------------------

def _build_split_dict(smiles_list: list, y_col: np.ndarray,
                       featurize_ecfp: bool = False, featurize_3d: bool = False,
                       featurize_unimol: bool = False,
                       dataset: str = '', seed: int = 0) -> dict:
    """Canonicalize + featurize. Returns aligned dict."""
    can_smiles, can_valid = canonicalize_and_filter(smiles_list)
    y_can = y_col[np.array(can_valid)]

    dc_graphs, graph_valid = featurize_smiles_to_graphs(can_smiles)
    y_graph = y_can[np.array(graph_valid)]
    smi_graph = [can_smiles[i] for i in graph_valid]
    X_graph = build_pyg_list(dc_graphs, y_graph)

    result = {
        'X_graph': X_graph,
        'X_ecfp':  None,
        'X_3d':    None,
        'X_3d_valid_idx': None,
        'X_unimol': None,
        'X_unimol_valid_idx': None,
        'y':   y_graph,
        'ids': smi_graph,
        'n':   len(smi_graph),
    }

    if featurize_ecfp:
        ecfp_mat, _ = featurize_smiles_to_ecfp(smi_graph)
        result['X_ecfp'] = ecfp_mat

    if featurize_3d:
        if dataset == 'qm9':
            # Use DFT-optimized coordinates from SDF (more accurate than ETKDG)
            pyg_3d, valid_3d = load_qm9_3d_from_sdf(smi_graph)
        else:
            pyg_3d, valid_3d = featurize_smiles_to_3d(smi_graph, seed=seed, dataset=dataset)
        result['X_3d'] = pyg_3d
        result['X_3d_valid_idx'] = valid_3d

    if featurize_unimol:
        if dataset == 'qm9':
            pyg_um, valid_um = load_qm9_unimol_from_sdf(smi_graph)
        else:
            pyg_um, valid_um = featurize_smiles_to_unimol(smi_graph, seed=seed, dataset=dataset)
        result['X_unimol'] = pyg_um
        result['X_unimol_valid_idx'] = valid_um

    return result


# ---------------------------------------------------------------------------
# Public: raw-only loader (call once, reuse across seeds/train_sizes)
# ---------------------------------------------------------------------------

def load_raw_data(dataset: str, data_dir: str, target: str = None):
    """
    Load raw (smiles, y_col, task_pos) without splitting or featurizing.
    Use this to pre-load once and pass as preloaded_raw= to load_dataset_splits.
    """
    dataset = dataset.lower()
    cfg = DATASET_CONFIGS[dataset]
    if target is None:
        target = cfg['target_tasks'][0]

    if dataset == 'qm9':
        smiles, y_matrix, task_names = _load_qm9_raw(data_dir)
        task_pos = task_names.index(target)
        y_col = y_matrix[:, task_pos]
    elif dataset == 'esol':
        smiles, y_matrix, task_names = _load_molnet_raw(dc.molnet.load_delaney, data_dir, target)
        task_pos = 0
        y_col = y_matrix[:, 0]
    elif dataset == 'lipo':
        smiles, y_matrix, task_names = _load_molnet_raw(dc.molnet.load_lipo, data_dir, target)
        task_pos = 0
        y_col = y_matrix[:, 0]
    elif dataset == 'bace':
        smiles, y_matrix, task_names = _load_molnet_raw(dc.molnet.load_bace_regression, data_dir, target)
        task_pos = 0
        y_col = y_matrix[:, 0]
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    return smiles, y_col, task_pos


# ---------------------------------------------------------------------------
# Public: unified loader
# ---------------------------------------------------------------------------

def load_dataset_splits(
    dataset: str = 'qm9',
    data_dir: str = './data',
    train_size: int = 100,
    val_size: int = 100,
    test_size: int = 10000,
    seed: int = 42,
    target: str = None,
    featurize_ecfp: bool = False,
    featurize_3d: bool = False,
    featurize_unimol: bool = False,
    preloaded_raw=None,  # (smiles, y_col, task_pos) from load_raw_data()
) -> dict:
    """
    Unified dataset loader. Returns:
    {
      'train': {'X_graph', 'X_ecfp', 'X_3d', 'y', 'ids', 'n'},
      'val':   {...},
      'test':  {...},
      'stats': (mean, std),
      'task_pos': int,
      'dataset': str,
      'target': str,
    }
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset = dataset.lower()

    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIGS)}")

    cfg = DATASET_CONFIGS[dataset]
    if target is None:
        target = cfg['target_tasks'][0]

    print(f"[data_loader] Loading {cfg['name']} (target={target}, seed={seed}) ...")

    if preloaded_raw is not None:
        smiles, y_col, task_pos = preloaded_raw
    else:
        if dataset == 'qm9':
            smiles, y_matrix, task_names = _load_qm9_raw(data_dir)
            if target not in task_names:
                raise ValueError(f"Target '{target}' not in QM9 tasks: {task_names}")
            task_pos = task_names.index(target)
            y_col = y_matrix[:, task_pos]
        elif dataset == 'esol':
            smiles, y_matrix, task_names = _load_molnet_raw(dc.molnet.load_delaney, data_dir, target)
            task_pos = 0
            y_col = y_matrix[:, 0]
        elif dataset == 'lipo':
            smiles, y_matrix, task_names = _load_molnet_raw(dc.molnet.load_lipo, data_dir, target)
            task_pos = 0
            y_col = y_matrix[:, 0]
        elif dataset == 'bace':
            smiles, y_matrix, task_names = _load_molnet_raw(dc.molnet.load_bace_regression, data_dir, target)
            task_pos = 0
            y_col = y_matrix[:, 0]

    print(f"[data_loader] Raw size: {len(smiles)}")

    avail = len(smiles)
    train_size = min(train_size, avail // 3)
    val_size   = min(val_size,   avail // 5)
    test_size  = min(test_size,  avail - train_size - val_size)

    scaffold_cache = os.path.join(data_dir, f'{dataset}_scaffold_groups.pkl')
    train_idx, val_idx, test_idx = _scaffold_split(
        smiles, y_col.reshape(-1, 1), train_size, val_size, test_size, seed,
        cache_path=scaffold_cache,
    )
    print(f"[data_loader] Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_smi = [smiles[i] for i in train_idx]
    val_smi   = [smiles[i] for i in val_idx]
    test_smi  = [smiles[i] for i in test_idx]
    train_y   = y_col[np.array(train_idx)]
    val_y     = y_col[np.array(val_idx)]
    test_y    = y_col[np.array(test_idx)]

    # Z-score normalization
    mean = float(np.mean(train_y))
    std  = float(np.std(train_y)) or 1.0
    train_y_n = (train_y - mean) / std
    val_y_n   = (val_y   - mean) / std
    test_y_n  = (test_y  - mean) / std

    train_ds = _build_split_dict(train_smi, train_y_n, featurize_ecfp, featurize_3d, featurize_unimol, dataset, seed)
    val_ds   = _build_split_dict(val_smi,   val_y_n,   featurize_ecfp, featurize_3d, featurize_unimol, dataset, seed)
    test_ds  = _build_split_dict(test_smi,  test_y_n,  featurize_ecfp, featurize_3d, featurize_unimol, dataset, seed)

    print(f"[data_loader] After feat: train={train_ds['n']}, val={val_ds['n']}, test={test_ds['n']}")

    return {
        'train':    train_ds,
        'val':      val_ds,
        'test':     test_ds,
        'stats':    (mean, std),
        'task_pos': task_pos,
        'dataset':  dataset,
        'target':   target,
    }


# ---------------------------------------------------------------------------
# Backward compat: legacy API for main.py
# ---------------------------------------------------------------------------

def load_qm9_splits(
    data_dir: str = './data',
    train_size: int = 100,
    val_size: int = 100,
    test_size: int = 10000,
    seed: int = 42,
    target: str = 'homo',
):
    """Legacy function. Returns (train_ds, val_ds, test_ds, stats, task_pos)."""
    result = load_dataset_splits(
        dataset='qm9',
        data_dir=data_dir,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        target=target,
        featurize_ecfp=False,
    )
    tr, va, te = result['train'], result['val'], result['test']
    return (
        {'X': tr['X_graph'], 'y': tr['y'].reshape(-1,1), 'ids': tr['ids']},
        {'X': va['X_graph'], 'y': va['y'].reshape(-1,1), 'ids': va['ids']},
        {'X': te['X_graph'], 'y': te['y'].reshape(-1,1), 'ids': te['ids']},
        result['stats'],
        result['task_pos'],
    )


def build_pyg_dataset(dataset: dict, task_pos: int = 0) -> list:
    """Legacy: extract PyG list from old-format dataset dict."""
    if 'X_graph' in dataset:
        return dataset['X_graph']
    if 'X' in dataset:
        return dataset['X']
    return []


# ---------------------------------------------------------------------------
# MTL: QM9 multi-task loader (12 properties)
# ---------------------------------------------------------------------------

# 12 QM9 molecular properties for MTL.
# Key: DeepChem task name. Name: pretty label for CSV / paper.
_QM9_MTL_TASK_KEYS = ['homo', 'lumo', 'gap', 'mu', 'alpha', 'zpve',
                       'u0', 'u298', 'h298', 'g298', 'cv', 'r2']
_QM9_MTL_TARGET_NAMES = ['homo', 'lumo', 'gap', 'mu', 'alpha', 'ZPVE',
                          'U0', 'U', 'H', 'G', 'Cv', 'R2']
assert len(_QM9_MTL_TASK_KEYS) == len(_QM9_MTL_TARGET_NAMES) == 12


def load_qm9_multitask(
    train_size: int,
    seed: int,
    data_dir: str = './data',
    val_size: int = 100,
    test_size: int = 10000,
) -> dict:
    """Load QM9 with 12 properties simultaneously for multi-task learning.

    Fairness guarantee: uses the *same scaffold cache + same seed* as the
    single-task loader → train/val/test contain the *same molecules* as
    load_dataset_splits(dataset='qm9', target='homo', train_size=train_size,
    seed=seed).  MTL sees the same N molecules; the advantage is purely
    the 12× label supervision per molecule.

    Per-task z-score normalization: each of the 12 columns is normalized
    independently using train-set statistics only.

    Adversarial guards:
      - NaN-safeguard: each task std clamped to ≥ 1e-6
      - stats/target_names alignment assertion
      - Column index assertion for QM9 task names

    Returns
    -------
    {
      'train': {'X_graph': list[PyG Data], 'y': np.ndarray(N_train, 12)},
      'val':   {'X_graph': list[PyG Data], 'y': np.ndarray(N_val, 12)},
      'test':  {'X_graph': list[PyG Data], 'y': np.ndarray(N_test, 12)},
      'stats': [(mean_0, std_0), ..., (mean_11, std_11)],  # per-task
      'target_names': ['homo', 'lumo', 'gap', 'mu', 'alpha', 'ZPVE',
                        'U0', 'U', 'H', 'G', 'Cv', 'R2'],
    }
    """
    n_tasks = len(_QM9_MTL_TASK_KEYS)

    # ── Load raw QM9 ─────────────────────────────────────────────────────────
    smiles, y_matrix_raw, task_names = _load_qm9_raw(data_dir)
    # Locate the 12 column indices (asserts all are found)
    try:
        col_indices = [task_names.index(k) for k in _QM9_MTL_TASK_KEYS]
    except ValueError as e:
        raise ValueError(
            f"QM9 task not found: {e}. Available: {task_names}. "
            "QM9 dataset may need to be re-featurized."
        )
    y_matrix = y_matrix_raw[:, col_indices].astype(np.float64)  # (N, 12)

    # ── Scaffold split (same cache + same seed → same molecules) ─────────────
    avail      = len(smiles)
    train_size = min(train_size, avail // 3)
    val_size   = min(val_size,   avail // 5)
    test_size  = min(test_size,  avail - train_size - val_size)

    scaffold_cache = os.path.join(data_dir, 'qm9_scaffold_groups.pkl')
    # Pass a dummy 1-col y — only used for type-check inside _scaffold_split
    train_idx, val_idx, test_idx = _scaffold_split(
        smiles, y_matrix[:, 0:1], train_size, val_size, test_size, seed,
        cache_path=scaffold_cache,
    )

    # ── Per-task z-score (train statistics only) ─────────────────────────────
    train_y_raw = y_matrix[np.array(train_idx)]   # (N_train, 12)
    stats = []
    for i in range(n_tasks):
        mu    = float(np.mean(train_y_raw[:, i]))
        sigma = max(float(np.std(train_y_raw[:, i])), 1e-6)
        stats.append((mu, sigma))

    # Adversarial: assert stats/target_names alignment
    assert len(stats) == len(_QM9_MTL_TARGET_NAMES), (
        f"stats ({len(stats)}) / target_names ({len(_QM9_MTL_TARGET_NAMES)}) mismatch"
    )

    def _normalize(y_raw):
        out = np.zeros_like(y_raw, dtype=np.float32)
        for i, (mu, sigma) in enumerate(stats):
            out[:, i] = ((y_raw[:, i] - mu) / sigma).astype(np.float32)
        return out

    train_y_n = _normalize(y_matrix[np.array(train_idx)])
    val_y_n   = _normalize(y_matrix[np.array(val_idx)])
    test_y_n  = _normalize(y_matrix[np.array(test_idx)])

    # ── Graph featurization ───────────────────────────────────────────────────
    def _build_mtl_split(smi_list, y_norm):
        can_smiles, can_valid = canonicalize_and_filter(smi_list)
        y_can  = y_norm[np.array(can_valid)]
        dc_graphs, graph_valid = featurize_smiles_to_graphs(can_smiles)
        y_graph = y_can[np.array(graph_valid)]  # (N_graph, 12)
        smi_graph = [can_smiles[i] for i in graph_valid]
        X_graph = build_pyg_list_mtl(dc_graphs, y_graph)
        return {'X_graph': X_graph, 'y': y_graph, 'ids': smi_graph, 'n': len(smi_graph)}

    train_smi = [smiles[i] for i in train_idx]
    val_smi   = [smiles[i] for i in val_idx]
    test_smi  = [smiles[i] for i in test_idx]

    train_ds = _build_mtl_split(train_smi, train_y_n)
    val_ds   = _build_mtl_split(val_smi,   val_y_n)
    test_ds  = _build_mtl_split(test_smi,  test_y_n)

    print(f"  [MTL] QM9 12-task split: train={train_ds['n']}, "
          f"val={val_ds['n']}, test={test_ds['n']}")

    return {
        'train':        train_ds,
        'val':          val_ds,
        'test':         test_ds,
        'stats':        stats,           # list[(mean_i, std_i)], index matches target_names
        'target_names': _QM9_MTL_TARGET_NAMES,
    }
