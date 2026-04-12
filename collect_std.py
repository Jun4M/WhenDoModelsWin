"""
collect_std.py
Scan existing raw_data files to find all (dataset, target, train_size, seed) combinations,
then compute train-set std for each without re-training.
Saves to std_lookup.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_raw_data, _scaffold_split

RESULTS_DIR = './results'
DATA_DIR    = './data'
OUTPUT_PATH = './std_lookup.csv'

FOLDER_TO_DATASET = {
    '01_QM9':  'qm9',
    '02_ESOL': 'esol',
    '03_Lipo': 'lipo',
    '04_BACE': 'bace',
}

DATASET_TARGETS = {
    'qm9':  ['homo', 'lumo', 'gap'],
    'esol': ['measured log solubility in mols per litre'],
    'lipo': ['exp'],
    'bace': ['pIC50'],
}

VAL_SIZE  = 100
TEST_SIZE = 10000


def parse_seed_target(fname, targets):
    """Extract (seed, target) from filename like '{model}_{depth}_{seed}_{target}.csv'."""
    for target in targets:
        suffix = f'_{target}.csv'
        if fname.endswith(suffix):
            prefix = fname[:-len(suffix)]
            parts = prefix.split('_')
            try:
                seed = int(parts[-1])
                return seed, target
            except ValueError:
                continue
    return None, None


def scan_raw_dirs(folder, dataset):
    """Return dict: (target, seed) -> set of train_sizes from all raw_data files."""
    targets = DATASET_TARGETS[dataset]
    combos  = {}

    dirs_to_scan = [
        os.path.join(RESULTS_DIR, folder, 'raw_data'),
        os.path.join(RESULTS_DIR, folder, 'fusion_study', 'raw_data'),
    ]

    for d in dirs_to_scan:
        if not os.path.exists(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.csv') or 'failures' in fname:
                continue
            seed, target = parse_seed_target(fname, targets)
            if seed is None:
                continue
            fpath = os.path.join(d, fname)
            try:
                df = pd.read_csv(fpath)
                sizes = set(df['train_size'].dropna().astype(int).tolist())
                combos.setdefault((target, seed), set()).update(sizes)
            except Exception:
                continue

    return combos


def compute_std(smiles, y_col, train_size, seed, cache_path):
    """Scaffold-split (no featurization), return std of train labels."""
    avail        = len(smiles)
    actual_train = min(train_size, avail // 3)
    actual_val   = min(VAL_SIZE,   avail // 5)
    actual_test  = min(TEST_SIZE,  avail - actual_train - actual_val)

    train_idx, _, _ = _scaffold_split(
        smiles,
        y_col.reshape(-1, 1),
        actual_train, actual_val, actual_test,
        seed,
        cache_path=cache_path,
    )

    train_y = y_col[np.array(train_idx)]
    std = float(np.std(train_y))
    return std if std > 0 else 1.0


def main():
    rows = []

    for folder, dataset in FOLDER_TO_DATASET.items():
        print(f'\n=== {dataset} ===')
        combos = scan_raw_dirs(folder, dataset)
        if not combos:
            print('  No raw_data files found, skipping.')
            continue

        targets      = DATASET_TARGETS[dataset]
        cache_path   = os.path.join(DATA_DIR, f'{dataset}_scaffold_groups.pkl')

        for target in targets:
            # Collect all (seed, train_size) pairs for this target
            pairs = []
            for (tgt, seed), sizes in combos.items():
                if tgt != target:
                    continue
                for ts in sizes:
                    pairs.append((seed, ts))
            if not pairs:
                continue

            print(f'  Loading raw data for target={target} ...')
            smiles, y_col, _ = load_raw_data(dataset, DATA_DIR, target=target)

            for seed, train_size in sorted(set(pairs)):
                std = compute_std(smiles, y_col, train_size, seed, cache_path)
                rows.append({
                    'dataset':    dataset,
                    'target':     target,
                    'train_size': train_size,
                    'seed':       seed,
                    'std_train':  std,
                })

            n = len(set(pairs))
            print(f'    {n} (seed, train_size) combos processed.')

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved {len(df)} rows → {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
