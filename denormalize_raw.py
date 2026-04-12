"""
denormalize_raw.py
De-normalize RMSE and MAE in raw_data CSV files.

Steps:
1. Load std_lookup.csv
2. For each dataset:
   - Rename raw_data/ → raw_data_normalized/  (backup)
   - Rename fusion_study/raw_data/ → fusion_study/raw_data_normalized/
   - Create new raw_data/ with RMSE *= std, MAE *= std
   - Create new fusion_study/raw_data/ same
3. Rebuild summary CSVs

Pearson_R and R2 are scale-invariant → unchanged.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = './results'
STD_LOOKUP  = './std_lookup.csv'

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_seed_target(fname, targets):
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


def get_std(lookup_df, dataset, target, train_size, seed):
    row = lookup_df[
        (lookup_df['dataset']    == dataset) &
        (lookup_df['target']     == target) &
        (lookup_df['train_size'] == train_size) &
        (lookup_df['seed']       == seed)
    ]
    if row.empty:
        print(f'  [WARN] std not found: {dataset}/{target}/size={train_size}/seed={seed} → using 1.0')
        return 1.0
    return float(row['std_train'].iloc[0])


# ---------------------------------------------------------------------------
# Core: rename + denorm one directory
# ---------------------------------------------------------------------------

def denorm_directory(raw_dir, dataset, targets, lookup_df):
    norm_dir = raw_dir + '_normalized'

    if not os.path.exists(raw_dir) and not os.path.exists(norm_dir):
        return  # nothing to do

    # Already processed in a previous run → work from norm_dir
    if os.path.exists(norm_dir) and os.path.exists(raw_dir):
        print(f'  [SKIP] Both {raw_dir} and {norm_dir} exist — already processed?')
        return

    # Step 1: rename original → backup
    if os.path.exists(raw_dir):
        shutil.move(raw_dir, norm_dir)
        print(f'  Backed up: {os.path.basename(raw_dir)} → {os.path.basename(norm_dir)}')

    # Step 2: create new raw_dir
    os.makedirs(raw_dir, exist_ok=True)

    n_ok, n_skip, n_warn = 0, 0, 0

    for fname in sorted(os.listdir(norm_dir)):
        src = os.path.join(norm_dir, fname)
        dst = os.path.join(raw_dir,  fname)

        # Non-target CSVs (failures.csv etc.) and non-CSV files: copy as-is
        if not fname.endswith('.csv') or 'failures' in fname:
            shutil.copy2(src, dst)
            n_skip += 1
            continue

        seed, target = parse_seed_target(fname, targets)
        if seed is None:
            shutil.copy2(src, dst)
            n_skip += 1
            continue

        df = pd.read_csv(src)

        for i, row in df.iterrows():
            train_size = int(row['train_size'])
            std = get_std(lookup_df, dataset, target, train_size, seed)
            if std == 1.0:
                n_warn += 1
            df.at[i, 'RMSE'] = row['RMSE'] * std
            df.at[i, 'MAE']  = row['MAE']  * std
            # Pearson_R, R2 unchanged

        df.to_csv(dst, index=False)
        n_ok += 1

    print(f'  Denormed {n_ok} files, copied {n_skip}, warnings {n_warn}')


# ---------------------------------------------------------------------------
# Rebuild summaries
# ---------------------------------------------------------------------------

def rebuild_summaries():
    from src.summary import rebuild_summary_baselines, BASELINE_MODELS

    for folder, dataset in FOLDER_TO_DATASET.items():
        targets  = DATASET_TARGETS[dataset]
        base_dir = os.path.join(RESULTS_DIR, folder)
        raw_dir  = os.path.join(base_dir, 'raw_data')
        sum_dir  = os.path.join(base_dir, 'summary')

        for target in targets:
            print(f'  Rebuilding baselines summary: {dataset}/{target}')
            if os.path.exists(raw_dir):
                rebuild_summary_baselines(raw_dir, sum_dir, target)

    print('\n  NOTE: fusion_study and depth_study summaries need separate rebuild.')
    print('  Run run_final_comparison.py / run_depth_study.py, or rebuild manually.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(STD_LOOKUP):
        print(f'ERROR: {STD_LOOKUP} not found. Run collect_std.py first.')
        sys.exit(1)

    lookup_df = pd.read_csv(STD_LOOKUP)
    print(f'Loaded std_lookup: {len(lookup_df)} rows\n')

    for folder, dataset in FOLDER_TO_DATASET.items():
        print(f'=== {dataset} ===')
        targets  = DATASET_TARGETS[dataset]
        base_dir = os.path.join(RESULTS_DIR, folder)

        # Baseline raw_data
        denorm_directory(
            os.path.join(base_dir, 'raw_data'),
            dataset, targets, lookup_df,
        )

        # Fusion study raw_data
        denorm_directory(
            os.path.join(base_dir, 'fusion_study', 'raw_data'),
            dataset, targets, lookup_df,
        )

    print('\n=== Rebuilding baseline summaries ===')
    rebuild_summaries()

    print('\nDone. Backup folders: raw_data_normalized/ in each dataset dir.')


if __name__ == '__main__':
    main()
