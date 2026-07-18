"""Verify RMSE unit consistency across models per dataset.

Background:
- All models train in normalized (z-score) space.
- _apply_denorm() in run_learning_curve.py multiplies metrics by std before save_run_csv,
  so raw_data CSVs SHOULD be in real-unit:
    QM9: Hartree (×27.2114 → eV later in paper_csv)
    ESOL: log mol/L
    Lipo: logD
    BACE: pIC50
- Any model whose RMSE is ~1.0 (z-score std) likely escaped denormalize — silent failure.

Check approach:
1. For each (dataset, model), compute median RMSE at the LARGEST train_size.
2. Compare against expected dataset scale.
3. Flag models that look normalized (RMSE near 1.0) or way off others.
"""
import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict

EXPECTED = {
    '01_QM9':  {'unit': 'Hartree',   'rmse_low': 0.005, 'rmse_high': 0.10,
                'targets': ['homo', 'lumo', 'gap']},
    '02_ESOL': {'unit': 'log mol/L', 'rmse_low': 0.4,   'rmse_high': 3.0,
                'targets': None},
    '03_Lipo': {'unit': 'logD',      'rmse_low': 0.4,   'rmse_high': 1.5,
                'targets': None},
    '04_BACE': {'unit': 'pIC50',     'rmse_low': 0.4,   'rmse_high': 1.6,
                'targets': None},
}

# Models with known different scales (MTL has 12 tasks with mixed Hartree/Cv scales)
MULTI_SCALE_MODELS = {'attentivefp_mtl', 'gcn_mtl'}


def parse_model(stem):
    if '_na_' in stem:
        return stem.split('_na_')[0]
    parts = stem.split('_')
    if parts[0] == 'gtca':
        return 'gtca'
    return parts[0]


def main():
    print(f'\n{"="*80}')
    print(f'Unit Consistency Check — raw_data CSVs should be REAL-UNIT (post-denorm)')
    print(f'{"="*80}')
    print(f'\nExpected ranges per dataset:')
    for ds, info in EXPECTED.items():
        print(f'  {ds:10s} {info["unit"]:12s}  RMSE ∈ [{info["rmse_low"]}, {info["rmse_high"]}]')

    print(f'\nNormalized data warning signal: RMSE ≈ 1.0 (z-score std=1)')

    for ds, info in EXPECTED.items():
        ds_dir = f'results/{ds}/raw_data'
        if not os.path.isdir(ds_dir):
            continue

        # Group by model: collect (target, max_train_size_rows, RMSE values)
        per_model = defaultdict(list)
        for f in glob.glob(f'{ds_dir}/*.csv'):
            if '/_' in f or 'failures' in f or '.overflow_backup' in f:
                continue
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if 'train_size' not in df.columns or 'RMSE' not in df.columns:
                continue
            if df.empty:
                continue

            stem = os.path.basename(f).replace('.csv', '')
            model = parse_model(stem)

            # Take max train_size row(s) — averaged across seeds
            max_n = df['train_size'].max()
            max_rows = df[df['train_size'] == max_n]
            for _, row in max_rows.iterrows():
                per_model[model].append({
                    'target': stem.split('_')[-1],  # rough
                    'train_size': int(row['train_size']),
                    'RMSE': row['RMSE'],
                })

        print(f'\n{"="*80}')
        print(f'{ds}  (expected: {info["unit"]}, RMSE ∈ [{info["rmse_low"]}, {info["rmse_high"]}])')
        print(f'{"="*80}')
        print(f'{"model":25s} {"n_obs":>6s}  {"median_RMSE_at_maxN":>22s}  status')

        suspicious = []
        for model in sorted(per_model.keys()):
            obs = per_model[model]
            rmses = [o['RMSE'] for o in obs]
            med = float(np.median(rmses))
            in_range = info['rmse_low'] <= med <= info['rmse_high']

            # MTL has mixed scales — skip strict range check, just flag if all in normalize range
            if model in MULTI_SCALE_MODELS:
                status = f'⏭️  MTL (multi-scale, skipping range check)'
            elif 0.7 <= med <= 1.3:
                status = f'🔴 NORMALIZED? (RMSE ≈ 1.0)'
                suspicious.append((model, med))
            elif not in_range:
                if med < info['rmse_low']:
                    status = f'⚠️  too small (factor {info["rmse_low"]/med:.1f}× too small)'
                else:
                    status = f'⚠️  too large (factor {med/info["rmse_high"]:.1f}× too large)'
                suspicious.append((model, med))
            else:
                status = f'✓ in expected range'

            print(f'  {model:25s} {len(obs):>6d}  {med:>22.4f}  {status}')

        if suspicious:
            print(f'\n  🚩 Suspicious for {ds}:')
            for m, med in suspicious:
                print(f'     {m}: median RMSE = {med:.4f}')

    print(f'\n{"="*80}')
    print(f'Cross-check: lookup train std files (if available)')
    print(f'{"="*80}')
    # Try to infer expected std from std_lookup.csv
    if os.path.exists('std_lookup.csv'):
        std_df = pd.read_csv('std_lookup.csv')
        print(f'std_lookup.csv loaded: {len(std_df)} rows')
        print(f'Approximate train std per dataset (averaged across train_size, seed):')
        if 'dataset' in std_df.columns and 'train_std' in std_df.columns:
            per_ds_std = std_df.groupby('dataset')['train_std'].agg(['mean','std','min','max'])
            print(per_ds_std.to_string())
    else:
        print('(no std_lookup.csv — cannot cross-check)')


if __name__ == '__main__':
    main()
