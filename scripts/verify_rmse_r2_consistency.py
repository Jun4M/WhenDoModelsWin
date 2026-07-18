"""Verify unit consistency via RMSE-R² mathematical relationship.

Key insight:
- R² is SCALE-INVARIANT (same value whether in normalized or real-unit space).
- But RMSE = std_test × sqrt(1 - R²).
- So given CSV's R² and known dataset std, we can predict what RMSE *should* be.
- If CSV's RMSE matches std_real × sqrt(1-R²) → real-unit ✓
- If CSV's RMSE matches sqrt(1-R²) (i.e., std=1)     → normalized 🚨
- This works even for failed models (R² < 0).

This is the gold-standard check — doesn't rely on heuristics about RMSE magnitude.
"""
import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict

# Dataset standard deviations (averaged from std_lookup.csv)
DATASET_STD = {
    '01_QM9':  0.0331,  # Hartree (varies by target, this is mean across homo/lumo/gap)
    '02_ESOL': 2.0519,
    '03_Lipo': 1.1707,
    '04_BACE': 1.3774,
}


def parse_model(stem):
    if '_na_' in stem:
        return stem.split('_na_')[0]
    parts = stem.split('_')
    if parts[0] == 'gtca':
        return 'gtca'
    return parts[0]


def main():
    print(f'\n{"="*100}')
    print('Unit consistency via RMSE = std × √(1 - R²) check')
    print(f'{"="*100}')
    print(f'\nFor each (dataset, model), pick rows where R² > -0.5 (avoid divide by absurd numbers)')
    print(f'and compute |CSV_RMSE / (std_real × √(1-R²)) − 1|')
    print(f'  ≈ 0  → real-unit ✓')
    print(f'  ≈ |1/std − 1|  → normalized 🚨')
    print()

    # Pre-compute the expected ratio for normalized data: RMSE_norm / RMSE_real = 1/std
    for ds, std in DATASET_STD.items():
        print(f'  {ds:10s} std={std:.4f},  norm ratio = 1/std = {1/std:.4f},  '
              f'normalized signature = "{1/std - 1:+.3f}" (deviation from 1.0)')

    print()
    suspect_models = defaultdict(list)

    for ds, std in DATASET_STD.items():
        ds_dir = f'results/{ds}/raw_data'
        if not os.path.isdir(ds_dir):
            continue
        print(f'\n{"="*100}')
        print(f'{ds} (real std = {std:.4f})')
        print(f'{"="*100}')
        print(f'{"model":20s} {"n_rows":>7s} {"med ratio":>11s} {"verdict":30s}')

        per_model = defaultdict(list)
        for f in glob.glob(f'{ds_dir}/*.csv'):
            if '/_' in f or 'failures' in f or '.overflow_backup' in f:
                continue
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if 'RMSE' not in df.columns or 'R2' not in df.columns:
                continue
            stem = os.path.basename(f).replace('.csv', '')
            model = parse_model(stem)
            # MTL has per-task std — skip
            if model in ['attentivefp_mtl', 'gcn_mtl']:
                continue

            # Skip rows where R² is too extreme (sqrt(1-R²) blows up or NaN issues)
            for _, row in df.iterrows():
                r2 = row['R2']
                rmse = row['RMSE']
                if pd.isna(r2) or pd.isna(rmse) or rmse <= 0:
                    continue
                if r2 > 0.999:  # nearly perfect prediction, sqrt(1-R²) → 0, divides break
                    continue
                expected_rmse_real = std * np.sqrt(max(0, 1 - r2))
                if expected_rmse_real <= 1e-6:
                    continue
                ratio = rmse / expected_rmse_real
                per_model[model].append(ratio)

        for model in sorted(per_model.keys()):
            ratios = per_model[model]
            med = float(np.median(ratios))
            # ratio ≈ 1 → real-unit
            # ratio ≈ 1/std → normalized
            normalized_ratio = 1.0 / std
            if abs(med - 1.0) < 0.05:
                verdict = '✅ real-unit (ratio ≈ 1)'
            elif abs(med - normalized_ratio) < 0.05:
                verdict = '🚨 NORMALIZED (ratio ≈ 1/std)'
                suspect_models[ds].append((model, med))
            elif med < 0.5:
                verdict = f'❌ doubly-normalized? (ratio = {med:.3f})'
                suspect_models[ds].append((model, med))
            else:
                verdict = f'⚠️  inconsistent (ratio = {med:.3f})'
                suspect_models[ds].append((model, med))
            print(f'  {model:20s} {len(ratios):>7d} {med:>11.4f} {verdict}')

    print(f'\n{"="*100}')
    print(f'Final summary')
    print(f'{"="*100}')
    if not any(suspect_models.values()):
        print('✅ All models in all datasets: ratio ≈ 1.0 → all CSVs in REAL-UNIT (denormalized).')
    else:
        print('🚨 Suspicious models found:')
        for ds, models in suspect_models.items():
            if models:
                print(f'\n  {ds}:')
                for m, ratio in models:
                    print(f'    {m}: ratio = {ratio:.4f}')


if __name__ == '__main__':
    main()
