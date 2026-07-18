"""
compare_cross_gpu.py
T4 vs L4 cross-GPU RMSE/R² variance check for foundation model baselines.

Usage:
    python compare_cross_gpu.py

Expects:
    ./results_cross_gpu_T4/{01_QM9,02_ESOL,03_Lipo,04_BACE}/raw_data/
    ./results_cross_gpu_L4/{01_QM9,02_ESOL,03_Lipo,04_BACE}/raw_data/

Output:
    cross_gpu_verification.csv   (12 rows)
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
MODELS = ['chemberta2', 'molformer', 'selformer']

CONFIGS = [
    ('01_QM9',  'homo'),
    ('02_ESOL', 'measured log solubility in mols per litre'),
    ('03_Lipo', 'exp'),
    ('04_BACE', 'pIC50'),
]

DATASET_LABELS = {
    '01_QM9':  'QM9',
    '02_ESOL': 'ESOL',
    '03_Lipo': 'Lipo',
    '04_BACE': 'BACE',
}

SEED   = 0
N_TRAIN = 200
ROOT_A   = './results_cross_MPS'
ROOT_B   = './results_cross_CPU'
LABEL_A  = 'MPS'
LABEL_B  = 'CPU'
OUT_CSV  = './cross_gpu_verification.csv'
# ---------------------------------------------------------------------------


def _csv_path(root, ds_dir, model, target):
    return os.path.join(root, ds_dir, 'raw_data',
                        f'{model}_na_{SEED}_{target}.csv')


def _read_row(path, n_train):
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    df = pd.read_csv(path)
    row = df[df['train_size'] == n_train]
    if row.empty:
        return None, f"n={n_train} not in {path}"
    return row.iloc[0], None


def main():
    # Sanity check roots
    missing = [r for r in (ROOT_A, ROOT_B) if not os.path.isdir(r)]
    if missing:
        print("ERROR: The following result folders are missing:")
        for m in missing:
            print(f"  {m}")
        print()
        print("Run the verification experiments first:")
        print(f"  {LABEL_A}:")
        print("    python run_colab_test.py \\")
        print("      --dataset qm9 esol lipo bace --train_sizes 200 --seeds 0 \\")
        print("      --skip_gcn --skip_attentivefp --skip_gps --skip_transformer \\")
        print("      --skip_chemprop --skip_krovex --skip_painn \\")
        print("      --skip_rf --skip_xgb --skip_gpr --skip_svr --skip_lgbm \\")
        print(f"      --results_root {ROOT_A}")
        print(f"  {LABEL_B}: same command with --results_root {ROOT_B}")
        sys.exit(1)

    rows = []
    errors = []

    for model in MODELS:
        for ds_dir, target in CONFIGS:
            path_a = _csv_path(ROOT_A, ds_dir, model, target)
            path_b = _csv_path(ROOT_B, ds_dir, model, target)

            row_a, err_a = _read_row(path_a, N_TRAIN)
            row_b, err_b = _read_row(path_b, N_TRAIN)

            if err_a or err_b:
                if err_a: errors.append(f"{LABEL_A} | {model} | {DATASET_LABELS[ds_dir]}: {err_a}")
                if err_b: errors.append(f"{LABEL_B} | {model} | {DATASET_LABELS[ds_dir]}: {err_b}")
                continue

            rmse_a, r2_a = float(row_a['RMSE']), float(row_a['R2'])
            rmse_b, r2_b = float(row_b['RMSE']), float(row_b['R2'])

            rows.append({
                'model':          model,
                'dataset':        DATASET_LABELS[ds_dir],
                'target':         target,
                'seed':           SEED,
                'n_train':        N_TRAIN,
                f'RMSE_{LABEL_A}':        round(rmse_a, 6),
                f'RMSE_{LABEL_B}':        round(rmse_b, 6),
                'abs_delta_RMSE': round(abs(rmse_a - rmse_b), 6),
                f'R2_{LABEL_A}':          round(r2_a, 6),
                f'R2_{LABEL_B}':          round(r2_b, 6),
                'abs_delta_R2':   round(abs(r2_a - r2_b), 6),
            })

    if errors:
        print("WARNINGS (missing files):")
        for e in errors:
            print(f"  {e}")
        print()

    if not rows:
        print("No comparison rows generated. Check that both result folders are populated.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # Console summary
    ra, rb = f'RMSE_{LABEL_A}', f'RMSE_{LABEL_B}'
    qa, qb = f'R2_{LABEL_A}',   f'R2_{LABEL_B}'
    header = (f"{'model':<14} {'dataset':<8} "
              f"{ra:>10} {rb:>10} {'|ΔRMSE|':>9}  "
              f"{qa:>8} {qb:>8} {'|ΔR²|':>8}")
    print(header)
    print('-' * len(header))
    for _, r in df.iterrows():
        print(f"{r['model']:<14} {r['dataset']:<8} "
              f"{r[ra]:>10.4f} {r[rb]:>10.4f} {r['abs_delta_RMSE']:>9.4f}  "
              f"{r[qa]:>8.4f} {r[qb]:>8.4f} {r['abs_delta_R2']:>8.4f}")

    print()
    max_drmse = df['abs_delta_RMSE'].max()
    max_dr2   = df['abs_delta_R2'].max()
    seed_std_ref = 0.03
    print(f"Max |ΔRMSE|  = {max_drmse:.4f}  (seed-to-seed std ref ≈ {seed_std_ref})")
    print(f"Max |ΔR²|    = {max_dr2:.4f}")
    flag = max_drmse > seed_std_ref
    print(f"Cross-GPU variance {'FLAG: exceeds seed std — investigate' if flag else 'OK: within seed std'}")
    print(f"\nSaved: {OUT_CSV}  ({len(df)} rows)")


if __name__ == '__main__':
    main()
