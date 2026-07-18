"""
compare_cross_device.py
N-way cross-device variance check (T4, L4, MPS, CPU) for foundation models.

This generalizes compare_cross_gpu.py to compare 2 or more devices, including
Mac MPS/CPU in addition to Colab T4/L4. Useful for verifying that backend
choice does not affect reported metrics across all hardware used in the
benchmark.

Usage:
    # Default: compare T4 L4 MPS CPU (expects all four folders)
    python compare_cross_device.py

    # Subset: only T4 vs L4 (mimics compare_cross_gpu.py)
    python compare_cross_device.py --devices T4 L4

    # Mac-internal: MPS vs CPU only, stricter threshold (bit-identical expected)
    python compare_cross_device.py --devices MPS CPU --seed_std_ref 0.005

    # Colab + Mac MPS (skip Mac CPU)
    python compare_cross_device.py --devices T4 L4 MPS

Expects:
    ./results_cross_{DEVICE}/{01_QM9,02_ESOL,03_Lipo,04_BACE}/raw_data/

Output:
    cross_device_verification.csv  -- long-format, one row per (model,dataset)
                                     with RMSE/R² columns for each device
    cross_device_pairs.csv         -- pair-format, one row per (model,dataset,pair)
                                     with abs_delta_RMSE / abs_delta_R²
"""

import os
import sys
import argparse
import itertools
import pandas as pd

# ---------------------------------------------------------------------------
MODELS = ['chemberta2', 'molformer', 'selformer']

CONFIGS = [
    ('01_QM9',  'homo'),
    ('01_QM9',  'lumo'),
    ('01_QM9',  'gap'),
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
# ---------------------------------------------------------------------------


def csv_path(device, ds_dir, model, target, seed):
    return os.path.join(f'./results_cross_{device}', ds_dir, 'raw_data',
                        f'{model}_na_{seed}_{target}.csv')


def read_row(path, n_train):
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    df = pd.read_csv(path)
    row = df[df['train_size'] == n_train]
    if row.empty:
        return None, f"n_train={n_train} not in {path}"
    return row.iloc[0], None


def main():
    p = argparse.ArgumentParser(
        description="N-way cross-device variance check for foundation models.")
    p.add_argument('--devices', nargs='+', default=['T4', 'L4', 'MPS', 'CPU'],
                   help="Device tags to compare (folder: ./results_cross_{TAG}/). "
                        "Default: T4 L4 MPS CPU")
    p.add_argument('--n_train', type=int, default=200,
                   help="Train size to compare (must exist in CSVs). Default: 200")
    p.add_argument('--seed', type=int, default=0,
                   help="Seed to compare. Default: 0")
    p.add_argument('--seed_std_ref', type=float, default=0.03,
                   help="Reference seed-to-seed std for flagging. Default 0.03")
    p.add_argument('--out_long', default='cross_device_verification.csv',
                   help="Long-format CSV (per-device columns).")
    p.add_argument('--out_pairs', default='cross_device_pairs.csv',
                   help="Pair-format CSV (per-pair rows).")
    args = p.parse_args()

    if len(args.devices) < 2:
        print("ERROR: Need at least 2 devices to compare. Got:", args.devices)
        sys.exit(1)

    # Verify folders
    missing = [d for d in args.devices if not os.path.isdir(f'./results_cross_{d}')]
    if missing:
        print(f"ERROR: Missing folders for devices: {missing}")
        print()
        print("Expected layout:")
        for d in missing:
            print(f"  ./results_cross_{d}/01_QM9/raw_data/...")
        print()
        print("Run experiments for each device first. Example commands:")
        print()
        for d in missing:
            device_flag = (
                "--device mps" if d == 'MPS' else
                "--device cpu" if d == 'CPU' else
                ""  # Colab T4/L4 use default cuda
            )
            print(f"  # {d}:")
            print(f"  # Run twice — once for QM9 (3 targets), once for the others (default targets)")
            print(f"  python run_colab_test.py \\")
            print(f"    --dataset qm9 --target homo lumo gap \\")
            print(f"    --train_sizes {args.n_train} --seeds {args.seed} \\")
            print(f"    --skip_gcn --skip_attentivefp --skip_gps --skip_transformer \\")
            print(f"    --skip_chemprop --skip_krovex --skip_painn \\")
            print(f"    --skip_rf --skip_xgb --skip_gpr --skip_svr --skip_lgbm \\")
            if device_flag:
                print(f"    {device_flag} \\")
            print(f"    --results_root ./results_cross_{d}")
            print()
            print(f"  python run_colab_test.py \\")
            print(f"    --dataset esol lipo bace \\")
            print(f"    --train_sizes {args.n_train} --seeds {args.seed} \\")
            print(f"    --skip_gcn --skip_attentivefp --skip_gps --skip_transformer \\")
            print(f"    --skip_chemprop --skip_krovex --skip_painn \\")
            print(f"    --skip_rf --skip_xgb --skip_gpr --skip_svr --skip_lgbm \\")
            if device_flag:
                print(f"    {device_flag} \\")
            print(f"    --results_root ./results_cross_{d}")
            print()
        sys.exit(1)

    # Read all values
    data = {}  # data[(model, ds_dir, target)][device] = {'RMSE': ..., 'R2': ...}
    warnings = []
    for model in MODELS:
        for ds_dir, target in CONFIGS:
            data[(model, ds_dir, target)] = {}
            for device in args.devices:
                path = csv_path(device, ds_dir, model, target, args.seed)
                row, err = read_row(path, args.n_train)
                if err:
                    warnings.append(
                        f"{device:5} | {model:11} | {DATASET_LABELS[ds_dir]:5} ({target[:10]}): {err}")
                    continue
                data[(model, ds_dir, target)][device] = {
                    'RMSE': float(row['RMSE']),
                    'R2':   float(row['R2']),
                }

    if warnings:
        print("WARNINGS (missing files):")
        for w in warnings:
            print(f"  {w}")
        print()

    # ---------- Long-format CSV ----------
    # NOTE: keys are (model, ds_dir, target) to distinguish QM9's 3 targets
    long_rows = []
    for model in MODELS:
        for ds_dir, target in CONFIGS:
            row = {'model': model, 'dataset': DATASET_LABELS[ds_dir], 'target': target}
            for device in args.devices:
                v = data[(model, ds_dir, target)].get(device)
                row[f'RMSE_{device}'] = round(v['RMSE'], 6) if v else None
                row[f'R2_{device}']   = round(v['R2'], 6) if v else None
            long_rows.append(row)
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(args.out_long, index=False)

    # ---------- Pair-format CSV ----------
    pair_rows = []
    for model in MODELS:
        for ds_dir, target in CONFIGS:
            entries = data.get((model, ds_dir, target), {})
            for dev_a, dev_b in itertools.combinations(args.devices, 2):
                if dev_a not in entries or dev_b not in entries:
                    continue
                rmse_a, r2_a = entries[dev_a]['RMSE'], entries[dev_a]['R2']
                rmse_b, r2_b = entries[dev_b]['RMSE'], entries[dev_b]['R2']
                pair_rows.append({
                    'model':           model,
                    'dataset':         DATASET_LABELS[ds_dir],
                    'target':          target,
                    'pair':            f'{dev_a} vs {dev_b}',
                    f'RMSE_{dev_a}':   round(rmse_a, 6),
                    f'RMSE_{dev_b}':   round(rmse_b, 6),
                    'abs_delta_RMSE':  round(abs(rmse_a - rmse_b), 6),
                    f'R2_{dev_a}':     round(r2_a, 6),
                    f'R2_{dev_b}':     round(r2_b, 6),
                    'abs_delta_R2':    round(abs(r2_a - r2_b), 6),
                })

    if not pair_rows:
        print("ERROR: No comparison pairs generated. Check folder population.")
        sys.exit(1)

    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(args.out_pairs, index=False)

    # ---------- Console summary: per-device measurements ----------
    print(f"\n=== Per-device measurements (seed={args.seed}, n_train={args.n_train}) ===")
    header = f"{'model':<12} {'dataset':<6} {'target':<14}  " + "  ".join(
        f"{'RMSE_' + d:>10}" for d in args.devices)
    print(header)
    print('-' * len(header))
    for _, r in df_long.iterrows():
        cells = []
        for d in args.devices:
            v = r.get(f'RMSE_{d}')
            cells.append(f"{v:>10.4f}" if (v is not None and not pd.isna(v))
                         else f"{'—':>10}")
        tgt = str(r['target'])[:13]  # truncate long target names like ESOL's
        print(f"{r['model']:<12} {r['dataset']:<6} {tgt:<14}  " + "  ".join(cells))

    # ---------- Pair-wise summary ----------
    print(f"\n=== Pairwise device comparison (|ΔRMSE|, seed-std ref ≈ {args.seed_std_ref}) ===")
    flagged_any = False
    for pair in df_pairs['pair'].unique():
        sub = df_pairs[df_pairs['pair'] == pair]
        max_drmse = sub['abs_delta_RMSE'].max()
        max_dr2 = sub['abs_delta_R2'].max()
        n_above = int((sub['abs_delta_RMSE'] > args.seed_std_ref).sum())
        flag = " (above seed std — investigate)" if max_drmse > args.seed_std_ref else " (within seed std)"
        if max_drmse > args.seed_std_ref:
            flagged_any = True
        print(f"  {pair:<20}  max |ΔRMSE| = {max_drmse:.5f}  "
              f"max |ΔR²| = {max_dr2:.5f}  "
              f"({n_above}/{len(sub)} above seed std){flag}")

    print()
    if flagged_any:
        print("RESULT: at least one pair exceeded the seed-to-seed std reference.")
        print("  Inspect cross_device_pairs.csv for the specific (model, dataset, pair) cells.")
    else:
        print("RESULT: all device pairs within seed-to-seed std reference.")
        print("  Cross-device variance is bounded; hardware choice does not affect reported metrics.")

    print(f"\nSaved: {args.out_long}  ({len(df_long)} rows, long format)")
    print(f"Saved: {args.out_pairs} ({len(df_pairs)} rows, pair format)")


if __name__ == '__main__':
    main()
