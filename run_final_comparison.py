"""
run_final_comparison.py
Merge best GTCA depth variant with baseline results for final paper figure.

After running run_depth_study.py and choosing the best bert_depth, run this to:
  1. Load summary_baselines.csv + summary_gtca_depth.csv
  2. Filter to the selected GTCA depth
  3. Merge and save summary_final.csv
  4. Generate plot_combined_final_{target}.png

Usage:
  python run_final_comparison.py --dataset qm9 --target homo --best_depth 4
  python run_final_comparison.py --dataset qm9 --target homo lumo gap --best_depth 4
  python run_final_comparison.py --dataset esol --target measured_log_solubility --best_depth 6

  # Show depth summary table first (no --best_depth needed)
  python run_final_comparison.py --dataset qm9 --target homo --show_summary
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import DATASET_CONFIGS
from src.visualization import plot_combined_final


# ---------------------------------------------------------------------------
# Dataset directory mapping
# ---------------------------------------------------------------------------

DATASET_DIRS = {
    'qm9':  '01_QM9',
    'esol': '02_ESOL',
    'lipo': '03_Lipo',
    'bace': '04_BACE',
}


# ---------------------------------------------------------------------------
# Show depth summary
# ---------------------------------------------------------------------------

def show_depth_summary(summary_dir: str, target: str):
    """Print a quick depth comparison table to help choose best_depth."""
    path = os.path.join(summary_dir, 'summary_gtca_depth.csv')
    if not os.path.exists(path):
        print(f"  [warn] {path} not found — run run_depth_study.py first.")
        return

    df = pd.read_csv(path)
    df = df[df['target'] == target] if 'target' in df.columns else df

    if df.empty:
        print(f"  [warn] No data for target={target} in {path}")
        return

    print(f"\n  GTCA depth summary ({target}):")
    print(f"  {'model':<20} {'train_size':>10} {'RMSE_mean':>12} {'MAE_mean':>10} {'R2_mean':>10} {'n_seeds':>8}")
    print(f"  {'-'*72}")

    df_sorted = df.sort_values(['model', 'train_size'])
    for _, row in df_sorted.iterrows():
        print(f"  {row['model']:<20} {int(row['train_size']):>10} "
              f"{row['RMSE_mean']:>12.4f} {row['MAE_mean']:>10.4f} "
              f"{row.get('R2_mean', float('nan')):>10.4f} {int(row.get('n_seeds', 0)):>8}")


# ---------------------------------------------------------------------------
# Merge and produce final summary
# ---------------------------------------------------------------------------

def merge_final(
    summary_dir: str,
    target: str,
    best_depth: int,
    plots_dir: str,
    baseline_models: list = None,
):
    baseline_path = os.path.join(summary_dir, 'summary_baselines.csv')
    depth_path    = os.path.join(summary_dir, 'summary_gtca_depth.csv')

    frames = []

    # Load baselines
    if os.path.exists(baseline_path):
        df = pd.read_csv(baseline_path)
        if 'target' in df.columns:
            df = df[df['target'] == target]
        if baseline_models:
            df = df[df['model'].isin(baseline_models)]
        frames.append(df)
    else:
        print(f"  [warn] {baseline_path} not found")

    # Load GTCA depth, filter to best_depth
    if os.path.exists(depth_path):
        df = pd.read_csv(depth_path)
        if 'target' in df.columns:
            df = df[df['target'] == target]
        # Keep only the selected depth
        depth_label = f'gtca_depth_{best_depth}'
        df = df[df['model'] == depth_label]
        # Rename to 'gtca' for clean display in the final figure
        df = df.copy()
        df['model'] = f'GTCA (depth={best_depth})'
        frames.append(df)
    else:
        print(f"  [warn] {depth_path} not found")

    if not frames:
        print("  [error] No data to merge.")
        return None

    merged = pd.concat(frames, ignore_index=True)

    # Save
    out_path = os.path.join(summary_dir, f'summary_final_{target}.csv')
    merged.to_csv(out_path, index=False)
    print(f"  [saved] {out_path}")

    # Plot
    try:
        os.makedirs(plots_dir, exist_ok=True)
        plot_combined_final(merged, plots_dir, target, best_depth=best_depth)
    except Exception as e:
        print(f"  [ERROR] plot_combined_final: {e}")

    return merged


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Final comparison: baselines + best GTCA depth")
    p.add_argument('--dataset', nargs='+', default=['qm9'],
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument('--target', nargs='+', default=None,
                   help="Targets to run. Default: all targets for each dataset.")
    p.add_argument('--best_depth', type=int, default=None,
                   help="Selected GTCA bert_depth to include in final figure.")
    p.add_argument('--baseline_models', nargs='+', default=None,
                   help="Which baselines to include (default: all available).")
    p.add_argument('--results_root', default='./results')
    p.add_argument('--show_summary', action='store_true',
                   help="Print depth summary table and exit (no best_depth required).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    for dataset in args.dataset:
        cfg     = DATASET_CONFIGS[dataset]
        targets = args.target if args.target else cfg['target_tasks']

        dataset_dir = os.path.join(args.results_root, DATASET_DIRS.get(dataset, dataset))
        summary_dir = os.path.join(dataset_dir, 'summary')
        plots_dir   = os.path.join(dataset_dir, 'plots')

        for target in targets:
            print(f"\n{'='*60}")
            print(f" Dataset={dataset} | Target={target}")
            print(f"{'='*60}")

            if args.show_summary or args.best_depth is None:
                show_depth_summary(summary_dir, target)
                if args.best_depth is None:
                    print("  Pass --best_depth N to generate the final comparison figure.")
                    continue

            merge_final(
                summary_dir=summary_dir,
                target=target,
                best_depth=args.best_depth,
                plots_dir=plots_dir,
                baseline_models=args.baseline_models,
            )

    print("\nDone!")


if __name__ == '__main__':
    main()
