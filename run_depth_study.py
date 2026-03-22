"""
run_depth_study.py
GTCA bert_depth ablation study: bert_depth in [2, 4, 6] (ChemBERTa-zinc-base-v1, 6 layers).

Results structure:
  results/{DATASET_DIR}/
  ├── raw_data/        {model}_{depth}_{seed}_{target}.csv  (shared with baselines)
  ├── depth_study/     depth_comparison.csv
  ├── summary/         summary_gtca_depth.csv
  └── plots/           plot_gtca_depth_lc_{target}.png

Usage:
  python run_depth_study.py --dataset qm9 --target homo --device mps
  python run_depth_study.py --dataset qm9 --target homo lumo gap \\
      --depths 2 4 6 --epochs_gtca 5 --train_sizes 50 100  # smoke test
  python run_depth_study.py --resume --dataset esol --device mps
"""

import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset_splits, DATASET_CONFIGS
from src.train import train_gtca
from src.summary import (
    save_run_csv, run_already_done, rebuild_summary_gtca_depth,
)
from src.visualization import plot_gtca_depth_lc


# ---------------------------------------------------------------------------
# Adaptive train sizes (same as run_learning_curve.py)
# ---------------------------------------------------------------------------

def get_train_sizes() -> list:
    sizes = list(range(50, 501, 25))
    sizes += list(range(600, 1001, 100))
    sizes += list(range(1500, 3001, 500))
    return sorted(set(sizes))


def get_seed_schedule(train_size: int) -> list:
    return list(range(10)) if train_size <= 500 else list(range(3))


# ---------------------------------------------------------------------------
# Dataset directory mapping
# ---------------------------------------------------------------------------

DATASET_DIRS = {
    'qm9':  '01_QM9',
    'esol': '02_ESOL',
    'lipo': '03_Lipo',
    'bace': '04_BACE',
}

GTCA_DEPTHS = [2, 4, 6]


# ---------------------------------------------------------------------------
# Single GTCA depth experiment
# ---------------------------------------------------------------------------

def run_one_depth(
    train_size: int,
    target: str,
    seed: int,
    bert_depth: int,
    dataset: str,
    raw_dir: str,
    data_dir: str,
    device: str,
    epochs_gtca: int,
    gcn_layers: int,
    log_dir: str = None,
) -> dict:
    """Run one (train_size, seed, bert_depth) GTCA experiment."""

    data = load_dataset_splits(
        dataset=dataset,
        data_dir=data_dir,
        train_size=train_size,
        val_size=100,
        test_size=10000,
        seed=seed,
        target=target,
        featurize_ecfp=False,
        featurize_3d=False,
    )

    tr, va, te = data['train'], data['val'], data['test']
    train_pyg, val_pyg, test_pyg = tr['X_graph'], va['X_graph'], te['X_graph']
    train_smi, val_smi, test_smi = tr['ids'],     va['ids'],     te['ids']
    train_y,   val_y,   test_y   = tr['y'],        va['y'],       te['y']

    node_feat_dim = train_pyg[0].x.shape[1] if train_pyg else 30
    n_test        = len(test_pyg)

    log_path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"gtca_{bert_depth}_{seed}_{target}_traininglog.csv")

    res = train_gtca(
        train_pyg, val_pyg, test_pyg,
        train_smi, val_smi, test_smi,
        target_name=target,
        node_feat_dim=node_feat_dim,
        gcn_layers=gcn_layers,
        bert_depth=bert_depth,
        epochs=epochs_gtca,
        device=device,
        log_path=log_path,
        seed=seed,
    )

    save_run_csv(raw_dir, 'gtca', bert_depth, seed, target, train_size,
                 res['metrics'], n_test)
    return res


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GTCA depth ablation study")
    p.add_argument('--dataset', nargs='+', default=['qm9'],
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument('--target', nargs='+', default=None,
                   help="Targets to run. Default: all targets for each dataset.")
    p.add_argument('--depths', nargs='+', type=int, default=GTCA_DEPTHS,
                   help=f"bert_depth values to test (default: {GTCA_DEPTHS})")
    def _default_device():
        if torch.cuda.is_available():    return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'
    p.add_argument('--device', default=_default_device())
    p.add_argument('--results_root', default='./results')
    p.add_argument('--data_dir',     default='./data')

    p.add_argument('--epochs_gtca', type=int, default=300)
    p.add_argument('--gcn_layers',  type=int, default=3)

    p.add_argument('--resume',    action='store_true', help="Skip already-done runs")
    p.add_argument('--save_logs', action='store_true', help="Save training logs")
    p.add_argument('--train_sizes', nargs='+', type=int, default=None,
                   help="Override train sizes")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    train_sizes = args.train_sizes if args.train_sizes else get_train_sizes()
    depths      = sorted(set(args.depths))

    print(f"Device:      {args.device}")
    print(f"Datasets:    {args.dataset}")
    print(f"BERT depths: {depths}")
    print(f"Train sizes ({len(train_sizes)}): {train_sizes[:5]}{'...' if len(train_sizes) > 5 else ''}")
    print()

    for dataset in args.dataset:
        cfg     = DATASET_CONFIGS[dataset]
        targets = args.target if args.target else cfg['target_tasks']

        dataset_dir = os.path.join(args.results_root, DATASET_DIRS.get(dataset, dataset))
        raw_dir     = os.path.join(dataset_dir, 'raw_data')
        summary_dir = os.path.join(dataset_dir, 'summary')
        plots_dir   = os.path.join(dataset_dir, 'plots')
        depth_dir   = os.path.join(dataset_dir, 'depth_study')
        log_dir     = os.path.join(dataset_dir, 'training_logs') if args.save_logs else None

        for d in [raw_dir, summary_dir, plots_dir, depth_dir]:
            os.makedirs(d, exist_ok=True)

        for target in targets:
            print(f"\n{'='*60}")
            print(f" Dataset={dataset} | Target={target} | Depths={depths}")
            print(f"{'='*60}")

            depth_rows = []  # collect all results for depth_comparison.csv

            for bert_depth in depths:
                print(f"\n  --- bert_depth={bert_depth} ---")

                for train_size in train_sizes:
                    seeds = get_seed_schedule(train_size)

                    for seed in seeds:
                        if args.resume and run_already_done(raw_dir, 'gtca', bert_depth, seed, target, train_size):
                            print(f"    [resume] depth={bert_depth} size={train_size} seed={seed} — skip")
                            continue

                        print(f"    size={train_size:4d} | seed={seed}")
                        try:
                            res = run_one_depth(
                                train_size=train_size,
                                target=target,
                                seed=seed,
                                bert_depth=bert_depth,
                                dataset=dataset,
                                raw_dir=raw_dir,
                                data_dir=args.data_dir,
                                device=args.device,
                                epochs_gtca=args.epochs_gtca,
                                gcn_layers=args.gcn_layers,
                                log_dir=log_dir,
                            )
                            m = res['metrics']
                            depth_rows.append({
                                'bert_depth': bert_depth,
                                'train_size': train_size,
                                'seed':       seed,
                                'RMSE':       m.get('RMSE'),
                                'MAE':        m.get('MAE'),
                                'Pearson_R':  m.get('Pearson_R'),
                                'R2':         m.get('R2'),
                            })
                        except Exception as e:
                            print(f"    [ERROR] depth={bert_depth} size={train_size} seed={seed}: {e}")
                            traceback.print_exc()

            # Save depth_comparison.csv for this target
            if depth_rows:
                depth_df = pd.DataFrame(depth_rows)
                depth_path = os.path.join(depth_dir, f"depth_comparison_{target}.csv")
                if os.path.exists(depth_path):
                    existing = pd.read_csv(depth_path)
                    depth_df = pd.concat([existing, depth_df], ignore_index=True)
                    depth_df = depth_df.drop_duplicates(
                        subset=['bert_depth', 'train_size', 'seed'], keep='last'
                    )
                depth_df.to_csv(depth_path, index=False)
                print(f"\n  [saved] {depth_path}")

            # Rebuild summary_gtca_depth.csv + plot
            try:
                summary_df = rebuild_summary_gtca_depth(
                    raw_dir, summary_dir, target, depths=depths
                )
                if not summary_df.empty:
                    plot_gtca_depth_lc(summary_df, plots_dir, target)
            except Exception as e:
                print(f"  [ERROR] Summary/plot: {e}")

    print(f"\nDone! Results in {args.results_root}/")


if __name__ == '__main__':
    main()
