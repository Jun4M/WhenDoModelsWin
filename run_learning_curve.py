"""
run_learning_curve.py
Multi-dataset, multi-model, multi-seed adaptive learning curve experiment.

Adaptive train sizes:
  50~500  : step 25
  600~1000: step 100
  1500~3000: step 500

Seed schedule:
  train_size <= 500 : 10 seeds (0-9)
  train_size >  500 : 3  seeds (0-2)

Models (baselines):
  gcn, transformer, rf, xgb, gpr, attentivefp, gps

Results structure:
  results/
  └── 01_QM9/
      ├── raw_data/   {model}_{depth}_{seed}_{target}.csv
      ├── summary/    summary_baselines.csv
      └── plots/      plot_baselines_lc_{target}.png

Usage:
  python run_learning_curve.py --dataset qm9 --target homo lumo gap --device mps
  python run_learning_curve.py --dataset esol --target all --device mps --resume
  python run_learning_curve.py --dataset qm9 --skip_gps \\
      --epochs_gcn 5 --epochs_transformer 5 --epochs_gtca 5  # smoke test
"""

import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset_splits, load_raw_data, DATASET_CONFIGS
from src.train import (
    train_gcn, train_transformer, train_attentivefp, train_gps, train_sklearn,
)
from src.analysis import save_failure_data_csv, group_analysis
from src.summary import (
    save_run_csv, run_already_done, rebuild_summary_baselines,
    BASELINE_MODELS,
)
from src.visualization import plot_baselines_lc, plot_group_analysis


# ---------------------------------------------------------------------------
# Adaptive train sizes
# ---------------------------------------------------------------------------

DATASET_MAX_TRAIN = {
    'qm9':  3000,
    'esol':  375,
    'lipo': 1000,
    'bace':  500,
}

def get_train_sizes(dataset: str = 'qm9') -> list:
    max_size = DATASET_MAX_TRAIN.get(dataset, 3000)
    sizes = list(range(50, min(501, max_size + 1), 25))
    if max_size >= 600:
        sizes += list(range(600, min(1001, max_size + 1), 100))
    if max_size >= 1500:
        sizes += list(range(1500, max_size + 1, 500))
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


# ---------------------------------------------------------------------------
# Single experiment: one (train_size, target, seed, dataset)
# ---------------------------------------------------------------------------

def run_one(
    train_size: int,
    target: str,
    seed: int,
    dataset: str,
    raw_dir: str,
    data_dir: str,
    device: str,
    epochs_transformer: int,
    epochs_gcn: int,
    epochs_gcn_layers: int,
    epochs_attfp: int,
    epochs_gps: int,
    skip_transformer: bool,
    skip_gcn: bool,
    skip_rf: bool,
    skip_xgb: bool,
    skip_gpr: bool,
    skip_svr: bool,
    skip_lgbm: bool,
    skip_attentivefp: bool,
    skip_gps: bool,
    log_dir: str = None,
    resume: bool = False,
    preloaded_raw=None,  # (smiles, y_col, task_pos) pre-loaded once per target
) -> dict:

    import gc

    def clear_memory():
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    need_ecfp = not (skip_rf and skip_xgb and skip_gpr and skip_svr and skip_lgbm)

    data = load_dataset_splits(
        dataset=dataset,
        data_dir=data_dir,
        train_size=train_size,
        val_size=100,
        test_size=10000,
        seed=seed,
        target=target,
        featurize_ecfp=need_ecfp,
        preloaded_raw=preloaded_raw,
    )

    tr, va, te = data['train'], data['val'], data['test']
    train_pyg, val_pyg, test_pyg = tr['X_graph'], va['X_graph'], te['X_graph']
    train_smi, val_smi, test_smi = tr['ids'], va['ids'], te['ids']
    train_y, val_y, test_y       = tr['y'], va['y'], te['y']

    node_feat_dim = train_pyg[0].x.shape[1] if train_pyg else 30
    edge_dim      = train_pyg[0].edge_attr.shape[1] if (train_pyg and train_pyg[0].edge_attr is not None) else 11
    n_test        = len(test_pyg)

    results = {}

    def log_path_for(model_name):
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            return os.path.join(log_dir, f"{model_name}_na_{seed}_{target}_traininglog.csv")
        return None

    # GCN
    if not skip_gcn and not (resume and run_already_done(raw_dir, 'gcn', None, seed, target, train_size)):
        try:
            res = train_gcn(
                train_pyg, val_pyg, test_pyg,
                target_name=target, node_feat_dim=node_feat_dim,
                num_layers=epochs_gcn_layers, epochs=epochs_gcn, device=device,
                log_path=log_path_for('gcn'), seed=seed,
            )
            save_run_csv(raw_dir, 'gcn', None, seed, target, train_size, res['metrics'], n_test)
            save_failure_data_csv(test_smi, res['test_true'], res['test_preds'],
                                  'gcn', target, raw_dir, model=None, device=device,
                                  pyg_data_list=test_pyg)
            results['gcn'] = res
        except Exception as e:
            print(f"  [ERROR] GCN: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # Transformer (CPU only — MPS causes segfault during ChemBERTa weight loading)
    if not skip_transformer and not (resume and run_already_done(raw_dir, 'transformer', None, seed, target, train_size)):
        try:
            res = train_transformer(
                train_smi, train_y, val_smi, val_y, test_smi, test_y,
                target_name=target, epochs=epochs_transformer, device='cpu',
                log_path=log_path_for('transformer'), seed=seed,
            )
            save_run_csv(raw_dir, 'transformer', None, seed, target, train_size, res['metrics'], n_test)
            results['transformer'] = res
        except Exception as e:
            print(f"  [ERROR] Transformer: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # AttentiveFP
    if not skip_attentivefp and not (resume and run_already_done(raw_dir, 'attentivefp', None, seed, target, train_size)):
        try:
            res = train_attentivefp(
                train_pyg, val_pyg, test_pyg,
                target_name=target, node_feat_dim=node_feat_dim, edge_dim=edge_dim,
                epochs=epochs_attfp, device=device, log_path=log_path_for('attentivefp'),
                seed=seed,
            )
            save_run_csv(raw_dir, 'attentivefp', None, seed, target, train_size, res['metrics'], n_test)
            results['attentivefp'] = res
        except Exception as e:
            print(f"  [ERROR] AttentiveFP: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # GPS
    if not skip_gps and not (resume and run_already_done(raw_dir, 'gps', None, seed, target, train_size)):
        try:
            res = train_gps(
                train_pyg, val_pyg, test_pyg,
                target_name=target, node_feat_dim=node_feat_dim,
                epochs=epochs_gps, device=device, log_path=log_path_for('gps'),
                seed=seed,
            )
            save_run_csv(raw_dir, 'gps', None, seed, target, train_size, res['metrics'], n_test)
            results['gps'] = res
        except Exception as e:
            print(f"  [ERROR] GPS: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # RF
    if not skip_rf and not (resume and run_already_done(raw_dir, 'rf', None, seed, target, train_size)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='rf', seed=seed)
            if res:
                save_run_csv(raw_dir, 'rf', None, seed, target, train_size, res['metrics'], n_test)
                results['rf'] = res
        except Exception as e:
            print(f"  [ERROR] RF: {e}"); traceback.print_exc()

    # XGBoost
    if not skip_xgb and not (resume and run_already_done(raw_dir, 'xgb', None, seed, target, train_size)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='xgb', seed=seed)
            if res:
                save_run_csv(raw_dir, 'xgb', None, seed, target, train_size, res['metrics'], n_test)
                results['xgb'] = res
        except Exception as e:
            print(f"  [ERROR] XGBoost: {e}"); traceback.print_exc()

    # GPR
    if not skip_gpr and not (resume and run_already_done(raw_dir, 'gpr', None, seed, target, train_size)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='gpr', seed=seed)
            if res:
                save_run_csv(raw_dir, 'gpr', None, seed, target, train_size, res['metrics'], n_test)
                results['gpr'] = res
        except Exception as e:
            print(f"  [ERROR] GPR: {e}"); traceback.print_exc()

    # SVR
    if not skip_svr and not (resume and run_already_done(raw_dir, 'svr', None, seed, target, train_size)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='svr', seed=seed)
            if res:
                save_run_csv(raw_dir, 'svr', None, seed, target, train_size, res['metrics'], n_test)
                results['svr'] = res
        except Exception as e:
            print(f"  [ERROR] SVR: {e}"); traceback.print_exc()

    # LightGBM
    if not skip_lgbm and not (resume and run_already_done(raw_dir, 'lgbm', None, seed, target, train_size)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='lgbm', seed=seed)
            if res:
                save_run_csv(raw_dir, 'lgbm', None, seed, target, train_size, res['metrics'], n_test)
                results['lgbm'] = res
        except Exception as e:
            print(f"  [ERROR] LightGBM: {e}"); traceback.print_exc()

    # Group analysis for completed models
    group_rows = []
    for model_name, res in results.items():
        try:
            rows = group_analysis(
                test_smi, res['test_true'], res['test_preds'],
                model_name=model_name, target_name=target,
                train_size=train_size, seed=seed,
            )
            group_rows.extend(rows)
        except Exception:
            pass

    return results, group_rows


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QM9 Adaptive Learning Curve (multi-model)")
    p.add_argument('--dataset', nargs='+', default=['qm9'],
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Datasets to run (default: qm9)")
    p.add_argument('--target', nargs='+', default=None,
                   help="Targets to run. Default: all targets for each dataset.")
    def _default_device():
        if torch.cuda.is_available():    return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'
    p.add_argument('--device', default=_default_device())
    p.add_argument('--results_root', default='./results')
    p.add_argument('--data_dir',     default='./data')

    # Epochs
    p.add_argument('--epochs_transformer', type=int, default=200)
    p.add_argument('--epochs_gcn',         type=int, default=300)
    p.add_argument('--gcn_layers',         type=int, default=3)
    p.add_argument('--epochs_attfp',       type=int, default=300)
    p.add_argument('--epochs_gps',         type=int, default=300)

    # Skip flags
    p.add_argument('--skip_transformer', action='store_true')
    p.add_argument('--skip_gcn',         action='store_true')
    p.add_argument('--skip_rf',          action='store_true')
    p.add_argument('--skip_xgb',         action='store_true')
    p.add_argument('--skip_gpr',         action='store_true')
    p.add_argument('--skip_svr',         action='store_true')
    p.add_argument('--skip_lgbm',        action='store_true')
    p.add_argument('--skip_attentivefp', action='store_true')
    p.add_argument('--skip_gps',         action='store_true')

    # Control
    p.add_argument('--resume',    action='store_true', help="Skip already-done runs")
    p.add_argument('--save_logs', action='store_true', help="Save training logs")
    p.add_argument('--train_sizes', nargs='+', type=int, default=None,
                   help="Override train sizes (e.g. --train_sizes 50 100 500)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    train_sizes = args.train_sizes if args.train_sizes else None  # per-dataset below

    # Active models list for display
    active_models = []
    for m in BASELINE_MODELS:
        flag = getattr(args, f'skip_{m}', False)
        if not flag:
            active_models.append(m)

    print(f"Device: {args.device}")
    print(f"Datasets: {args.dataset}")
    print(f"Train sizes: {'per-dataset' if train_sizes is None else f'({len(train_sizes)}) {train_sizes}'}")
    print(f"Active models: {active_models}")
    print()

    for dataset in args.dataset:
        cfg = DATASET_CONFIGS[dataset]
        targets = args.target if args.target else cfg['target_tasks']
        ds_sizes = train_sizes if train_sizes is not None else get_train_sizes(dataset)

        dataset_dir  = os.path.join(args.results_root, DATASET_DIRS.get(dataset, dataset))
        raw_dir      = os.path.join(dataset_dir, 'raw_data')
        summary_dir  = os.path.join(dataset_dir, 'summary')
        plots_dir    = os.path.join(dataset_dir, 'plots')
        log_dir      = os.path.join(dataset_dir, 'training_logs') if args.save_logs else None

        os.makedirs(raw_dir,     exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(plots_dir,   exist_ok=True)

        group_master_path = os.path.join(dataset_dir, 'master_group_summary.csv')
        group_all_rows = []

        for target in targets:
            print(f"\n{'='*60}")
            print(f" Dataset={dataset} | Target={target}")
            print(f"{'='*60}")

            # Pre-load raw data once per (dataset, target) to avoid re-reading
            # 130k QM9 molecules on every run_one() call
            print(f"  [data_loader] Pre-loading raw data for {dataset}/{target} ...")
            preloaded_raw = load_raw_data(dataset, args.data_dir, target)
            print(f"  [data_loader] Raw data ready ({len(preloaded_raw[0])} molecules)")

            for train_size in ds_sizes:
                seeds = get_seed_schedule(train_size)

                for seed in seeds:
                    # Check resume for all active models
                    if args.resume:
                        all_done = all(
                            run_already_done(raw_dir, m, None, seed, target, train_size)
                            for m in active_models
                        )
                        if all_done:
                            print(f"  [resume] size={train_size} seed={seed} target={target} — skip")
                            continue

                    print(f"\n  --- size={train_size:4d} | seed={seed} | target={target} ---")

                    try:
                        results, group_rows = run_one(
                            train_size=train_size,
                            target=target,
                            seed=seed,
                            dataset=dataset,
                            raw_dir=raw_dir,
                            data_dir=args.data_dir,
                            device=args.device,
                            epochs_transformer=args.epochs_transformer,
                            epochs_gcn=args.epochs_gcn,
                            epochs_gcn_layers=args.gcn_layers,
                            epochs_attfp=args.epochs_attfp,
                            epochs_gps=args.epochs_gps,
                            skip_transformer=args.skip_transformer,
                            skip_gcn=args.skip_gcn,
                            skip_rf=args.skip_rf,
                            skip_xgb=args.skip_xgb,
                            skip_gpr=args.skip_gpr,
                            skip_svr=args.skip_svr,
                            skip_lgbm=args.skip_lgbm,
                            skip_attentivefp=args.skip_attentivefp,
                            skip_gps=args.skip_gps,
                            log_dir=log_dir,
                            resume=args.resume,
                            preloaded_raw=preloaded_raw,
                        )
                        group_all_rows.extend(group_rows)

                    except Exception as e:
                        print(f"  [ERROR] size={train_size} seed={seed}: {e}")
                        traceback.print_exc()

            # After all sizes/seeds for this target: rebuild summary + plots
            try:
                baseline_models = [m for m in BASELINE_MODELS
                                   if not getattr(args, f'skip_{m}', False)]
                summary_df = rebuild_summary_baselines(
                    raw_dir, summary_dir, target,
                    baseline_models=baseline_models,
                )
                if not summary_df.empty:
                    plot_baselines_lc(summary_df, plots_dir, target)
            except Exception as e:
                print(f"  [ERROR] Summary/plot: {e}")

        # Save group summary
        if group_all_rows:
            new_df = pd.DataFrame(group_all_rows)
            if os.path.exists(group_master_path):
                existing = pd.read_csv(group_master_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=['train_size','model','target','seed','group_type','category'],
                    keep='last'
                )
            else:
                combined = new_df
            combined.to_csv(group_master_path, index=False)
            print(f"\n[saved] {group_master_path}")

            try:
                group_df = pd.read_csv(group_master_path)
                for target in targets:
                    plot_group_analysis(group_df, plots_dir, target)
            except Exception as e:
                print(f"  [ERROR] Group plot: {e}")

    print(f"\nDone! Results in {args.results_root}/")


if __name__ == '__main__':
    main()
