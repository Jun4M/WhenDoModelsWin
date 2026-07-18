"""Ensemble analysis from saved test predictions (R1.5 reviewer response).

Loads npz files from results/{dataset}/predictions/ and computes:
  1. Ensemble RMSE (mean of model predictions) vs best single-model RMSE
  2. Family-based ensembles (GNN, Transformer, 3D, Tree)
  3. Inter-model prediction correlation (diversity)

Usage:
    python scripts/ensemble_analysis.py --dataset lipo
    python scripts/ensemble_analysis.py --dataset all   # 4 datasets

Output:
    results/paper_csv/ensemble_{dataset}.csv
    results/paper_csv/ensemble_diversity_{dataset}.csv

Fixes (2026-06-29):
  B1 - cell key now includes target → QM9 homo/lumo/gap kept separate
  B2 - QM9 RMSE converted Hartree→eV (×27.2114), matching rebuild_paper_csv.py
"""
import argparse
import os
import glob
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

DATASET_DIRS = {
    'lipo': '03_Lipo',
    'esol': '02_ESOL',
    'bace': '04_BACE',
    'qm9':  '01_QM9',
}

HARTREE_TO_EV = 27.2114   # same constant as rebuild_paper_csv.py
QM9_TARGETS   = {'homo', 'lumo', 'gap'}

# Model families for grouped ensembles
FAMILIES = {
    'GNN':         ['gcn', 'attentivefp', 'gps', 'krovex'],
    'Transformer': ['transformer', 'chemberta2', 'molformer', 'selformer'],
    '3D':          ['painn', 'unimol_scratch', 'unimol_pt'],
    'Tree':        ['rf', 'xgb', 'lgbm'],
    'Other':       ['gpr', 'svr', 'chemprop'],
}
ALL_MODELS_FLAT = sum(FAMILIES.values(), [])


def parse_npz_filename(fname):
    """Parse {model}_na_{seed}_{target}_n{train_size}.npz."""
    m = re.match(r'(.+)_na_(\d+)_(.+)_n(\d+)\.npz$', fname)
    if not m:
        return None
    return {
        'model':      m.group(1),
        'seed':       int(m.group(2)),
        'target':     m.group(3),
        'train_size': int(m.group(4)),
    }


def load_predictions(pred_dir):
    """Load all npz → dict[(model, seed, size, target)] = {'preds': arr, 'true': arr, ...}."""
    data = {}
    for f in os.listdir(pred_dir):
        if not f.endswith('.npz'):
            continue
        meta = parse_npz_filename(f)
        if meta is None:
            continue
        try:
            npz = np.load(os.path.join(pred_dir, f))
            # B1 fix: key includes target so QM9 homo/lumo/gap are kept separate
            key = (meta['model'], meta['seed'], meta['train_size'], meta['target'])
            data[key] = {
                'preds':  npz['test_preds'],
                'true':   npz['test_true'],
                'y_mean': float(npz['y_mean']) if npz['y_mean'].ndim == 0 else None,
                'y_std':  float(npz['y_std'])  if npz['y_std'].ndim  == 0 else None,
                'target': meta['target'],
            }
        except Exception as e:
            print(f"  [warn] failed to load {f}: {e}")
    return data


def rmse(preds, true, y_std=None):
    """RMSE; if y_std given, denormalize from z-score space."""
    err = np.sqrt(np.mean((preds - true) ** 2))
    return err * y_std if y_std is not None else err


def ensemble_rmse(model_preds_list, true, y_std):
    """Mean ensemble of multiple model predictions."""
    stacked = np.stack(model_preds_list, axis=0)  # (n_models, n_test)
    ens = stacked.mean(axis=0)
    return rmse(ens, true, y_std)


def analyze_dataset(dataset_name):
    ds_folder = DATASET_DIRS[dataset_name]
    pred_dir = f'results/{ds_folder}/predictions'
    if not os.path.exists(pred_dir):
        print(f"  [skip] {dataset_name}: no predictions dir")
        return None

    print(f"\n=== {dataset_name.upper()} ensemble analysis ===")
    data = load_predictions(pred_dir)
    print(f"Loaded {len(data)} prediction files")

    if not data:
        return None

    # B1 fix: group by (target, size, seed) → keep QM9 homo/lumo/gap separate
    cells = defaultdict(dict)  # (target, size, seed) → {model: pred_dict}
    for (model, seed, size, target), pred in data.items():
        cells[(target, size, seed)][model] = pred

    # Cells with at least 2 models (otherwise no ensemble possible)
    cells = {k: v for k, v in cells.items() if len(v) >= 2}
    print(f"Cells with ≥2 models: {len(cells)}")

    # Per-cell metrics
    rows = []
    n_excluded = defaultdict(int)
    for (target, size, seed), models_preds in cells.items():
        # 3D models (PaiNN/UniMol) drop ETKDG-failed molecules → different test_true.
        # Group models by test_true signature, ensemble only within the majority group.
        sig_to_models = defaultdict(list)
        for m, p in models_preds.items():
            sig = (p['true'].shape[0], float(p['true'][:5].sum()), float(p['true'][-5:].sum()))
            sig_to_models[sig].append(m)

        # Pick the largest group (most models with matching test_true)
        majority_sig = max(sig_to_models, key=lambda s: len(sig_to_models[s]))
        majority_models = sig_to_models[majority_sig]
        excluded_models = [m for s, ms in sig_to_models.items() if s != majority_sig for m in ms]
        for em in excluded_models:
            n_excluded[em] += 1

        if len(majority_models) < 2:
            continue

        models_preds = {m: models_preds[m] for m in majority_models}
        true = models_preds[majority_models[0]]['true']
        y_stds = [v['y_std'] for v in models_preds.values() if v['y_std'] is not None]
        y_std = y_stds[0] if y_stds else 1.0

        # Single-model RMSEs
        single_rmse = {m: rmse(p['preds'], true, y_std)
                       for m, p in models_preds.items()}
        best_model = min(single_rmse, key=single_rmse.get)
        best_rmse = single_rmse[best_model]

        # All-model ensemble
        all_preds_list = [models_preds[m]['preds'] for m in models_preds]
        all_ens_rmse = ensemble_rmse(all_preds_list, true, y_std)

        # Top-K ensemble (K=3, 5)
        top_k_rmse = {}
        for K in [3, 5]:
            if len(single_rmse) >= K:
                top_k_models = sorted(single_rmse, key=single_rmse.get)[:K]
                top_k_preds = [models_preds[m]['preds'] for m in top_k_models]
                top_k_rmse[f'top{K}'] = ensemble_rmse(top_k_preds, true, y_std)
            else:
                top_k_rmse[f'top{K}'] = np.nan

        # Family-based ensembles (one per family if 2+ models present)
        family_rmse = {}
        for fam_name, fam_models in FAMILIES.items():
            available = [m for m in fam_models if m in models_preds]
            if len(available) >= 2:
                fam_preds = [models_preds[m]['preds'] for m in available]
                family_rmse[f'fam_{fam_name}'] = ensemble_rmse(fam_preds, true, y_std)
            else:
                family_rmse[f'fam_{fam_name}'] = np.nan

        # Cross-family (one model per family, the best single from each)
        cross_fam_preds = []
        for fam_name, fam_models in FAMILIES.items():
            available = [m for m in fam_models if m in models_preds]
            if available:
                best_in_fam = min(available, key=lambda m: single_rmse[m])
                cross_fam_preds.append(models_preds[best_in_fam]['preds'])
        cross_fam_rmse = ensemble_rmse(cross_fam_preds, true, y_std) if len(cross_fam_preds) >= 2 else np.nan

        # improvement_pct computed before eV conversion (ratio is scale-invariant)
        improvement_pct = 100 * (best_rmse - all_ens_rmse) / best_rmse

        # B2 fix: convert QM9 RMSE from Hartree → eV (matching rebuild_paper_csv.py)
        if dataset_name == 'qm9' and target in QM9_TARGETS:
            def _to_ev(v):
                return v * HARTREE_TO_EV if (v is not None and not np.isnan(v)) else v
            best_rmse     = _to_ev(best_rmse)
            all_ens_rmse  = _to_ev(all_ens_rmse)
            cross_fam_rmse = _to_ev(cross_fam_rmse)
            top_k_rmse    = {k: _to_ev(v) for k, v in top_k_rmse.items()}
            family_rmse   = {k: _to_ev(v) for k, v in family_rmse.items()}

        row = {
            'dataset': dataset_name,
            'target': target,          # B1 fix: include target column
            'train_size': size,
            'seed': seed,
            'n_models': len(models_preds),
            'best_single_rmse': best_rmse,
            'best_single_model': best_model,
            'all_ensemble_rmse': all_ens_rmse,
            'all_ens_improvement_pct': improvement_pct,
            'cross_fam_rmse': cross_fam_rmse,
            **top_k_rmse,
            **family_rmse,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = f'results/paper_csv/ensemble_{dataset_name}.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} ensemble cells)")
    if n_excluded:
        print(f"\nModels excluded from ensemble (test_true mismatch — typically 3D models with ETKDG drops):")
        for m, n in sorted(n_excluded.items(), key=lambda x: -x[1]):
            print(f"  {m}: excluded from {n} cells")

    # Per-size summary (group by target + train_size for multi-target datasets)
    print(f"\n--- Per-size aggregate (mean across seeds) ---")
    group_cols = ['target', 'train_size'] if 'target' in df.columns else ['train_size']
    agg = df.groupby(group_cols).agg(
        n_seeds=('seed', 'count'),
        best_single=('best_single_rmse', 'mean'),
        all_ens=('all_ensemble_rmse', 'mean'),
        improvement_pct=('all_ens_improvement_pct', 'mean'),
    ).round(4)
    print(agg.to_string())

    # Diversity: average pairwise correlation at largest size, first (lexically) target
    print(f"\n--- Inter-model prediction correlation (at n=largest, seed=0) ---")
    all_sizes   = [s for (_, s, _) in cells]
    all_targets = sorted(set(t for (t, _, _) in cells))
    max_size    = max(all_sizes) if all_sizes else 0
    first_target = all_targets[0] if all_targets else ''
    if (first_target, max_size, 0) in cells:
        models_at = cells[(first_target, max_size, 0)]
        # Filter to models with matching test_true (majority shape)
        from collections import Counter
        shapes = Counter(v['true'].shape for v in models_at.values())
        majority_shape = shapes.most_common(1)[0][0]
        models_at = {m: v for m, v in models_at.items()
                     if v['true'].shape == majority_shape}
        sorted_models = sorted(models_at)
        n = len(sorted_models)
        if n < 2:
            print(f"  (skipped — only {n} models share the majority test shape)")
        else:
            corr_matrix = np.zeros((n, n))
            for i, m1 in enumerate(sorted_models):
                for j, m2 in enumerate(sorted_models):
                    p1 = models_at[m1]['preds']
                    p2 = models_at[m2]['preds']
                    if p1.shape != p2.shape:
                        corr_matrix[i, j] = np.nan
                        continue
                    corr_matrix[i, j] = np.corrcoef(p1, p2)[0, 1]
            corr_df = pd.DataFrame(corr_matrix, index=sorted_models, columns=sorted_models)
            out_corr_path = f'results/paper_csv/ensemble_diversity_{dataset_name}.csv'
            corr_df.to_csv(out_corr_path)
            print(f"Saved diversity matrix: {out_corr_path} ({n} models)")
            # Off-diagonal mean (ignore NaN)
            off_diag = corr_matrix[~np.eye(n, dtype=bool)]
            off_diag = off_diag[~np.isnan(off_diag)]
            print(f"Mean pairwise correlation: {off_diag.mean():.3f} (std {off_diag.std():.3f})")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='lipo',
                        choices=['lipo', 'esol', 'bace', 'qm9', 'all'])
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in ['lipo', 'esol', 'bace', 'qm9']:
            analyze_dataset(ds)
    else:
        analyze_dataset(args.dataset)


if __name__ == '__main__':
    main()
