"""
summary.py
CSV aggregation utilities.
  - save_run_csv: append one run result to raw_data/{model}_{depth}_{seed}_{target}.csv
  - rebuild_summary_baselines: aggregate mean±std → summary_baselines.csv
  - rebuild_summary_gtca_depth: aggregate mean±std → summary_gtca_depth.csv
"""

import os
import glob
import numpy as np
import pandas as pd


METRICS = ['RMSE', 'MAE', 'Pearson_R', 'R2']

BASELINE_MODELS = ['gcn', 'transformer', 'chemberta2', 'molformer', 'selformer', 'chemprop', 'krovex', 'rf', 'xgb', 'gpr', 'lgbm', 'svr', 'attentivefp', 'attentivefp_mtl', 'gcn_mtl', 'unimol_scratch', 'unimol_pt', 'gps', 'painn']
GTCA_DEPTHS     = [2, 4, 6]


# ---------------------------------------------------------------------------
# File naming convention
# ---------------------------------------------------------------------------

def raw_csv_filename(model: str, depth, seed: int, target: str) -> str:
    """
    Returns filename (not full path): {model}_{depth}_{seed}_{target}.csv
    depth is 'na' for non-GTCA models.
    """
    d = 'na' if depth is None else str(depth)
    return f"{model}_{d}_{seed}_{target}.csv"


def raw_csv_path(raw_dir: str, model: str, depth, seed: int, target: str) -> str:
    return os.path.join(raw_dir, raw_csv_filename(model, depth, seed, target))


# ---------------------------------------------------------------------------
# Save one run result
# ---------------------------------------------------------------------------

def save_run_csv(
    raw_dir: str,
    model: str,
    depth,       # int or None
    seed: int,
    target: str,
    train_size: int,
    metrics: dict,
    n_test: int = None,
):
    """
    Appends one row to raw_data/{model}_{depth}_{seed}_{target}.csv.
    Deduplicates by train_size (keeps last).
    """
    os.makedirs(raw_dir, exist_ok=True)
    path = raw_csv_path(raw_dir, model, depth, seed, target)

    row = {
        'train_size': train_size,
        'RMSE':       float(metrics.get('RMSE', float('nan'))),
        'MAE':        float(metrics.get('MAE',  float('nan'))),
        'Pearson_R':  float(metrics.get('Pearson_R', float('nan'))),
        'R2':         float(metrics.get('R2',   float('nan'))),
    }
    if n_test is not None:
        row['n_test'] = n_test

    new_df = pd.DataFrame([row])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['train_size'], keep='last')
    else:
        combined = new_df

    combined.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Check if a run is already done
# ---------------------------------------------------------------------------

def run_already_done(raw_dir: str, model: str, depth, seed: int,
                      target: str, train_size: int,
                      *, check_predictions: bool = False,
                      pred_dir: str = None,
                      target_safe: str = None) -> bool:
    """Returns True if this (model, seed, target, train_size) cell can be skipped.

    When check_predictions=True (i.e., --resume --save_predictions),
    requires both the CSV row AND prediction npz to exist.
    CSV-only cells are re-trained so the npz gets created.
    """
    path = raw_csv_path(raw_dir, model, depth, seed, target)
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        csv_done = int(train_size) in df['train_size'].values
    except Exception:
        return False

    if not csv_done:
        return False
    if not check_predictions:
        return True  # original behaviour

    # Also require prediction npz
    if target_safe is None:
        target_safe = target.replace(' ', '_').replace('/', '_')
    kind = str(depth) if depth is not None else 'na'
    fname = f'{model}_{kind}_{seed}_{target_safe}_n{train_size}.npz'
    return os.path.exists(os.path.join(pred_dir, fname))


# ---------------------------------------------------------------------------
# Save prediction arrays (--save_predictions flag)
# ---------------------------------------------------------------------------

def save_predictions_npz(res, *, model, pred_dir, model_kind,
                          seed, target, train_size, y_mean=None, y_std=None):
    """Save normalized test_preds + test_true (+ denorm stats) to .npz for ensemble.

    Output: {pred_dir}/{model}_{model_kind}_{seed}_{target_safe}_n{train_size}.npz
    NPZ keys: test_preds, test_true (float32, normalized), y_mean, y_std (scalar or array).
    Silently skips if res is None or keys are missing (e.g. GPR > 500).
    """
    if res is None:
        return
    if 'test_preds' not in res or 'test_true' not in res:
        return

    preds = np.asarray(res['test_preds'])
    truth = np.asarray(res['test_true'])
    if preds.shape != truth.shape:
        print(f'  [warn] save_predictions shape mismatch {model}: '
              f'preds={preds.shape}, true={truth.shape}')
        return

    os.makedirs(pred_dir, exist_ok=True)
    target_safe = target.replace(' ', '_').replace('/', '_')
    fname = f'{model}_{model_kind}_{seed}_{target_safe}_n{train_size}.npz'
    out_path = os.path.join(pred_dir, fname)

    np.savez_compressed(
        out_path,
        test_preds=preds.astype(np.float32),
        test_true=truth.astype(np.float32),
        y_mean=np.asarray(y_mean) if y_mean is not None else np.array(np.nan),
        y_std =np.asarray(y_std)  if y_std  is not None else np.array(np.nan),
    )


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------

def _aggregate_seeds(raw_dir: str, model: str, depth, target: str,
                      seeds: list) -> pd.DataFrame:
    """Read all seed CSVs for one (model, depth, target), return aggregated df."""
    frames = []
    for seed in seeds:
        path = raw_csv_path(raw_dir, model, depth, seed, target)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['seed'] = seed
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    agg = combined.groupby('train_size').agg(
        RMSE_mean=('RMSE', 'mean'),
        RMSE_std= ('RMSE', 'std'),
        MAE_mean= ('MAE',  'mean'),
        MAE_std=  ('MAE',  'std'),
        Pearson_R_mean=('Pearson_R', 'mean'),
        Pearson_R_std= ('Pearson_R', 'std'),
        R2_mean= ('R2',   'mean'),
        R2_std=  ('R2',   'std'),
        n_seeds= ('seed', 'count'),
    ).reset_index()

    if 'n_test' in combined.columns:
        agg['n_test_mean'] = combined.groupby('train_size')['n_test'].mean().values

    return agg


# ---------------------------------------------------------------------------
# Rebuild summary_baselines.csv
# ---------------------------------------------------------------------------

def rebuild_summary_baselines(
    raw_dir: str,
    summary_dir: str,
    target: str,
    baseline_models: list = None,
    max_seeds_small: int = 10,
    max_seeds_large: int = 3,
    size_boundary: int = 500,
) -> pd.DataFrame:
    """
    Reads raw CSVs for all baseline models, aggregates mean±std.
    Writes summary_baselines.csv.
    """
    if baseline_models is None:
        baseline_models = BASELINE_MODELS

    os.makedirs(summary_dir, exist_ok=True)
    rows = []

    for model in baseline_models:
        # Determine seeds from available files
        pattern = os.path.join(raw_dir, f"{model}_na_*_{target}.csv")
        files = glob.glob(pattern)
        if not files:
            continue

        # Collect all train_sizes across all seeds
        all_sizes = set()
        all_seeds = set()
        for f in files:
            try:
                parts = os.path.basename(f).split('_')
                seed = int(parts[2])
                all_seeds.add(seed)
                df = pd.read_csv(f)
                all_sizes.update(df['train_size'].tolist())
            except Exception:
                continue

        seeds = sorted(all_seeds)
        agg = _aggregate_seeds(raw_dir, model, None, target, seeds)
        if agg.empty:
            continue

        agg['model'] = model
        agg['target'] = target
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(summary_dir, 'summary_baselines.csv')
    result.to_csv(out_path, index=False)
    print(f"  [summary] → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Rebuild summary_gtca_depth.csv
# ---------------------------------------------------------------------------

def rebuild_summary_gtca_depth(
    raw_dir: str,
    summary_dir: str,
    target: str,
    depths: list = None,
    max_seeds_small: int = 10,
    max_seeds_large: int = 3,
) -> pd.DataFrame:
    """
    Reads raw CSVs for GTCA depth variants, aggregates mean±std.
    Writes summary_gtca_depth.csv.
    """
    if depths is None:
        depths = GTCA_DEPTHS

    os.makedirs(summary_dir, exist_ok=True)
    rows = []

    for depth in depths:
        pattern = os.path.join(raw_dir, f"gtca_{depth}_*_{target}.csv")
        files = glob.glob(pattern)
        if not files:
            continue

        all_seeds = set()
        for f in files:
            try:
                parts = os.path.basename(f).split('_')
                seed = int(parts[2])
                all_seeds.add(seed)
            except Exception:
                continue

        seeds = sorted(all_seeds)
        agg = _aggregate_seeds(raw_dir, 'gtca', depth, target, seeds)
        if agg.empty:
            continue

        agg['model'] = f'gtca_depth_{depth}'
        agg['target'] = target
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(summary_dir, 'summary_gtca_depth.csv')
    result.to_csv(out_path, index=False)
    print(f"  [summary] → {out_path}")
    return result
