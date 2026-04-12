"""
rebuild_paper_csv.py
De-normalized raw_data로부터 paper_csv 파일들을 전부 재생성.

생성 파일:
  paper_csv/lc_{dataset}_all_models.csv       — baselines + gtca_cat
  paper_csv/ablation_gtca_fusion_qm9.csv      — gtca_cat vs gtca_ca (QM9)
  paper_csv/ablation_gtca_depth_qm9.csv       — depth ablation (QM9)
  paper_csv/ablation_gtca_depth_ci_qm9.csv    — depth CI version
  paper_csv/stats_fusion_welch_qm9.csv        — Welch t-test fusion
  paper_csv/stats_depth_welch_qm9.csv         — Welch t-test depth

Pearson_R, R2는 scale-invariant → 재계산 불필요.
RMSE, MAE는 이미 raw_data에서 de-normalize 완료.
QM9(homo/lumo/gap)은 DeepChem이 Hartree 반환 → paper_csv 생성 시 ×27.2114로 eV 변환.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.summary import rebuild_summary_baselines, rebuild_summary_gtca_depth, BASELINE_MODELS

RESULTS_DIR = './results'
PAPER_DIR   = './results/paper_csv'

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

QM9_DIR      = os.path.join(RESULTS_DIR, '01_QM9')
HARTREE_TO_EV = 27.2114
QM9_TARGETS   = {'homo', 'lumo', 'gap'}


# ---------------------------------------------------------------------------
# Helper: Hartree → eV conversion for QM9 RMSE/MAE columns
# ---------------------------------------------------------------------------

def convert_to_ev(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Multiply RMSE/MAE columns by 27.2114 for QM9 targets (in-place safe)."""
    if target not in QM9_TARGETS:
        return df
    df = df.copy()
    for col in df.columns:
        if col.startswith('RMSE') or col.startswith('MAE'):
            df[col] = df[col] * HARTREE_TO_EV
    return df


# ---------------------------------------------------------------------------
# Helper: aggregate raw_data CSVs with CI95
# ---------------------------------------------------------------------------

def aggregate_with_ci(frames: list, model_col: str = 'model') -> pd.DataFrame:
    """Aggregate list of per-seed DataFrames → mean ± std ± CI95 per train_size."""
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for (model, ts), grp in combined.groupby([model_col, 'train_size']):
        n = len(grp)
        ci_factor = 1.96 / np.sqrt(n) if n > 1 else 0.0
        row = {
            model_col:   model,
            'train_size': ts,
            'n_seeds':    n,
        }
        for col in ['RMSE', 'MAE', 'Pearson_R', 'R2']:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f'{col}_mean'] = float(vals.mean())
            row[f'{col}_std']  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f'{col}_CI95'] = float(vals.std(ddof=1) * ci_factor) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def read_raw_dir(raw_dir: str, pattern: str, model_label_fn=None) -> list:
    """
    Read all matching CSVs from raw_dir, return list of DataFrames
    with 'model' and 'seed' columns added.
    pattern: glob pattern for filenames.
    model_label_fn(fname) → model label string.
    """
    frames = []
    for fpath in sorted(glob.glob(os.path.join(raw_dir, pattern))):
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath)
            # extract seed from filename (last numeric token before .csv/{target}.csv)
            stem = fname.replace('.csv', '')
            seed = int(stem.split('_')[-1])
            df['seed'] = seed
            if model_label_fn:
                df['model'] = model_label_fn(fname)
            frames.append(df)
        except Exception as e:
            print(f'  [WARN] skip {fname}: {e}')
    return frames


# ---------------------------------------------------------------------------
# 1. lc_{dataset}_all_models.csv
# ---------------------------------------------------------------------------

def build_lc_all_models(folder, dataset, targets):
    """raw_data baselines + gtca_cat from fusion_study/raw_data."""
    base_dir   = os.path.join(RESULTS_DIR, folder)
    raw_dir    = os.path.join(base_dir, 'raw_data')
    fusion_raw = os.path.join(base_dir, 'fusion_study', 'raw_data')

    for target in targets:
        # --- baselines: aggregate directly from raw_data ---
        baseline_frames = []
        for model in BASELINE_MODELS:
            for fpath in sorted(glob.glob(os.path.join(raw_dir, f'{model}_na_*_{target}.csv'))):
                fname = os.path.basename(fpath)
                stem  = fname[:-len(f'_{target}.csv')]
                try:
                    seed = int(stem.split('_')[-1])
                except ValueError:
                    continue
                df = pd.read_csv(fpath)
                df['seed']   = seed
                df['model']  = model
                df['target'] = target
                baseline_frames.append(df)

        if not baseline_frames:
            print(f'  [WARN] No baseline raw_data for {dataset}/{target}')
            continue

        combined_b = pd.concat(baseline_frames, ignore_index=True)
        rows_b = []
        for (model, ts), grp in combined_b.groupby(['model', 'train_size']):
            n = len(grp)
            ci_factor = 1.96 / np.sqrt(n) if n > 1 else 0.0
            row = {'model': model, 'target': target, 'train_size': ts,
                   'n_seeds': n}
            for col in ['RMSE', 'MAE', 'Pearson_R', 'R2']:
                if col not in grp.columns:
                    continue
                vals = grp[col].dropna()
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f'{col}_mean'] = float(vals.mean())
                row[f'{col}_std']  = std
                row[f'{col}_CI95'] = std * ci_factor
            if 'n_test' in grp.columns:
                row['n_test_mean'] = float(grp['n_test'].mean())
            rows_b.append(row)
        baselines = pd.DataFrame(rows_b)

        # --- gtca_cat from fusion_study/raw_data ---
        gtca_frames = []
        for fpath in sorted(glob.glob(os.path.join(fusion_raw, f'gtca_cat_*_{target}.csv'))):
            fname = os.path.basename(fpath)
            stem  = fname[:-len(f'_{target}.csv')]
            parts = stem.split('_')
            try:
                seed = int(parts[-1])
            except ValueError:
                continue
            df = pd.read_csv(fpath)
            df['seed'] = seed
            gtca_frames.append(df)

        if gtca_frames:
            combined = pd.concat(gtca_frames, ignore_index=True)
            agg = combined.groupby('train_size').agg(
                RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
                MAE_mean=('MAE', 'mean'),   MAE_std=('MAE', 'std'),
                Pearson_R_mean=('Pearson_R', 'mean'), Pearson_R_std=('Pearson_R', 'std'),
                R2_mean=('R2', 'mean'),     R2_std=('R2', 'std'),
                n_seeds=('seed', 'count'),
            ).reset_index()
            for col in ['RMSE', 'MAE', 'Pearson_R', 'R2']:
                agg[f'{col}_CI95'] = agg[f'{col}_std'] * (1.96 / np.sqrt(agg['n_seeds'].clip(lower=2)))
            if 'n_test' in combined.columns:
                agg['n_test_mean'] = combined.groupby('train_size')['n_test'].mean().values
            agg['model']  = 'gtca_cat'
            agg['target'] = target
            merged = pd.concat([baselines, agg], ignore_index=True)
        else:
            merged = baselines

        merged = convert_to_ev(merged, target)

        out = os.path.join(PAPER_DIR, f'lc_{dataset}_all_models.csv')
        # append if multiple targets (qm9)
        if os.path.exists(out) and dataset == 'qm9':
            existing = pd.read_csv(out)
            merged   = pd.concat([existing, merged], ignore_index=True)
        merged.to_csv(out, index=False)
        print(f'  → {out}  (target={target}, models={merged["model"].unique().tolist()})')


# ---------------------------------------------------------------------------
# 2. ablation_gtca_fusion_qm9.csv  (cat vs ca, all 3 QM9 targets)
# ---------------------------------------------------------------------------

def build_fusion_ablation():
    fusion_raw = os.path.join(QM9_DIR, 'fusion_study', 'raw_data')
    all_frames = []

    for target in DATASET_TARGETS['qm9']:
        for fusion in ['cat', 'ca']:
            for fpath in sorted(glob.glob(os.path.join(fusion_raw, f'gtca_{fusion}_6_*_{target}.csv'))):
                fname = os.path.basename(fpath)
                stem  = fname[:-len(f'_{target}.csv')]
                parts = stem.split('_')
                try:
                    seed = int(parts[-1])
                except ValueError:
                    continue
                df = pd.read_csv(fpath)
                df['seed']   = seed
                df['model']  = f'gtca_{fusion}'
                df['target'] = target
                all_frames.append(df)

    if not all_frames:
        print('  [WARN] No fusion_study raw_data found for QM9')
        return

    combined = pd.concat(all_frames, ignore_index=True)

    rows = []
    for (target, model, ts), grp in combined.groupby(['target', 'model', 'train_size']):
        n = len(grp)
        ci_factor = 1.96 / np.sqrt(n) if n > 1 else 0.0
        row = {'target': target, 'model': model, 'train_size': ts, 'n_seeds': n}
        for col in ['RMSE', 'MAE', 'Pearson_R', 'R2']:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f'{col}_mean'] = float(vals.mean())
            row[f'{col}_std']  = std
            row[f'{col}_CI95'] = std * ci_factor
        rows.append(row)

    result = pd.DataFrame(rows)
    # convert per target
    parts = []
    for tgt, grp in result.groupby('target'):
        parts.append(convert_to_ev(grp, tgt))
    out = os.path.join(PAPER_DIR, 'ablation_gtca_fusion_qm9.csv')
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f'  → {out}')


# ---------------------------------------------------------------------------
# 3. ablation_gtca_depth_qm9.csv  (depth 2/4/6)
# ---------------------------------------------------------------------------

def build_depth_ablation():
    """Aggregate directly from raw_data for all 3 QM9 targets (avoids overwrite bug)."""
    raw_dir    = os.path.join(QM9_DIR, 'raw_data')
    all_frames = []

    for target in DATASET_TARGETS['qm9']:
        for depth in [2, 4, 6]:
            for fpath in sorted(glob.glob(os.path.join(raw_dir, f'gtca_{depth}_*_{target}.csv'))):
                fname = os.path.basename(fpath)
                stem  = fname[:-len(f'_{target}.csv')]
                try:
                    seed = int(stem.split('_')[-1])
                except ValueError:
                    continue
                df = pd.read_csv(fpath)
                df['seed']   = seed
                df['model']  = f'gtca_depth_{depth}'
                df['target'] = target
                all_frames.append(df)

    if not all_frames:
        print('  [WARN] No depth raw_data found')
        return

    combined = pd.concat(all_frames, ignore_index=True)
    rows = []
    for (target, model, ts), grp in combined.groupby(['target', 'model', 'train_size']):
        n = len(grp)
        row = {'model': model, 'target': target, 'train_size': ts, 'n_seeds': n}
        for col in ['RMSE', 'MAE', 'Pearson_R', 'R2']:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f'{col}_mean'] = float(vals.mean())
            row[f'{col}_std']  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if 'n_test' in grp.columns:
            row['n_test_mean'] = float(grp['n_test'].mean())
        rows.append(row)

    result = pd.DataFrame(rows)
    parts  = []
    for tgt, grp in result.groupby('target'):
        parts.append(convert_to_ev(grp, tgt))
    df_ev = pd.concat(parts, ignore_index=True)
    out = os.path.join(PAPER_DIR, 'ablation_gtca_depth_qm9.csv')
    df_ev.to_csv(out, index=False)
    print(f'  → {out}')
    return df_ev


# ---------------------------------------------------------------------------
# 4. ablation_gtca_depth_ci_qm9.csv
# ---------------------------------------------------------------------------

def build_depth_ci():
    raw_dir = os.path.join(QM9_DIR, 'raw_data')
    all_frames = []

    for target in DATASET_TARGETS['qm9']:
        for depth in [2, 4, 6]:
            for fpath in sorted(glob.glob(os.path.join(raw_dir, f'gtca_{depth}_*_{target}.csv'))):
                fname = os.path.basename(fpath)
                stem  = fname[:-len(f'_{target}.csv')]
                parts = stem.split('_')
                try:
                    seed = int(parts[-1])
                except ValueError:
                    continue
                df = pd.read_csv(fpath)
                df['seed']       = seed
                df['bert_depth'] = depth
                df['target']     = target
                all_frames.append(df)

    if not all_frames:
        return

    combined = pd.concat(all_frames, ignore_index=True)
    rows = []
    for (target, depth, ts), grp in combined.groupby(['target', 'bert_depth', 'train_size']):
        n = len(grp)
        ci_factor = 1.96 / np.sqrt(n) if n > 1 else 0.0
        row = {'target': target, 'bert_depth': depth, 'train_size': ts, 'n_seeds': n}
        for col in ['RMSE', 'MAE', 'Pearson_R', 'R2']:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f'{col}_mean'] = float(vals.mean())
            row[f'{col}_std']  = std
            row[f'{col}_CI95'] = std * ci_factor
        rows.append(row)

    result = pd.DataFrame(rows)
    parts = []
    for tgt, grp in result.groupby('target'):
        parts.append(convert_to_ev(grp, tgt))
    out = os.path.join(PAPER_DIR, 'ablation_gtca_depth_ci_qm9.csv')
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f'  → {out}')


# ---------------------------------------------------------------------------
# 5 & 6. Welch t-test stats
# ---------------------------------------------------------------------------

def _welch_tests(combined_df, group_col: str, ref_model: str, other_models: list,
                 metric: str = 'RMSE') -> pd.DataFrame:
    """Welch t-test: ref_model vs each other at each train_size."""
    rows = []
    for ts, grp in combined_df.groupby('train_size'):
        ref = grp[grp[group_col] == ref_model][metric].dropna().values
        for other in other_models:
            oth = grp[grp[group_col] == other][metric].dropna().values
            if len(ref) < 2 or len(oth) < 2:
                continue
            t, p = ttest_ind(ref, oth, equal_var=False)
            rows.append({
                'train_size': ts,
                'model_a': ref_model,
                'model_b': other,
                'metric':  metric,
                't_stat':  float(t),
                'p_value': float(p),
                'significant_005': p < 0.05,
            })
    return pd.DataFrame(rows)


def build_fusion_welch():
    fusion_raw = os.path.join(QM9_DIR, 'fusion_study', 'raw_data')
    all_frames = []

    for target in DATASET_TARGETS['qm9']:
        for fusion in ['cat', 'ca']:
            for fpath in sorted(glob.glob(os.path.join(fusion_raw, f'gtca_{fusion}_6_*_{target}.csv'))):
                fname = os.path.basename(fpath)
                stem  = fname[:-len(f'_{target}.csv')]
                try:
                    seed = int(stem.split('_')[-1])
                except ValueError:
                    continue
                df = pd.read_csv(fpath)
                df['seed']   = seed
                df['model']  = f'gtca_{fusion}'
                df['target'] = target
                all_frames.append(df)

    if not all_frames:
        return

    combined = pd.concat(all_frames, ignore_index=True)
    results  = []
    for target in DATASET_TARGETS['qm9']:
        sub = combined[combined['target'] == target]
        for metric in ['RMSE', 'MAE']:
            df_w = _welch_tests(sub, 'model', 'gtca_cat', ['gtca_ca'], metric)
            df_w['target'] = target
            results.append(df_w)

    out = os.path.join(PAPER_DIR, 'stats_fusion_welch_qm9.csv')
    pd.concat(results, ignore_index=True).to_csv(out, index=False)
    print(f'  → {out}')


def build_depth_welch():
    raw_dir    = os.path.join(QM9_DIR, 'raw_data')
    all_frames = []

    for target in DATASET_TARGETS['qm9']:
        for depth in [2, 4, 6]:
            for fpath in sorted(glob.glob(os.path.join(raw_dir, f'gtca_{depth}_*_{target}.csv'))):
                fname = os.path.basename(fpath)
                stem  = fname[:-len(f'_{target}.csv')]
                try:
                    seed = int(stem.split('_')[-1])
                except ValueError:
                    continue
                df = pd.read_csv(fpath)
                df['seed']       = seed
                df['model']      = f'gtca_depth_{depth}'
                df['target']     = target
                all_frames.append(df)

    if not all_frames:
        return

    combined = pd.concat(all_frames, ignore_index=True)
    results  = []
    for target in DATASET_TARGETS['qm9']:
        sub = combined[combined['target'] == target]
        for metric in ['RMSE', 'MAE']:
            df_w = _welch_tests(sub, 'model', 'gtca_depth_6',
                                ['gtca_depth_2', 'gtca_depth_4'], metric)
            df_w['target'] = target
            results.append(df_w)

    out = os.path.join(PAPER_DIR, 'stats_depth_welch_qm9.csv')
    pd.concat(results, ignore_index=True).to_csv(out, index=False)
    print(f'  → {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PAPER_DIR, exist_ok=True)

    # Remove existing lc_*_all_models.csv so we rebuild fresh
    for f in glob.glob(os.path.join(PAPER_DIR, 'lc_*_all_models.csv')):
        os.remove(f)

    print('\n=== 1. lc_*_all_models.csv ===')
    for folder, dataset in FOLDER_TO_DATASET.items():
        print(f'  {dataset}')
        build_lc_all_models(folder, dataset, DATASET_TARGETS[dataset])

    print('\n=== 2. ablation_gtca_fusion_qm9.csv ===')
    build_fusion_ablation()

    print('\n=== 3. ablation_gtca_depth_qm9.csv ===')
    build_depth_ablation()

    print('\n=== 4. ablation_gtca_depth_ci_qm9.csv ===')
    build_depth_ci()

    print('\n=== 5. stats_fusion_welch_qm9.csv ===')
    build_fusion_welch()

    print('\n=== 6. stats_depth_welch_qm9.csv ===')
    build_depth_welch()

    print('\nDone.')


if __name__ == '__main__':
    main()
