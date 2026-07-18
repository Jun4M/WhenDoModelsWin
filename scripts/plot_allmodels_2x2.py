"""
scripts/plot_allmodels_2x2.py

Per-target all-model 2×2 learning-curve figures (4 figures):
  QM9 HOMO  → allmodels_2x2_homo.{png,pdf}
  QM9 LUMO  → allmodels_2x2_lumo.{png,pdf}
  QM9 gap   → allmodels_2x2_gap.{png,pdf}
  ESOL      → allmodels_2x2_esol.{png,pdf}

2×2 panels: RMSE | MAE / Pearson R | R²
Error bars: ±1 SD via ax.errorbar — fill_between 금지 (Spec 11).
Shared figure legend at bottom (ncol=5, fontsize 7).
Style: regenerate_plots.py (linewidth=1.8, markersize=4, grid alpha=0.3, dpi=300).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PAPER_CSV   = 'results/paper_csv'
PAPER_PLOTS = 'results/paper_plots'

# ---------------------------------------------------------------------------
# Color / label maps — extended from regenerate_plots.py
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    # ── original regenerate_plots.py ──────────────────────────────────────
    'gcn':           '#3B82C4',
    'transformer':   '#E07B39',
    'gtca_cat':      '#2EAB6E',
    'rf':            '#9B59B6',
    'xgb':           '#E74C3C',
    'gpr':           '#F39C12',
    'attentivefp':   '#1ABC9C',
    'lgbm':          '#C0392B',
    'svr':           '#95A5A6',
    'gps':           '#7F8C8D',
    'painn':         '#56B4E9',
    # ── R1 revision additions ─────────────────────────────────────────────
    'chemberta2':      '#D4880A',   # dark amber (transformer family)
    'molformer':       '#8E44AD',   # medium purple
    'selformer':       '#1A5276',   # dark navy
    'chemprop':        '#0B6623',   # forest green
    'krovex':          '#B7950B',   # dark gold
    'unimol_scratch':  '#7FB3D3',   # muted sky-blue
    'unimol_pt':       '#154360',   # deep blue
    # ── MTL variants (QM9 only) ───────────────────────────────────────────
    'attentivefp_mtl': '#117A65',   # dark teal
    'gcn_mtl':         '#922B21',   # dark crimson
}

MODEL_LABELS = {
    'gcn':             'GCN',
    'transformer':     'ChemBERTa',
    'gtca_cat':        'GTCA-Cat',
    'rf':              'Random Forest',
    'xgb':             'XGBoost',
    'gpr':             'GPR',
    'attentivefp':     'AttentiveFP',
    'lgbm':            'LightGBM',
    'svr':             'SVR',
    'gps':             'GPS',
    'painn':           'PaiNN',
    'chemberta2':      'ChemBERTa-2',
    'molformer':       'MoLFormer',
    'selformer':       'SELFormer',
    'chemprop':        'Chemprop',
    'krovex':          'KROVEX',
    'unimol_scratch':  'UniMol-Scratch',
    'unimol_pt':       'UniMol-PT',
    'attentivefp_mtl': 'AttentiveFP-MTL',
    'gcn_mtl':         'GCN-MTL',
}

# Draw order: GNN → Seq → 3D → Tree → MTL
MODEL_ORDER = [
    'attentivefp', 'gcn', 'gps', 'krovex', 'gtca_cat',
    'transformer', 'chemberta2', 'molformer', 'selformer', 'chemprop',
    'painn', 'unimol_scratch', 'unimol_pt',
    'rf', 'xgb', 'lgbm', 'svr', 'gpr',
    'attentivefp_mtl', 'gcn_mtl',
]

METRICS = ['RMSE', 'MAE', 'Pearson_R', 'R2']
METRIC_TITLES = {
    'RMSE':      'RMSE',
    'MAE':       'MAE',
    'Pearson_R': 'Pearson R',
    'R2':        'R²',
}

DATASET_YLABELS = {
    'qm9':  {
        'RMSE':      'RMSE (eV)',
        'MAE':       'MAE (eV)',
        'Pearson_R': 'Pearson R',
        'R2':        'R²',
    },
    'esol': {
        'RMSE':      'RMSE (log mol/L)',
        'MAE':       'MAE (log mol/L)',
        'Pearson_R': 'Pearson R',
        'R2':        'R²',
    },
    'lipo': {
        'RMSE':      'RMSE (log D)',
        'MAE':       'MAE (log D)',
        'Pearson_R': 'Pearson R',
        'R2':        'R²',
    },
    'bace': {
        'RMSE':      'RMSE (pIC50)',
        'MAE':       'MAE (pIC50)',
        'Pearson_R': 'Pearson R',
        'R2':        'R²',
    },
}

LW = 1.8
MS = 4
CAPSIZE = 2


# ---------------------------------------------------------------------------
# Panel draw
# ---------------------------------------------------------------------------

def _clip_bounds(df, metric):
    """Return (ymin, ymax) clip bounds for the panel.

    RMSE/MAE : [0,  95th-pct × 1.3]   — excludes diverged-run outliers
    Pearson_R: [-0.15, 1.05]
    R2       : [max(5th-pct × 1.3, -2), 1.05]
    """
    mean_col = f'{metric}_mean'
    vals = df[mean_col].dropna().values
    if len(vals) == 0:
        return None, None
    if metric in ('RMSE', 'MAE'):
        return 0.0, float(np.percentile(vals, 95)) * 1.3
    if metric == 'Pearson_R':
        return -0.15, 1.05
    if metric == 'R2':
        return -1.0, 1.05
    return None, None


def _draw_panel(ax, df, metric, ylabel, title):
    """Draw one learning-curve panel with ±1 SD error bars.

    Out-of-range values are replaced with NaN so the line breaks cleanly
    at diverged points instead of shooting a segment to the axes boundary.
    """
    mean_col = f'{metric}_mean'
    std_col  = f'{metric}_std'
    if mean_col not in df.columns:
        return

    # Compute clip bounds FIRST (before drawing) so NaN masking is consistent
    ymin, ymax = _clip_bounds(df, metric)

    models_in_data = set(df['model'].unique())
    ordered = [m for m in MODEL_ORDER if m in models_in_data]
    ordered += sorted(models_in_data - set(ordered))  # catch unknowns

    for model in ordered:
        sub = df[df['model'] == model].sort_values('train_size')
        if sub.empty:
            continue
        x    = sub['train_size'].values
        y    = sub[mean_col].values.astype(float)
        yerr = (sub[std_col].fillna(0).values.astype(float)
                if std_col in sub.columns else np.zeros_like(y))

        # NaN-mask out-of-range → clean line breaks at diverged points
        if ymin is not None and ymax is not None:
            bad = (y > ymax) | (y < ymin)
            y    = np.where(bad, np.nan, y)
            yerr = np.where(bad, 0.0, yerr)

        color = MODEL_COLORS.get(model, 'black')
        label = MODEL_LABELS.get(model, model)

        ax.errorbar(
            x, y, yerr=yerr,
            label=label,
            color=color,
            linewidth=LW,
            marker='o',
            markersize=MS,
            capsize=CAPSIZE,
            elinewidth=0.7,
        )

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Training set size', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# Per-target figure
# ---------------------------------------------------------------------------

def make_figure(df, dataset, target_label, suptitle, out_stem):
    """Build 2×2 figure and save PNG + PDF."""
    ylabels = DATASET_YLABELS[dataset]
    n_models = df['model'].nunique()
    ncol = 5

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, metric in zip(axes.flat, METRICS):
        _draw_panel(ax, df, metric, ylabels[metric], METRIC_TITLES[metric])

    # ── shared figure legend ──────────────────────────────────────────────
    # collect from the panel that rendered the most lines (RMSE, top-left)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    n_rows = -(-len(handles) // ncol)  # ceiling division
    # rough per-row height in figure-fraction units (fontsize 7 ≈ 0.022 fig height per row)
    legend_frac = n_rows * 0.028 + 0.015

    fig.tight_layout(rect=[0, legend_frac, 1, 1.0])

    leg = fig.legend(
        handles, labels,
        loc='upper center',
        ncol=ncol,
        fontsize=7,
        bbox_to_anchor=(0.5, legend_frac),
        framealpha=0.9,
        columnspacing=0.8,
        handlelength=1.5,
        borderpad=0.6,
    )

    os.makedirs(PAPER_PLOTS, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(PAPER_PLOTS, f'{out_stem}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'  → {path}')
    plt.close(fig)
    print(f'     ({n_models} models, {len(df["train_size"].unique())} train sizes)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    lc_qm9  = pd.read_csv(os.path.join(PAPER_CSV, 'lc_qm9_all_models.csv'))
    lc_esol = pd.read_csv(os.path.join(PAPER_CSV, 'lc_esol_all_models.csv'))
    lc_lipo = pd.read_csv(os.path.join(PAPER_CSV, 'lc_lipo_all_models.csv'))
    lc_bace = pd.read_csv(os.path.join(PAPER_CSV, 'lc_bace_all_models.csv'))

    ESOL_TGT = 'measured log solubility in mols per litre'
    LIPO_TGT = lc_lipo['target'].iloc[0]
    BACE_TGT = lc_bace['target'].iloc[0]

    tasks = [
        # (df_subset, dataset, target_label, suptitle, out_stem)
        (
            lc_qm9[lc_qm9['target'] == 'homo'].copy(),
            'qm9', 'homo',
            'QM9 — HOMO: All Models Learning Curves',
            'allmodels_2x2_homo',
        ),
        (
            lc_qm9[lc_qm9['target'] == 'lumo'].copy(),
            'qm9', 'lumo',
            'QM9 — LUMO: All Models Learning Curves',
            'allmodels_2x2_lumo',
        ),
        (
            lc_qm9[lc_qm9['target'] == 'gap'].copy(),
            'qm9', 'gap',
            'QM9 — HOMO–LUMO Gap: All Models Learning Curves',
            'allmodels_2x2_gap',
        ),
        (
            lc_esol[lc_esol['train_size'] <= 375].copy(),
            'esol', ESOL_TGT,
            'ESOL: All Models Learning Curves',
            'allmodels_2x2_esol',
        ),
        (
            lc_lipo[lc_lipo['train_size'] <= 1000].copy(),
            'lipo', LIPO_TGT,
            'Lipophilicity: All Models Learning Curves',
            'allmodels_2x2_lipo',
        ),
        (
            lc_bace[lc_bace['train_size'] <= 500].copy(),
            'bace', BACE_TGT,
            'BACE: All Models Learning Curves',
            'allmodels_2x2_bace',
        ),
    ]

    for df, dataset, target_label, suptitle, out_stem in tasks:
        print(f'\n{suptitle}')
        make_figure(df, dataset, target_label, suptitle, out_stem)

    print('\nDone.')


if __name__ == '__main__':
    main()
