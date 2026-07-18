"""
scripts/plot_ensemble_analysis.py

6-panel ensemble analysis figure (R1.5, Reviewer 1 #5).

Panels  (3 rows × 2 cols):
  row 1 — QM9 HOMO      | QM9 LUMO
  row 2 — QM9 Gap       | ESOL
  row 3 — Lipophilicity | BACE

Per panel 8 lines:
  Comparison models (from lc_{ds}_all_models.csv): ±1 SD error bar
    - AttentiveFP   #1ABC9C  solid
    - MoLFormer     #E07B39  solid
    - KROVEX        #8E44AD  solid
    - Random Forest #7F8C8D  dashed  (de-emphasized reference)

  Ensemble (from ensemble_{ds}.csv, seed-mean ± seed-std per train_size):
    - best single (ref)    black    solid    ● circle
    - top-3 ensemble       #2C7FB8  solid    ▲ filled triangle
    - top-5 ensemble       #2C7FB8  dashed   △ open triangle
    - cross-family         #D7301F  solid    ■ square

Style reference: scripts/plot_allmodels_2x2.py
  LW=1.8, MS=4/6, CAPSIZE=2, elinewidth=0.8, grid alpha=0.3, dpi=300
  x-axis: log scale; y-axis: linear RMSE
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAPER_CSV   = 'results/paper_csv'
PAPER_PLOTS = 'results/paper_plots'

# ---------------------------------------------------------------------------
# Style constants — mirror plot_allmodels_2x2.py
# ---------------------------------------------------------------------------
LW      = 1.8
MS      = 4      # comparison model markers
MS_ENS  = 6      # ensemble markers (slightly larger for emphasis)
CAPSIZE = 2
ELW     = 0.8   # elinewidth

COMP_MODELS = {
    'attentivefp': {'label': 'AttentiveFP',  'color': '#1ABC9C', 'ls': '-',  'marker': 'o'},
    'molformer':   {'label': 'MoLFormer',    'color': '#E07B39', 'ls': '-',  'marker': 'o'},
    'krovex':      {'label': 'KROVEX',       'color': '#8E44AD', 'ls': '-',  'marker': 'o'},
    'rf':          {'label': 'Random Forest','color': '#7F8C8D', 'ls': '--', 'marker': 'o'},
}

ENS_LINES = {
    'best_single_rmse': {'label': 'Best single (ref)',     'color': 'black',   'ls': '-',  'marker': 'o', 'fillstyle': 'full'},
    'top3':             {'label': 'Top-3 ensemble',        'color': '#2C7FB8', 'ls': '-',  'marker': '^', 'fillstyle': 'full'},
    'top5':             {'label': 'Top-5 ensemble',        'color': '#2C7FB8', 'ls': '--', 'marker': '^', 'fillstyle': 'none'},
    'cross_fam_rmse':   {'label': 'Cross-family ensemble', 'color': '#D7301F', 'ls': '-',  'marker': 's', 'fillstyle': 'full'},
}

DATASET_YLABEL = {
    'qm9':  'RMSE (eV)',
    'esol': 'RMSE (log mol/L)',
    'lipo': 'RMSE (log D)',
    'bace': 'RMSE (pIC50)',
}

# ensemble_*.csv target strings → lc_*_all_models.csv target strings
LC_TARGET = {
    'qm9':  {'homo': 'homo', 'lumo': 'lumo', 'gap': 'gap'},
    'esol': {'measured_log_solubility_in_mols_per_litre':
             'measured log solubility in mols per litre'},
    'lipo': {'exp': 'exp'},
    'bace': {'pIC50': 'pIC50'},
}

# max train_size to show per dataset (canonical limits)
MAX_SIZE = {'qm9': 3000, 'esol': 375, 'lipo': 1000, 'bace': 500}

# exact train_sizes to include in ensemble panels
ENSEMBLE_SIZES = {
    'qm9':  [50, 100, 200, 500, 1000, 3000],
    'esol': [50, 100, 200, 375],
    'lipo': [50, 100, 200, 500, 1000],
    'bace': [50, 100, 200, 500],
}

PANELS = [
    # (dataset, ens_target, panel_title)
    ('qm9',  'homo',                                       'QM9 — HOMO'),
    ('qm9',  'lumo',                                       'QM9 — LUMO'),
    ('qm9',  'gap',                                        'QM9 — HOMO–LUMO Gap'),
    ('esol', 'measured_log_solubility_in_mols_per_litre',  'ESOL'),
    ('lipo', 'exp',                                        'Lipophilicity'),
    ('bace', 'pIC50',                                      'BACE'),
]

# legend label order
LEGEND_ORDER = [
    'AttentiveFP', 'MoLFormer', 'KROVEX', 'Random Forest',
    'Best single (ref)', 'Top-3 ensemble', 'Top-5 ensemble', 'Cross-family ensemble',
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_lc(dataset: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PAPER_CSV, f'lc_{dataset}_all_models.csv'))


def load_ensemble(dataset: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(PAPER_CSV, f'ensemble_{dataset}.csv'))


# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------

def _draw_comparison(ax, lc_df, lc_target_str, sizes):
    """4 comparison-model lines with ±1 SD error bars."""
    sub = lc_df[(lc_df['target'] == lc_target_str) &
                (lc_df['train_size'].isin(sizes))]
    handles = {}
    for key, cfg in COMP_MODELS.items():
        m_sub = sub[sub['model'] == key].sort_values('train_size')
        if m_sub.empty:
            continue
        x    = m_sub['train_size'].values
        y    = m_sub['RMSE_mean'].values.astype(float)
        yerr = m_sub['RMSE_std'].fillna(0).values.astype(float)
        c = ax.errorbar(
            x, y, yerr=yerr,
            label=cfg['label'], color=cfg['color'], linestyle=cfg['ls'],
            linewidth=LW, marker=cfg['marker'], markersize=MS,
            capsize=CAPSIZE, elinewidth=ELW,
        )
        handles[cfg['label']] = c
    return handles


def _draw_ensemble(ax, ens_df, ens_target_str, sizes):
    """4 ensemble lines: seed-mean ± seed-std aggregated per train_size."""
    sub = ens_df[(ens_df['target'] == ens_target_str) &
                 (ens_df['train_size'].isin(sizes))]
    if sub.empty:
        return {}
    handles = {}
    for col, cfg in ENS_LINES.items():
        if col not in sub.columns:
            continue
        agg = (sub.groupby('train_size')[col]
                   .agg(['mean', 'std'])
                   .dropna(subset=['mean'])
                   .sort_index())
        if agg.empty:
            continue
        x    = agg.index.values
        y    = agg['mean'].values.astype(float)
        yerr = agg['std'].fillna(0).values.astype(float)
        c = ax.errorbar(
            x, y, yerr=yerr,
            label=cfg['label'], color=cfg['color'], linestyle=cfg['ls'],
            linewidth=LW, marker=cfg['marker'], markersize=MS_ENS,
            fillstyle=cfg['fillstyle'],
            capsize=CAPSIZE, elinewidth=ELW,
            zorder=5,
        )
        handles[cfg['label']] = c
    return handles


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure():
    lc_cache  = {ds: load_lc(ds)       for ds in ['qm9', 'esol', 'lipo', 'bace']}
    ens_cache = {ds: load_ensemble(ds)  for ds in ['qm9', 'esol', 'lipo', 'bace']}

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    all_handles: dict = {}   # label → handle, dedup across panels

    for ax, (dataset, ens_tgt, title) in zip(axes.flat, PANELS):
        lc_tgt = LC_TARGET[dataset].get(ens_tgt, ens_tgt)
        sizes  = ENSEMBLE_SIZES[dataset]

        h1 = _draw_comparison(ax, lc_cache[dataset], lc_tgt, sizes)
        h2 = _draw_ensemble(ax, ens_cache[dataset], ens_tgt, sizes)
        for lbl, h in {**h1, **h2}.items():
            all_handles.setdefault(lbl, h)

        ax.set_xscale('log')
        ax.set_xlabel('Training set size', fontsize=10)
        ax.set_ylabel(DATASET_YLABEL[dataset], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(labelsize=9)

    # Shared figure legend — ordered
    ordered_labels  = [l for l in LEGEND_ORDER if l in all_handles]
    ordered_labels += [l for l in all_handles  if l not in LEGEND_ORDER]
    ordered_handles = [all_handles[l] for l in ordered_labels]

    fig.tight_layout(rect=[0, 0.06, 1, 1.0])

    fig.legend(
        ordered_handles, ordered_labels,
        loc='upper center', ncol=4, fontsize=8,
        bbox_to_anchor=(0.5, 0.06),
        framealpha=0.9, columnspacing=0.8,
        handlelength=2.0, borderpad=0.6,
    )
    return fig


def save_figure(fig):
    os.makedirs(PAPER_PLOTS, exist_ok=True)
    base = os.path.join(PAPER_PLOTS, 'ensemble_analysis')
    for ext in ('png', 'pdf'):
        out = f'{base}.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'  → {out}')
    plt.close(fig)


def main():
    print('Building 6-panel ensemble analysis figure ...')
    fig = make_figure()
    save_figure(fig)
    print('Done.')


if __name__ == '__main__':
    main()
