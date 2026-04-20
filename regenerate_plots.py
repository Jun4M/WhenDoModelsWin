"""
regenerate_plots.py
새 de-normalized paper_csv 기반으로 논문 모든 그래프 재생성.

생성 파일:
  results/01_QM9/plots/  — baselines LC (homo/lumo/gap), depth ablation, fusion, final
  results/02_ESOL/plots/ — baselines LC + combined
  results/03_Lipo/plots/ — baselines LC + combined
  results/04_BACE/plots/ — baselines LC + combined
  results/paper_plots/   — 논문 제출용 고해상도 PNG (각 데이터셋 RMSE만)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PAPER_CSV = './results/paper_csv'
RESULTS   = './results'
PAPER_PLOTS = './results/paper_plots'

# ---------------------------------------------------------------------------
# Color / label maps (extended)
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    'gcn':          '#3B82C4',
    'transformer':  '#E07B39',
    'gtca_cat':     '#2EAB6E',
    'gtca_ca':      '#E74C3C',
    'rf':           '#9B59B6',
    'xgb':          '#E74C3C',
    'gpr':          '#F39C12',
    'attentivefp':  '#1ABC9C',
    'lgbm':         '#C0392B',
    'svr':          '#95A5A6',
    'gps':          '#7F8C8D',
    'gtca_depth_2': '#E74C3C',
    'gtca_depth_4': '#3B82C4',
    'gtca_depth_6': '#2EAB6E',
}

MODEL_LABELS = {
    'gcn':          'GCN',
    'transformer':  'ChemBERTa',
    'gtca_cat':     'GTCA-Cat',
    'gtca_ca':      'GTCA-CA',
    'rf':           'Random Forest',
    'xgb':          'XGBoost',
    'gpr':          'GPR',
    'attentivefp':  'AttentiveFP',
    'lgbm':         'LightGBM',
    'svr':          'SVR',
    'gps':          'GPS',
    'gtca_depth_2': 'GTCA (depth=2)',
    'gtca_depth_4': 'GTCA (depth=4)',
    'gtca_depth_6': 'GTCA (depth=6)',
}

DATASET_YLABEL = {
    'qm9':  'RMSE (eV)',
    'esol': 'RMSE (log mol/L)',
    'lipo': 'RMSE (log D)',
    'bace': 'RMSE (pIC50)',
}

METRICS = ['RMSE', 'MAE', 'Pearson_R', 'R2']
METRIC_LABELS = {'RMSE': 'RMSE', 'MAE': 'MAE',
                 'Pearson_R': 'Pearson R', 'R2': 'R²'}

# Model display order for legend
BASELINE_ORDER = ['gcn', 'transformer', 'rf', 'xgb', 'gpr',
                  'attentivefp', 'lgbm', 'svr', 'gps', 'gtca_cat']


# ---------------------------------------------------------------------------
# Core draw helpers
# ---------------------------------------------------------------------------

def _draw_lc(ax, df, metric, ylabel=None, title='', use_ci=False, show_legend=True):
    """
    Draw learning curve panel. df must have {metric}_mean and {metric}_std (or _CI95).
    """
    mean_col = f'{metric}_mean'
    ci_col   = f'{metric}_CI95' if use_ci else f'{metric}_std'

    if mean_col not in df.columns:
        return

    # respect BASELINE_ORDER if possible, otherwise sorted
    models_in_data = df['model'].unique().tolist()
    ordered = [m for m in BASELINE_ORDER if m in models_in_data]
    ordered += [m for m in models_in_data if m not in ordered]

    for model in ordered:
        sub = df[df['model'] == model].sort_values('train_size')
        if sub.empty:
            continue
        color = MODEL_COLORS.get(model, 'gray')
        label = MODEL_LABELS.get(model, model)
        x = sub['train_size'].values
        y = sub[mean_col].values

        ax.plot(x, y, color=color, label=label, linewidth=1.8,
                marker='o', markersize=4)

        if ci_col in sub.columns:
            yerr = sub[ci_col].fillna(0).values
            if not np.all(yerr == 0):
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

    ax.set_xlabel('Training set size', fontsize=10)
    ax.set_ylabel(ylabel or METRIC_LABELS.get(metric, metric), fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    if show_legend:
        ax.legend(fontsize=7, loc='upper right', ncol=1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())


def _save(fig, path, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    base = os.path.splitext(path)[0]
    for ext in ['png', 'pdf']:
        out = f'{base}.{ext}'
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        print(f'  → {out}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Baseline LC: per dataset (all metrics, combined 2×2, and RMSE-only)
# ---------------------------------------------------------------------------

def plot_baselines(df, plots_dir, dataset, target, title_prefix=''):
    """4 panels (RMSE/MAE/PearsonR/R2) + individual RMSE PNG."""
    ylabel = DATASET_YLABEL.get(dataset, 'RMSE')

    # Individual metric PNGs
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_lc(ax, df, metric, ylabel=(ylabel if metric in ('RMSE','MAE') else None),
                 title=f'{title_prefix} — {METRIC_LABELS[metric]}')
        _save(fig, os.path.join(plots_dir, f'plot_baselines_{target}_{metric}.png'))

    # Combined 2×2
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'Baseline Models — {title_prefix}', fontsize=13)
    for ax, metric in zip(axes.flat, METRICS):
        _draw_lc(ax, df, metric,
                 ylabel=(ylabel if metric in ('RMSE','MAE') else None),
                 title=METRIC_LABELS[metric])
    _save(fig, os.path.join(plots_dir, f'plot_baselines_lc_{target}.png'))


# ---------------------------------------------------------------------------
# 2. GTCA Depth Ablation LC
# ---------------------------------------------------------------------------

def plot_depth_ablation(df, plots_dir, target, dataset='qm9'):
    ylabel = DATASET_YLABEL.get(dataset, 'RMSE')
    title  = f'GTCA Depth Ablation — {target.upper()}'

    # Combined 2×2
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=13)
    for ax, metric in zip(axes.flat, METRICS):
        _draw_lc(ax, df, metric,
                 ylabel=(ylabel if metric in ('RMSE','MAE') else None),
                 title=METRIC_LABELS[metric])
    _save(fig, os.path.join(plots_dir, f'plot_gtca_depth_lc_{target}.png'))

    # RMSE only (논문 Fig용)
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_lc(ax, df, 'RMSE', ylabel=ylabel, title=title)
    _save(fig, os.path.join(plots_dir, f'plot_gtca_depth_rmse_{target}.png'))


# ---------------------------------------------------------------------------
# 3. Fusion Comparison (Cat vs CA)
# ---------------------------------------------------------------------------

def plot_fusion(df, plots_dir, target, dataset='qm9'):
    ylabel = DATASET_YLABEL.get(dataset, 'RMSE')
    title  = f'GTCA Fusion: Cat vs CA — {target.upper()}'

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=13)
    for ax, metric in zip(axes.flat, METRICS):
        _draw_lc(ax, df, metric, use_ci=True,
                 ylabel=(ylabel if metric in ('RMSE','MAE') else None),
                 title=METRIC_LABELS[metric])
    _save(fig, os.path.join(plots_dir, f'plot_fusion_comparison_{target}.png'))

    # RMSE only
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_lc(ax, df, 'RMSE', ylabel=ylabel, title=title, use_ci=True)
    _save(fig, os.path.join(plots_dir, f'plot_fusion_rmse_{target}.png'))


# ---------------------------------------------------------------------------
# 4. Final: baselines + GTCA-Cat (one combined plot per target)
# ---------------------------------------------------------------------------

def plot_final(df_baselines, df_gtca_cat, plots_dir, dataset, target, title_prefix):
    """Merge baselines + gtca_cat and plot all together."""
    merged = pd.concat([df_baselines, df_gtca_cat], ignore_index=True)
    ylabel = DATASET_YLABEL.get(dataset, 'RMSE')
    title  = f'All Models — {title_prefix}'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=13)
    for ax, metric in zip(axes.flat, METRICS):
        _draw_lc(ax, merged, metric,
                 ylabel=(ylabel if metric in ('RMSE','MAE') else None),
                 title=METRIC_LABELS[metric])
    _save(fig, os.path.join(plots_dir, f'plot_final_{target}_depth6.png'))


# ---------------------------------------------------------------------------
# 5. Paper-quality RMSE-only plots (single panel, larger font)
# ---------------------------------------------------------------------------

def plot_paper_rmse(df, out_path, title, ylabel, use_ci=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_lc(ax, df, 'RMSE', ylabel=ylabel, title=title, use_ci=use_ci)
    ax.set_xlabel('Training set size', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc='upper right')
    _save(fig, out_path, dpi=200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PAPER_PLOTS, exist_ok=True)

    # ---- Load all paper CSVs ----
    lc_qm9  = pd.read_csv(os.path.join(PAPER_CSV, 'lc_qm9_all_models.csv'))
    lc_esol = pd.read_csv(os.path.join(PAPER_CSV, 'lc_esol_all_models.csv'))
    lc_lipo = pd.read_csv(os.path.join(PAPER_CSV, 'lc_lipo_all_models.csv'))
    lc_bace = pd.read_csv(os.path.join(PAPER_CSV, 'lc_bace_all_models.csv'))
    abl_dep = pd.read_csv(os.path.join(PAPER_CSV, 'ablation_gtca_depth_qm9.csv'))
    abl_fus = pd.read_csv(os.path.join(PAPER_CSV, 'ablation_gtca_fusion_qm9.csv'))

    ESOL_TGT = 'measured log solubility in mols per litre'

    # ====================================================================
    print('\n=== QM9 ===')
    qm9_plots = os.path.join(RESULTS, '01_QM9', 'plots')

    for target in ['homo', 'lumo', 'gap']:
        sub_all = lc_qm9[lc_qm9['target'] == target].copy()
        # baselines only (excl. gtca_cat for separate layer)
        sub_base = sub_all[sub_all['model'] != 'gtca_cat'].copy()
        sub_gtca = sub_all[sub_all['model'] == 'gtca_cat'].copy()

        print(f'  QM9/{target} baselines')
        plot_baselines(sub_all, qm9_plots, 'qm9', target, title_prefix=f'QM9 {target.upper()}')

        print(f'  QM9/{target} final (baselines+GTCA-Cat)')
        plot_final(sub_base, sub_gtca, qm9_plots, 'qm9', target, f'QM9 {target.upper()}')

        print(f'  QM9/{target} depth ablation')
        sub_dep = abl_dep[abl_dep['target'] == target].copy()
        plot_depth_ablation(sub_dep, qm9_plots, target, dataset='qm9')

        print(f'  QM9/{target} fusion comparison')
        sub_fus = abl_fus[abl_fus['target'] == target].copy()
        plot_fusion(sub_fus, qm9_plots, target, dataset='qm9')

        # Paper-quality RMSE
        plot_paper_rmse(
            sub_all, os.path.join(PAPER_PLOTS, f'qm9_{target}_rmse.png'),
            title=f'QM9 {target.upper()} Learning Curve',
            ylabel=DATASET_YLABEL['qm9'],
        )
        plot_paper_rmse(
            sub_dep, os.path.join(PAPER_PLOTS, f'qm9_{target}_depth_rmse.png'),
            title=f'QM9 {target.upper()} — GTCA Depth Ablation',
            ylabel=DATASET_YLABEL['qm9'],
        )
        plot_paper_rmse(
            sub_fus, os.path.join(PAPER_PLOTS, f'qm9_{target}_fusion_rmse.png'),
            title=f'QM9 {target.upper()} — Fusion: Cat vs CA',
            ylabel=DATASET_YLABEL['qm9'],
            use_ci=True,
        )

    # ====================================================================
    print('\n=== ESOL ===')
    esol_plots = os.path.join(RESULTS, '02_ESOL', 'plots')
    sub_esol   = lc_esol[lc_esol['train_size'] <= 375].copy()
    sub_esol_base = sub_esol[sub_esol['model'] != 'gtca_cat'].copy()
    sub_esol_gtca = sub_esol[sub_esol['model'] == 'gtca_cat'].copy()

    plot_baselines(sub_esol, esol_plots, 'esol', ESOL_TGT,
                   title_prefix='ESOL')
    plot_final(sub_esol_base, sub_esol_gtca, esol_plots, 'esol',
               'esol', title_prefix='ESOL')

    # legacy names used by paper
    for metric, short in [('RMSE','rmse'),('MAE','mae'),('Pearson_R','pearson_r'),('R2','r2')]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_lc(ax, sub_esol, metric,
                 ylabel=(DATASET_YLABEL['esol'] if metric in ('RMSE','MAE') else None),
                 title=f'ESOL — {METRIC_LABELS[metric]}')
        _save(fig, os.path.join(esol_plots, f'plot_esol_{short}.png'))

    # combined 4-panel with name "plot_esol_combined.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('ESOL — All Baselines', fontsize=13)
    for ax, metric in zip(axes.flat, METRICS):
        _draw_lc(ax, sub_esol, metric,
                 ylabel=(DATASET_YLABEL['esol'] if metric in ('RMSE','MAE') else None),
                 title=METRIC_LABELS[metric])
    _save(fig, os.path.join(esol_plots, 'plot_esol_combined.png'))

    plot_paper_rmse(sub_esol, os.path.join(PAPER_PLOTS, 'esol_rmse.png'),
                    title='ESOL Learning Curve', ylabel=DATASET_YLABEL['esol'])

    # ====================================================================
    print('\n=== Lipophilicity ===')
    lipo_plots = os.path.join(RESULTS, '03_Lipo', 'plots')
    sub_lipo      = lc_lipo[lc_lipo['train_size'] <= 1000].copy()
    sub_lipo_base = sub_lipo[sub_lipo['model'] != 'gtca_cat'].copy()
    sub_lipo_gtca = sub_lipo[sub_lipo['model'] == 'gtca_cat'].copy()

    plot_baselines(sub_lipo, lipo_plots, 'lipo', 'exp', title_prefix='Lipophilicity')
    plot_final(sub_lipo_base, sub_lipo_gtca, lipo_plots, 'lipo', 'exp',
               title_prefix='Lipophilicity')

    for metric, short in [('RMSE','rmse'),('MAE','mae'),('Pearson_R','pearson_r'),('R2','r2')]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_lc(ax, sub_lipo, metric,
                 ylabel=(DATASET_YLABEL['lipo'] if metric in ('RMSE','MAE') else None),
                 title=f'Lipophilicity — {METRIC_LABELS[metric]}')
        _save(fig, os.path.join(lipo_plots, f'plot_lipo_{short}.png'))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Lipophilicity — All Baselines', fontsize=13)
    for ax, metric in zip(axes.flat, METRICS):
        _draw_lc(ax, sub_lipo, metric,
                 ylabel=(DATASET_YLABEL['lipo'] if metric in ('RMSE','MAE') else None),
                 title=METRIC_LABELS[metric])
    _save(fig, os.path.join(lipo_plots, 'plot_lipo_combined.png'))

    plot_paper_rmse(sub_lipo, os.path.join(PAPER_PLOTS, 'lipo_rmse.png'),
                    title='Lipophilicity Learning Curve', ylabel=DATASET_YLABEL['lipo'])

    # ====================================================================
    print('\n=== BACE ===')
    bace_plots = os.path.join(RESULTS, '04_BACE', 'plots')
    sub_bace      = lc_bace[lc_bace['train_size'] <= 500].copy()
    sub_bace_base = sub_bace[sub_bace['model'] != 'gtca_cat'].copy()
    sub_bace_gtca = sub_bace[sub_bace['model'] == 'gtca_cat'].copy()

    plot_baselines(sub_bace, bace_plots, 'bace', 'pIC50', title_prefix='BACE')
    plot_final(sub_bace_base, sub_bace_gtca, bace_plots, 'bace', 'pIC50',
               title_prefix='BACE')

    for metric, short in [('RMSE','rmse'),('MAE','mae'),('Pearson_R','pearson_r'),('R2','r2')]:
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_lc(ax, sub_bace, metric,
                 ylabel=(DATASET_YLABEL['bace'] if metric in ('RMSE','MAE') else None),
                 title=f'BACE — {METRIC_LABELS[metric]}')
        _save(fig, os.path.join(bace_plots, f'plot_bace_{short}.png'))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('BACE — All Baselines', fontsize=13)
    for ax, metric in zip(axes.flat, METRICS):
        _draw_lc(ax, sub_bace, metric,
                 ylabel=(DATASET_YLABEL['bace'] if metric in ('RMSE','MAE') else None),
                 title=METRIC_LABELS[metric])
    _save(fig, os.path.join(bace_plots, 'plot_bace_combined.png'))

    plot_paper_rmse(sub_bace, os.path.join(PAPER_PLOTS, 'bace_rmse.png'),
                    title='BACE Learning Curve', ylabel=DATASET_YLABEL['bace'])

    # ====================================================================
    print('\n=== Paper summary: 4-dataset 2×2 per metric ===')
    qm9_homo = lc_qm9[lc_qm9['target'] == 'homo']
    panels_data = [
        (qm9_homo, 'QM9 HOMO'),
        (sub_esol, 'ESOL'),
        (sub_lipo, 'Lipophilicity'),
        (sub_bace, 'BACE'),
    ]
    DATASET_YLABEL_MAE = {
        'QM9 HOMO':      'MAE (eV)',
        'ESOL':          'MAE (log mol/L)',
        'Lipophilicity': 'MAE (log D)',
        'BACE':          'MAE (pIC50)',
    }
    DATASET_YLABEL_RMSE = {
        'QM9 HOMO':      'RMSE (eV)',
        'ESOL':          'RMSE (log mol/L)',
        'Lipophilicity': 'RMSE (log D)',
        'BACE':          'RMSE (pIC50)',
    }
    metric_configs = [
        ('RMSE',      'all_datasets_rmse_2x2.png',      DATASET_YLABEL_RMSE),
        ('MAE',       'all_datasets_mae_2x2.png',       DATASET_YLABEL_MAE),
        ('Pearson_R', 'all_datasets_pearsonr_2x2.png',  None),
        ('R2',        'all_datasets_r2_2x2.png',        None),
    ]
    for metric, fname, ylabel_map in metric_configs:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Learning Curves — All Datasets ({METRIC_LABELS[metric]})', fontsize=14)
        for ax, (df, ttl) in zip(axes.flat, panels_data):
            yl = ylabel_map[ttl] if ylabel_map else None
            _draw_lc(ax, df, metric, ylabel=yl, title=ttl)
        _save(fig, os.path.join(PAPER_PLOTS, fname), dpi=200)

    # ====================================================================
    print('\n=== Paper Figures (Fig 3–9, 11–13) ===')

    YLABEL_QM9_RMSE = DATASET_YLABEL['qm9']        # 'RMSE (eV)'
    YLABEL_QM9_MAE  = 'MAE (eV)'

    def _paper_2x2(df, title, fname, ylabel_rmse, ylabel_mae, use_ci=False):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ylabels = {'RMSE': ylabel_rmse, 'MAE': ylabel_mae,
                   'Pearson_R': None, 'R2': None}
        for ax, metric in zip(axes.flat, METRICS):
            _draw_lc(ax, df, metric,
                     ylabel=ylabels[metric],
                     title=METRIC_LABELS[metric],
                     use_ci=use_ci,
                     show_legend=(metric == 'RMSE'))
        _save(fig, os.path.join(PAPER_PLOTS, fname))

    # Fig 3–5: Depth ablation (HOMO / LUMO / GAP)
    for figno, target, tname in [(3,'homo','HOMO'), (4,'lumo','LUMO'), (5,'gap','HOMO-LUMO Gap')]:
        sub = abl_dep[abl_dep['target'] == target].copy()
        _paper_2x2(sub,
                   title=f'Figure {figno}. QM9 Depth Ablation — {tname}',
                   fname=f'fig{figno:02d}_depth_{target}.png',
                   ylabel_rmse=YLABEL_QM9_RMSE,
                   ylabel_mae=YLABEL_QM9_MAE,
                   use_ci=False)

    # Fig 6–8: Fusion comparison (HOMO / LUMO / GAP)
    for figno, target, tname in [(6,'homo','HOMO'), (7,'lumo','LUMO'), (8,'gap','HOMO-LUMO Gap')]:
        sub = abl_fus[abl_fus['target'] == target].copy()
        _paper_2x2(sub,
                   title=f'Figure {figno}. QM9 Fusion: Cat vs CA — {tname}',
                   fname=f'fig{figno:02d}_fusion_{target}.png',
                   ylabel_rmse=YLABEL_QM9_RMSE,
                   ylabel_mae=YLABEL_QM9_MAE,
                   use_ci=False)

    # Fig 9, 11, 12: Full model LC — QM9 (HOMO / LUMO / GAP)
    for figno, target, tname in [(9,'homo','HOMO'), (11,'lumo','LUMO'), (12,'gap','HOMO-LUMO Gap')]:
        sub = lc_qm9[lc_qm9['target'] == target].copy()
        _paper_2x2(sub,
                   title=f'Figure {figno}. QM9 ({tname}) — All Models (GTCA depth=6)',
                   fname=f'fig{figno:02d}_qm9_{target}.png',
                   ylabel_rmse=YLABEL_QM9_RMSE,
                   ylabel_mae=YLABEL_QM9_MAE,
                   use_ci=False)

    # Fig 13: ESOL full model LC
    _paper_2x2(sub_esol,
               title='Figure 13. ESOL — All Models (N=50–375)',
               fname='fig13_esol.png',
               ylabel_rmse=DATASET_YLABEL['esol'],
               ylabel_mae='MAE (log mol/L)',
               use_ci=False)

    # Fig 15: Lipophilicity full model LC
    _paper_2x2(sub_lipo,
               title='Figure 15. Lipophilicity — All Models (N=50–1,000)',
               fname='fig15_lipo.png',
               ylabel_rmse=DATASET_YLABEL['lipo'],
               ylabel_mae='MAE (log D)',
               use_ci=False)

    # Fig 16: BACE full model LC
    _paper_2x2(sub_bace,
               title='Figure 16. BACE — All Models (N=50–500)',
               fname='fig16_bace.png',
               ylabel_rmse=DATASET_YLABEL['bace'],
               ylabel_mae='MAE (pIC50)',
               use_ci=False)

    # ====================================================================
    print('\n=== Paper Figures (Fig 18–23): XAI saliency pairs ===')

    XAI_DIR = os.path.join(RESULTS, 'scatter_plots')

    xai_figures = [
        (18, 'esol_mol0', 'ESOL mol0 — CCOP(=S)(OCC)SCSCC\nTrue=−4.110 log(mol/L)  GTCA err=0.27  GCN err=2.47'),
        (19, 'esol_mol1', 'ESOL mol1 — CCOP(=S)(OCC)SCSC(C)(C)C\nTrue=−4.755 log(mol/L)  GTCA err=0.14  GCN err=1.12'),
        (20, 'esol_mol2', 'ESOL mol2 — CC(C)CCC(C)(C)C\nTrue=−5.050 log(mol/L)  GTCA err=−0.11  GCN err=−1.75'),
        (21, 'bace_mol0', 'BACE mol0 — COCC#Cc1cc(...)ccc1F\nTrue=7.921 pIC50  GPR err=0.274  AFP err=0.616'),
        (22, 'bace_mol1', 'BACE mol1 — CC#Cc1cccc(...)c1\nTrue=7.699 pIC50  GPR err=−0.067  AFP err=0.620'),
        (23, 'bace_mol2', 'BACE mol2 — CC(C)C#Cc1cccc(...)c1\nTrue=7.699 pIC50  GPR err=0.119  AFP err=0.512'),
    ]

    for figno, mol_label, suptitle in xai_figures:
        gtca_path = os.path.join(XAI_DIR, f'xai_gtca_ig_{mol_label}.png')
        afp_path  = os.path.join(XAI_DIR, f'xai_attentivefp_{mol_label}.png')

        if not os.path.exists(gtca_path) or not os.path.exists(afp_path):
            print(f"  [skip] Fig {figno}: missing XAI images")
            continue

        from PIL import Image as _PIL
        gtca_img = _PIL.open(gtca_path)
        afp_img  = _PIL.open(afp_path)

        w = gtca_img.width + afp_img.width
        h = max(gtca_img.height, afp_img.height)
        combined = _PIL.new('RGB', (w, h), (255, 255, 255))
        combined.paste(gtca_img, (0, 0))
        combined.paste(afp_img,  (gtca_img.width, 0))

        fig, ax = plt.subplots(figsize=(w/300, h/300 + 0.3))
        ax.imshow(combined)
        ax.axis('off')
        ax.text(0.25, -0.03, '(a)', transform=ax.transAxes,
                ha='center', fontsize=10, fontweight='bold')
        ax.text(0.75, -0.03, '(b)', transform=ax.transAxes,
                ha='center', fontsize=10, fontweight='bold')
        _save(fig, os.path.join(PAPER_PLOTS, f'fig{figno:02d}_{mol_label}.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
