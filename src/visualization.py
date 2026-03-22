"""
visualization.py
Learning curve plots with error bars for all models.
  - plot_baselines_lc: baseline models (GCN, RF, XGB, ...) learning curves
  - plot_gtca_depth_lc: GTCA depth variants comparison
  - plot_combined_final: baselines + selected GTCA depth
  - plot_group_analysis: chemical group bar charts
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color / label maps
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    'transformer':  '#E07B39',
    'gcn':          '#3B82C4',
    'gtca':         '#2EAB6E',
    'rf':           '#9B59B6',
    'xgb':          '#E74C3C',
    'gpr':          '#F39C12',
    'attentivefp':  '#1ABC9C',
    'painn':        '#2C3E50',
    'gps':          '#7F8C8D',
    'gtca_depth_2': '#E74C3C',
    'gtca_depth_4': '#3B82C4',
    'gtca_depth_6': '#2EAB6E',
}

MODEL_LABELS = {
    'transformer':  'Transformer (ChemBERTa)',
    'gcn':          'GCN',
    'gtca':         'GTCA',
    'rf':           'Random Forest',
    'xgb':          'XGBoost',
    'gpr':          'GPR',
    'attentivefp':  'AttentiveFP',
    'painn':        'PaiNN',
    'gps':          'GPS',
    'gtca_depth_2': 'GTCA (depth=2)',
    'gtca_depth_4': 'GTCA (depth=4)',
    'gtca_depth_6': 'GTCA (depth=6)',
}

METRICS = ['RMSE', 'MAE', 'Pearson_R', 'R2']
METRIC_LABELS = {'RMSE': 'RMSE', 'MAE': 'MAE', 'Pearson_R': 'Pearson R', 'R2': 'R²'}


# ---------------------------------------------------------------------------
# Internal: draw one LC panel with error bars
# ---------------------------------------------------------------------------

def _draw_lc_panel(ax, df: pd.DataFrame, metric: str, title: str = ''):
    """
    df must have columns: train_size, model, {metric}_mean, {metric}_std
    """
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"
    if mean_col not in df.columns:
        return

    for model in df['model'].unique():
        sub = df[df['model'] == model].sort_values('train_size')
        color = MODEL_COLORS.get(model, 'gray')
        label = MODEL_LABELS.get(model, model)
        x = sub['train_size'].values
        y = sub[mean_col].values
        yerr = sub[std_col].fillna(0).values if std_col in sub.columns else None

        ax.plot(x, y, color=color, label=label, linewidth=1.8, marker='o', markersize=4)
        if yerr is not None and not np.all(yerr == 0):
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

    ax.set_xlabel('Training set size', fontsize=10)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    # Vertical boundary lines
    for boundary in [500, 1000]:
        if ax.get_xlim()[1] > boundary:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)


# ---------------------------------------------------------------------------
# plot_baselines_lc
# ---------------------------------------------------------------------------

def plot_baselines_lc(
    summary_df: pd.DataFrame,
    plots_dir: str,
    target: str,
    metrics: list = None,
):
    """
    Per-metric learning curve plots for baseline models.
    Also generates a combined 2×2 figure.
    """
    if metrics is None:
        metrics = METRICS

    os.makedirs(plots_dir, exist_ok=True)
    df = summary_df[summary_df['target'] == target] if 'target' in summary_df.columns else summary_df

    # Individual plots
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_lc_panel(ax, df, metric, title=f'{target} — {METRIC_LABELS.get(metric, metric)}')
        out = os.path.join(plots_dir, f'plot_baselines_{target}_{metric}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  [viz] → {out}")

    # Combined 2×2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Baseline Models — {target}', fontsize=13)
    for ax, metric in zip(axes.flat, metrics):
        _draw_lc_panel(ax, df, metric, title=METRIC_LABELS.get(metric, metric))
    plt.tight_layout()
    out = os.path.join(plots_dir, f'plot_baselines_lc_{target}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [viz] → {out}")


# ---------------------------------------------------------------------------
# plot_gtca_depth_lc
# ---------------------------------------------------------------------------

def plot_gtca_depth_lc(
    summary_df: pd.DataFrame,
    plots_dir: str,
    target: str,
    metrics: list = None,
):
    """Learning curves for GTCA depth variants."""
    if metrics is None:
        metrics = METRICS

    os.makedirs(plots_dir, exist_ok=True)
    df = summary_df[summary_df['target'] == target] if 'target' in summary_df.columns else summary_df

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'GTCA Depth Study — {target}', fontsize=13)
    for ax, metric in zip(axes.flat, metrics):
        _draw_lc_panel(ax, df, metric, title=METRIC_LABELS.get(metric, metric))
    plt.tight_layout()
    out = os.path.join(plots_dir, f'plot_gtca_depth_lc_{target}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [viz] → {out}")


# ---------------------------------------------------------------------------
# plot_combined_final (called from run_final_comparison.py)
# ---------------------------------------------------------------------------

def plot_combined_final(
    baselines_df: pd.DataFrame,
    best_gtca_depth: int,
    gtca_depth_df: pd.DataFrame,
    plots_dir: str,
    target: str,
    metrics: list = None,
):
    """
    Merge selected GTCA depth with baselines and plot all models together.
    """
    if metrics is None:
        metrics = METRICS

    os.makedirs(plots_dir, exist_ok=True)

    # Filter GTCA to selected depth and rename
    model_key = f'gtca_depth_{best_gtca_depth}'
    gtca_sel = gtca_depth_df[gtca_depth_df['model'] == model_key].copy()
    gtca_sel['model'] = model_key

    bdf = baselines_df[baselines_df['target'] == target] if 'target' in baselines_df.columns else baselines_df
    gdf = gtca_sel[gtca_sel['target'] == target] if 'target' in gtca_sel.columns else gtca_sel

    combined = pd.concat([bdf, gdf], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'All Models — {target} (GTCA depth={best_gtca_depth})', fontsize=13)
    for ax, metric in zip(axes.flat, metrics):
        _draw_lc_panel(ax, combined, metric, title=METRIC_LABELS.get(metric, metric))
    plt.tight_layout()
    out = os.path.join(plots_dir, f'plot_final_{target}_depth{best_gtca_depth}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [viz] → {out}")


# ---------------------------------------------------------------------------
# plot_group_analysis
# ---------------------------------------------------------------------------

def plot_group_analysis(
    group_df: pd.DataFrame,
    plots_dir: str,
    target: str,
    metric: str = 'Mean_MAE',
):
    """Bar charts: mean MAE per chemical group for all models."""
    os.makedirs(plots_dir, exist_ok=True)
    df = group_df[group_df['target'] == target]

    for group_type in df['group_type'].unique():
        gdf = df[df['group_type'] == group_type]
        categories = sorted(gdf['category'].unique())
        models = sorted(gdf['model'].unique())

        fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.5), 5))
        x = np.arange(len(categories))
        width = 0.8 / max(len(models), 1)

        for i, model in enumerate(models):
            mdf = gdf[gdf['model'] == model]
            vals = [mdf[mdf['category'] == c][metric].mean() if c in mdf['category'].values else 0
                    for c in categories]
            color = MODEL_COLORS.get(model, 'gray')
            label = MODEL_LABELS.get(model, model)
            ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.8)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f'{target} — {group_type}')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        out = os.path.join(plots_dir, f'group_{target}_{group_type}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  [viz] → {out}")


# ---------------------------------------------------------------------------
# Legacy: plot_learning_curves (kept for backward compat)
# ---------------------------------------------------------------------------

def plot_learning_curves(df: pd.DataFrame, results_dir: str, targets: list = None):
    """Legacy wrapper. df has columns: train_size, model, target, RMSE, MAE, Pearson_R."""
    if targets is None:
        targets = df['target'].unique().tolist() if 'target' in df.columns else []
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Build mock summary_df from raw (no std)
    for target in targets:
        sub = df[df['target'] == target].copy() if 'target' in df.columns else df.copy()
        # Create mean columns (no repetition → mean = value, std = 0)
        mock = sub.copy()
        for m in ['RMSE', 'MAE', 'Pearson_R']:
            if m in mock.columns:
                mock[f'{m}_mean'] = mock[m]
                mock[f'{m}_std']  = 0.0
        plot_baselines_lc(mock, plots_dir, target, metrics=['RMSE', 'MAE', 'Pearson_R'])
