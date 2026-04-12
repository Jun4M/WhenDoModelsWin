"""
figure1_overview.py
===================
Generate Figure 1: Conceptual Overview (2x2 panel A/B/C/D)
Panel A: Molecular Data Context & Scarcity
Panel B: Scaffold-Based Learning Curve Workflow
Panel C: Multimodal Representations & Model Zoo
Panel D: Regime-Dependent Optimality
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

# ── RDKit optional ──────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    import io
    from PIL import Image
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ── Font setup ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
})

PAPER_PLOTS = 'results/paper_plots'
os.makedirs(PAPER_PLOTS, exist_ok=True)

# ── Colors ───────────────────────────────────────────────────────────────────
C_BLUE   = '#4B72B8'
C_GREEN  = '#2EAB6E'
C_RED    = '#E74C3C'
C_ORANGE = '#E8942A'
C_PURPLE = '#9467BD'
C_GRAY   = '#888888'
C_LIGHT  = '#F0F4F8'
C_DARK   = '#2C3E50'

# ============================================================
# Helper: draw a rounded box with text
# ============================================================
def rounded_box(ax, x, y, w, h, text, facecolor='#EAF2FB', edgecolor='#2980B9',
                fontsize=8, bold=False, text_color='black', linestyle='-', lw=1.2,
                va='center', wrap=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle='round,pad=0.02', linewidth=lw,
                         edgecolor=edgecolor, facecolor=facecolor, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va=va, fontsize=fontsize,
            fontweight=weight, color=text_color, zorder=4, wrap=wrap,
            multialignment='center')

def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.4, arrowsize=8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=arrowsize),
                zorder=4)


# ============================================================
# PANEL A: Molecular Data Context & Scarcity
# ============================================================
def draw_panel_A(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # ── Molecule drawing (left side) ──────────────────────────
    mol_drawn = False
    if HAS_RDKIT:
        try:
            smi = 'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34'  # pyrene
            mol = Chem.MolFromSmiles(smi)
            if mol:
                drawer = rdMolDraw2D.MolDraw2DCairo(220, 180)
                drawer.drawOptions().addStereoAnnotation = False
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                bio = io.BytesIO(drawer.GetDrawingText())
                img = Image.open(bio)
                img_ax = ax.inset_axes([0.01, 0.58, 0.32, 0.38])
                img_ax.imshow(img)
                img_ax.axis('off')
                mol_drawn = True
        except Exception:
            pass

    if not mol_drawn:
        # Placeholder hexagon molecule
        mol_ax = ax.inset_axes([0.01, 0.58, 0.32, 0.38])
        mol_ax.set_xlim(-1.5, 1.5)
        mol_ax.set_ylim(-1.5, 1.5)
        mol_ax.axis('off')
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        xs = np.cos(angles)
        ys = np.sin(angles)
        mol_ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 'k-', lw=2)
        for i, (x, y) in enumerate(zip(xs, ys)):
            mol_ax.text(x*1.35, y*1.35, ['C','C','C','N','C','O'][i],
                        ha='center', va='center', fontsize=9, fontweight='bold')
        mol_ax.text(0, 0, 'molecule', ha='center', va='center',
                    fontsize=7, color='gray', style='italic')

    # ── Dataset category boxes (right side) ──────────────────
    categories = [
        ('Quantum\nProperties', '#D5E8D4', '#82B366', 'QM9\n(HOMO/LUMO/Gap)'),
        ('Experimental\nBioactivity', '#DAE8FC', '#6C8EBF', 'BACE\n(pIC50)'),
        ('Physical\nChemistry', '#FFF2CC', '#D6B656', 'ESOL, Lipo\n(logS, logP)'),
    ]
    for i, (cat, fc, ec, dsname) in enumerate(categories):
        y_pos = 9.2 - i * 2.8
        rounded_box(ax, 5.5, y_pos, 2.6, 0.9, cat,
                    facecolor=fc, edgecolor=ec, fontsize=8, bold=True)
        ax.text(8.2, y_pos, dsname, ha='center', va='center',
                fontsize=7.5, color='#444444', style='italic')
        ax.annotate('', xy=(6.9, y_pos), xytext=(7.5, y_pos),
                    arrowprops=dict(arrowstyle='<-', color=ec, lw=1.2, mutation_scale=7))

    # ── Bar chart: dataset sizes ──────────────────────────────
    bar_ax = ax.inset_axes([0.01, 0.02, 0.88, 0.42])
    datasets = ['ESOL', 'Lipo', 'BACE', 'QM9']
    sizes    = [1128, 4200, 1513, 134000]
    colors   = ['#FFF2CC', '#FCE5CD', '#DAE8FC', '#D5E8D4']
    edges    = ['#D6B656', '#E8942A', '#6C8EBF', '#82B366']
    log_sizes = np.log10(sizes)

    y_pos_bars = np.arange(len(datasets))
    bars = bar_ax.barh(y_pos_bars, log_sizes, color=colors, edgecolor=edges,
                       linewidth=1.3, height=0.55)
    for bar, s in zip(bars, sizes):
        w = bar.get_width()
        label = f'{s:,}' if s < 10000 else f'{s//1000}k'
        bar_ax.text(w + 0.05, bar.get_y() + bar.get_height()/2,
                    label, va='center', fontsize=8, color='#333333')

    bar_ax.set_xlabel('Dataset size (log10 scale)', fontsize=8)
    bar_ax.set_xlim(0, 6.3)
    bar_ax.set_xticks([2, 3, 4, 5])
    bar_ax.set_xticklabels(['100', '1k', '10k', '100k'], fontsize=7.5)
    bar_ax.set_yticks(y_pos_bars)
    bar_ax.set_yticklabels(datasets, fontsize=8)
    bar_ax.spines[['top', 'right']].set_visible(False)
    # Highlight "data scarcity" region
    bar_ax.axvspan(0, 3.3, alpha=0.08, color='red', zorder=0)
    bar_ax.text(1.65, 3.65, 'data-scarce\nregime', ha='center', va='center',
                fontsize=7, color='#C0392B', style='italic')

    ax.text(5.0, 9.85, 'Molecular property prediction spans\norders of magnitude in data availability.',
            ha='center', va='top', fontsize=8.5, color=C_DARK,
            multialignment='center')


# ============================================================
# PANEL B: Scaffold-Based Learning Curve Workflow
# ============================================================
def draw_panel_B(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Step boxes
    steps = [
        (1.2, 8.5,  'Full\nDataset',        '#D5E8D4', '#82B366'),
        (3.8, 8.5,  'Scaffold\nSplit',       '#DAE8FC', '#6C8EBF'),
        (6.5, 8.5,  'Train / Val / Test',    '#FFF2CC', '#D6B656'),
        (1.2, 5.8,  'Subsample\nN from train','#FCE5CD','#E8942A'),
        (4.2, 5.8,  'Train\nModel',          '#E1D5E7', '#9673A6'),
        (7.2, 5.8,  'Evaluate\non Test',     '#DAE8FC', '#6C8EBF'),
    ]
    for (x, y, label, fc, ec) in steps:
        rounded_box(ax, x, y, 1.9, 0.85, label,
                    facecolor=fc, edgecolor=ec, fontsize=8.5, bold=True)

    # Arrows top row
    arrow(ax, 2.15, 8.5, 2.85, 8.5)
    arrow(ax, 4.75, 8.5, 5.55, 8.5)

    # Down arrow
    arrow(ax, 6.5, 8.07, 6.5, 7.2)
    ax.text(6.85, 7.65, 'test set\nfixed', ha='left', va='center',
            fontsize=7.5, color='#555555', style='italic')

    # Arrows middle row
    arrow(ax, 2.15, 5.8, 3.25, 5.8)
    arrow(ax, 5.15, 5.8, 6.25, 5.8)

    # Loop arrow (repeat for N and seed)
    ax.annotate('', xy=(1.2, 6.22), xytext=(1.2, 7.4),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2,
                                connectionstyle='arc3,rad=-0.5', mutation_scale=8))
    ax.text(0.25, 6.85, 'N in {50,100,\n200,...}\n10 seeds', ha='center', va='center',
            fontsize=7.5, color='#555555', style='italic')

    # Learning curve schematic
    lc_ax = ax.inset_axes([0.03, 0.03, 0.93, 0.43])
    N = np.array([50, 100, 200, 300, 500, 1000])
    np.random.seed(42)
    rmse_gcn  = 1.8 / np.sqrt(N/50) + np.random.normal(0, 0.03, len(N))
    rmse_gtca = 1.5 / np.sqrt(N/50) + np.random.normal(0, 0.03, len(N))
    rmse_afp  = 1.6 / np.sqrt(N/50) + np.random.normal(0, 0.03, len(N))

    lc_ax.plot(N, rmse_gcn,  'o-', color=C_BLUE,   lw=1.8, ms=5, label='GCN')
    lc_ax.plot(N, rmse_gtca, 's-', color=C_GREEN,  lw=1.8, ms=5, label='GTCA-Cat')
    lc_ax.plot(N, rmse_afp,  '^-', color=C_ORANGE, lw=1.8, ms=5, label='AttentiveFP')
    lc_ax.fill_between(N, rmse_gtca*0.93, rmse_gtca*1.07, alpha=0.15, color=C_GREEN)

    lc_ax.set_xlabel('Training set size N', fontsize=8.5)
    lc_ax.set_ylabel('RMSE', fontsize=8.5)
    lc_ax.set_xscale('log')
    lc_ax.set_xticks(N)
    lc_ax.set_xticklabels([str(n) for n in N], fontsize=7.5)
    lc_ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8)
    lc_ax.spines[['top', 'right']].set_visible(False)
    lc_ax.set_title('Resulting learning curve (schematic)', fontsize=8.5, pad=3)

    ax.text(0.5, 0.99, 'Scaffold split ensures structural generalization;\nN is systematically varied across seeds.',
            ha='center', va='top', fontsize=8.5, color=C_DARK,
            transform=ax.transAxes, multialignment='center')


# ============================================================
# PANEL C: Multimodal Representations & Model Zoo
# ============================================================
def draw_panel_C(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Central molecule
    circ = Circle((5.0, 6.9), 0.55, color='#F8F9FA', ec='#555555', lw=1.5, zorder=3)
    ax.add_patch(circ)
    ax.text(5.0, 6.9, 'MOL', ha='center', va='center', fontsize=9,
            color='#2C3E50', zorder=4)
    ax.text(5.0, 6.1, 'Molecule', ha='center', va='center',
            fontsize=8, color='#444444', style='italic')

    # Representation streams
    repr_items = [
        (1.6, 8.7, 'SMILES\n(ChemBERTa)', '#EAF2FB', '#2980B9'),
        (5.0, 9.5, 'Molecular\nGraph (PyG)', '#E8F8F5', '#27AE60'),
        (8.4, 8.7, 'ECFP\nFingerprint', '#FEF9E7', '#F39C12'),
    ]
    for (x, y, label, fc, ec) in repr_items:
        rounded_box(ax, x, y, 2.3, 0.75, label,
                    facecolor=fc, edgecolor=ec, fontsize=8, bold=True)
        # Arrow from box to center
        dx = 5.0 - x; dy = 6.9 - y
        length = np.sqrt(dx**2 + dy**2)
        scale = 0.55 / length
        ax.annotate('', xy=(5.0 - dx*scale, 6.9 - dy*scale),
                    xytext=(x + dx*0.55, y + dy*0.45),
                    arrowprops=dict(arrowstyle='->', color=ec, lw=1.3, mutation_scale=8))

    # Model zoo boxes
    models = [
        # (x,    y,   label,               color,    ec)
        (1.2,  5.0,  'GCN',               '#DAE8FC', '#6C8EBF'),
        (1.2,  3.7,  'AttentiveFP',        '#DAE8FC', '#6C8EBF'),
        (1.2,  2.4,  'PaiNN',             '#DAE8FC', '#6C8EBF'),
        (1.2,  1.1,  'GPS',               '#DAE8FC', '#6C8EBF'),
        (5.0,  5.0,  'ChemBERTa\n(Transf.)','#E1D5E7','#9673A6'),
        (5.0,  3.5,  'GTCA-Cat\n(Fusion)', '#D5E8D4', '#82B366'),
        (5.0,  2.0,  'GTCA-CA\n(Fusion)',  '#FFDCE1', '#E74C3C'),
        (8.5,  5.0,  'Random\nForest',     '#FFF2CC', '#D6B656'),
        (8.5,  3.7,  'XGBoost',           '#FFF2CC', '#D6B656'),
        (8.5,  2.4,  'GPR',               '#FFF2CC', '#D6B656'),
    ]
    for (x, y, label, fc, ec) in models:
        rounded_box(ax, x, y, 1.85, 0.65, label,
                    facecolor=fc, edgecolor=ec, fontsize=7.5, bold=False)

    # Category labels
    ax.text(1.2, 5.75, 'Graph Neural\nNetworks', ha='center', va='center',
            fontsize=7.5, color='#2980B9', fontweight='bold')
    ax.text(5.0, 5.75, 'Seq / Fusion\nModels', ha='center', va='center',
            fontsize=7.5, color='#9673A6', fontweight='bold')
    ax.text(8.5, 5.75, 'Classical\nML', ha='center', va='center',
            fontsize=7.5, color='#D6B656', fontweight='bold')

    # Arrows from molecule to category groups
    for (x, ec) in [(1.2, '#6C8EBF'), (5.0, '#82B366'), (8.5, '#D6B656')]:
        dx = x - 5.0; dy = 6.35 - 6.9
        ax.annotate('', xy=(x, 6.15), xytext=(5.0 + dx*0.15, 6.9 + dy*0.15),
                    arrowprops=dict(arrowstyle='->', color=ec, lw=1.2, mutation_scale=7))

    ax.text(0.5, 0.99, 'Nine model families span GNNs, language models,\nfusion architectures, and classical ML.',
            ha='center', va='top', fontsize=8.5, color=C_DARK,
            transform=ax.transAxes, multialignment='center')


# ============================================================
# PANEL D: Regime-Dependent Optimality
# ============================================================
def draw_panel_D(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    lc_ax = ax.inset_axes([0.07, 0.1, 0.90, 0.72])

    N = np.linspace(20, 1200, 200)
    np.random.seed(7)

    # Classical ML: good at small N, plateaus
    rmse_classical = 0.9 + 2.5 * np.exp(-N/300)
    # GNN: needs more data, eventually wins
    rmse_gnn       = 0.7 + 3.5 * np.exp(-N/600)
    # Fusion GTCA: best of both worlds
    rmse_fusion    = 0.6 + 2.2 * np.exp(-N/450)
    # ChemBERTa: good with mid data
    rmse_bert      = 0.85 + 2.8 * np.exp(-N/500)

    lc_ax.plot(N, rmse_classical, '--', color=C_ORANGE, lw=2,   label='Classical ML (GPR/RF)')
    lc_ax.plot(N, rmse_gnn,       '-',  color=C_BLUE,   lw=2,   label='GCN / AttentiveFP')
    lc_ax.plot(N, rmse_bert,      ':',  color=C_PURPLE, lw=1.8, label='ChemBERTa')
    lc_ax.plot(N, rmse_fusion,    '-',  color=C_GREEN,  lw=2.5, label='GTCA-Cat (Fusion)', zorder=5)

    # Crossover classical vs GNN
    cross1_idx = np.argmin(np.abs(rmse_classical - rmse_gnn))
    cx1 = N[cross1_idx]
    cy1 = rmse_classical[cross1_idx]
    lc_ax.axvline(cx1, color='#888888', lw=1, linestyle=':', alpha=0.7)
    lc_ax.scatter([cx1], [cy1], color='#333333', s=55, zorder=6)
    lc_ax.text(cx1 + 25, cy1 + 0.08, f'crossover\n(N≈{int(cx1)})', fontsize=7.5,
               color='#333333', va='bottom')

    # Shade regimes
    lc_ax.axvspan(20,  cx1,   alpha=0.06, color=C_ORANGE)
    lc_ax.axvspan(cx1, 1200,  alpha=0.06, color=C_BLUE)
    lc_ax.text(cx1*0.4, 3.25, 'Data-Scarce\nRegime', ha='center', va='center',
               fontsize=8, color=C_ORANGE, fontweight='bold', alpha=0.85)
    lc_ax.text((cx1 + 1200)*0.5, 3.25, 'Data-Rich\nRegime', ha='center', va='center',
               fontsize=8, color=C_BLUE, fontweight='bold', alpha=0.85)

    lc_ax.set_xlabel('Training set size N', fontsize=9)
    lc_ax.set_ylabel('RMSE (normalized)', fontsize=9)
    lc_ax.set_xlim(20, 1200)
    lc_ax.set_ylim(0.5, 3.5)
    lc_ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    lc_ax.spines[['top', 'right']].set_visible(False)
    lc_ax.tick_params(labelsize=8)

    ax.text(0.5, 0.99, 'No single model dominates across all regimes;\nfusion models show consistent advantage.',
            ha='center', va='top', fontsize=8.5, color=C_DARK,
            transform=ax.transAxes, multialignment='center')


# ============================================================
# ASSEMBLE FIGURE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

panels = [
    (axes[0, 0], 'A', draw_panel_A),
    (axes[0, 1], 'B', draw_panel_B),
    (axes[1, 0], 'C', draw_panel_C),
    (axes[1, 1], 'D', draw_panel_D),
]

for ax, label, draw_fn in panels:
    ax.set_facecolor('white')
    draw_fn(ax)
    # Bold panel label
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left', color=C_DARK)

plt.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.04, hspace=0.25, wspace=0.15)

# Save
out_base = os.path.join(PAPER_PLOTS, 'figure1_overview')
fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_base + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {out_base}.png")
print(f"Saved: {out_base}.pdf")
