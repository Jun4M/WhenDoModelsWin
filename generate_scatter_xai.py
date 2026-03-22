"""
generate_scatter_xai.py
=======================
Part 1: Scatter plots (ESOL n=50, BACE n=500, QM9-homo n=3000)
Part 2: Representative molecule selection
Part 3: XAI visualizations (GTCA grad saliency, BERT attention, AttentiveFP attention)
"""

import sys
import os
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.data_loader import load_dataset_splits, load_raw_data
from src.train import train_gcn, train_gtca, train_attentivefp, train_sklearn

OUTDIR = 'results/scatter_plots'
os.makedirs(OUTDIR, exist_ok=True)
DEVICE = 'cpu'
SEED = 0

# ============================================================
# Helper: denormalize
# ============================================================
def denorm(arr, stats):
    mean, std = stats
    return arr * std + mean


# ============================================================
# PART 1 — SCATTER PLOTS
# ============================================================

print("\n" + "="*60)
print("PART 1: SCATTER PLOTS")
print("="*60)

# ---- 1A. ESOL n=50 -----------------------------------------
print("\n[1A] Loading ESOL n=50 ...")
esol_data = load_dataset_splits(
    dataset='esol',
    data_dir='./data',
    train_size=50,
    val_size=50,
    test_size=200,
    seed=SEED,
    featurize_ecfp=True,
)
tr_esol = esol_data['train']
va_esol = esol_data['val']
te_esol = esol_data['test']
stats_esol = esol_data['stats']

print("  Training GCN on ESOL n=50 ...")
gcn_esol = train_gcn(
    tr_esol['X_graph'], va_esol['X_graph'], te_esol['X_graph'],
    target_name='esol', device=DEVICE, seed=SEED,
    node_feat_dim=tr_esol['X_graph'][0].x.shape[1],
)

print("  Training GTCA-Cat (depth=6) on ESOL n=50 ...")
gtca_esol = train_gtca(
    tr_esol['X_graph'], va_esol['X_graph'], te_esol['X_graph'],
    tr_esol['ids'], va_esol['ids'], te_esol['ids'],
    target_name='esol', device=DEVICE, seed=SEED, bert_depth=6,
    node_feat_dim=tr_esol['X_graph'][0].x.shape[1],
)

print("  Training AttentiveFP on ESOL n=50 ...")
afp_esol = train_attentivefp(
    tr_esol['X_graph'], va_esol['X_graph'], te_esol['X_graph'],
    target_name='esol', device=DEVICE, seed=SEED,
    node_feat_dim=tr_esol['X_graph'][0].x.shape[1],
    edge_dim=tr_esol['X_graph'][0].edge_attr.shape[1] if tr_esol['X_graph'][0].edge_attr is not None else 11,
)

# ---- 1B. BACE n=500 ----------------------------------------
print("\n[1B] Loading BACE n=500 ...")
bace_data = load_dataset_splits(
    dataset='bace',
    data_dir='./data',
    train_size=500,
    val_size=100,
    test_size=300,
    seed=SEED,
    featurize_ecfp=True,
)
tr_bace = bace_data['train']
va_bace = bace_data['val']
te_bace = bace_data['test']
stats_bace = bace_data['stats']

print("  Training GPR on BACE n=500 ...")
gpr_bace = train_sklearn(
    tr_bace['X_ecfp'], tr_bace['y'],
    va_bace['X_ecfp'], va_bace['y'],
    te_bace['X_ecfp'], te_bace['y'],
    model_type='gpr', seed=SEED,
)

print("  Training AttentiveFP on BACE n=500 ...")
afp_bace = train_attentivefp(
    tr_bace['X_graph'], va_bace['X_graph'], te_bace['X_graph'],
    target_name='bace', device=DEVICE, seed=SEED,
    node_feat_dim=tr_bace['X_graph'][0].x.shape[1],
    edge_dim=tr_bace['X_graph'][0].edge_attr.shape[1] if tr_bace['X_graph'][0].edge_attr is not None else 11,
)

print("  Training GTCA-Cat (depth=6) on BACE n=500 ...")
gtca_bace = train_gtca(
    tr_bace['X_graph'], va_bace['X_graph'], te_bace['X_graph'],
    tr_bace['ids'], va_bace['ids'], te_bace['ids'],
    target_name='bace', device=DEVICE, seed=SEED, bert_depth=6,
    node_feat_dim=tr_bace['X_graph'][0].x.shape[1],
)

# ---- 1C. QM9 homo n=3000 -----------------------------------
print("\n[1C] Loading QM9 homo n=3000 ...")
raw_qm9 = load_raw_data('qm9', './data', 'homo')
qm9_data = load_dataset_splits(
    dataset='qm9',
    data_dir='./data',
    train_size=3000,
    val_size=1000,
    test_size=5000,
    seed=SEED,
    target='homo',
    preloaded_raw=raw_qm9,
)
tr_qm9 = qm9_data['train']
va_qm9 = qm9_data['val']
te_qm9 = qm9_data['test']
stats_qm9 = qm9_data['stats']

print("  Training GCN on QM9 homo n=3000 ...")
gcn_qm9 = train_gcn(
    tr_qm9['X_graph'], va_qm9['X_graph'], te_qm9['X_graph'],
    target_name='qm9_homo', device=DEVICE, seed=SEED,
    node_feat_dim=tr_qm9['X_graph'][0].x.shape[1],
)

print("  Training AttentiveFP on QM9 homo n=3000 ...")
afp_qm9 = train_attentivefp(
    tr_qm9['X_graph'], va_qm9['X_graph'], te_qm9['X_graph'],
    target_name='qm9_homo', device=DEVICE, seed=SEED,
    node_feat_dim=tr_qm9['X_graph'][0].x.shape[1],
    edge_dim=tr_qm9['X_graph'][0].edge_attr.shape[1] if tr_qm9['X_graph'][0].edge_attr is not None else 11,
)

print("  Training GTCA-Cat (depth=6) on QM9 homo n=3000 ...")
gtca_qm9 = train_gtca(
    tr_qm9['X_graph'], va_qm9['X_graph'], te_qm9['X_graph'],
    tr_qm9['ids'], va_qm9['ids'], te_qm9['ids'],
    target_name='qm9_homo', device=DEVICE, seed=SEED, bert_depth=6,
    node_feat_dim=tr_qm9['X_graph'][0].x.shape[1],
)


# ---- Draw scatter plots ------------------------------------
def scatter_subplot(ax, y_true, y_pred, title, use_density=False):
    """Draw one scatter subplot."""
    from sklearn.metrics import r2_score
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    r2   = float(r2_score(y_true, y_pred))

    if use_density and len(y_true) > 10:
        xy = np.vstack([y_true, y_pred])
        try:
            kde = gaussian_kde(xy)
            z = kde(xy)
            idx = z.argsort()
            sc = ax.scatter(y_true[idx], y_pred[idx], c=z[idx], cmap='viridis',
                            alpha=0.3, s=8, rasterized=True)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        except Exception:
            ax.scatter(y_true, y_pred, alpha=0.3, s=8)
    else:
        ax.scatter(y_true, y_pred, alpha=0.3, s=8)

    lim_min = min(y_true.min(), y_pred.min()) - 0.1
    lim_max = max(y_true.max(), y_pred.max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=1.2)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('True', fontsize=9)
    ax.set_ylabel('Predicted', fontsize=9)
    ax.text(0.05, 0.93, f'RMSE={rmse:.3f}\nR²={r2:.3f}',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


# -- ESOL scatter --
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('ESOL (train_size=50, seed=0)', fontsize=13)

gcn_esol_true_real  = denorm(gcn_esol['test_true'],  stats_esol)
gcn_esol_pred_real  = denorm(gcn_esol['test_preds'], stats_esol)
gtca_esol_true_real = denorm(gtca_esol['test_true'],  stats_esol)
gtca_esol_pred_real = denorm(gtca_esol['test_preds'], stats_esol)
afp_esol_true_real  = denorm(afp_esol['test_true'],  stats_esol)
afp_esol_pred_real  = denorm(afp_esol['test_preds'], stats_esol)

scatter_subplot(axes[0], gcn_esol_true_real,  gcn_esol_pred_real,  'GCN')
scatter_subplot(axes[1], gtca_esol_true_real, gtca_esol_pred_real, 'GTCA-Cat (depth=6)')
scatter_subplot(axes[2], afp_esol_true_real,  afp_esol_pred_real,  'AttentiveFP')

plt.tight_layout()
path_esol = os.path.join(OUTDIR, 'scatter_esol_n50.png')
plt.savefig(path_esol, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {path_esol}")

# -- BACE scatter --
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('BACE (train_size=500, seed=0)', fontsize=13)

gpr_bace_true_real  = denorm(gpr_bace['test_true'],  stats_bace)
gpr_bace_pred_real  = denorm(gpr_bace['test_preds'], stats_bace)
afp_bace_true_real  = denorm(afp_bace['test_true'],  stats_bace)
afp_bace_pred_real  = denorm(afp_bace['test_preds'], stats_bace)
gtca_bace_true_real = denorm(gtca_bace['test_true'],  stats_bace)
gtca_bace_pred_real = denorm(gtca_bace['test_preds'], stats_bace)

scatter_subplot(axes[0], gpr_bace_true_real,  gpr_bace_pred_real,  'GPR')
scatter_subplot(axes[1], afp_bace_true_real,  afp_bace_pred_real,  'AttentiveFP')
scatter_subplot(axes[2], gtca_bace_true_real, gtca_bace_pred_real, 'GTCA-Cat (depth=6)')

plt.tight_layout()
path_bace = os.path.join(OUTDIR, 'scatter_bace_n500.png')
plt.savefig(path_bace, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path_bace}")

# -- QM9 scatter --
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('QM9-homo (train_size=3000, seed=0)', fontsize=13)

gcn_qm9_true_real  = denorm(gcn_qm9['test_true'],  stats_qm9)
gcn_qm9_pred_real  = denorm(gcn_qm9['test_preds'], stats_qm9)
afp_qm9_true_real  = denorm(afp_qm9['test_true'],  stats_qm9)
afp_qm9_pred_real  = denorm(afp_qm9['test_preds'], stats_qm9)
gtca_qm9_true_real = denorm(gtca_qm9['test_true'],  stats_qm9)
gtca_qm9_pred_real = denorm(gtca_qm9['test_preds'], stats_qm9)

scatter_subplot(axes[0], gcn_qm9_true_real,  gcn_qm9_pred_real,  'GCN',             use_density=True)
scatter_subplot(axes[1], afp_qm9_true_real,  afp_qm9_pred_real,  'AttentiveFP',     use_density=True)
scatter_subplot(axes[2], gtca_qm9_true_real, gtca_qm9_pred_real, 'GTCA-Cat (depth=6)', use_density=True)

plt.tight_layout()
path_qm9 = os.path.join(OUTDIR, 'scatter_qm9_homo_n3000.png')
plt.savefig(path_qm9, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path_qm9}")


# ============================================================
# PART 2 — REPRESENTATIVE MOLECULE SELECTION
# ============================================================

print("\n" + "="*60)
print("PART 2: REPRESENTATIVE MOLECULE SELECTION")
print("="*60)

# -- ESOL: GCN large error, GTCA small error --
# Note: the two models may have slightly different test sets due to validity filtering,
# but we use the GTCA test set (more conservative) as the common reference.
# Actually both share te_esol['X_graph'], so test_true should align.
# We use gcn_esol['test_true'] as canonical true values.

# The test sets come from te_esol which is the same PyG list for both.
# Residuals are in normalized space; convert to real for thresholding.
gcn_res_real  = np.abs(gcn_esol_true_real  - gcn_esol_pred_real)
gtca_res_real = np.abs(gtca_esol_true_real - gtca_esol_pred_real)

# Both models trained on same test split but the internal indices may differ slightly
# (graph featurization filters). Use the SMILES to align.
# Simpler: both use the same te_esol['X_graph'], so results should be index-aligned.
# gcn and gtca test_true are both from te_esol (same loader).
# They should be the same length if featurization is deterministic.
# Use the shorter length to be safe.
n_common_esol = min(len(gcn_res_real), len(gtca_res_real))
gcn_res_real  = gcn_res_real[:n_common_esol]
gtca_res_real = gtca_res_real[:n_common_esol]
gcn_true_real = gcn_esol_true_real[:n_common_esol]
gcn_pred_real = gcn_esol_pred_real[:n_common_esol]
gtca_pred_real = gtca_esol_pred_real[:n_common_esol]

test_smiles_esol = te_esol['ids'][:n_common_esol]

# Find where GCN is bad but GTCA is good
esol_mask = (gcn_res_real > 1.0) & (gtca_res_real < 0.5)
esol_indices = np.where(esol_mask)[0]
print(f"\nESOL candidates (GCN |res|>1.0, GTCA |res|<0.5): {len(esol_indices)} found")

if len(esol_indices) == 0:
    # Relax thresholds
    esol_mask = (gcn_res_real > 0.7) & (gtca_res_real < 0.6)
    esol_indices = np.where(esol_mask)[0]
    print(f"  (relaxed) ESOL candidates: {len(esol_indices)} found")

if len(esol_indices) == 0:
    # Just pick top-3 where GCN_res - GTCA_res is largest
    diff = gcn_res_real - gtca_res_real
    esol_indices = np.argsort(diff)[-3:][::-1]
    print(f"  (fallback) Taking top-3 by GCN_res - GTCA_res difference")

esol_indices = esol_indices[:3]
print(f"  Selected indices: {esol_indices.tolist()}")

rep_esol_rows = []
for idx in esol_indices:
    rep_esol_rows.append({
        'smiles': test_smiles_esol[idx],
        'y_true': float(gcn_true_real[idx]),
        'gcn_pred': float(gcn_pred_real[idx]),
        'gtca_pred': float(gtca_pred_real[idx]),
        'gcn_residual': float(gcn_esol_true_real[idx] - gcn_esol_pred_real[idx]),
        'gtca_residual': float(gtca_esol_true_real[idx] - gtca_esol_pred_real[idx]),
    })

rep_esol_df = pd.DataFrame(rep_esol_rows)
esol_rep_path = os.path.join(OUTDIR, 'representative_esol.csv')
rep_esol_df.to_csv(esol_rep_path, index=False)
print(f"Saved: {esol_rep_path}")
print(rep_esol_df.to_string())

# -- BACE: GPR small error, AFP large error --
gpr_res_real  = np.abs(gpr_bace_true_real  - gpr_bace_pred_real)
afp_res_real  = np.abs(afp_bace_true_real  - afp_bace_pred_real)

n_common_bace = min(len(gpr_res_real), len(afp_res_real))
gpr_res_real  = gpr_res_real[:n_common_bace]
afp_res_real  = afp_res_real[:n_common_bace]
gpr_true_real = gpr_bace_true_real[:n_common_bace]
gpr_pred_real = gpr_bace_pred_real[:n_common_bace]
afp_pred_real_ = afp_bace_pred_real[:n_common_bace]

test_smiles_bace = te_bace['ids'][:n_common_bace]

bace_mask = (gpr_res_real < 0.3) & (afp_res_real > 0.5)
bace_indices = np.where(bace_mask)[0]
print(f"\nBACE candidates (GPR |res|<0.3, AFP |res|>0.5): {len(bace_indices)} found")

if len(bace_indices) == 0:
    bace_mask = (gpr_res_real < 0.4) & (afp_res_real > 0.4)
    bace_indices = np.where(bace_mask)[0]
    print(f"  (relaxed) BACE candidates: {len(bace_indices)} found")

if len(bace_indices) == 0:
    diff = afp_res_real - gpr_res_real
    bace_indices = np.argsort(diff)[-3:][::-1]
    print(f"  (fallback) Taking top-3 by AFP_res - GPR_res difference")

bace_indices = bace_indices[:3]
print(f"  Selected indices: {bace_indices.tolist()}")

rep_bace_rows = []
for idx in bace_indices:
    rep_bace_rows.append({
        'smiles': test_smiles_bace[idx],
        'y_true': float(gpr_true_real[idx]),
        'gpr_pred': float(gpr_pred_real[idx]),
        'attentivefp_pred': float(afp_pred_real_[idx]),
        'gpr_residual': float(gpr_bace_true_real[idx] - gpr_bace_pred_real[idx]),
        'attentivefp_residual': float(afp_bace_true_real[idx] - afp_bace_pred_real[idx]),
    })

rep_bace_df = pd.DataFrame(rep_bace_rows)
bace_rep_path = os.path.join(OUTDIR, 'representative_bace.csv')
rep_bace_df.to_csv(bace_rep_path, index=False)
print(f"Saved: {bace_rep_path}")
print(rep_bace_df.to_string())


# ============================================================
# PART 3 — XAI VISUALIZATIONS
# ============================================================

print("\n" + "="*60)
print("PART 3: XAI VISUALIZATIONS")
print("="*60)

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    HAS_RDKIT_DRAW = True
except Exception as e:
    print(f"  [warn] RDKit drawing not available: {e}")
    HAS_RDKIT_DRAW = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from src.models import GTCAHybrid, get_tokenizer, tokenize_smiles
from torch_geometric.loader import DataLoader as PyGDataLoader


def get_mol_color(saliency_vals):
    """Map saliency values [0,1] to RGB tuples (red=high, white=low)."""
    colors = {}
    smin = saliency_vals.min()
    smax = saliency_vals.max() + 1e-8
    for i, s in enumerate(saliency_vals):
        t = float((s - smin) / (smax - smin))
        # white (1,1,1) → red (1,0,0)
        colors[i] = (1.0, 1.0 - t, 1.0 - t)
    return colors


def draw_mol_with_atom_colors(smiles, atom_colors, save_path, title=''):
    """Draw molecule with per-atom colors using RDKit, save as PNG."""
    if not HAS_RDKIT_DRAW:
        print(f"  [skip] RDKit draw not available: {save_path}")
        return

    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [warn] Invalid SMILES: {smiles[:40]}")
            return
        rdDepictor.Compute2DCoords(mol)

        n_atoms = mol.GetNumAtoms()
        # atom_colors: dict mapping atom_idx -> (r,g,b)
        highlight_atoms = list(range(n_atoms))
        highlight_colors = {i: atom_colors.get(i, (1.0, 1.0, 1.0)) for i in range(n_atoms)}

        try:
            from rdkit.Chem.Draw import rdMolDraw2D as rmd
            drawer = rmd.MolDraw2DCairo(400, 300)
        except Exception:
            drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)

        drawer.drawOptions().addAtomIndices = False
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
            highlightBonds=[],
        )
        drawer.FinishDrawing()

        if hasattr(drawer, 'GetDrawingText'):
            data = drawer.GetDrawingText()
            with open(save_path, 'wb') as f:
                f.write(data)
            print(f"  Saved: {save_path}")
        else:
            print(f"  [warn] drawer has no GetDrawingText")
    except Exception as ex:
        print(f"  [warn] draw_mol failed: {ex}")


def compute_gtca_grad_saliency(model, pyg_data, tokenizer, device='cpu'):
    """
    Compute GCN-branch gradient saliency for a single molecule.
    Returns numpy array of shape (n_atoms,) with saliency per atom.
    """
    model.eval()
    from torch_geometric.data import Batch

    # Create a batch of 1
    batch = Batch.from_data_list([pyg_data]).to(device)
    batch.x.requires_grad_(True)

    ids, mask = tokenize_smiles([pyg_data.smiles], tokenizer, device=device)

    # Forward pass
    out = model(batch.x, batch.edge_index, batch.batch, ids, mask)
    out = out.squeeze()

    # Backprop
    model.zero_grad()
    out.backward()

    grad = batch.x.grad  # (n_atoms, feat_dim)
    if grad is None:
        return np.zeros(pyg_data.x.shape[0])

    # L2 norm per atom
    saliency = grad.norm(dim=-1).detach().cpu().numpy()
    return saliency


def compute_gtca_bert_attention(model, smiles, tokenizer, device='cpu'):
    """
    Extract last-layer attention from GTCAHybrid BERT.
    Returns (tokens, attention_weights) where attention_weights shape = (seq_len,).
    """
    model.eval()
    ids, mask = tokenize_smiles([smiles], tokenizer, device=device)

    with torch.no_grad():
        out = model.bert(
            input_ids=ids,
            attention_mask=mask,
            output_attentions=True,
            return_dict=True,
        )

    # out.attentions: tuple of (batch, heads, seq, seq) per layer
    # Use last used layer (index _bert_depth - 1)
    layer_idx = min(model._bert_depth - 1, len(out.attentions) - 1)
    attn = out.attentions[layer_idx]  # (1, heads, seq, seq)
    attn = attn.squeeze(0)           # (heads, seq, seq)
    attn_mean = attn.mean(0)          # (seq, seq)
    cls_attn = attn_mean[0].cpu().numpy()  # CLS → all tokens

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist())

    # Filter padding
    mask_np = mask[0].cpu().numpy()
    valid_len = int(mask_np.sum())
    tokens = tokens[:valid_len]
    cls_attn = cls_attn[:valid_len]

    return tokens, cls_attn


def compute_attentivefp_saliency(model, pyg_data, device='cpu'):
    """
    Compute AttentiveFP atom-level saliency via gradient of output w.r.t. x.
    Returns numpy array of shape (n_atoms,).
    """
    model.eval()
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([pyg_data]).to(device)
    batch.x = batch.x.clone().requires_grad_(True)

    ea = batch.edge_attr
    if ea is None:
        ea = torch.zeros(batch.edge_index.shape[1], 1, device=device)

    out = model(batch.x, batch.edge_index, ea, batch.batch)
    out = out.squeeze()

    model.zero_grad()
    out.backward()

    grad = batch.x.grad
    if grad is None:
        return np.zeros(pyg_data.x.shape[0])

    saliency = grad.norm(dim=-1).detach().cpu().numpy()
    return saliency


# --- XAI for ESOL representative molecules ---

gtca_esol_model = gtca_esol['model']
afp_esol_model  = afp_esol['model']
tokenizer = get_tokenizer()

# Build a lookup from smiles → pyg data for test set
esol_test_smiles_all = te_esol['ids']
esol_test_pyg_all    = te_esol['X_graph']

for row_idx, row in rep_esol_df.iterrows():
    smi = row['smiles']
    mol_label = f"esol_mol{row_idx}"
    print(f"\n  Processing ESOL representative: {smi[:50]} ...")

    # Find pyg data for this SMILES
    pyg_item = None
    for i, s in enumerate(esol_test_smiles_all):
        if s == smi:
            pyg_item = esol_test_pyg_all[i]
            break

    if pyg_item is None:
        print(f"    [warn] SMILES not found in test_pyg. Skipping XAI.")
        continue

    # Make sure smiles is attached (needed for GTCA forward)
    if not hasattr(pyg_item, 'smiles'):
        pyg_item.smiles = smi

    # 3A. GTCA Integrated Gradients (grad saliency)
    print("    Computing GTCA grad saliency ...")
    try:
        saliency_gtca = compute_gtca_grad_saliency(
            gtca_esol_model, pyg_item, tokenizer, device=DEVICE
        )
        atom_colors_gtca = get_mol_color(saliency_gtca)
        xai_gtca_path = os.path.join(OUTDIR, f'xai_gtca_ig_{mol_label}.png')
        draw_mol_with_atom_colors(smi, atom_colors_gtca, xai_gtca_path)
    except Exception as ex:
        print(f"    [warn] GTCA grad saliency failed: {ex}")

    # 3B. GTCA BERT attention heatmap
    print("    Computing GTCA BERT attention ...")
    try:
        tokens, cls_attn = compute_gtca_bert_attention(
            gtca_esol_model, smi, tokenizer, device=DEVICE
        )
        fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5), 2.5))
        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(
                cls_attn.reshape(1, -1),
                xticklabels=tokens,
                yticklabels=['CLS attn'],
                ax=ax, cmap='Blues', cbar=True, annot=False
            )
        else:
            im = ax.imshow(cls_attn.reshape(1, -1), cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
            ax.set_yticks([0])
            ax.set_yticklabels(['CLS attn'])
            plt.colorbar(im, ax=ax)
        ax.set_title(f'GTCA BERT Last-Layer Attention (CLS→tokens)\n{smi[:50]}', fontsize=8)
        plt.tight_layout()
        attn_path = os.path.join(OUTDIR, f'xai_gtca_attn_{mol_label}.png')
        plt.savefig(attn_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {attn_path}")
    except Exception as ex:
        print(f"    [warn] GTCA BERT attention failed: {ex}")

    # 3C. AttentiveFP saliency
    print("    Computing AttentiveFP saliency ...")
    try:
        saliency_afp = compute_attentivefp_saliency(
            afp_esol_model, pyg_item, device=DEVICE
        )
        atom_colors_afp = get_mol_color(saliency_afp)
        xai_afp_path = os.path.join(OUTDIR, f'xai_attentivefp_{mol_label}.png')
        draw_mol_with_atom_colors(smi, atom_colors_afp, xai_afp_path)
    except Exception as ex:
        print(f"    [warn] AttentiveFP saliency failed: {ex}")


# --- XAI for BACE representative molecules ---

gtca_bace_model = gtca_bace['model']
afp_bace_model  = afp_bace['model']

bace_test_smiles_all = te_bace['ids']
bace_test_pyg_all    = te_bace['X_graph']

for row_idx, row in rep_bace_df.iterrows():
    smi = row['smiles']
    mol_label = f"bace_mol{row_idx}"
    print(f"\n  Processing BACE representative: {smi[:50]} ...")

    pyg_item = None
    for i, s in enumerate(bace_test_smiles_all):
        if s == smi:
            pyg_item = bace_test_pyg_all[i]
            break

    if pyg_item is None:
        print(f"    [warn] SMILES not found in test_pyg. Skipping XAI.")
        continue

    if not hasattr(pyg_item, 'smiles'):
        pyg_item.smiles = smi

    # GTCA grad saliency
    print("    Computing GTCA grad saliency ...")
    try:
        saliency_gtca = compute_gtca_grad_saliency(
            gtca_bace_model, pyg_item, tokenizer, device=DEVICE
        )
        atom_colors_gtca = get_mol_color(saliency_gtca)
        xai_gtca_path = os.path.join(OUTDIR, f'xai_gtca_ig_{mol_label}.png')
        draw_mol_with_atom_colors(smi, atom_colors_gtca, xai_gtca_path)
    except Exception as ex:
        print(f"    [warn] GTCA grad saliency failed: {ex}")

    # GTCA BERT attention
    print("    Computing GTCA BERT attention ...")
    try:
        tokens, cls_attn = compute_gtca_bert_attention(
            gtca_bace_model, smi, tokenizer, device=DEVICE
        )
        fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5), 2.5))
        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(
                cls_attn.reshape(1, -1),
                xticklabels=tokens,
                yticklabels=['CLS attn'],
                ax=ax, cmap='Blues', cbar=True, annot=False
            )
        else:
            im = ax.imshow(cls_attn.reshape(1, -1), cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
            ax.set_yticks([0])
            ax.set_yticklabels(['CLS attn'])
            plt.colorbar(im, ax=ax)
        ax.set_title(f'GTCA BERT Last-Layer Attention (CLS→tokens)\n{smi[:50]}', fontsize=8)
        plt.tight_layout()
        attn_path = os.path.join(OUTDIR, f'xai_gtca_attn_{mol_label}.png')
        plt.savefig(attn_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {attn_path}")
    except Exception as ex:
        print(f"    [warn] GTCA BERT attention failed: {ex}")

    # AttentiveFP saliency
    print("    Computing AttentiveFP saliency ...")
    try:
        saliency_afp = compute_attentivefp_saliency(
            afp_bace_model, pyg_item, device=DEVICE
        )
        atom_colors_afp = get_mol_color(saliency_afp)
        xai_afp_path = os.path.join(OUTDIR, f'xai_attentivefp_{mol_label}.png')
        draw_mol_with_atom_colors(smi, atom_colors_afp, xai_afp_path)
    except Exception as ex:
        print(f"    [warn] AttentiveFP saliency failed: {ex}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("OUTPUT SUMMARY")
print("="*60)
created = []
for fname in sorted(os.listdir(OUTDIR)):
    fpath = os.path.join(OUTDIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    created.append((fname, f"{size_kb:.1f} KB"))
    print(f"  {fname:55s} {size_kb:7.1f} KB")

print(f"\nAll outputs in: {os.path.abspath(OUTDIR)}")
print("Done.")
