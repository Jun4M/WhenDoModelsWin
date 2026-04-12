"""
generate_scatter_xai.py
=======================
Reads predictions from results/scatter_predictions/ (saved by run_scatter_predictions.py)
and generates:
  Part 1: Scatter plots (Fig 10, 14, 17) — PNG + PDF, 300 dpi
  Part 2: Representative molecule selection → representative_esol/bace.csv
  Part 3: XAI visualizations (Fig 18-23) — gradient saliency + attention heatmap

Run run_scatter_predictions.py first to generate the CSV files.
XAI (Part 3) requires re-loading models from run_scatter_predictions.py.
"""

import sys, os
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

PRED_DIR = 'results/scatter_predictions'
OUTDIR   = 'results/scatter_plots'
DEVICE   = 'cpu'
SEED     = 0
HARTREE_TO_EV = 27.2114

os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Check prediction CSVs exist
# ============================================================
ESOL_CSV = os.path.join(PRED_DIR, 'esol_n50_seed0.csv')
BACE_CSV = os.path.join(PRED_DIR, 'bace_n500_seed0.csv')
QM9_CSV  = os.path.join(PRED_DIR, 'qm9_homo_n3000_seed0.csv')

missing = [f for f in [ESOL_CSV, BACE_CSV, QM9_CSV] if not os.path.exists(f)]
if missing:
    print("ERROR: Missing prediction CSVs:")
    for f in missing:
        print(f"  {f}")
    print("Run: python run_scatter_predictions.py")
    sys.exit(1)

esol_df = pd.read_csv(ESOL_CSV)
bace_df = pd.read_csv(BACE_CSV)
qm9_df  = pd.read_csv(QM9_CSV)

print(f"Loaded ESOL:  {len(esol_df)} molecules")
print(f"Loaded BACE:  {len(bace_df)} molecules")
print(f"Loaded QM9:   {len(qm9_df)} molecules")


# ============================================================
# PART 1 — SCATTER PLOTS
# ============================================================

print("\n" + "="*60)
print("PART 1: SCATTER PLOTS")
print("="*60)


def scatter_subplot(ax, y_true, y_pred, title,
                    use_density=False,
                    xlabel='True value', ylabel='Predicted value'):
    from sklearn.metrics import r2_score
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    r2   = float(r2_score(y_true, y_pred))

    if use_density and len(y_true) > 10:
        xy = np.vstack([y_true, y_pred])
        try:
            kde = gaussian_kde(xy)
            z   = kde(xy)
            idx = z.argsort()
            sc  = ax.scatter(y_true[idx], y_pred[idx], c=z[idx], cmap='viridis',
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
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.text(0.05, 0.93, f'RMSE={rmse:.3f}\nR²={r2:.3f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def save_fig(fig, path):
    base = os.path.splitext(path)[0]
    for ext in ['png', 'pdf']:
        out = f'{base}.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out}")
    plt.close(fig)


# -- Fig 14: ESOL n=50 --
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
xl = 'True solubility (log mol/L)'
yl = 'Predicted solubility (log mol/L)'
scatter_subplot(axes[0], esol_df['y_true'], esol_df['gcn'],         'GCN',              xlabel=xl, ylabel=yl)
scatter_subplot(axes[1], esol_df['y_true'], esol_df['gtca_cat'],    'GTCA-Cat (depth=6)', xlabel=xl, ylabel=yl)
scatter_subplot(axes[2], esol_df['y_true'], esol_df['attentivefp'], 'AttentiveFP',       xlabel=xl, ylabel=yl)
plt.tight_layout()
save_fig(fig, os.path.join(OUTDIR, 'scatter_esol_n50.png'))

# -- Fig 17: BACE n=500 --
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
xl = 'True pIC50'
yl = 'Predicted pIC50'
scatter_subplot(axes[0], bace_df['y_true'], bace_df['gpr'],         'GPR',              xlabel=xl, ylabel=yl)
scatter_subplot(axes[1], bace_df['y_true'], bace_df['attentivefp'], 'AttentiveFP',      xlabel=xl, ylabel=yl)
scatter_subplot(axes[2], bace_df['y_true'], bace_df['gtca_cat'],    'GTCA-Cat (depth=6)', xlabel=xl, ylabel=yl)
plt.tight_layout()
save_fig(fig, os.path.join(OUTDIR, 'scatter_bace_n500.png'))

# -- Fig 10: QM9 HOMO n=3000 --
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
xl = 'True HOMO energy (eV)'
yl = 'Predicted HOMO energy (eV)'
scatter_subplot(axes[0], qm9_df['y_true_ev'], qm9_df['gcn_ev'],         'GCN',              use_density=True, xlabel=xl, ylabel=yl)
scatter_subplot(axes[1], qm9_df['y_true_ev'], qm9_df['attentivefp_ev'], 'AttentiveFP',      use_density=True, xlabel=xl, ylabel=yl)
scatter_subplot(axes[2], qm9_df['y_true_ev'], qm9_df['gtca_cat_ev'],    'GTCA-Cat (depth=6)', use_density=True, xlabel=xl, ylabel=yl)
plt.tight_layout()
save_fig(fig, os.path.join(OUTDIR, 'scatter_qm9_homo_n3000.png'))


# ============================================================
# PART 2 — REPRESENTATIVE MOLECULE SELECTION
# ============================================================

print("\n" + "="*60)
print("PART 2: REPRESENTATIVE MOLECULE SELECTION")
print("="*60)

# -- ESOL: GCN large error, GTCA small error --
gcn_res  = np.abs(esol_df['y_true'] - esol_df['gcn'])
gtca_res = np.abs(esol_df['y_true'] - esol_df['gtca_cat'])

mask = (gcn_res > 1.0) & (gtca_res < 0.5)
idxs = np.where(mask)[0]
if len(idxs) == 0:
    mask = (gcn_res > 0.7) & (gtca_res < 0.6)
    idxs = np.where(mask)[0]
if len(idxs) == 0:
    idxs = np.argsort(gcn_res - gtca_res)[-3:][::-1]
idxs = idxs[:3]
print(f"ESOL representative indices: {idxs.tolist()}")

rep_esol = esol_df.iloc[idxs][['smiles', 'y_true', 'gcn', 'gtca_cat']].copy()
rep_esol['gcn_residual']  = esol_df.iloc[idxs]['y_true'].values - esol_df.iloc[idxs]['gcn'].values
rep_esol['gtca_residual'] = esol_df.iloc[idxs]['y_true'].values - esol_df.iloc[idxs]['gtca_cat'].values
rep_esol = rep_esol.rename(columns={'gcn': 'gcn_pred', 'gtca_cat': 'gtca_pred'})
rep_esol.to_csv(os.path.join(OUTDIR, 'representative_esol.csv'), index=False)
print(rep_esol.to_string())

# -- BACE: GPR small error, AFP large error --
gpr_res = np.abs(bace_df['y_true'] - bace_df['gpr'])
afp_res = np.abs(bace_df['y_true'] - bace_df['attentivefp'])

mask = (gpr_res < 0.3) & (afp_res > 0.5)
idxs = np.where(mask)[0]
if len(idxs) == 0:
    mask = (gpr_res < 0.4) & (afp_res > 0.4)
    idxs = np.where(mask)[0]
if len(idxs) == 0:
    idxs = np.argsort(afp_res - gpr_res)[-3:][::-1]
idxs = idxs[:3]
print(f"\nBACE representative indices: {idxs.tolist()}")

rep_bace = bace_df.iloc[idxs][['smiles', 'y_true', 'gpr', 'attentivefp']].copy()
rep_bace['gpr_residual']          = bace_df.iloc[idxs]['y_true'].values - bace_df.iloc[idxs]['gpr'].values
rep_bace['attentivefp_residual']  = bace_df.iloc[idxs]['y_true'].values - bace_df.iloc[idxs]['attentivefp'].values
rep_bace = rep_bace.rename(columns={'gpr': 'gpr_pred', 'attentivefp': 'attentivefp_pred'})
rep_bace.to_csv(os.path.join(OUTDIR, 'representative_bace.csv'), index=False)
print(rep_bace.to_string())


# ============================================================
# PART 3 — XAI VISUALIZATIONS
# ============================================================

print("\n" + "="*60)
print("PART 3: XAI VISUALIZATIONS")
print("="*60)
print("Re-training models for XAI (gradient computation needs live model)...")

from src.data_loader import load_dataset_splits, load_raw_data
from src.train import train_gcn, train_gtca, train_attentivefp, train_sklearn
from src.models import GTCAHybrid, get_tokenizer, tokenize_smiles

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


def get_mol_color(saliency_vals):
    colors = {}
    smin = saliency_vals.min()
    smax = saliency_vals.max() + 1e-8
    for i, s in enumerate(saliency_vals):
        t = float((s - smin) / (smax - smin))
        colors[i] = (1.0, 1.0 - t, 1.0 - t)
    return colors


def draw_mol_with_atom_colors(smiles, atom_colors, save_path, title=''):
    if not HAS_RDKIT_DRAW:
        print(f"  [skip] RDKit draw not available: {save_path}")
        return
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [warn] Invalid SMILES: {smiles[:40]}")
            return
        rdDepictor.Compute2DCoords(mol)
        n_atoms = mol.GetNumAtoms()
        highlight_atoms  = list(range(n_atoms))
        highlight_colors = {i: atom_colors.get(i, (1.0, 1.0, 1.0)) for i in range(n_atoms)}
        try:
            from rdkit.Chem.Draw import rdMolDraw2D as rmd
            drawer = rmd.MolDraw2DCairo(1200, 900)
        except Exception:
            drawer = rdMolDraw2D.MolDraw2DSVG(1200, 900)
        drawer.drawOptions().addAtomIndices = False
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                            highlightAtomColors=highlight_colors, highlightBonds=[])
        drawer.FinishDrawing()
        if hasattr(drawer, 'GetDrawingText'):
            data = drawer.GetDrawingText()
            with open(save_path, 'wb') as f:
                f.write(data)
            print(f"  Saved: {save_path}")
            try:
                import io
                from PIL import Image as _PIL
                pdf_path = os.path.splitext(save_path)[0] + '.pdf'
                img_buf = _PIL.open(io.BytesIO(data))
                fig_pdf, ax_pdf = plt.subplots(figsize=(img_buf.width/300, img_buf.height/300))
                ax_pdf.imshow(img_buf)
                ax_pdf.axis('off')
                fig_pdf.savefig(pdf_path, dpi=300, bbox_inches='tight')
                plt.close(fig_pdf)
                print(f"  Saved: {pdf_path}")
            except Exception as _e:
                print(f"  [warn] PDF export failed: {_e}")
    except Exception as ex:
        print(f"  [warn] draw_mol failed: {ex}")


def compute_gtca_grad_saliency(model, pyg_data, tokenizer, device='cpu'):
    from torch_geometric.data import Batch
    model.eval()
    batch = Batch.from_data_list([pyg_data]).to(device)
    batch.x.requires_grad_(True)
    ids, mask = tokenize_smiles([pyg_data.smiles], tokenizer, device=device)
    out = model(batch.x, batch.edge_index, batch.batch, ids, mask).squeeze()
    model.zero_grad()
    out.backward()
    grad = batch.x.grad
    if grad is None:
        return np.zeros(pyg_data.x.shape[0])
    return grad.norm(dim=-1).detach().cpu().numpy()


def compute_gtca_bert_attention(model, smiles, tokenizer, device='cpu'):
    model.eval()
    ids, mask = tokenize_smiles([smiles], tokenizer, device=device)
    with torch.no_grad():
        out = model.bert(input_ids=ids, attention_mask=mask,
                         output_attentions=True, return_dict=True)
    layer_idx = min(model._bert_depth - 1, len(out.attentions) - 1)
    attn = out.attentions[layer_idx].squeeze(0).mean(0)
    cls_attn = attn[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist())
    valid_len = int(mask[0].cpu().numpy().sum())
    return tokens[:valid_len], cls_attn[:valid_len]


def compute_attentivefp_saliency(model, pyg_data, device='cpu'):
    from torch_geometric.data import Batch
    model.eval()
    batch = Batch.from_data_list([pyg_data]).to(device)
    batch.x = batch.x.clone().requires_grad_(True)
    ea = batch.edge_attr
    if ea is None:
        ea = torch.zeros(batch.edge_index.shape[1], 1, device=device)
    out = model(batch.x, batch.edge_index, ea, batch.batch).squeeze()
    model.zero_grad()
    out.backward()
    grad = batch.x.grad
    if grad is None:
        return np.zeros(pyg_data.x.shape[0])
    return grad.norm(dim=-1).detach().cpu().numpy()


def run_xai(dataset, train_size, val_size, test_size, target_name,
            rep_df, model_types, te_data, gtca_result, afp_result, tokenizer):
    """Run XAI for representative molecules of one dataset."""
    test_smiles = te_data['ids']
    test_pyg    = te_data['X_graph']
    gtca_model  = gtca_result['model']
    afp_model   = afp_result['model']

    for row_idx in range(len(rep_df)):
        row = rep_df.iloc[row_idx]
        smi = row['smiles']
        mol_label = f"{dataset}_mol{row_idx}"
        print(f"\n  [{dataset}] mol{row_idx}: {smi[:50]} ...")

        pyg_item = next((test_pyg[i] for i, s in enumerate(test_smiles) if s == smi), None)
        if pyg_item is None:
            print(f"    [warn] SMILES not found in test set. Skipping.")
            continue
        if not hasattr(pyg_item, 'smiles'):
            pyg_item.smiles = smi

        # GTCA grad saliency
        try:
            sal = compute_gtca_grad_saliency(gtca_model, pyg_item, tokenizer, device=DEVICE)
            draw_mol_with_atom_colors(smi, get_mol_color(sal),
                                      os.path.join(OUTDIR, f'xai_gtca_ig_{mol_label}.png'))
        except Exception as ex:
            print(f"    [warn] GTCA saliency failed: {ex}")

        # GTCA BERT attention heatmap
        try:
            tokens, cls_attn = compute_gtca_bert_attention(gtca_model, smi, tokenizer, device=DEVICE)
            fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5), 2.5))
            if HAS_SEABORN:
                sns.heatmap(cls_attn.reshape(1, -1), xticklabels=tokens,
                            yticklabels=['CLS attn'], ax=ax, cmap='Blues', cbar=True)
            else:
                im = ax.imshow(cls_attn.reshape(1, -1), cmap='Blues', aspect='auto')
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
                ax.set_yticks([0]); ax.set_yticklabels(['CLS attn'])
                plt.colorbar(im, ax=ax)
            ax.set_title(f'GTCA BERT Attention (CLS→tokens)\n{smi[:50]}', fontsize=8)
            plt.tight_layout()
            attn_path = os.path.join(OUTDIR, f'xai_gtca_attn_{mol_label}')
            for ext in ['png', 'pdf']:
                plt.savefig(f'{attn_path}.{ext}', dpi=300, bbox_inches='tight')
                print(f"    Saved: {attn_path}.{ext}")
            plt.close()
        except Exception as ex:
            print(f"    [warn] GTCA BERT attention failed: {ex}")

        # AttentiveFP saliency
        try:
            sal = compute_attentivefp_saliency(afp_model, pyg_item, device=DEVICE)
            draw_mol_with_atom_colors(smi, get_mol_color(sal),
                                      os.path.join(OUTDIR, f'xai_attentivefp_{mol_label}.png'))
        except Exception as ex:
            print(f"    [warn] AFP saliency failed: {ex}")


import json

def load_or_train_models(dataset, train_size, val_size, test_size, target_name, featurize_ecfp=True):
    """Load model weights from scatter_predictions/ if available, else re-train."""
    gtca_w = os.path.join(PRED_DIR, f'{dataset}_gtca_cat_weights.pt')
    afp_w  = os.path.join(PRED_DIR, f'{dataset}_attentivefp_weights.pt')
    cfg_f  = os.path.join(PRED_DIR, f'{dataset}_model_config.json')

    data = load_dataset_splits(dataset=dataset, data_dir='./data',
                               train_size=train_size, val_size=val_size,
                               test_size=test_size, seed=SEED,
                               featurize_ecfp=featurize_ecfp)
    tr, va, te = data['train'], data['val'], data['test']

    if os.path.exists(gtca_w) and os.path.exists(afp_w) and os.path.exists(cfg_f):
        print(f"  [XAI] Loading saved weights for {dataset} ...")
        with open(cfg_f) as f:
            cfg = json.load(f)
        from src.models import GTCAHybrid, AttentiveFPRegressor
        gtca_model = GTCAHybrid(node_feat_dim=cfg['node_feat_dim'],
                                bert_depth=cfg['bert_depth']).to(DEVICE)
        gtca_model.load_state_dict(torch.load(gtca_w, map_location=DEVICE))
        gtca_model.eval()
        afp_model = AttentiveFPRegressor(in_channels=cfg['node_feat_dim'],
                                         edge_dim=cfg['edge_dim']).to(DEVICE)
        afp_model.load_state_dict(torch.load(afp_w, map_location=DEVICE))
        afp_model.eval()
        gtca_result = {'model': gtca_model}
        afp_result  = {'model': afp_model}
    else:
        print(f"  [XAI] No saved weights found for {dataset}, re-training ...")
        gtca_result = train_gtca(tr['X_graph'], va['X_graph'], te['X_graph'],
                                  tr['ids'], va['ids'], te['ids'],
                                  target_name=target_name, device=DEVICE, seed=SEED, bert_depth=6,
                                  node_feat_dim=tr['X_graph'][0].x.shape[1])
        afp_result  = train_attentivefp(tr['X_graph'], va['X_graph'], te['X_graph'],
                                         target_name=target_name, device=DEVICE, seed=SEED,
                                         node_feat_dim=tr['X_graph'][0].x.shape[1],
                                         edge_dim=tr['X_graph'][0].edge_attr.shape[1] if tr['X_graph'][0].edge_attr is not None else 11)
    return te, gtca_result, afp_result


tokenizer = get_tokenizer()

# -- ESOL XAI --
print("\n[XAI] ESOL ...")
te_esol, gtca_esol, afp_esol = load_or_train_models('esol', 50, 50, 200, 'esol')
run_xai('esol', 50, 50, 200, 'esol', rep_esol, None, te_esol, gtca_esol, afp_esol, tokenizer)

# -- BACE XAI --
print("\n[XAI] BACE ...")
te_bace, gtca_bace, afp_bace = load_or_train_models('bace', 500, 100, 300, 'bace')
run_xai('bace', 500, 100, 300, 'bace', rep_bace, None, te_bace, gtca_bace, afp_bace, tokenizer)


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("OUTPUT SUMMARY")
print("="*60)
for fname in sorted(os.listdir(OUTDIR)):
    fpath = os.path.join(OUTDIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname:55s} {size_kb:7.1f} KB")
print(f"\nAll outputs in: {os.path.abspath(OUTDIR)}")
print("Done.")
