"""
run_scatter_predictions.py
==========================
Train models and save predictions to scatter_predictions/ as CSV.
Run this ONCE — then generate_scatter_xai.py reads from these CSVs.

Saves:
  scatter_predictions/esol_n50_seed0.csv      — smiles, y_true, gcn, gtca_cat, attentivefp
  scatter_predictions/bace_n500_seed0.csv     — smiles, y_true, gpr, gtca_cat, attentivefp
  scatter_predictions/qm9_homo_n3000_seed0.csv — smiles, y_true_ev, gcn_ev, gtca_cat_ev, attentivefp_ev
"""

import sys, os
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from src.data_loader import load_dataset_splits, load_raw_data
from src.train import train_gcn, train_gtca, train_attentivefp, train_sklearn

PRED_DIR = 'results/scatter_predictions'
os.makedirs(PRED_DIR, exist_ok=True)
DEVICE = 'cpu'
SEED = 0
HARTREE_TO_EV = 27.2114


def denorm(arr, stats):
    mean, std = stats
    return arr * std + mean


# ============================================================
# ESOL n=50
# ============================================================
print("\n" + "="*60)
print("[1/3] ESOL n=50, seed=0")
print("="*60)

esol_data = load_dataset_splits(
    dataset='esol', data_dir='./data',
    train_size=50, val_size=50, test_size=200,
    seed=SEED, featurize_ecfp=True,
)
tr = esol_data['train']
va = esol_data['val']
te = esol_data['test']
stats = esol_data['stats']

print("  Training GCN ...")
gcn = train_gcn(tr['X_graph'], va['X_graph'], te['X_graph'],
                target_name='esol', device=DEVICE, seed=SEED,
                node_feat_dim=tr['X_graph'][0].x.shape[1])

print("  Training GTCA-Cat (depth=6) ...")
gtca = train_gtca(tr['X_graph'], va['X_graph'], te['X_graph'],
                  tr['ids'], va['ids'], te['ids'],
                  target_name='esol', device=DEVICE, seed=SEED, bert_depth=6,
                  node_feat_dim=tr['X_graph'][0].x.shape[1])

print("  Training AttentiveFP ...")
afp = train_attentivefp(tr['X_graph'], va['X_graph'], te['X_graph'],
                        target_name='esol', device=DEVICE, seed=SEED,
                        node_feat_dim=tr['X_graph'][0].x.shape[1],
                        edge_dim=tr['X_graph'][0].edge_attr.shape[1] if tr['X_graph'][0].edge_attr is not None else 11)

n = min(len(gcn['test_true']), len(gtca['test_true']), len(afp['test_true']))
df = pd.DataFrame({
    'smiles':      te['ids'][:n],
    'y_true':      denorm(gcn['test_true'][:n],  stats),
    'gcn':         denorm(gcn['test_preds'][:n], stats),
    'gtca_cat':    denorm(gtca['test_preds'][:n], stats),
    'attentivefp': denorm(afp['test_preds'][:n],  stats),
})
out = os.path.join(PRED_DIR, 'esol_n50_seed0.csv')
df.to_csv(out, index=False)
print(f"  Saved: {out}  ({len(df)} test molecules)")

# Save model weights for XAI
import torch
torch.save(gtca['model'].state_dict(), os.path.join(PRED_DIR, 'esol_gtca_cat_weights.pt'))
torch.save(afp['model'].state_dict(),  os.path.join(PRED_DIR, 'esol_attentivefp_weights.pt'))
# Save model config for reconstruction
import json
esol_model_config = {
    'node_feat_dim': tr['X_graph'][0].x.shape[1],
    'edge_dim': tr['X_graph'][0].edge_attr.shape[1] if tr['X_graph'][0].edge_attr is not None else 11,
    'bert_depth': 6,
}
with open(os.path.join(PRED_DIR, 'esol_model_config.json'), 'w') as f:
    json.dump(esol_model_config, f)
print(f"  Saved: model weights + config (esol)")


# ============================================================
# BACE n=500
# ============================================================
print("\n" + "="*60)
print("[2/3] BACE n=500, seed=0")
print("="*60)

bace_data = load_dataset_splits(
    dataset='bace', data_dir='./data',
    train_size=500, val_size=100, test_size=300,
    seed=SEED, featurize_ecfp=True,
)
tr = bace_data['train']
va = bace_data['val']
te = bace_data['test']
stats = bace_data['stats']

print("  Training GPR ...")
gpr = train_sklearn(tr['X_ecfp'], tr['y'], va['X_ecfp'], va['y'],
                    te['X_ecfp'], te['y'], model_type='gpr', seed=SEED)

print("  Training AttentiveFP ...")
afp = train_attentivefp(tr['X_graph'], va['X_graph'], te['X_graph'],
                        target_name='bace', device=DEVICE, seed=SEED,
                        node_feat_dim=tr['X_graph'][0].x.shape[1],
                        edge_dim=tr['X_graph'][0].edge_attr.shape[1] if tr['X_graph'][0].edge_attr is not None else 11)

print("  Training GTCA-Cat (depth=6) ...")
gtca = train_gtca(tr['X_graph'], va['X_graph'], te['X_graph'],
                  tr['ids'], va['ids'], te['ids'],
                  target_name='bace', device=DEVICE, seed=SEED, bert_depth=6,
                  node_feat_dim=tr['X_graph'][0].x.shape[1])

n = min(len(gpr['test_true']), len(afp['test_true']), len(gtca['test_true']))
df = pd.DataFrame({
    'smiles':      te['ids'][:n],
    'y_true':      denorm(gpr['test_true'][:n],  stats),
    'gpr':         denorm(gpr['test_preds'][:n], stats),
    'attentivefp': denorm(afp['test_preds'][:n],  stats),
    'gtca_cat':    denorm(gtca['test_preds'][:n], stats),
})
out = os.path.join(PRED_DIR, 'bace_n500_seed0.csv')
df.to_csv(out, index=False)
print(f"  Saved: {out}  ({len(df)} test molecules)")

# Save model weights for XAI
torch.save(gtca['model'].state_dict(), os.path.join(PRED_DIR, 'bace_gtca_cat_weights.pt'))
torch.save(afp['model'].state_dict(),  os.path.join(PRED_DIR, 'bace_attentivefp_weights.pt'))
bace_model_config = {
    'node_feat_dim': tr['X_graph'][0].x.shape[1],
    'edge_dim': tr['X_graph'][0].edge_attr.shape[1] if tr['X_graph'][0].edge_attr is not None else 11,
    'bert_depth': 6,
}
with open(os.path.join(PRED_DIR, 'bace_model_config.json'), 'w') as f:
    json.dump(bace_model_config, f)
print(f"  Saved: model weights + config (bace)")


# ============================================================
# QM9 HOMO n=3000
# ============================================================
print("\n" + "="*60)
print("[3/3] QM9 HOMO n=3000, seed=0")
print("="*60)

raw_qm9 = load_raw_data('qm9', './data', 'homo')
qm9_data = load_dataset_splits(
    dataset='qm9', data_dir='./data',
    train_size=3000, val_size=1000, test_size=5000,
    seed=SEED, target='homo', preloaded_raw=raw_qm9,
)
tr = qm9_data['train']
va = qm9_data['val']
te = qm9_data['test']
stats = qm9_data['stats']

print("  Training GCN ...")
gcn = train_gcn(tr['X_graph'], va['X_graph'], te['X_graph'],
                target_name='qm9_homo', device=DEVICE, seed=SEED,
                node_feat_dim=tr['X_graph'][0].x.shape[1])

print("  Training AttentiveFP ...")
afp = train_attentivefp(tr['X_graph'], va['X_graph'], te['X_graph'],
                        target_name='qm9_homo', device=DEVICE, seed=SEED,
                        node_feat_dim=tr['X_graph'][0].x.shape[1],
                        edge_dim=tr['X_graph'][0].edge_attr.shape[1] if tr['X_graph'][0].edge_attr is not None else 11)

print("  Training GTCA-Cat (depth=6) ...")
gtca = train_gtca(tr['X_graph'], va['X_graph'], te['X_graph'],
                  tr['ids'], va['ids'], te['ids'],
                  target_name='qm9_homo', device=DEVICE, seed=SEED, bert_depth=6,
                  node_feat_dim=tr['X_graph'][0].x.shape[1])

n = min(len(gcn['test_true']), len(afp['test_true']), len(gtca['test_true']))
df = pd.DataFrame({
    'smiles':         te['ids'][:n],
    'y_true_ev':      denorm(gcn['test_true'][:n],  stats) * HARTREE_TO_EV,
    'gcn_ev':         denorm(gcn['test_preds'][:n], stats) * HARTREE_TO_EV,
    'attentivefp_ev': denorm(afp['test_preds'][:n],  stats) * HARTREE_TO_EV,
    'gtca_cat_ev':    denorm(gtca['test_preds'][:n], stats) * HARTREE_TO_EV,
})
out = os.path.join(PRED_DIR, 'qm9_homo_n3000_seed0.csv')
df.to_csv(out, index=False)
print(f"  Saved: {out}  ({len(df)} test molecules)")

print("\n" + "="*60)
print("Done. All predictions saved to results/scatter_predictions/")
print("="*60)
