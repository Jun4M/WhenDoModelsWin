"""
analysis.py
- save_failure_data_csv: worst/best-10 with per-atom saliency scores (no images)
- group_analysis: chemical group MAE breakdown
- compile_summary: aggregate metrics to summary CSV (legacy)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# ---------------------------------------------------------------------------
# RDKit descriptor extraction
# ---------------------------------------------------------------------------

def _get_rdkit_desc(smiles: str) -> dict:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            'MW':            Descriptors.MolWt(mol),
            'LogP':          Descriptors.MolLogP(mol),
            'TPSA':          Descriptors.TPSA(mol),
            'HeavyAtomCount': mol.GetNumHeavyAtoms(),
            'FractionCSP3':  rdMolDescriptors.CalcFractionCSP3(mol),
            'HasF':          int(any(a.GetAtomicNum() == 9  for a in mol.GetAtoms())),
            'HasN':          int(any(a.GetAtomicNum() == 7  for a in mol.GetAtoms())),
            'HasO':          int(any(a.GetAtomicNum() == 8  for a in mol.GetAtoms())),
            'RingCount':     rdMolDescriptors.CalcNumRings(mol),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Per-atom saliency (input gradient w.r.t. node features)
# ---------------------------------------------------------------------------

def _compute_atom_saliency_gcn(model, pyg_data, device: str = 'cpu') -> list:
    """
    Gradient of output w.r.t. node features (GCN / AttentiveFP / GPS).
    Returns list of per-atom saliency scores (float).
    """
    try:
        import torch
        from torch_geometric.data import Batch
        model.eval()
        data = pyg_data.to(device)
        data.x = data.x.requires_grad_(True)
        batch_vec = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

        if hasattr(model, 'forward') and 'edge_attr' in model.forward.__code__.co_varnames:
            out = model(data.x, data.edge_index, data.edge_attr, batch_vec)
        else:
            out = model(data.x, data.edge_index, batch_vec)

        out.sum().backward()
        scores = data.x.grad.norm(dim=-1).detach().cpu().tolist()
        return scores
    except Exception:
        return []


def _compute_token_saliency_transformer(model, tokenizer, smiles: str, device: str = 'cpu') -> list:
    """
    Attention weights from last encoder layer, averaged over heads.
    Returns list of per-token attention scores (float).
    """
    try:
        from src.models import tokenize_smiles
        model.eval()
        ids, mask = tokenize_smiles([smiles], tokenizer, device=device)
        with torch.no_grad():
            out = model.encoder(
                input_ids=ids, attention_mask=mask,
                output_attentions=True,
            )
        # Last layer attention: (batch, heads, seq, seq) → average over heads → CLS row
        attn = out.attentions[-1][0]  # (heads, seq, seq)
        cls_attn = attn.mean(0)[0]    # (seq,) CLS → all tokens
        return cls_attn.cpu().tolist()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Failure analysis: save CSV with saliency (no images)
# ---------------------------------------------------------------------------

def save_failure_data_csv(
    smiles_list: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_name: str,
    raw_dir: str,
    model=None,
    device: str = 'cpu',
    top_k: int = 10,
    pyg_data_list: list = None,
    tokenizer=None,
):
    """
    Saves worst/best top_k molecules as CSV with RDKit descriptors
    and per-atom/per-token saliency scores as JSON column.
    No image files are generated here.

    Output: raw_dir/{model_name}_{target_name}_failures.csv
    """
    residuals = np.abs(y_true - y_pred)
    sorted_idx = np.argsort(residuals)

    best_idx  = sorted_idx[:top_k]
    worst_idx = sorted_idx[-top_k:][::-1]

    rows = []
    for split, indices in [('best', best_idx), ('worst', worst_idx)]:
        for rank, idx in enumerate(indices):
            smi = smiles_list[idx]
            desc = _get_rdkit_desc(smi)
            row = {
                'rank':      rank + 1,
                'split':     split,
                'smiles':    smi,
                'y_true':    float(y_true[idx]),
                'y_pred':    float(y_pred[idx]),
                'residual':  float(residuals[idx]),
                'model':     model_name,
                'target':    target_name,
                **desc,
            }

            # Saliency scores
            atom_scores = []
            if model is not None:
                if model_name in ('gcn', 'attentivefp', 'gps') and pyg_data_list is not None:
                    atom_scores = _compute_atom_saliency_gcn(model, pyg_data_list[idx], device)
                elif model_name == 'transformer' and tokenizer is not None:
                    atom_scores = _compute_token_saliency_transformer(model, tokenizer, smi, device)
            row['atom_scores'] = json.dumps(atom_scores)
            rows.append(row)

    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f"{model_name}_{target_name}_failures.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [analysis] failures → {out_path}")
    return df


# ---------------------------------------------------------------------------
# Chemical group analysis
# ---------------------------------------------------------------------------

def group_analysis(
    smiles_list: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_name: str,
    train_size: int,
    seed: int = 0,
) -> list:
    """
    Groups test molecules by chemical properties and computes Mean_MAE per group.
    Returns list of row dicts for master_group_summary.csv.
    """
    residuals = np.abs(y_true - y_pred)
    descs = [_get_rdkit_desc(s) for s in smiles_list]

    mw_vals   = np.array([d.get('MW', np.nan) for d in descs])
    hac_vals  = np.array([d.get('HeavyAtomCount', np.nan) for d in descs])
    csp3_vals = np.array([d.get('FractionCSP3', np.nan) for d in descs])
    hasF_vals = np.array([d.get('HasF', 0) for d in descs])
    hasN_vals = np.array([d.get('HasN', 0) for d in descs])
    hasO_vals = np.array([d.get('HasO', 0) for d in descs])

    rows = []
    base = dict(train_size=train_size, model=model_name, target=target_name, seed=seed)

    # MW bins
    bins = [0, 150, 250, 350, 500, 10000]
    labels = ['<150', '150-250', '250-350', '350-500', '>500']
    for lo, hi, lbl in zip(bins, bins[1:], labels):
        mask = (~np.isnan(mw_vals)) & (mw_vals >= lo) & (mw_vals < hi)
        if mask.sum() > 0:
            rows.append({**base, 'group_type': 'MW_bin', 'category': lbl,
                         'Mean_MAE': float(residuals[mask].mean()), 'Count': int(mask.sum())})

    # Element presence
    for elem, vals in [('HasF', hasF_vals), ('HasN', hasN_vals), ('HasO', hasO_vals)]:
        for cat, flag in [('yes', 1), ('no', 0)]:
            mask = vals == flag
            if mask.sum() > 0:
                rows.append({**base, 'group_type': elem, 'category': cat,
                             'Mean_MAE': float(residuals[mask].mean()), 'Count': int(mask.sum())})

    # Heavy atom count bins
    hac_bins   = [0, 10, 20, 30, 50, 1000]
    hac_labels = ['<10', '10-20', '20-30', '30-50', '>50']
    for lo, hi, lbl in zip(hac_bins, hac_bins[1:], hac_labels):
        mask = (~np.isnan(hac_vals)) & (hac_vals >= lo) & (hac_vals < hi)
        if mask.sum() > 0:
            rows.append({**base, 'group_type': 'HeavyAtomCount', 'category': lbl,
                         'Mean_MAE': float(residuals[mask].mean()), 'Count': int(mask.sum())})

    # FractionCSP3 bins
    csp3_bins   = [0.0, 0.25, 0.5, 0.75, 1.01]
    csp3_labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']
    for lo, hi, lbl in zip(csp3_bins, csp3_bins[1:], csp3_labels):
        mask = (~np.isnan(csp3_vals)) & (csp3_vals >= lo) & (csp3_vals < hi)
        if mask.sum() > 0:
            rows.append({**base, 'group_type': 'FractionCSP3', 'category': lbl,
                         'Mean_MAE': float(residuals[mask].mean()), 'Count': int(mask.sum())})

    return rows


# ---------------------------------------------------------------------------
# Legacy: compile_summary (kept for main.py compat)
# ---------------------------------------------------------------------------

def compile_summary(all_results: dict, results_dir: str) -> pd.DataFrame:
    rows = []
    for target, model_dict in all_results.items():
        for model_name, res in model_dict.items():
            if res is None:
                continue
            m = res.get('metrics', {})
            rows.append({
                'target': target, 'model': model_name,
                'RMSE': m.get('RMSE'), 'MAE': m.get('MAE'),
                'Pearson_R': m.get('Pearson_R'), 'R2': m.get('R2'),
            })
    df = pd.DataFrame(rows)
    out = os.path.join(results_dir, 'summary_metrics.csv')
    df.to_csv(out, index=False)
    print(f"[summary] → {out}")
    return df


# ---------------------------------------------------------------------------
# Legacy: failure_analysis (kept for main.py compat, no images)
# ---------------------------------------------------------------------------

def failure_analysis(
    smiles_list, y_true, y_pred,
    model_name, target_name, results_dir,
    train_size=None, save_images=False, top_k=10,
    model=None, device='cpu', pyg_data_list=None, tokenizer=None,
):
    """Legacy wrapper → delegates to save_failure_data_csv."""
    return save_failure_data_csv(
        smiles_list=smiles_list, y_true=y_true, y_pred=y_pred,
        model_name=model_name, target_name=target_name,
        raw_dir=results_dir, model=model, device=device,
        top_k=top_k, pyg_data_list=pyg_data_list, tokenizer=tokenizer,
    )
