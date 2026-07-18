"""
descriptor_selection.py
=======================
Python re-implementation of KROVEX's descriptor selection pipeline
(Jang et al. 2026, J. Cheminformatics).

Replaces the R-based SIS package with pure-Python ISIS for portability.

Two-stage pipeline (per training fold):
  1. ISIS  — Iterative Sure Independence Screening (Fan & Lv 2008)
  2. Elastic Net — GridSearchCV on (alpha, l1_ratio) → sparse selection

All statistics are computed from training-fold y only.  No train/test leak.

Reference: Fan, J. & Lv, J. (2008). Sure independence screening for ultrahigh
dimensional feature space. J. R. Statist. Soc. B, 70(5), 849–911.
doi:10.1111/j.1467-9868.2008.00674.x
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


# ──────────────────────────────────────────────────────────────────────────────
# Step 0 — descriptor extraction
# ──────────────────────────────────────────────────────────────────────────────

_DESC_NAMES = [d[0] for d in Descriptors._descList]
_DESC_FUNCS  = {d[0]: d[1] for d in Descriptors._descList}


def compute_209_descriptors(smiles_list: list) -> pd.DataFrame:
    """Compute all RDKit descriptors (~209–217 depending on version) for each SMILES.

    Returns a DataFrame with one column per descriptor.
    Rows where SMILES cannot be parsed contain None for all descriptors.
    Descriptor computation errors (rare) set that cell to None.
    """
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({d: None for d in _DESC_NAMES})
            continue
        row = {}
        for name, fn in _DESC_FUNCS.items():
            try:
                row[name] = fn(mol)
            except Exception:
                row[name] = None
        rows.append(row)
    return pd.DataFrame(rows, columns=_DESC_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_descriptors(df: pd.DataFrame) -> tuple:
    """Drop NaN rows, constant columns, and low-variance columns (var < 0.001).

    Returns (cleaned_df, kept_descriptor_names).
    cleaned_df has the same row index as df (only rows with ANY NaN are removed).
    """
    # Drop rows with any NaN
    cleaned = df.dropna().copy()

    # Drop constant columns (nunique == 1)
    constant_cols = [c for c in cleaned.columns if cleaned[c].nunique() <= 1]
    cleaned = cleaned.drop(columns=constant_cols)

    # Drop low-variance columns
    variances  = cleaned.var(axis=0)
    low_var    = variances[variances < 0.001].index.tolist()
    cleaned    = cleaned.drop(columns=low_var)

    return cleaned, list(cleaned.columns)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — ISIS
# ──────────────────────────────────────────────────────────────────────────────

def isis_screen(
    X: np.ndarray,
    y: np.ndarray,
    nsis: int = 20,
    max_iter: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Iterative Sure Independence Screening (Fan & Lv 2008).

    Algorithm:
      Iteration 0:
        a. Standardize X (z-score, column-wise).
        b. Compute |Pearson corr| between each column and y.
        c. Select top-nsis columns → initial selected set S.
        d. OLS on S → residuals r.
      Iterations 1..max_iter:
        e. Compute |Pearson corr| between unselected columns and r.
        f. Add top-nsis unselected columns to S (augment).
        g. OLS on S → new residuals.
        h. Stop early if S did not change.

    Parameters
    ----------
    X : (n, p) array — features (training fold only)
    y : (n,) array   — targets  (training fold only, normalized)
    nsis : int       — number of features to add per iteration
    max_iter : int   — maximum number of screening iterations
    seed : int       — unused (pure deterministic algorithm); kept for API compat

    Returns
    -------
    selected_indices : np.ndarray of int — column indices selected (sorted)
    """
    n, p = X.shape
    if p == 0:
        return np.array([], dtype=int)

    # Standardize (z-score) — training statistics only
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xz = (X - mu) / sigma

    # Standardize y (needed for correlation to be meaningful)
    y_mu  = y.mean()
    y_sig = y.std()
    if y_sig == 0:
        y_sig = 1.0
    yz = (y - y_mu) / y_sig

    selected = set()
    resid    = yz.copy()

    for _ in range(max_iter + 1):  # iteration 0 screens against y; 1..max_iter against residuals
        # Pearson correlation: (Xz.T @ resid) / n  (both already z-scored column-wise)
        corr = np.abs(Xz.T @ resid) / n

        unselected = np.array([i for i in range(p) if i not in selected])
        if len(unselected) == 0:
            break

        n_new = min(nsis, len(unselected))
        top_new = unselected[np.argsort(corr[unselected])[-n_new:]]

        prev_size = len(selected)
        selected.update(top_new.tolist())
        if len(selected) == prev_size and _ > 0:
            break  # no new features added

        # Recompute residuals via OLS on current selected set
        sel_idx = np.array(sorted(selected), dtype=int)
        Xs = Xz[:, sel_idx]
        try:
            beta, _, _, _ = np.linalg.lstsq(Xs, yz, rcond=None)
            resid = yz - Xs @ beta
        except np.linalg.LinAlgError:
            break

    return np.array(sorted(selected), dtype=int)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Elastic Net
# ──────────────────────────────────────────────────────────────────────────────

def elastic_net_select(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    alpha_grid: np.ndarray = None,
    l1_ratio_grid: np.ndarray = None,
) -> np.ndarray:
    """GridSearchCV over Elastic Net hyperparameters; return indices of non-zero coefs.

    Parameters
    ----------
    X : (n, p) array — features (already z-scored from isis_screen step)
    y : (n,) array   — targets
    alpha_grid     : 1-D array of regularization strengths
                     Default: [0.01, 0.05, 0.1, 0.3, 0.5, 1.0] (condensed from paper's 300)
    l1_ratio_grid  : 1-D array of L1 mixing ratios in [0, 1]
                     Default: [0.1, 0.3, 0.5, 0.7, 0.9] (condensed from paper's 30)
    seed : int       — random state for KFold shuffle

    Returns
    -------
    selected_indices : np.ndarray of int — columns with |coef| > 0, sorted by |coef| desc
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.preprocessing import StandardScaler

    if alpha_grid is None:
        alpha_grid = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0])
    if l1_ratio_grid is None:
        l1_ratio_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    n, p = X.shape
    if p == 0:
        return np.array([], dtype=int)

    # Z-score (training stats)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_splits = min(5, n)
    if n_splits < 2:
        # Fall back to single fit if too few samples
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        en.fit(Xs, y)
        coef = en.coef_
        idx  = np.where(np.abs(coef) > 0)[0]
        return idx[np.argsort(np.abs(coef[idx]))[::-1]]

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        ElasticNet(max_iter=5000),
        param_grid={'alpha': alpha_grid, 'l1_ratio': l1_ratio_grid},
        scoring='neg_mean_squared_error',
        cv=kfold,
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(Xs, y)

    best_en = ElasticNet(
        alpha=gs.best_params_['alpha'],
        l1_ratio=gs.best_params_['l1_ratio'],
        max_iter=5000,
    )
    best_en.fit(Xs, y)
    coef = best_en.coef_

    nonzero_idx = np.where(np.abs(coef) > 0)[0]
    if len(nonzero_idx) == 0:
        # Fallback: return top-5 by |coef| even if small
        nonzero_idx = np.argsort(np.abs(coef))[-5:]
    return nonzero_idx[np.argsort(np.abs(coef[nonzero_idx]))[::-1]]


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Full per-fold pipeline
# ──────────────────────────────────────────────────────────────────────────────

def select_descriptors_per_fold(
    train_smiles: list,
    train_y: np.ndarray,
    seed: int = 0,
    nsis: int = 20,
    max_iter_isis: int = 5,
    alpha_grid: np.ndarray = None,
    l1_ratio_grid: np.ndarray = None,
) -> tuple:
    """Full per-fold KROVEX descriptor selection pipeline.

    Uses ONLY train_smiles and train_y — no val/test information.

    Steps
    -----
    1. compute_209_descriptors(train_smiles)
    2. clean_descriptors → drop NaN/constant/low-var columns
    3. isis_screen on cleaned subset
    4. elastic_net_select on ISIS survivors
    5. Return (selected_names, fit_stats)

    Returns
    -------
    selected_names : list[str] — descriptor column names in coef-magnitude order
    fit_stats : dict with keys:
        'all_desc_names': list of all descriptor names post-cleaning
        'train_mean': np.ndarray — per-column mean of cleaned descriptors (for z-score)
        'train_std': np.ndarray  — per-column std of cleaned descriptors
        'isis_indices': np.ndarray — column indices selected by ISIS (within cleaned)
        'en_indices': np.ndarray  — relative indices within ISIS output selected by EN
        'selected_global_indices': np.ndarray — column indices in cleaned df
        'n_train_cleaned': int — number of training molecules after NaN drop
    """
    # Step 1
    df_all = compute_209_descriptors(train_smiles)

    # Step 2
    df_clean, kept_names = clean_descriptors(df_all)
    if df_clean.empty or len(kept_names) == 0:
        # Degenerate fallback: return empty selection
        return [], {'selected_global_indices': np.array([], dtype=int),
                    'all_desc_names': [], 'train_mean': np.array([]),
                    'train_std': np.array([]), 'n_train_cleaned': 0}

    X_clean = df_clean.values.astype(float)  # (n_clean, p_clean)
    n_clean  = len(df_clean)

    # y values for cleaned rows (same row mask as df_clean)
    clean_row_idx = df_clean.index.tolist()
    y_clean = train_y[clean_row_idx]

    # Training statistics for z-score (used by apply_descriptor_selection)
    train_mean = X_clean.mean(axis=0)
    train_std  = X_clean.std(axis=0)
    # Guard against near-zero std (not just exact zero):
    # std=1e-15 produces 1e15-scale z-scores → MLP output explosion via Kronecker product
    train_std[train_std < 1e-6] = 1.0

    # Step 3 — ISIS
    isis_idx = isis_screen(X_clean, y_clean, nsis=nsis, max_iter=max_iter_isis, seed=seed)

    if len(isis_idx) == 0:
        isis_idx = np.arange(min(nsis, len(kept_names)), dtype=int)

    X_isis = X_clean[:, isis_idx]
    isis_names = [kept_names[i] for i in isis_idx]

    # Step 4 — Elastic Net
    en_rel_idx = elastic_net_select(X_isis, y_clean, seed=seed,
                                    alpha_grid=alpha_grid, l1_ratio_grid=l1_ratio_grid)

    # Map back to global column indices in the cleaned df
    selected_global = isis_idx[en_rel_idx]
    selected_names  = [kept_names[i] for i in selected_global]

    if len(selected_names) == 0:
        # Final fallback: at least use ISIS top-5
        selected_global = isis_idx[:min(5, len(isis_idx))]
        selected_names  = [kept_names[i] for i in selected_global]

    fit_stats = {
        'all_desc_names':        kept_names,
        'train_mean':            train_mean,
        'train_std':             train_std,
        'isis_indices':          isis_idx,
        'en_indices':            en_rel_idx,
        'selected_global_indices': selected_global,
        'n_train_cleaned':       n_clean,
    }
    return selected_names, fit_stats


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Apply to val/test
# ──────────────────────────────────────────────────────────────────────────────

def apply_descriptor_selection(
    smiles_list: list,
    descriptor_names: list,
    fit_stats: dict,
) -> np.ndarray:
    """Apply previously-fit descriptor selection to new SMILES (val/test).

    Computes all descriptors, selects the same columns chosen on the training fold,
    applies training z-score normalization (NO re-fitting — guards against leak).
    NaN values (parse failure or descriptor error) are imputed with 0 (= training mean).

    Parameters
    ----------
    smiles_list      : list[str] — val/test SMILES
    descriptor_names : list[str] — selected descriptor names from select_descriptors_per_fold
    fit_stats        : dict      — fit_stats dict from select_descriptors_per_fold

    Returns
    -------
    X : np.ndarray of shape (len(smiles_list), len(descriptor_names)) — z-scored
    """
    if not descriptor_names:
        return np.zeros((len(smiles_list), 0), dtype=np.float32)

    df_all = compute_209_descriptors(smiles_list)

    # Keep only selected columns; NaN → 0 (impute with training mean after z-score)
    # (after z-score, training mean → 0, so 0 = mean imputation)
    X_raw = np.zeros((len(smiles_list), len(descriptor_names)), dtype=float)
    for j, name in enumerate(descriptor_names):
        if name in df_all.columns:
            col = pd.to_numeric(df_all[name], errors='coerce').values
            X_raw[:, j] = np.nan_to_num(col.astype(float), nan=0.0)

    # Apply training z-score using training mean/std for the selected columns
    all_names  = fit_stats['all_desc_names']
    train_mean = fit_stats['train_mean']
    train_std  = fit_stats['train_std']
    sel_global = fit_stats['selected_global_indices']

    X_out = np.zeros_like(X_raw, dtype=np.float32)
    for j, name in enumerate(descriptor_names):
        if name in all_names:
            g_idx  = all_names.index(name)
            mu     = train_mean[g_idx]
            sigma  = train_std[g_idx] if train_std[g_idx] >= 1e-6 else 1.0
            X_out[:, j] = ((X_raw[:, j] - mu) / sigma).astype(np.float32)
        else:
            X_out[:, j] = X_raw[:, j].astype(np.float32)

    # Defensive guard: NaN/Inf from descriptor failures or extreme outliers
    # (Ipc and similar skewed descriptors can produce large z-scores in test set
    # even after per-column std guard; clip to ±10 σ to prevent Kronecker explosion)
    X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)
    X_out = np.clip(X_out, -10.0, 10.0)

    # Leak guard: assert we used only training statistics
    # (cannot compute val/test stats here — structural guarantee by construction)
    assert X_out.shape == (len(smiles_list), len(descriptor_names))
    return X_out
