"""
tests/test_inline_denormalize.py
==================================
Unit tests for run_learning_curve._apply_denorm().

(a) Math identity: RMSE_n * std == RMSE_raw  (and same for MAE)
(b) stats=None → metric dict unchanged
(c) task_type='classification' → ×std not applied
(d) std=0 → ValueError
(e) std=NaN → ValueError
(f) R² and Pearson_R unchanged by denorm (scale-invariant)

Run:
    pytest tests/test_inline_denormalize.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np
import pytest

from run_learning_curve import _apply_denorm


# ---------------------------------------------------------------------------
# (a) Math identity: RMSE_n * std == RMSE_raw
# ---------------------------------------------------------------------------

def test_math_identity_rmse_mae():
    """
    Generate y_raw with (mean=-1500, std=300), add noise → pred_raw.
    Normalize with the same (mean, std) → y_n, pred_n.
    Compute RMSE/MAE in normalized space, then denorm → must match raw RMSE/MAE.
    """
    rng = np.random.default_rng(seed=42)
    mean_true, std_true = -1500.0, 300.0
    n = 100

    y_raw   = rng.normal(loc=mean_true, scale=std_true, size=n)
    noise   = rng.normal(loc=0.0, scale=50.0, size=n)
    pred_raw = y_raw + noise

    # Raw-space metrics
    rmse_raw = float(np.sqrt(np.mean((y_raw - pred_raw) ** 2)))
    mae_raw  = float(np.mean(np.abs(y_raw - pred_raw)))

    # Normalize
    mean_fit = float(np.mean(y_raw))
    std_fit  = float(np.std(y_raw))
    y_n    = (y_raw   - mean_fit) / std_fit
    pred_n = (pred_raw - mean_fit) / std_fit

    # Normalized-space metrics
    rmse_n = float(np.sqrt(np.mean((y_n - pred_n) ** 2)))
    mae_n  = float(np.mean(np.abs(y_n  - pred_n)))

    metrics_n = {'RMSE': rmse_n, 'MAE': mae_n, 'R2': 0.9, 'Pearson_R': 0.95}
    stats = (mean_fit, std_fit)

    result = _apply_denorm(metrics_n, stats)

    assert abs(result['RMSE'] - rmse_raw) < 1e-10, (
        f"RMSE mismatch: denormed={result['RMSE']:.8f}, raw={rmse_raw:.8f}"
    )
    assert abs(result['MAE'] - mae_raw) < 1e-10, (
        f"MAE mismatch: denormed={result['MAE']:.8f}, raw={mae_raw:.8f}"
    )


# ---------------------------------------------------------------------------
# (b) stats=None → passthrough
# ---------------------------------------------------------------------------

def test_stats_none_passthrough():
    metrics = {'RMSE': 0.5, 'MAE': 0.3, 'R2': 0.8, 'Pearson_R': 0.9}
    result  = _apply_denorm(metrics, stats=None)
    assert result == metrics
    assert result is not metrics   # returns new dict


# ---------------------------------------------------------------------------
# (c) task_type='classification' → ×std not applied
# ---------------------------------------------------------------------------

def test_classification_skips_scaling():
    metrics = {'AUROC': 0.82, 'RMSE': 0.4, 'MAE': 0.3}
    stats   = (0.0, 5.0)   # std=5, would produce large RMSE if applied
    result  = _apply_denorm(metrics, stats, task_type='classification')
    assert result['RMSE'] == 0.4, "RMSE must not be scaled for classification"
    assert result['MAE']  == 0.3, "MAE must not be scaled for classification"
    assert result['AUROC'] == 0.82


# ---------------------------------------------------------------------------
# (d) std=0 → ValueError
# ---------------------------------------------------------------------------

def test_std_zero_raises():
    metrics = {'RMSE': 0.5, 'MAE': 0.3}
    with pytest.raises(ValueError, match="std == 0"):
        _apply_denorm(metrics, stats=(0.0, 0.0))


# ---------------------------------------------------------------------------
# (e) std=NaN → ValueError
# ---------------------------------------------------------------------------

def test_std_nan_raises():
    metrics = {'RMSE': 0.5, 'MAE': 0.3}
    with pytest.raises(ValueError, match="NaN"):
        _apply_denorm(metrics, stats=(0.0, float('nan')))


# ---------------------------------------------------------------------------
# (f) R² and Pearson_R unchanged (scale-invariant)
# ---------------------------------------------------------------------------

def test_scale_invariant_metrics_unchanged():
    """R² and Pearson_R computed from normalized preds == those from raw preds."""
    rng = np.random.default_rng(seed=7)
    n   = 80
    y_raw    = rng.normal(loc=200.0, scale=40.0, size=n)
    pred_raw = y_raw + rng.normal(scale=10.0, size=n)

    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score

    mean_fit = float(np.mean(y_raw))
    std_fit  = float(np.std(y_raw))
    y_n    = (y_raw    - mean_fit) / std_fit
    pred_n = (pred_raw - mean_fit) / std_fit

    r2_raw,  _  = pearsonr(y_raw, pred_raw)
    r2_norm, _  = pearsonr(y_n,   pred_n)
    r2_sk_raw   = r2_score(y_raw, pred_raw)
    r2_sk_norm  = r2_score(y_n,   pred_n)

    # Normalized metrics before denorm
    rmse_n = float(np.sqrt(np.mean((y_n - pred_n)**2)))
    metrics_n = {
        'RMSE':      rmse_n,
        'MAE':       float(np.mean(np.abs(y_n - pred_n))),
        'R2':        r2_sk_norm,
        'Pearson_R': r2_norm,
    }
    stats  = (mean_fit, std_fit)
    result = _apply_denorm(metrics_n, stats)

    # R² and Pearson should be unchanged (scale-invariant)
    assert abs(result['R2']        - metrics_n['R2'])        < 1e-12
    assert abs(result['Pearson_R'] - metrics_n['Pearson_R']) < 1e-12

    # Also confirm raw and normalized Pearson/R² are numerically identical
    assert abs(r2_raw  - r2_norm)   < 1e-10
    assert abs(r2_sk_raw - r2_sk_norm) < 1e-10


# ---------------------------------------------------------------------------
# Adversarial review — 5 silent failure modes
# ---------------------------------------------------------------------------

"""
ADVERSARIAL REVIEW: how _apply_denorm can silently produce wrong results

[1] New metric key added later (e.g. Spearman_R) — not scaled, but could be
    if someone adds it to the ×std block by mistake.
    Detection: for every key NOT in {'RMSE', 'MAE'}, assert result[k] == metrics[k].
    Guard:
        scaled_keys = {'RMSE', 'MAE'}
        for k, v in result.items():
            if k not in scaled_keys:
                assert v == metrics[k], f"{k} should not be scaled"

[2] Classification task accidentally receives ×std because task_type default
    is 'regression' and caller forgets to pass task_type='classification'.
    Detection: assert AUROC-like value is still in [0, 1] after denorm.
    Guard: if metric name contains 'AUC' or 'AUROC', assert result < 2.0.

[3] R²/Pearson accidentally ×std (copy-paste error in the result dict).
    Detection: test_scale_invariant_metrics_unchanged() already guards this.
    Extra guard: assert result['R2'] == metrics['R2'].

[4] stats changes from tuple (mean, std) to dict {'mean': …, 'std': …} in
    a future refactor → unpack `_, std = stats` silently takes dict values.
    Detection: explicitly check type at entry:
        assert isinstance(stats, (tuple, list)) and len(stats) == 2

[5] Empty test set (0 samples) → RMSE=NaN from compute_metrics, then
    denorm propagates NaN and saves it silently if the NaN check is skipped.
    Detection: the existing NaN check after multiplication catches this;
    add a test:
        metrics = {'RMSE': float('nan'), 'MAE': 0.1}
        with pytest.raises(ValueError):
            _apply_denorm(metrics, stats=(0.0, 1.5))
"""
