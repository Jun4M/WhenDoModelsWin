"""
Tests for train_unimol_pretrained() — pretrained UniMol fine-tuning.

All CUDA-dependent tests are skipped on macOS / CPU-only machines.
Run on Colab L4:
    pytest tests/test_unimol_pretrained_smoke.py -v
"""
import numpy as np
import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")

try:
    from unimol_tools import MolTrain, MolPredict  # noqa: F401
    UNIMOL_TOOLS_AVAILABLE = True
except ImportError:
    UNIMOL_TOOLS_AVAILABLE = False

skip_no_unimol = pytest.mark.skipif(
    not UNIMOL_TOOLS_AVAILABLE,
    reason="unimol-tools not installed"
)

# ---------------------------------------------------------------------------
# Shared tiny dataset (ESOL-like, 60 molecules)
# ---------------------------------------------------------------------------

_SMILES = [
    "c1ccccc1", "CC(=O)O", "CCO", "c1ccncc1", "CC(C)O",
    "CCCC", "CCC(=O)O", "c1ccc(N)cc1", "CC(N)=O", "CCCO",
    "c1ccc(O)cc1", "CC(C)(C)O", "CCOCCO", "c1cccc2ccccc12", "CCN",
    "CC1=CC=CC=C1", "CCOCC", "c1ccsc1", "CC(=O)c1ccccc1", "CCCCCO",
] * 3  # 60 SMILES


def _make_data(n=50, seed=42):
    rng = np.random.default_rng(seed)
    smiles = _SMILES[:n]
    y = rng.standard_normal(n).astype(np.float32)
    split = int(0.8 * n)
    return smiles[:split], y[:split], smiles[split:], y[split:]


# ---------------------------------------------------------------------------
# Test 1 (CUDA): smoke fine-tune completes, returns expected dict shape
# ---------------------------------------------------------------------------

@skip_no_cuda
@skip_no_unimol
def test_smoke_finetune_runs():
    """2-epoch ESOL fine-tune completes without error, returns expected dict."""
    from src.train import train_unimol_pretrained

    train_smi, train_y, test_smi, test_y = _make_data(n=50)

    result = train_unimol_pretrained(
        train_smiles=train_smi, train_y=train_y,
        val_smiles=test_smi,   val_y=test_y,   # ignored by unimol-tools
        test_smiles=test_smi,  test_y=test_y,
        target_name='solubility',
        epochs=2,
        batch_size=16,
        patience=5,
        device='cuda',
        seed=0,
    )

    assert isinstance(result, dict)
    assert set(result.keys()) >= {'model', 'metrics', 'test_preds', 'test_true'}
    assert len(result['test_preds']) == len(test_smi)
    assert len(result['test_true']) == len(test_y)
    assert not np.any(np.isnan(result['test_preds'])), "test_preds contains NaN"

    m = result['metrics']
    assert 'RMSE' in m and 'R2' in m
    assert np.isfinite(m['RMSE'])


# ---------------------------------------------------------------------------
# Test 2 (CUDA): seed determinism — same seed → RMSE delta < 0.01
# ---------------------------------------------------------------------------

@skip_no_cuda
@skip_no_unimol
def test_seed_determinism():
    """Same seed run twice → RMSE difference < 0.01.

    Note: unimol-tools may retain non-determinism from CUDA kernels even with
    seeding. We use a loose threshold (0.01) and only flag gross divergence.
    """
    from src.train import train_unimol_pretrained

    train_smi, train_y, test_smi, test_y = _make_data(n=50)
    kwargs = dict(
        train_smiles=train_smi, train_y=train_y,
        val_smiles=test_smi,   val_y=test_y,
        test_smiles=test_smi,  test_y=test_y,
        target_name='solubility',
        epochs=2, batch_size=16, patience=5, device='cuda', seed=42,
    )

    r1 = train_unimol_pretrained(**kwargs)
    r2 = train_unimol_pretrained(**kwargs)

    delta = abs(r1['metrics']['RMSE'] - r2['metrics']['RMSE'])
    assert delta < 0.01, (
        f"RMSE drift between same-seed runs: {delta:.4f} "
        f"(run1={r1['metrics']['RMSE']:.4f}, run2={r2['metrics']['RMSE']:.4f}). "
        "Investigate unimol-tools seed coverage."
    )


# ---------------------------------------------------------------------------
# Test 3 (CUDA): no double-normalize — predictions stay in normalized scale
# ---------------------------------------------------------------------------

@skip_no_cuda
@skip_no_unimol
def test_no_double_normalize():
    """Verify target_normalize=False actually disables internal scaling.

    Strategy: pass pre-normalized data (mean≈0, std≈1). If internal scaling
    fires, predictions will be near-zero (scaled twice) or wildly large.
    We check predictions stay within a ±10 band around the true values' range.

    This is the most critical correctness test — silent double-normalize would
    produce plausible-looking but systematically wrong results.
    """
    from src.train import train_unimol_pretrained

    rng = np.random.default_rng(0)
    n = 50
    smiles = _SMILES[:n]
    # Pre-normalized: mean=0, std=1 (mimics our z-score protocol)
    y = rng.standard_normal(n).astype(np.float32)
    split = 40

    result = train_unimol_pretrained(
        train_smiles=smiles[:split], train_y=y[:split],
        val_smiles=smiles[split:],   val_y=y[split:],
        test_smiles=smiles[split:],  test_y=y[split:],
        target_name='normalized_target',
        epochs=2, batch_size=16, patience=5, device='cuda', seed=0,
    )

    preds = result['test_preds']
    # Predictions should be in roughly the same range as y (mean≈0, std~1)
    assert np.abs(preds).max() < 10.0, (
        f"Predictions out of normalized range (max |pred|={np.abs(preds).max():.2f}). "
        "Likely double-normalization: check target_normalize=False in MolTrain."
    )
    assert preds.std() > 1e-3, (
        f"Predictions collapsed to near-constant (std={preds.std():.4f}). "
        "Model may not have learned, or double-normalization squashed outputs."
    )


# ---------------------------------------------------------------------------
# Test 4 (CPU): CUDA guard fires on non-CUDA device
# ---------------------------------------------------------------------------

def test_cuda_required_guard():
    """CPU/MPS device → RuntimeError raised before any compute (no unimol-tools needed)."""
    from src.train import train_unimol_pretrained

    train_smi, train_y, test_smi, test_y = _make_data(n=10)

    with pytest.raises(RuntimeError, match="CUDA"):
        train_unimol_pretrained(
            train_smiles=train_smi, train_y=train_y,
            val_smiles=test_smi,   val_y=test_y,
            test_smiles=test_smi,  test_y=test_y,
            target_name='solubility',
            epochs=1, device='cpu', seed=0,
        )
