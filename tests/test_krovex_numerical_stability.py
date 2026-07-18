"""
Regression tests for KROVEX numerical stability fix (spec 10).

Verify that:
1. train_krovex training loop applies gradient clipping
2. descriptor_selection guards against near-zero std
3. apply_descriptor_selection also guards near-zero std from fit_stats
"""
import numpy as np
import pytest


def test_train_krovex_uses_grad_clipping():
    """Source-level check: train_krovex must call clip_grad_norm_."""
    import inspect
    from src.train import train_krovex
    source = inspect.getsource(train_krovex)
    assert 'clip_grad_norm_' in source, \
        'train_krovex missing gradient clipping (root cause of spec 10 issue)'


def test_descriptor_selection_handles_near_zero_std():
    """Near-zero std (1e-15) in training data must not produce inflated z-scores."""
    from src.descriptor_selection import select_descriptors_per_fold, apply_descriptor_selection

    # Build SMILES with one near-constant RDKit descriptor by using nearly-identical mols
    # Use ethanol variants — small, parseable, stable SMILES
    smiles = ['CCO'] * 80 + ['CCCO'] * 20  # 100 total

    np.random.seed(0)
    y = np.random.randn(len(smiles))

    selected_names, fit_stats = select_descriptors_per_fold(
        smiles, y, seed=0, nsis=5,
    )

    if not selected_names:
        pytest.skip('No descriptors selected — isis fallback, skip this test')

    X_norm = apply_descriptor_selection(smiles, selected_names, fit_stats)
    assert np.isfinite(X_norm).all(), 'Normalized features contain NaN/Inf'
    assert np.abs(X_norm).max() < 1e6, \
        f'Z-scores exceeded 1e6 (max={np.abs(X_norm).max():.2e}); std guard insufficient'


def test_apply_descriptor_selection_handles_near_zero_std_in_fit_stats():
    """apply_descriptor_selection with tiny sigma injected into fit_stats must not explode."""
    from src.descriptor_selection import select_descriptors_per_fold, apply_descriptor_selection

    smiles_train = ['CCO'] * 40 + ['CCCO'] * 10
    smiles_test  = ['CCO', 'CCCO', 'CCCCO']
    np.random.seed(1)
    y = np.random.randn(len(smiles_train))

    selected_names, fit_stats = select_descriptors_per_fold(
        smiles_train, y, seed=0, nsis=5,
    )

    if not selected_names:
        pytest.skip('No descriptors selected — skip')

    # Inject near-zero std into fit_stats to simulate the root-cause scenario
    fit_stats_bad = dict(fit_stats)
    bad_std = fit_stats_bad['train_std'].copy()
    bad_std[0] = 1e-15  # force the first selected descriptor to near-zero std
    fit_stats_bad['train_std'] = bad_std

    X_norm = apply_descriptor_selection(smiles_test, selected_names, fit_stats_bad)
    assert np.isfinite(X_norm).all(), 'NaN/Inf with near-zero sigma in fit_stats'
    assert np.abs(X_norm).max() < 1e6, \
        f'Z-scores exploded (max={np.abs(X_norm).max():.2e})'
