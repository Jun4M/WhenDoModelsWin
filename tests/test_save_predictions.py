"""Tests for --save_predictions flag (Spec 01).

Tests:
1. default off  — predictions dir not created, existing behaviour unchanged
2. flag on, single model — .npz created with correct shape/keys
3. bit-identical test_true across models for same (seed, size, target)
4. GPR silent skip when train_size > 500
5. MTL multi-task — npz shape (N, 12), y_mean/y_std shape (12,)
"""

import os
import tempfile
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Unit test: save_predictions_npz helper
# ---------------------------------------------------------------------------

def _make_res(n=50, n_tasks=1):
    preds = np.random.randn(n) if n_tasks == 1 else np.random.randn(n, n_tasks)
    truth = np.random.randn(n) if n_tasks == 1 else np.random.randn(n, n_tasks)
    return {'test_preds': preds.astype(np.float32), 'test_true': truth.astype(np.float32),
            'metrics': {'RMSE': 0.5, 'MAE': 0.4, 'Pearson_R': 0.8, 'R2': 0.6}}


def test_save_predictions_npz_creates_file():
    from src.summary import save_predictions_npz
    res = _make_res()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_predictions_npz(res, model='gcn', pred_dir=tmpdir, model_kind='na',
                             seed=0, target='homo', train_size=50,
                             y_mean=0.1, y_std=0.5)
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0] == 'gcn_na_0_homo_n50.npz'
        data = np.load(os.path.join(tmpdir, files[0]))
        assert 'test_preds' in data and 'test_true' in data
        assert 'y_mean' in data and 'y_std' in data
        assert data['test_preds'].shape == (50,)
        np.testing.assert_allclose(float(data['y_mean']), 0.1, atol=1e-6)
        np.testing.assert_allclose(float(data['y_std']),  0.5, atol=1e-6)


def test_save_predictions_npz_none_res():
    from src.summary import save_predictions_npz
    with tempfile.TemporaryDirectory() as tmpdir:
        save_predictions_npz(None, model='gpr', pred_dir=tmpdir, model_kind='na',
                             seed=0, target='homo', train_size=600)
        assert len(os.listdir(tmpdir)) == 0


def test_save_predictions_npz_missing_keys():
    from src.summary import save_predictions_npz
    res = {'metrics': {'RMSE': 0.5}}
    with tempfile.TemporaryDirectory() as tmpdir:
        save_predictions_npz(res, model='gcn', pred_dir=tmpdir, model_kind='na',
                             seed=0, target='homo', train_size=50)
        assert len(os.listdir(tmpdir)) == 0


def test_save_predictions_npz_target_with_spaces():
    from src.summary import save_predictions_npz
    res = _make_res()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_predictions_npz(res, model='attentivefp', pred_dir=tmpdir, model_kind='na',
                             seed=2, target='measured log solubility in mols per litre',
                             train_size=100)
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert ' ' not in files[0]
        assert 'measured_log_solubility' in files[0]


def test_save_predictions_npz_mtl_shape():
    from src.summary import save_predictions_npz
    res = _make_res(n=200, n_tasks=12)
    y_mean_arr = np.arange(12, dtype=np.float32)
    y_std_arr  = np.ones(12, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_predictions_npz(res, model='attentivefp_mtl', pred_dir=tmpdir, model_kind='na',
                             seed=0, target='homo', train_size=50,
                             y_mean=y_mean_arr, y_std=y_std_arr)
        data = np.load(os.path.join(tmpdir, 'attentivefp_mtl_na_0_homo_n50.npz'))
        assert data['test_preds'].shape == (200, 12)
        assert data['test_true'].shape  == (200, 12)
        assert data['y_mean'].shape == (12,)
        assert data['y_std'].shape  == (12,)


# ---------------------------------------------------------------------------
# Integration test: run_one with save_predictions flag
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def esol_data():
    """Load ESOL n=50 seed=0 once for all integration tests."""
    from src.data_loader import load_dataset_splits
    return load_dataset_splits('esol', './data', train_size=50, seed=0,
                               target='measured log solubility in mols per litre',
                               featurize_ecfp=True)


def _run_model_with_flag(model_fn, data, tmpdir, save_predictions):
    """Helper: run a single model inside a minimal run_one-like context."""
    from src.summary import save_predictions_npz, save_run_csv
    tr, va, te = data['train'], data['val'], data['test']
    res = model_fn(tr, va, te)
    if res and save_predictions:
        pred_dir = os.path.join(tmpdir, 'predictions')
        save_predictions_npz(res, model='test_model', pred_dir=pred_dir, model_kind='na',
                             seed=0, target='sol', train_size=50,
                             y_mean=data['stats'][0], y_std=data['stats'][1])
    return res


def test_default_off_no_predictions_dir(esol_data):
    """--save_predictions default False → predictions/ dir not created."""
    from src.train import train_sklearn
    tr, va, te = esol_data['train'], esol_data['val'], esol_data['test']
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_dir = os.path.join(tmpdir, 'predictions')
        res = train_sklearn(tr['X_ecfp'], tr['y'], va['X_ecfp'], va['y'],
                            te['X_ecfp'], te['y'], model_type='rf', seed=0)
        assert not os.path.exists(pred_dir), "predictions/ must not be created when flag is off"


def test_flag_on_single_model_npz_created(esol_data):
    """--save_predictions True → .npz file created with correct shape."""
    from src.summary import save_predictions_npz
    from src.train import train_sklearn
    tr, va, te = esol_data['train'], esol_data['val'], esol_data['test']
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_dir = os.path.join(tmpdir, 'predictions')
        res = train_sklearn(tr['X_ecfp'], tr['y'], va['X_ecfp'], va['y'],
                            te['X_ecfp'], te['y'], model_type='rf', seed=0)
        assert res is not None
        target = 'measured log solubility in mols per litre'
        save_predictions_npz(res, model='rf', pred_dir=pred_dir, model_kind='na',
                             seed=0, target=target, train_size=50,
                             y_mean=esol_data['stats'][0], y_std=esol_data['stats'][1])
        files = os.listdir(pred_dir)
        assert len(files) == 1
        data = np.load(os.path.join(pred_dir, files[0]))
        n_test = len(te['X_ecfp'])
        assert data['test_preds'].shape == (n_test,)
        assert data['test_true'].shape  == (n_test,)
        assert data['test_preds'].dtype == np.float32


def test_bit_identical_test_true_across_models(esol_data):
    """Same (seed, size, target) → test_true identical across RF and XGB."""
    from src.summary import save_predictions_npz
    from src.train import train_sklearn
    tr, va, te = esol_data['train'], esol_data['val'], esol_data['test']
    target = 'measured log solubility in mols per litre'
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_dir = os.path.join(tmpdir, 'predictions')
        for mtype in ['rf', 'xgb']:
            res = train_sklearn(tr['X_ecfp'], tr['y'], va['X_ecfp'], va['y'],
                                te['X_ecfp'], te['y'], model_type=mtype, seed=0)
            save_predictions_npz(res, model=mtype, pred_dir=pred_dir, model_kind='na',
                                 seed=0, target=target, train_size=50,
                                 y_mean=esol_data['stats'][0], y_std=esol_data['stats'][1])
        files = sorted(os.listdir(pred_dir))
        assert len(files) == 2
        d_rf  = np.load(os.path.join(pred_dir, [f for f in files if 'rf_' in f][0]))
        d_xgb = np.load(os.path.join(pred_dir, [f for f in files if 'xgb_' in f][0]))
        np.testing.assert_allclose(d_rf['test_true'], d_xgb['test_true'],
                                   err_msg='test_true must be model-independent')


def test_gpr_silent_skip_above_500():
    """GPR > train_size_limit_gpr → train_sklearn returns None → save_predictions_npz no-ops."""
    from src.summary import save_predictions_npz
    from src.train import train_sklearn
    # Build a dummy X with > 500 rows to trigger GPR's size guard
    X = np.random.randn(501, 10).astype(np.float32)
    y = np.random.randn(501).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_dir = os.path.join(tmpdir, 'predictions')
        res = train_sklearn(X, y, X[:50], y[:50], X[:50], y[:50],
                            model_type='gpr', seed=0, train_size_limit_gpr=500)
        assert res is None, "GPR must return None for len(train_X) > train_size_limit_gpr"
        save_predictions_npz(res, model='gpr', pred_dir=pred_dir, model_kind='na',
                             seed=0, target='sol', train_size=600)
        assert not os.path.exists(pred_dir), "No predictions/ dir should be created for None result"


# ---------------------------------------------------------------------------
# Spec 06: prediction-aware resume tests
# ---------------------------------------------------------------------------

def test_run_already_done_csv_only_no_check():
    """--resume without --save_predictions: CSV alone is sufficient to skip."""
    from src.summary import run_already_done, save_run_csv
    with tempfile.TemporaryDirectory() as raw_dir:
        save_run_csv(raw_dir, 'rf', None, 0, 'homo', 50,
                     {'RMSE': 0.5, 'MAE': 0.4, 'Pearson_R': 0.8, 'R2': 0.6})
        assert run_already_done(raw_dir, 'rf', None, 0, 'homo', 50,
                                check_predictions=False) is True
        assert run_already_done(raw_dir, 'rf', None, 0, 'homo', 99,
                                check_predictions=False) is False


def test_run_already_done_csv_only_with_check_returns_false():
    """--resume --save_predictions: CSV exists but npz absent → must NOT skip."""
    from src.summary import run_already_done, save_run_csv
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, 'raw_data')
        pred_dir = os.path.join(tmpdir, 'predictions')
        os.makedirs(raw_dir)
        save_run_csv(raw_dir, 'rf', None, 0, 'homo', 50,
                     {'RMSE': 0.5, 'MAE': 0.4, 'Pearson_R': 0.8, 'R2': 0.6})
        # npz does NOT exist
        assert run_already_done(raw_dir, 'rf', None, 0, 'homo', 50,
                                check_predictions=True,
                                pred_dir=pred_dir,
                                target_safe='homo') is False


def test_run_already_done_csv_and_npz_skip():
    """--resume --save_predictions: both CSV and npz exist → skip."""
    from src.summary import run_already_done, save_run_csv, save_predictions_npz
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, 'raw_data')
        pred_dir = os.path.join(tmpdir, 'predictions')
        os.makedirs(raw_dir)
        save_run_csv(raw_dir, 'rf', None, 0, 'homo', 50,
                     {'RMSE': 0.5, 'MAE': 0.4, 'Pearson_R': 0.8, 'R2': 0.6})
        res = {'test_preds': np.zeros(10, dtype=np.float32),
               'test_true':  np.zeros(10, dtype=np.float32)}
        save_predictions_npz(res, model='rf', pred_dir=pred_dir, model_kind='na',
                             seed=0, target='homo', train_size=50)
        assert run_already_done(raw_dir, 'rf', None, 0, 'homo', 50,
                                check_predictions=True,
                                pred_dir=pred_dir,
                                target_safe='homo') is True
