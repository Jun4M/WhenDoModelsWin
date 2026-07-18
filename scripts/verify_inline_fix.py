"""
scripts/verify_inline_fix.py
==============================
End-to-end verification: inline denormalize fix가 올바른 real-unit RMSE를
생성하는지 기존 archive와 비교한다.

Archive path: results_archive_20260510_pre_unify/01_QM9/raw_data/
  - 존재하지 않으면 SKIP (fail 아님)
  - QM9 homo, model=gcn, train_size=200, seed=0 항목과 비교
  - Adam stochasticity 때문에 절대 일치 불가 → |new/archive - 1| < 0.05 면 PASS

Usage:
    python scripts/verify_inline_fix.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARCHIVE_DIR   = './results_archive_20260510_pre_unify/01_QM9/raw_data'
TARGET_MODEL  = 'gcn'
TARGET_NAME   = 'homo'
TRAIN_SIZE    = 200
SEED          = 0
EPOCHS        = 5           # fast smoke run
RATIO_THRESH  = 0.05        # |new/archive - 1| < 5%


def find_archive_rmse() -> float | None:
    """Return RMSE from archive CSV, or None if not found."""
    if not os.path.exists(ARCHIVE_DIR):
        print(f'[SKIP] Archive dir not found: {ARCHIVE_DIR}')
        return None

    # filename pattern: gcn_na_{seed}_{target}.csv
    fname = f'{TARGET_MODEL}_na_{SEED}_{TARGET_NAME}.csv'
    fpath = os.path.join(ARCHIVE_DIR, fname)
    if not os.path.exists(fpath):
        print(f'[SKIP] Archive file not found: {fpath}')
        return None

    df = pd.read_csv(fpath)
    row = df[df['train_size'] == TRAIN_SIZE]
    if row.empty:
        print(f'[SKIP] train_size={TRAIN_SIZE} not in archive file')
        return None

    rmse = float(row['RMSE'].iloc[0])
    print(f'[archive] RMSE = {rmse:.6f}  (model={TARGET_MODEL}, target={TARGET_NAME}, '
          f'size={TRAIN_SIZE}, seed={SEED})')
    return rmse


def run_inline_experiment() -> float:
    """Run GCN for EPOCHS and return inline-denormed RMSE."""
    from src.data_loader import load_dataset_splits, load_raw_data
    from src.train import train_gcn

    preloaded = load_raw_data('qm9', './data', TARGET_NAME)

    data = load_dataset_splits(
        dataset='qm9',
        data_dir='./data',
        train_size=TRAIN_SIZE,
        val_size=100,
        test_size=10000,
        seed=SEED,
        target=TARGET_NAME,
        preloaded_raw=preloaded,
    )

    tr, va, te = data['train'], data['val'], data['test']
    stats = data['stats']

    res = train_gcn(
        tr['X_graph'], va['X_graph'], te['X_graph'],
        target_name=TARGET_NAME,
        node_feat_dim=tr['X_graph'][0].x.shape[1],
        epochs=EPOCHS,
        device='cpu',
        seed=SEED,
    )

    # Inline denorm (same logic as run_learning_curve._apply_denorm)
    from run_learning_curve import _apply_denorm
    denormed = _apply_denorm(res['metrics'], stats)

    rmse = denormed['RMSE']
    print(f'[inline]  RMSE = {rmse:.6f}  (model={TARGET_MODEL}, target={TARGET_NAME}, '
          f'size={TRAIN_SIZE}, seed={SEED}, epochs={EPOCHS})')
    return rmse


def main():
    print('=== verify_inline_fix.py ===\n')

    archive_rmse = find_archive_rmse()
    if archive_rmse is None:
        print('\nResult: SKIP (no archive to compare against)')
        return

    inline_rmse = run_inline_experiment()

    ratio = inline_rmse / archive_rmse
    diff  = abs(inline_rmse - archive_rmse)
    deviation = abs(ratio - 1.0)

    print(f'\n--- comparison ---')
    print(f'  archive RMSE : {archive_rmse:.6f}')
    print(f'  inline  RMSE : {inline_rmse:.6f}')
    print(f'  ratio        : {ratio:.4f}')
    print(f'  |ratio - 1|  : {deviation:.4f}  (threshold: {RATIO_THRESH})')

    if deviation < RATIO_THRESH:
        print(f'\nResult: PASS  (within {RATIO_THRESH*100:.0f}% of archive)')
    else:
        print(f'\nResult: FAIL')
        raise AssertionError(
            f'Inline RMSE ({inline_rmse:.6f}) deviates from archive '
            f'({archive_rmse:.6f}) by {deviation*100:.1f}% > {RATIO_THRESH*100:.0f}%\n'
            f'  absolute diff = {diff:.6f}'
        )


if __name__ == '__main__':
    main()
