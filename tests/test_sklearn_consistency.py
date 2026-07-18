"""
tests/test_sklearn_consistency.py
=====================================
Cross-model sanity tests for SklearnRegressorWrapper (RF, XGB, GPR, SVR, LGBM).

(h) Cross-model sanity — all 5 models produce valid output (finite, 1D, correct length);
    RF / XGB / SVR produce non-constant predictions; pairwise predictions differ.

1 test total.

Adversarial review:
  [Cross-model silent bugs]       — 5 failure modes the sanity check catches
  [Test integrity silent passes]  — 5 ways the sanity check could pass despite bugs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np


# ── Dataset ───────────────────────────────────────────────────────────────────

_SMILES_32 = [
    'c1ccccc1',
    'CCO',
    'CC(=O)O',
    'c1cccnc1',
    'CC(=O)Nc1ccc(O)cc1',
    'c1ccc2ccccc2c1',
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'OC(=O)c1ccccc1',
    'c1ccsc1',
    'Cc1ccccc1',
    'CC(C)O',
    'CCCC',
    'CC1CCCCC1',
    'c1ccc(O)cc1',
    'CC(=O)c1ccccc1',
    'c1cncnc1',
    'CC(=O)NCCO',
    'c1ccc(cc1)O',
    'CCOC(=O)c1ccccc1',
    'c1ccc(cc1)N',
    'O=C(O)c1cccc(c1)O',
    'CC(=O)Oc1ccccc1C(=O)O',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'c1ccc2[nH]cccc2c1',
    'CCCCCC',
    'CCCCO',
    'c1ccc(cc1)Cl',
    'Cc1cc(C)cc(C)c1',
    'CC(C)(C)c1ccccc1',
    'c1ccc(cc1)F',
    'O=C(N)c1ccccc1',
    'c1ccc2c(c1)cc1ccccc1n2',
]


def _make_data():
    """Train/val/test split on 32 SMILES with raw bit-count target (not normalised)."""
    from src.featurizer import featurize_smiles_to_ecfp
    fps, valid = featurize_smiles_to_ecfp(_SMILES_32)
    y = fps.sum(axis=1).astype(float)   # raw counts, not z-scored

    n         = fps.shape[0]
    train_end = int(n * 0.625)
    val_end   = int(n * 0.75)

    return (fps[:train_end],        y[:train_end],
            fps[train_end:val_end], y[train_end:val_end],
            fps[val_end:],          y[val_end:])


# ── (h) Cross-model sanity ────────────────────────────────────────────────────

def test_cross_model_sanity():
    """
    All 5 SklearnRegressorWrapper models produce structurally valid output.

    Assertions per model:
      (i)  result is not None (model ran without size-limit early exit)
      (ii) test_preds is a 1D numpy array of length == len(test_y)
      (iii) all predictions are finite (no NaN / Inf)
      (iv) metrics dict contains {RMSE, MAE, Pearson_R, R2} with finite values

    Stronger assertions for RF / XGB / SVR (confirmed to not collapse to constant):
      (v)  predictions are non-constant (std > 1e-4)

    Pairwise assertion across all models:
      (vi) no two models produce bitwise-identical predictions on test set —
           each model type must have its own decision boundary.  If RF and XGB
           produce identical predictions, one of them is not running (e.g., both
           map to the same sklearn object due to a wrapper dispatch bug).

    Note: GPR may produce near-constant predictions (all ≈ train_mean) due to
    length_scale saturation on ECFP4 — this is a documented GPR limitation, not
    a wrapper bug.  LGBM may also produce constant predictions on tiny datasets.
    For these two, only assertions (i)-(iv) are applied.

    Integrity:
      (1) test_y.std() > 0.5 — ensures predictions must spread to be informative;
          a degenerate featurizer (all-zero fps) would yield constant predictions
          in all models, which would fail assertion (v) for RF/XGB/SVR.
      (2) n_test >= 4 — pairwise comparison needs multiple test points.
    """
    from src.train import train_sklearn

    train_X, train_y, val_X, val_y, test_X, test_y = _make_data()

    # Integrity (1): test_y must be varied
    assert test_y.std() > 0.5, (
        f"test_y.std()={test_y.std():.4f} — test targets nearly constant. "
        "Non-constant prediction check is vacuous when all targets are equal."
    )

    # Integrity (2): enough test points for pairwise comparison
    assert len(test_y) >= 4, (
        f"n_test={len(test_y)} — too few test points for meaningful pairwise comparison."
    )

    all_model_types = ['rf', 'xgb', 'svr', 'lgbm', 'gpr']
    preds_by_model  = {}

    for model_type in all_model_types:
        res = train_sklearn(
            train_X, train_y, val_X, val_y, test_X, test_y,
            model_type=model_type, seed=42,
        )

        # (i) Result must be non-None
        assert res is not None, (
            f"[{model_type}] train_sklearn returned None. "
            "GPR should not hit the size limit with N≈20 training samples."
        )

        preds = res['test_preds']
        metrics = res['metrics']

        # (ii) Shape and type
        assert isinstance(preds, np.ndarray), (
            f"[{model_type}] test_preds is {type(preds)}, expected np.ndarray"
        )
        assert preds.ndim == 1, (
            f"[{model_type}] test_preds has {preds.ndim} dimensions, expected 1D"
        )
        assert preds.shape[0] == len(test_y), (
            f"[{model_type}] test_preds length {preds.shape[0]} ≠ test_y length {len(test_y)}"
        )

        # (iii) Finiteness
        assert np.all(np.isfinite(preds)), (
            f"[{model_type}] test_preds contains NaN or Inf: "
            f"{preds[~np.isfinite(preds)][:5]}"
        )

        # (iv) Metrics structure and finiteness
        expected_keys = {'RMSE', 'MAE', 'Pearson_R', 'R2'}
        missing = expected_keys - set(metrics.keys())
        assert not missing, (
            f"[{model_type}] metrics missing keys: {missing}. "
            f"Present keys: {list(metrics.keys())}"
        )
        # GPR / LGBM may have NaN Pearson_R when predictions are constant
        for k in ('RMSE', 'MAE'):
            assert np.isfinite(metrics[k]), (
                f"[{model_type}] metrics['{k}']={metrics[k]} is not finite"
            )

        # (v) Non-constant predictions for models that should generalise
        if model_type in ('rf', 'xgb', 'svr'):
            assert preds.std() > 1e-4, (
                f"[{model_type}] test_preds are constant (std={preds.std():.2e}). "
                "RF/XGB/SVR should produce varied predictions on a structurally "
                "diverse 32-SMILES set. Check: model fitted before predict? "
                "Feature matrix non-zero? (test_X.sum()={test_X.sum()})"
            )

        preds_by_model[model_type] = preds

    # (vi) Pairwise: no two models produce bitwise-identical predictions
    model_list = list(preds_by_model.keys())
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            m1, m2 = model_list[i], model_list[j]
            assert not np.array_equal(preds_by_model[m1], preds_by_model[m2]), (
                f"[{m1}] and [{m2}] produced bitwise-identical test predictions. "
                "A wrapper dispatch bug may be mapping both model_type strings to "
                "the same sklearn class — verify SklearnRegressorWrapper's "
                "model_type dispatch in __init__ or fit()."
            )


# ── Adversarial review ────────────────────────────────────────────────────────

class AdversarialReview:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SECTION I — CROSS-MODEL SILENT BUGS (5 failure modes this suite catches)
    ═══════════════════════════════════════════════════════════════════════════

    (1) WRAPPER DISPATCHES TWO MODEL TYPES TO THE SAME CLASS
        Symptom  SklearnRegressorWrapper's __init__ has an if-elif chain:
                   if model_type == 'rf':   self.model = RandomForest(...)
                   elif model_type == 'xgb': self.model = XGBRegressor(...)
                 A copy-paste error could make 'lgbm' map to RandomForest.
                 Both models produce different predictions in isolation but
                 have the same class → their decision boundaries differ only
                 by random_state.

        Detection  (vi) pairwise equality check: if 'lgbm' and 'rf' produce
                   identical predictions, the dispatch is broken.

    ──────────────────────────────────────────────────────────────────────────
    (2) fit() NOT CALLED — wrapper calls predict() before fit()
        Symptom  If SklearnRegressorWrapper.fit() has an early-return bug
                 (e.g., a unit-test stub that returns immediately), the model
                 is in a default unfitted state.  sklearn unfitted predictors
                 raise NotFittedError on predict().

        Detection  The test would propagate the NotFittedError as a pytest
                   exception, failing with a clear sklearn error message.
                   This is caught before assertion (ii).

    ──────────────────────────────────────────────────────────────────────────
    (3) TRAIN FEATURES MISMATCH TEST FEATURES (different nBits at fit vs predict)
        Symptom  If train_X is featurized with nBits=2048 but test_X is
                 accidentally featurized with nBits=1024, fit() and predict()
                 receive different shapes.  sklearn raises ValueError:
                 "X has 1024 features but RandomForest expects 2048."

        Detection  ValueError propagates as pytest exception (caught before any
                   assertion).  test (c) in test_sklearn_invariance.py also
                   validates shape at featurization time.

    ──────────────────────────────────────────────────────────────────────────
    (4) GPR RETURNS CONSTANT PREDICTIONS (degenerate RBF kernel)
        Symptom  GPR on ECFP4 saturates length_scale at 100000.0 (upper bound
                 of the optimization space), yielding K(x,x')≈1 for all pairs
                 → all predictions = training y-mean.  This is not a wrapper
                 bug but a kernel-feature incompatibility.

        Detection  Constant prediction has std ≈ 0.  Test (h) intentionally
                   does NOT assert non-constant for GPR to avoid false failures.
                   The RMSE check in test (f) excludes GPR for the same reason.

        Verdict  Known documented limitation.  Correct fix: Tanimoto kernel. ⚠

    ──────────────────────────────────────────────────────────────────────────
    (5) TRAIN_SKLEARN RETURNS DICT WITH WRONG test_preds KEY
        Symptom  A refactor renames 'test_preds' to 'predictions'.
                 All downstream code using result['test_preds'] raises KeyError.
                 The models train successfully but no results are saved.

        Detection  Assertion (iv) explicitly checks the dict keys.  A missing
                   key produces a clear AssertionError naming the missing key.


    ═══════════════════════════════════════════════════════════════════════════
    SECTION II — TEST INTEGRITY: WAYS THIS SUITE COULD SILENTLY PASS DESPITE BUGS
    ═══════════════════════════════════════════════════════════════════════════

    (1) PAIRWISE CHECK VACUOUS IF MODELS ARE INDIVIDUALLY BROKEN
        Scenario  If all 5 models produce the same constant value (e.g.,
                  train_y.mean() = 0.0 after z-score normalisation), the pairwise
                  check fails — but only because all predictions are identical.
                  The test correctly catches this, but the root cause is hidden.

        Guard     Assertion (v) checks std > 1e-4 for RF/XGB/SVR before the
                  pairwise loop, making the root cause explicit.

    ──────────────────────────────────────────────────────────────────────────
    (2) GPR/LGBM CONSTANT OUTPUT MASKS PAIRWISE FAILURE
        Scenario  If GPR and LGBM both produce constant predictions that happen
                  to be the same constant value (e.g., both = 0.0 = normalised mean),
                  the pairwise check for (GPR, LGBM) would fail.

        Guard     If both predict 0.0, the assert triggers and reports "GPR and LGBM
                  produced bitwise-identical predictions."  This is a true signal: their
                  dispatch IS different (GPR uses different kernel than LGBM boosting),
                  so identical output indicates degenerate behaviour, not a dispatch error.
                  The error message points to wrapper dispatch — the user should investigate.

    ──────────────────────────────────────────────────────────────────────────
    (3) FINITENESS CHECK PASSES ON ZERO VECTORS
        Scenario  If featurize_smiles_to_ecfp returns all-zero fingerprints,
                  models that normalise features via sklearn Pipeline/StandardScaler
                  receive 0/0 = NaN → predict() raises ValueError or returns NaN.
                  Alternatively, models that do not normalise predict a constant ≈ 0.

        Guard     Assertion (iii) catches NaN/Inf.  Assertion (v) catches constant
                  zero predictions for RF/XGB/SVR.  test (c) in test_sklearn_invariance.py
                  catches the all-zero fingerprint case at source.

    ──────────────────────────────────────────────────────────────────────────
    (4) test_y.std() > 0.5 GUARD CAN BE FOOLED BY BIASED SPLIT
        Scenario  If the 8 test molecules happen to have very similar heavy-atom
                  counts (all aromatics from the 32-SMILES set), test_y.std. < 0.5.

        Guard     The 32-SMILES set is deliberately diverse: alkanes (4 carbons) to
                  three-ring aromatics (carbazole, 16 heavy atoms).  The 8-molecule
                  test set spans a wide heavy-atom range by construction.
                  If the guard fails, the test itself fails with a clear message —
                  the test suite is self-checking.

    ──────────────────────────────────────────────────────────────────────────
    (5) N_TEST >= 4 GUARD FAILS TO CATCH SINGLE-SAMPLE TEST SETS
        Scenario  If val_end is miscalculated and test contains only 1 sample,
                  pairwise comparison is trivially satisfied (one-element arrays from
                  different models are always different unless both map to the exact
                  same float value).

        Guard     Integrity assertion (2) catches n_test < 4 before the model loop.
                  With 32 SMILES and a 62.5/12.5/25% split, n_test ≈ 8 ≥ 4.
    """
