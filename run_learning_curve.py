"""
run_learning_curve.py
Multi-dataset, multi-model, multi-seed adaptive learning curve experiment.

Adaptive train sizes:
  50~500  : step 25
  600~1000: step 100
  1500~3000: step 500

Seed schedule:
  train_size <= 500 : 10 seeds (0-9)
  train_size >  500 : 3  seeds (0-2)

Models (baselines):
  gcn, transformer, rf, xgb, gpr, attentivefp, painn, gps

Results structure:
  results/
  └── 01_QM9/
      ├── raw_data/   {model}_{depth}_{seed}_{target}.csv
      ├── summary/    summary_baselines.csv
      └── plots/      plot_baselines_lc_{target}.png

Usage:
  python run_learning_curve.py --dataset qm9 --target homo lumo gap --device mps
  python run_learning_curve.py --dataset esol --target all --device mps --resume
  python run_learning_curve.py --dataset qm9 --skip_painn --skip_gps \\
      --epochs_gcn 5 --epochs_transformer 5 --epochs_gtca 5  # smoke test
"""

import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset_splits, load_raw_data, DATASET_CONFIGS
from src.train import (
    train_gcn, train_transformer, train_chemberta2, train_molformer, train_selformer,
    train_attentivefp, train_attentivefp_mtl, train_gcn_mtl,
    train_gps, train_sklearn, train_chemprop, train_krovex, train_unimol,
    train_unimol_pretrained,
)
from src.analysis import save_failure_data_csv, group_analysis
from src.summary import (
    save_run_csv, run_already_done, rebuild_summary_baselines,
    save_predictions_npz, BASELINE_MODELS,
)
from src.visualization import plot_baselines_lc, plot_group_analysis


# ---------------------------------------------------------------------------
# Inline denormalize helper
# ---------------------------------------------------------------------------

def _apply_denorm(metrics: dict, stats, task_type: str = 'regression') -> dict:
    """
    Multiply RMSE and MAE by train std to convert normalized → real units.

    stats  : (mean, std) tuple from load_dataset_splits, or None to skip.
    task_type : 'classification' skips scaling (AUROC etc. are unit-free).
    Raises ValueError on std == 0 or NaN (silent corruption prevention).
    Returns a new dict; does not mutate the input.
    """
    import math
    if stats is None or task_type == 'classification':
        return dict(metrics)
    _, std = stats
    std = float(std)
    if math.isnan(std):
        raise ValueError(f"std is NaN — cannot denormalize metrics")
    if std == 0.0:
        raise ValueError(f"std == 0 — zero-variance target, cannot denormalize")
    result = dict(metrics)
    result['RMSE'] = metrics['RMSE'] * std
    result['MAE']  = metrics['MAE']  * std
    # R2, Pearson_R, Spearman are scale-invariant — left unchanged
    if math.isnan(result['RMSE']) or math.isnan(result['MAE']):
        raise ValueError(
            f"NaN after denorm (RMSE={result['RMSE']}, MAE={result['MAE']})"
        )
    return result


# ---------------------------------------------------------------------------
# Adaptive train sizes
# ---------------------------------------------------------------------------

DATASET_MAX_TRAIN = {
    'qm9':  3000,
    'esol':  375,
    'lipo': 1000,
    'bace':  500,
}

def get_train_sizes(dataset: str = 'qm9') -> list:
    max_size = DATASET_MAX_TRAIN.get(dataset, 3000)
    sizes = list(range(50, min(501, max_size + 1), 25))
    if max_size >= 600:
        sizes += list(range(600, min(1001, max_size + 1), 100))
    if max_size >= 1500:
        sizes += list(range(1500, max_size + 1, 500))
    return sorted(set(sizes))


def get_seed_schedule(train_size: int) -> list:
    return list(range(10)) if train_size <= 500 else list(range(3))


# ---------------------------------------------------------------------------
# Dataset directory mapping
# ---------------------------------------------------------------------------

DATASET_DIRS = {
    'qm9':  '01_QM9',
    'esol': '02_ESOL',
    'lipo': '03_Lipo',
    'bace': '04_BACE',
}


# ---------------------------------------------------------------------------
# Single experiment: one (train_size, target, seed, dataset)
# ---------------------------------------------------------------------------

def run_one(
    train_size: int,
    target: str,
    seed: int,
    dataset: str,
    raw_dir: str,
    data_dir: str,
    device: str,
    epochs_transformer: int,
    epochs_chemberta2: int,
    epochs_molformer: int,
    epochs_selformer: int,
    epochs_chemprop: int,
    epochs_krovex: int,
    epochs_gcn: int,
    epochs_gcn_layers: int,
    epochs_attfp: int,
    epochs_gps: int,
    skip_transformer: bool,
    skip_chemberta2: bool,
    skip_molformer: bool,
    skip_selformer: bool,
    skip_chemprop: bool,
    skip_krovex: bool,
    skip_gcn: bool,
    skip_rf: bool,
    skip_xgb: bool,
    skip_gpr: bool,
    skip_svr: bool,
    skip_lgbm: bool,
    skip_attentivefp: bool,
    skip_painn: bool,
    skip_gps: bool,
    skip_unimol: bool = True,
    epochs_unimol: int = 300,
    epochs_painn: int = 300,
    enable_unimol_pretrained: bool = False,
    epochs_unimol_pretrained: int = 50,
    enable_attentivefp_mtl: bool = False,
    epochs_attentivefp_mtl: int = 300,
    enable_gcn_mtl: bool = False,
    epochs_gcn_mtl: int = 300,
    log_dir: str = None,
    resume: bool = False,
    preloaded_raw=None,  # (smiles, y_col, task_pos) pre-loaded once per target
    save_predictions: bool = False,
    pred_dir: str = None,   # {results_root}/{ds_dir}/predictions/
    gpr_max_train_size: int = 500,
) -> dict:

    import gc

    def clear_memory():
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    need_ecfp   = not (skip_rf and skip_xgb and skip_gpr and skip_svr and skip_lgbm)
    need_3d     = not skip_painn
    need_unimol = not skip_unimol

    data = load_dataset_splits(
        dataset=dataset,
        data_dir=data_dir,
        train_size=train_size,
        val_size=100,
        test_size=10000,
        seed=seed,
        target=target,
        featurize_ecfp=need_ecfp,
        featurize_3d=need_3d,
        featurize_unimol=need_unimol,
        preloaded_raw=preloaded_raw,
    )

    tr, va, te = data['train'], data['val'], data['test']
    train_pyg, val_pyg, test_pyg = tr['X_graph'], va['X_graph'], te['X_graph']
    train_smi, val_smi, test_smi = tr['ids'], va['ids'], te['ids']
    train_y, val_y, test_y       = tr['y'], va['y'], te['y']

    # Inline denormalize: wrap stats so every save_run_csv call gets real-unit metrics
    _stats     = data['stats']
    _task_type = data.get('task_type', 'regression')
    def denorm(metrics):
        return _apply_denorm(metrics, _stats, _task_type)

    node_feat_dim = train_pyg[0].x.shape[1] if train_pyg else 30
    edge_dim      = train_pyg[0].edge_attr.shape[1] if (train_pyg and train_pyg[0].edge_attr is not None) else 11
    n_test        = len(test_pyg)

    target_safe = target.replace(' ', '_').replace('/', '_')

    results = {}

    def log_path_for(model_name):
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            return os.path.join(log_dir, f"{model_name}_na_{seed}_{target}_traininglog.csv")
        return None

    # GCN
    if not skip_gcn and not (resume and run_already_done(raw_dir, 'gcn', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_gcn(
                train_pyg, val_pyg, test_pyg,
                target_name=target, node_feat_dim=node_feat_dim,
                num_layers=epochs_gcn_layers, epochs=epochs_gcn, device=device,
                log_path=log_path_for('gcn'), seed=seed,
            )
            save_run_csv(raw_dir, 'gcn', None, seed, target, train_size, denorm(res['metrics']), n_test)
            save_failure_data_csv(test_smi, res['test_true'], res['test_preds'],
                                  'gcn', target, raw_dir, model=None, device=device,
                                  pyg_data_list=test_pyg)
            if save_predictions:
                save_predictions_npz(res, model='gcn', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['gcn'] = res
        except Exception as e:
            print(f"  [ERROR] GCN: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # Transformer (CPU only — MPS causes segfault during ChemBERTa weight loading)
    if not skip_transformer and not (resume and run_already_done(raw_dir, 'transformer', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_transformer(
                train_smi, train_y, val_smi, val_y, test_smi, test_y,
                target_name=target, epochs=epochs_transformer, device='cpu',
                log_path=log_path_for('transformer'), seed=seed,
            )
            save_run_csv(raw_dir, 'transformer', None, seed, target, train_size, denorm(res['metrics']), n_test)
            if save_predictions:
                save_predictions_npz(res, model='transformer', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['transformer'] = res
        except Exception as e:
            print(f"  [ERROR] Transformer: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # ChemBERTa-2 (CPU only — same reason as Transformer)
    if not skip_chemberta2 and not (resume and run_already_done(raw_dir, 'chemberta2', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_chemberta2(
                train_smi, train_y, val_smi, val_y, test_smi, test_y,
                target_name=target, epochs=epochs_chemberta2, device='cpu',
                log_path=log_path_for('chemberta2'), seed=seed,
            )
            save_run_csv(raw_dir, 'chemberta2', None, seed, target, train_size, denorm(res['metrics']), n_test)
            if save_predictions:
                save_predictions_npz(res, model='chemberta2', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['chemberta2'] = res
        except Exception as e:
            print(f"  [ERROR] ChemBERTa-2: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # MoLFormer (CPU only — trust_remote_code; same reason as Transformer)
    if not skip_molformer and not (resume and run_already_done(raw_dir, 'molformer', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_molformer(
                train_smi, train_y, val_smi, val_y, test_smi, test_y,
                target_name=target, epochs=epochs_molformer, device='cpu',
                log_path=log_path_for('molformer'), seed=seed,
            )
            save_run_csv(raw_dir, 'molformer', None, seed, target, train_size, denorm(res['metrics']), n_test)
            if save_predictions:
                save_predictions_npz(res, model='molformer', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['molformer'] = res
        except Exception as e:
            print(f"  [ERROR] MoLFormer: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # SELFormer (CPU only — large pretrained; SELFIES conversion done internally)
    if not skip_selformer and not (resume and run_already_done(raw_dir, 'selformer', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_selformer(
                train_smi, train_y, val_smi, val_y, test_smi, test_y,
                target_name=target, epochs=epochs_selformer, device='cpu',
                log_path=log_path_for('selformer'), seed=seed,
            )
            save_run_csv(raw_dir, 'selformer', None, seed, target, train_size, denorm(res['metrics']), n_test)
            if save_predictions:
                save_predictions_npz(res, model='selformer', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['selformer'] = res
        except Exception as e:
            print(f"  [ERROR] SELFormer: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # Chemprop (D-MPNN, CPU — pytorch-lightning managed)
    if not skip_chemprop and not (resume and run_already_done(raw_dir, 'chemprop', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_chemprop(
                train_smi, train_y, val_smi, val_y, test_smi, test_y,
                target_name=target, epochs=epochs_chemprop, seed=seed,
            )
            save_run_csv(raw_dir, 'chemprop', None, seed, target, train_size, denorm(res['metrics']), n_test)
            if save_predictions:
                save_predictions_npz(res, model='chemprop', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['chemprop'] = res
        except Exception as e:
            print(f"  [ERROR] Chemprop: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # KROVEX (ISIS + ElasticNet descriptor selection + GCN + Kronecker fusion)
    if not skip_krovex and not (resume and run_already_done(raw_dir, 'krovex', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_krovex(
                train_smi, train_y, val_smi, val_y, test_smi, test_y,
                target_name=target, epochs=epochs_krovex, device=device,
                seed=seed,
            )
            save_run_csv(raw_dir, 'krovex', None, seed, target, train_size, denorm(res['metrics']), n_test)
            if save_predictions:
                save_predictions_npz(res, model='krovex', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['krovex'] = res
            # Log selected descriptors for this fold (informational)
            if log_dir and res.get('selected_descriptors') is not None:
                import json as _json
                os.makedirs(log_dir, exist_ok=True)
                desc_log_path = os.path.join(log_dir, f"krovex_na_{seed}_{target}_descriptors.json")
                with open(desc_log_path, 'w') as _fh:
                    _json.dump({
                        'train_size': train_size,
                        'seed': seed,
                        'target': target,
                        'selected_descriptors': res['selected_descriptors'],
                    }, _fh, indent=2)
        except Exception as e:
            print(f"  [ERROR] KROVEX: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # AttentiveFP
    if not skip_attentivefp and not (resume and run_already_done(raw_dir, 'attentivefp', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_attentivefp(
                train_pyg, val_pyg, test_pyg,
                target_name=target, node_feat_dim=node_feat_dim, edge_dim=edge_dim,
                epochs=epochs_attfp, device=device, log_path=log_path_for('attentivefp'),
                seed=seed,
            )
            save_run_csv(raw_dir, 'attentivefp', None, seed, target, train_size, denorm(res['metrics']), n_test)
            if save_predictions:
                save_predictions_npz(res, model='attentivefp', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['attentivefp'] = res
        except Exception as e:
            print(f"  [ERROR] AttentiveFP: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # GPS
    if not skip_gps and not (resume and run_already_done(raw_dir, 'gps', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            res = train_gps(
                train_pyg, val_pyg, test_pyg,
                target_name=target, node_feat_dim=node_feat_dim,
                epochs=epochs_gps, device=device, log_path=log_path_for('gps'),
                seed=seed,
            )
            save_run_csv(raw_dir, 'gps', None, seed, target, train_size, denorm(res['metrics']), n_test)
            if save_predictions:
                save_predictions_npz(res, model='gps', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['gps'] = res
        except Exception as e:
            print(f"  [ERROR] GPS: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # PaiNN (3D)
    if not skip_painn and not (resume and run_already_done(raw_dir, 'painn', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)) and tr['X_3d'] is not None and len(tr['X_3d']) > 0:
        try:
            from src.train import train_painn
            vi_tr = tr['X_3d_valid_idx'] or list(range(len(tr['X_3d'])))
            vi_va = va['X_3d_valid_idx'] or list(range(len(va['X_3d'])))
            vi_te = te['X_3d_valid_idx'] or list(range(len(te['X_3d'])))
            res = train_painn(
                tr['X_3d'], va['X_3d'], te['X_3d'],
                train_y[np.array(vi_tr)], val_y[np.array(vi_va)], test_y[np.array(vi_te)],
                target_name=target, device=device, log_path=log_path_for('painn'),
                seed=seed, epochs=epochs_painn,
            )
            save_run_csv(raw_dir, 'painn', None, seed, target, train_size, denorm(res['metrics']),
                         n_test=len(te['X_graph']))
            if save_predictions:
                save_predictions_npz(res, model='painn', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['painn'] = res
        except Exception as e:
            print(f"  [ERROR] PaiNN: {e}"); traceback.print_exc()

    # UniMol-Scratch (SE(3)-invariant transformer, from-scratch implementation)
    # CSV prefix: unimol_scratch_*  (paper name: UniMol-Scratch)
    if not skip_unimol and not (resume and run_already_done(raw_dir, 'unimol_scratch', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)) and tr['X_unimol'] is not None and len(tr['X_unimol']) > 0:
        try:
            vi_tr = tr['X_unimol_valid_idx'] or list(range(len(tr['X_unimol'])))
            vi_va = va['X_unimol_valid_idx'] or list(range(len(va['X_unimol'])))
            vi_te = te['X_unimol_valid_idx'] or list(range(len(te['X_unimol'])))
            res = train_unimol(
                tr['X_unimol'], va['X_unimol'], te['X_unimol'],
                train_y[np.array(vi_tr)], val_y[np.array(vi_va)], test_y[np.array(vi_te)],
                target_name=target, device=device, log_path=log_path_for('unimol_scratch'),
                seed=seed, epochs=epochs_unimol,
            )
            save_run_csv(raw_dir, 'unimol_scratch', None, seed, target, train_size,
                         denorm(res['metrics']), n_test=len(te['X_graph']))
            if save_predictions:
                save_predictions_npz(res, model='unimol_scratch', pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target=target, train_size=train_size,
                                     y_mean=_stats[0], y_std=_stats[1])
            results['unimol_scratch'] = res
        except Exception as e:
            print(f"  [ERROR] UniMol-Scratch: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # UniMol-PT (fine-tuning official DPTechnology PubChem-209M checkpoint, CUDA only)
    # CSV prefix: unimol_pt_*  (paper name: UniMol-PT)
    if enable_unimol_pretrained and not (resume and run_already_done(raw_dir, 'unimol_pt', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)):
        try:
            if device != 'cuda':
                print(f"  [SKIP] unimol_pt requires CUDA; device={device}")
            else:
                res = train_unimol_pretrained(
                    train_smiles=train_smi, train_y=train_y,
                    val_smiles=val_smi,   val_y=val_y,
                    test_smiles=test_smi, test_y=test_y,
                    target_name=target,
                    epochs=epochs_unimol_pretrained,
                    device=device,
                    log_path=log_path_for('unimol_pt'),
                    seed=seed,
                )
                save_run_csv(raw_dir, 'unimol_pt', None, seed, target, train_size,
                             denorm(res['metrics']), n_test)
                if save_predictions:
                    save_predictions_npz(res, model='unimol_pt', pred_dir=pred_dir, model_kind='na',
                                         seed=seed, target=target, train_size=train_size,
                                         y_mean=_stats[0], y_std=_stats[1])
                results['unimol_pt'] = res
        except Exception as e:
            print(f"  [ERROR] UniMol-PT: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    # RF
    if not skip_rf and not (resume and run_already_done(raw_dir, 'rf', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='rf', seed=seed)
            if res:
                save_run_csv(raw_dir, 'rf', None, seed, target, train_size, denorm(res['metrics']), n_test)
                if save_predictions:
                    save_predictions_npz(res, model='rf', pred_dir=pred_dir, model_kind='na',
                                         seed=seed, target=target, train_size=train_size,
                                         y_mean=_stats[0], y_std=_stats[1])
                results['rf'] = res
        except Exception as e:
            print(f"  [ERROR] RF: {e}"); traceback.print_exc()

    # XGBoost
    if not skip_xgb and not (resume and run_already_done(raw_dir, 'xgb', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='xgb', seed=seed)
            if res:
                save_run_csv(raw_dir, 'xgb', None, seed, target, train_size, denorm(res['metrics']), n_test)
                if save_predictions:
                    save_predictions_npz(res, model='xgb', pred_dir=pred_dir, model_kind='na',
                                         seed=seed, target=target, train_size=train_size,
                                         y_mean=_stats[0], y_std=_stats[1])
                results['xgb'] = res
        except Exception as e:
            print(f"  [ERROR] XGBoost: {e}"); traceback.print_exc()

    # GPR
    if not skip_gpr and not (resume and run_already_done(raw_dir, 'gpr', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='gpr', seed=seed,
                                train_size_limit_gpr=gpr_max_train_size)
            if res:
                save_run_csv(raw_dir, 'gpr', None, seed, target, train_size, denorm(res['metrics']), n_test)
                if save_predictions:
                    save_predictions_npz(res, model='gpr', pred_dir=pred_dir, model_kind='na',
                                         seed=seed, target=target, train_size=train_size,
                                         y_mean=_stats[0], y_std=_stats[1])
                results['gpr'] = res
        except Exception as e:
            print(f"  [ERROR] GPR: {e}"); traceback.print_exc()

    # SVR
    if not skip_svr and not (resume and run_already_done(raw_dir, 'svr', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='svr', seed=seed)
            if res:
                save_run_csv(raw_dir, 'svr', None, seed, target, train_size, denorm(res['metrics']), n_test)
                if save_predictions:
                    save_predictions_npz(res, model='svr', pred_dir=pred_dir, model_kind='na',
                                         seed=seed, target=target, train_size=train_size,
                                         y_mean=_stats[0], y_std=_stats[1])
                results['svr'] = res
        except Exception as e:
            print(f"  [ERROR] SVR: {e}"); traceback.print_exc()

    # LightGBM
    if not skip_lgbm and not (resume and run_already_done(raw_dir, 'lgbm', None, seed, target, train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe=target_safe)) and tr['X_ecfp'] is not None:
        try:
            res = train_sklearn(tr['X_ecfp'], train_y, va['X_ecfp'], val_y,
                                te['X_ecfp'], test_y, model_type='lgbm', seed=seed)
            if res:
                save_run_csv(raw_dir, 'lgbm', None, seed, target, train_size, denorm(res['metrics']), n_test)
                if save_predictions:
                    save_predictions_npz(res, model='lgbm', pred_dir=pred_dir, model_kind='na',
                                         seed=seed, target=target, train_size=train_size,
                                         y_mean=_stats[0], y_std=_stats[1])
                results['lgbm'] = res
        except Exception as e:
            print(f"  [ERROR] LightGBM: {e}"); traceback.print_exc()

    # ── MTL shared helper (QM9 only, sentinel at target='homo') ──────────────
    def _do_mtl(model_key, train_fn, epochs_mtl):
        """Load QM9 multitask data once, train, save all 12 CSVs."""
        if not (dataset == 'qm9' and target == 'homo'):
            return
        if resume and run_already_done(raw_dir, model_key, None, seed, 'homo', train_size,
                check_predictions=save_predictions, pred_dir=pred_dir,
                target_safe='homo'):
            return
        try:
            from src.data_loader import load_qm9_multitask
            mtl_data = load_qm9_multitask(train_size=train_size, seed=seed, data_dir=data_dir)
            res = train_fn(
                mtl_data['train']['X_graph'],
                mtl_data['val']['X_graph'],
                mtl_data['test']['X_graph'],
                mtl_data['train']['y'],
                mtl_data['val']['y'],
                mtl_data['test']['y'],
                stats=mtl_data['stats'],
                target_names=mtl_data['target_names'],
                n_tasks=len(mtl_data['target_names']),
                node_feat_dim=node_feat_dim,
                edge_dim=edge_dim,
                epochs=epochs_mtl,
                device=device,
                seed=seed,
            )
            n_test_mtl = len(mtl_data['test']['y'])
            for i, tname in enumerate(mtl_data['target_names']):
                task_metrics_denorm = _apply_denorm(res['metrics_per_task'][tname],
                                                    mtl_data['stats'][i])
                save_run_csv(raw_dir, model_key, None, seed, tname,
                             train_size, task_metrics_denorm, n_test_mtl)
            if save_predictions:
                y_mean_arr = np.array([s[0] for s in mtl_data['stats']])
                y_std_arr  = np.array([s[1] for s in mtl_data['stats']])
                save_predictions_npz(res, model=model_key, pred_dir=pred_dir, model_kind='na',
                                     seed=seed, target='homo', train_size=train_size,
                                     y_mean=y_mean_arr, y_std=y_std_arr)
            results[model_key] = res
        except Exception as e:
            print(f"  [ERROR] {model_key}: {e}"); traceback.print_exc()
        finally:
            clear_memory()

    if enable_attentivefp_mtl:
        _do_mtl('attentivefp_mtl', train_attentivefp_mtl, epochs_attentivefp_mtl)

    if enable_gcn_mtl:
        _do_mtl('gcn_mtl', train_gcn_mtl, epochs_gcn_mtl)

    # Group analysis for completed models
    group_rows = []
    for model_name, res in results.items():
        try:
            rows = group_analysis(
                test_smi, res['test_true'], res['test_preds'],
                model_name=model_name, target_name=target,
                train_size=train_size, seed=seed,
            )
            group_rows.extend(rows)
        except Exception:
            pass

    return results, group_rows


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QM9 Adaptive Learning Curve (multi-model)")
    p.add_argument('--dataset', nargs='+', default=['qm9'],
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Datasets to run (default: qm9)")
    p.add_argument('--target', nargs='+', default=None,
                   help="Targets to run. Default: all targets for each dataset.")
    def _default_device():
        if torch.cuda.is_available():    return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'
    p.add_argument('--device', default=_default_device())
    p.add_argument('--results_root', default='./results')
    p.add_argument('--data_dir',     default='./data')

    # Epochs
    p.add_argument('--epochs_transformer',  type=int, default=200)
    p.add_argument('--epochs_chemberta2',   type=int, default=200)
    p.add_argument('--epochs_molformer',    type=int, default=200)
    p.add_argument('--epochs_selformer',    type=int, default=200)
    p.add_argument('--epochs_chemprop',     type=int, default=200)
    p.add_argument('--epochs_krovex',       type=int, default=300)
    p.add_argument('--epochs_gcn',          type=int, default=300)
    p.add_argument('--gcn_layers',          type=int, default=3)
    p.add_argument('--epochs_attfp',             type=int, default=300)
    p.add_argument('--epochs_gps',               type=int, default=300)
    p.add_argument('--epochs_attentivefp_mtl',   type=int, default=300)
    p.add_argument('--epochs_gcn_mtl',           type=int, default=300)
    p.add_argument('--epochs_unimol',            type=int, default=300)
    p.add_argument('--epochs_painn',             type=int, default=300)
    p.add_argument('--epochs_unimol_pretrained', type=int, default=50)

    # Skip flags
    p.add_argument('--skip_transformer',  action='store_true')
    p.add_argument('--skip_chemberta2',   action='store_true')
    p.add_argument('--skip_molformer',    action='store_true')
    p.add_argument('--skip_selformer',    action='store_true')
    p.add_argument('--skip_chemprop',     action='store_true')
    p.add_argument('--skip_krovex',       action='store_true')
    p.add_argument('--skip_gcn',          action='store_true')
    p.add_argument('--skip_rf',          action='store_true')
    p.add_argument('--skip_xgb',         action='store_true')
    p.add_argument('--skip_gpr',         action='store_true')
    p.add_argument('--skip_svr',         action='store_true')
    p.add_argument('--skip_lgbm',        action='store_true')
    p.add_argument('--skip_attentivefp',        action='store_true')
    p.add_argument('--skip_painn',              action='store_true')
    p.add_argument('--skip_gps',               action='store_true')
    p.add_argument('--enable_unimol',          action='store_true',
                   help="Enable UniMol SE(3)-invariant transformer (default: off, opt-in)")
    p.add_argument('--enable_attentivefp_mtl', action='store_true',
                   help="Enable QM9 12-task MTL AttentiveFP (default: off)")
    p.add_argument('--enable_gcn_mtl', action='store_true',
                   help="Enable QM9 12-task MTL GCN robustness check (default: off)")
    p.add_argument('--enable_unimol_pretrained', action='store_true',
                   help="Enable pretrained UniMol fine-tuning (CUDA required; default: off)")

    # Control
    p.add_argument('--resume',    action='store_true', help="Skip already-done runs")
    p.add_argument('--save_logs', action='store_true', help="Save training logs")
    p.add_argument('--train_sizes', nargs='+', type=int, default=None,
                   help="Override train sizes (e.g. --train_sizes 50 100 500)")
    p.add_argument('--seeds', nargs='+', type=int, default=None,
                   help="Override seed list (e.g. --seeds 0 1 2). Use only when results are "
                        "confirmed seed-invariant; otherwise the schedule-based 10/3 seeds is "
                        "recommended. Useful for seed-invariant models like UniMol-PT on QM9.")
    p.add_argument('--save_predictions', action='store_true',
                   help='Save normalized test_preds + test_true as .npz for ensemble analysis. '
                        'Output: {results_root}/{ds_dir}/predictions/'
                        '{model}_na_{seed}_{target}_n{train_size}.npz')
    p.add_argument('--gpr_max_train_size', type=int, default=500,
                   help='Maximum train_size for GPR (default 500, O(N³) practical limit). '
                        'Pass a larger value to extend GPR coverage, e.g. 3000 for full QM9 grid. '
                        'May take hours to days at large N.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    train_sizes = args.train_sizes if args.train_sizes else None  # per-dataset below

    # Active models list for display
    active_models = []
    for m in BASELINE_MODELS:
        flag = getattr(args, f'skip_{m}', False)
        if not flag:
            active_models.append(m)

    print(f"Device: {args.device}")
    print(f"Datasets: {args.dataset}")
    print(f"Train sizes: {'per-dataset' if train_sizes is None else f'({len(train_sizes)}) {train_sizes}'}")
    print(f"Active models: {active_models}")
    print()

    for dataset in args.dataset:
        cfg = DATASET_CONFIGS[dataset]
        targets = args.target if args.target else cfg['target_tasks']
        ds_sizes = train_sizes if train_sizes is not None else get_train_sizes(dataset)

        dataset_dir  = os.path.join(args.results_root, DATASET_DIRS.get(dataset, dataset))
        raw_dir      = os.path.join(dataset_dir, 'raw_data')
        summary_dir  = os.path.join(dataset_dir, 'summary')
        plots_dir    = os.path.join(dataset_dir, 'plots')
        log_dir      = os.path.join(dataset_dir, 'training_logs') if args.save_logs else None

        pred_dir     = os.path.join(dataset_dir, 'predictions') if args.save_predictions else None

        os.makedirs(raw_dir,     exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(plots_dir,   exist_ok=True)

        group_master_path = os.path.join(dataset_dir, 'master_group_summary.csv')
        group_all_rows = []

        for target in targets:
            print(f"\n{'='*60}")
            print(f" Dataset={dataset} | Target={target}")
            print(f"{'='*60}")

            # Pre-load raw data once per (dataset, target) to avoid re-reading
            # 130k QM9 molecules on every run_one() call
            print(f"  [data_loader] Pre-loading raw data for {dataset}/{target} ...")
            preloaded_raw = load_raw_data(dataset, args.data_dir, target)
            print(f"  [data_loader] Raw data ready ({len(preloaded_raw[0])} molecules)")

            for train_size in ds_sizes:
                seeds = args.seeds if args.seeds is not None else get_seed_schedule(train_size)

                for seed in seeds:
                    # Check resume for all active models
                    if args.resume:
                        _tgt_safe = target.replace(' ', '_').replace('/', '_')
                        all_done = all(
                            run_already_done(raw_dir, m, None, seed, target, train_size,
                                             check_predictions=args.save_predictions,
                                             pred_dir=pred_dir,
                                             target_safe=_tgt_safe)
                            for m in active_models
                        )
                        if all_done:
                            print(f"  [resume] size={train_size} seed={seed} target={target} — skip")
                            continue

                    print(f"\n  --- size={train_size:4d} | seed={seed} | target={target} ---")

                    try:
                        results, group_rows = run_one(
                            train_size=train_size,
                            target=target,
                            seed=seed,
                            dataset=dataset,
                            raw_dir=raw_dir,
                            data_dir=args.data_dir,
                            device=args.device,
                            epochs_transformer=args.epochs_transformer,
                            epochs_chemberta2=args.epochs_chemberta2,
                            epochs_molformer=args.epochs_molformer,
                            epochs_selformer=args.epochs_selformer,
                            epochs_chemprop=args.epochs_chemprop,
                            epochs_krovex=args.epochs_krovex,
                            epochs_gcn=args.epochs_gcn,
                            epochs_gcn_layers=args.gcn_layers,
                            epochs_attfp=args.epochs_attfp,
                            epochs_gps=args.epochs_gps,
                            skip_transformer=args.skip_transformer,
                            skip_chemberta2=args.skip_chemberta2,
                            skip_molformer=args.skip_molformer,
                            skip_selformer=args.skip_selformer,
                            skip_chemprop=args.skip_chemprop,
                            skip_krovex=args.skip_krovex,
                            skip_gcn=args.skip_gcn,
                            skip_rf=args.skip_rf,
                            skip_xgb=args.skip_xgb,
                            skip_gpr=args.skip_gpr,
                            skip_svr=args.skip_svr,
                            skip_lgbm=args.skip_lgbm,
                            skip_attentivefp=args.skip_attentivefp,
                            skip_painn=args.skip_painn,
                            skip_gps=args.skip_gps,
                            skip_unimol=not args.enable_unimol,
                            epochs_unimol=args.epochs_unimol,
                            epochs_painn=args.epochs_painn,
                            enable_unimol_pretrained=args.enable_unimol_pretrained,
                            epochs_unimol_pretrained=args.epochs_unimol_pretrained,
                            enable_attentivefp_mtl=args.enable_attentivefp_mtl,
                            epochs_attentivefp_mtl=args.epochs_attentivefp_mtl,
                            enable_gcn_mtl=args.enable_gcn_mtl,
                            epochs_gcn_mtl=args.epochs_gcn_mtl,
                            log_dir=log_dir,
                            resume=args.resume,
                            preloaded_raw=preloaded_raw,
                            save_predictions=args.save_predictions,
                            pred_dir=pred_dir,
                            gpr_max_train_size=args.gpr_max_train_size,
                        )
                        group_all_rows.extend(group_rows)

                    except Exception as e:
                        print(f"  [ERROR] size={train_size} seed={seed}: {e}")
                        traceback.print_exc()

            # After all sizes/seeds for this target: rebuild summary + plots
            try:
                baseline_models = [m for m in BASELINE_MODELS
                                   if not getattr(args, f'skip_{m}', False)]
                summary_df = rebuild_summary_baselines(
                    raw_dir, summary_dir, target,
                    baseline_models=baseline_models,
                )
                if not summary_df.empty:
                    plot_baselines_lc(summary_df, plots_dir, target)
            except Exception as e:
                print(f"  [ERROR] Summary/plot: {e}")

        # Save group summary
        if group_all_rows:
            new_df = pd.DataFrame(group_all_rows)
            if os.path.exists(group_master_path):
                existing = pd.read_csv(group_master_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=['train_size','model','target','seed','group_type','category'],
                    keep='last'
                )
            else:
                combined = new_df
            combined.to_csv(group_master_path, index=False)
            print(f"\n[saved] {group_master_path}")

            try:
                group_df = pd.read_csv(group_master_path)
                for target in targets:
                    plot_group_analysis(group_df, plots_dir, target)
            except Exception as e:
                print(f"  [ERROR] Group plot: {e}")

    print(f"\nDone! Results in {args.results_root}/")


if __name__ == '__main__':
    main()
