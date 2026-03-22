"""
main.py
Full QM9 low-data benchmarking pipeline.

Usage:
    python main.py [--target homo] [--device cpu]
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import (
    load_qm9_splits,
    build_pyg_dataset,
    TARGET_TASKS,
)
from src.train import train_transformer, train_gcn, train_gtca
from src.analysis import failure_analysis, compile_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_smiles_and_labels(dataset: dict, task_pos: int):
    """Extract SMILES list and labels array from dataset dict."""
    smiles = dataset['ids']
    labels = dataset['y'][:, task_pos].astype(np.float32)
    return smiles, labels


# ---------------------------------------------------------------------------
# Single-target pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    target: str,
    results_dir: str,
    data_dir: str,
    device: str,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    epochs_transformer: int,
    epochs_gcn: int,
    epochs_gtca: int,
    skip_transformer: bool,
    skip_gcn: bool,
    skip_gtca: bool,
):
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f" Target: {target.upper()}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    train_ds, val_ds, test_ds, transformers, task_pos = load_qm9_splits(
        data_dir=data_dir,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        target=target,
    )

    # SMILES + labels
    train_smiles, train_y = _get_smiles_and_labels(train_ds, task_pos)
    val_smiles,   val_y   = _get_smiles_and_labels(val_ds,   task_pos)
    test_smiles,  test_y  = _get_smiles_and_labels(test_ds,  task_pos)

    # PyG graphs
    train_pyg = build_pyg_dataset(train_ds, task_pos)
    val_pyg   = build_pyg_dataset(val_ds,   task_pos)
    test_pyg  = build_pyg_dataset(test_ds,  task_pos)

    # Infer node feature dim from first graph
    node_feat_dim = train_pyg[0].x.shape[1]
    print(f"  Node feature dim: {node_feat_dim}")

    # ------------------------------------------------------------------ #
    # 2. Training
    # ------------------------------------------------------------------ #
    target_results = {}

    # --- Model A: Transformer ---
    if not skip_transformer:
        res_a = train_transformer(
            train_smiles, train_y,
            val_smiles,   val_y,
            test_smiles,  test_y,
            target_name=target,
            results_dir=results_dir,
            epochs=epochs_transformer,
            device=device,
        )
        target_results["transformer"] = res_a["metrics"]

        failure_analysis(
            test_smiles, test_y, res_a["test_preds"],
            model_name="transformer", target_name=target,
            results_dir=results_dir,
        )
    else:
        print("  [skip] Transformer")

    # --- Model B: GCN ---
    if not skip_gcn:
        res_b = train_gcn(
            train_pyg, val_pyg, test_pyg,
            target_name=target,
            results_dir=results_dir,
            node_feat_dim=node_feat_dim,
            epochs=epochs_gcn,
            device=device,
        )
        target_results["gcn"] = res_b["metrics"]

        failure_analysis(
            test_smiles, res_b["test_true"], res_b["test_preds"],
            model_name="gcn", target_name=target,
            results_dir=results_dir,
        )
    else:
        print("  [skip] GCN")

    # --- Model C: GTCA ---
    if not skip_gtca:
        res_c = train_gtca(
            train_pyg, val_pyg, test_pyg,
            train_smiles, val_smiles, test_smiles,
            target_name=target,
            results_dir=results_dir,
            node_feat_dim=node_feat_dim,
            epochs=epochs_gtca,
            device=device,
        )
        target_results["gtca"] = res_c["metrics"]

        failure_analysis(
            test_smiles, res_c["test_true"], res_c["test_preds"],
            model_name="gtca", target_name=target,
            results_dir=results_dir,
        )
    else:
        print("  [skip] GTCA")

    return target_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QM9 Low-data Benchmark")
    p.add_argument("--target",  nargs="+", default=TARGET_TASKS,
                   help="Target(s): homo lumo gap (default: all)")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--train_size",  type=int, default=100)
    p.add_argument("--val_size",    type=int, default=100)
    p.add_argument("--test_size",   type=int, default=10000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--epochs_transformer", type=int, default=200)
    p.add_argument("--epochs_gcn",         type=int, default=300)
    p.add_argument("--epochs_gtca",        type=int, default=200)
    p.add_argument("--skip_transformer", action="store_true")
    p.add_argument("--skip_gcn",         action="store_true")
    p.add_argument("--skip_gtca",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"Targets: {args.target}")

    all_results = {}   # {model: {target: metrics}}

    for target in args.target:
        target_res = run_pipeline(
            target=target,
            results_dir=args.results_dir,
            data_dir=args.data_dir,
            device=args.device,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            epochs_transformer=args.epochs_transformer,
            epochs_gcn=args.epochs_gcn,
            epochs_gtca=args.epochs_gtca,
            skip_transformer=args.skip_transformer,
            skip_gcn=args.skip_gcn,
            skip_gtca=args.skip_gtca,
        )
        # Reorganise: all_results[model][target]
        for model, metrics in target_res.items():
            all_results.setdefault(model, {})[target] = metrics

    # Final summary table
    compile_summary(all_results, args.results_dir)

    # Save raw JSON (convert numpy types to Python native)
    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    out_json = os.path.join(args.results_dir, "all_metrics.json")
    with open(out_json, "w") as f:
        json.dump(_to_python(all_results), f, indent=2)
    print(f"\nAll metrics saved → {out_json}")


if __name__ == "__main__":
    main()
