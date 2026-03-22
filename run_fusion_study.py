"""
run_fusion_study.py
Ablation study: Concatenation fusion (GTCA-Cat) vs Cross-Attention fusion (GTCA-CA).

Run AFTER depth_study to know the best bert_depth.

Results structure:
  results/{DATASET_DIR}/fusion_study/
  ├── raw_data/    gtca_cat_{depth}_{seed}_{target}.csv
  │                gtca_ca_{depth}_{seed}_{target}.csv
  └── fusion_comparison_{target}.csv

Usage:
  # smoke test
  python run_fusion_study.py --dataset qm9 --best_depth 4 \\
      --epochs 5 --train_sizes 50 100 --device mps

  # full run (QM9 only, best_depth determined from depth_study)
  python run_fusion_study.py --dataset qm9 --best_depth 4 --device mps --resume
"""

import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset_splits, DATASET_CONFIGS
from src.train import train_gtca, train_gtca_ca


DATASET_DIRS = {
    'qm9':  '01_QM9',
    'esol': '02_ESOL',
    'lipo': '03_Lipo',
    'bace': '04_BACE',
}


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


def run_already_done(fusion_raw_dir: str, model_tag: str, depth: int,
                     seed: int, target: str, train_size: int) -> bool:
    """Check if this (model_tag, depth, seed, target, train_size) is saved."""
    csv = os.path.join(fusion_raw_dir, f"{model_tag}_{depth}_{seed}_{target}.csv")
    if not os.path.exists(csv):
        return False
    df = pd.read_csv(csv)
    return int(train_size) in df['train_size'].values


def save_fusion_csv(fusion_raw_dir: str, model_tag: str, depth: int,
                    seed: int, target: str, train_size: int,
                    metrics: dict, n_test: int):
    csv = os.path.join(fusion_raw_dir, f"{model_tag}_{depth}_{seed}_{target}.csv")
    row = {'train_size': train_size, 'n_test': n_test, **metrics}
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.drop_duplicates(subset=['train_size'], keep='last')
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv, index=False)


def load_data(dataset, data_dir, train_size, seed, target):
    data = load_dataset_splits(
        dataset=dataset,
        data_dir=data_dir,
        train_size=train_size,
        val_size=100,
        test_size=10000,
        seed=seed,
        target=target,
        featurize_ecfp=False,
        featurize_3d=False,
    )
    tr, va, te = data['train'], data['val'], data['test']
    return (
        tr['X_graph'], va['X_graph'], te['X_graph'],
        tr['ids'],     va['ids'],     te['ids'],
        tr['y'],       va['y'],       te['y'],
        len(te['X_graph']),
    )


def parse_args():
    p = argparse.ArgumentParser(description="GTCA fusion ablation: Cat vs Cross-Attention")
    p.add_argument('--dataset',    nargs='+', default=['qm9'],
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument('--target',     nargs='+', default=None)
    p.add_argument('--best_depth', type=int,  required=True,
                   help="Best bert_depth from depth_study (e.g. 4)")
    def _default_device():
        if torch.cuda.is_available():    return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'
    p.add_argument('--device', default=_default_device())
    p.add_argument('--results_root', default='./results')
    p.add_argument('--data_dir',     default='./data')
    p.add_argument('--epochs',     type=int, default=300)
    p.add_argument('--gcn_layers', type=int, default=3)
    p.add_argument('--ca_dim',     type=int, default=256,
                   help="Cross-attention projection dim (default: 256)")
    p.add_argument('--ca_heads',   type=int, default=4,
                   help="Cross-attention heads (default: 4)")
    p.add_argument('--resume',     action='store_true')
    p.add_argument('--train_sizes', nargs='+', type=int, default=None)
    p.add_argument('--skip_cat',   action='store_true', help="Skip GTCA-Cat runs")
    p.add_argument('--skip_ca',    action='store_true', help="Skip GTCA-CA runs")
    return p.parse_args()


def main():
    args   = parse_args()
    depth  = args.best_depth
    sizes  = args.train_sizes if args.train_sizes else None  # per-dataset below
    fusion_models = []
    if not args.skip_cat:
        fusion_models.append('gtca_cat')
    if not args.skip_ca:
        fusion_models.append('gtca_ca')

    print(f"Device:      {args.device}")
    print(f"Datasets:    {args.dataset}")
    print(f"Best depth:  {depth}")
    print(f"Fusion:      {fusion_models}")
    print(f"Train sizes ({len(sizes)}): {sizes[:5]}{'...' if len(sizes) > 5 else ''}")
    print()

    for dataset in args.dataset:
        cfg     = DATASET_CONFIGS[dataset]
        targets = args.target if args.target else cfg['target_tasks']
        ds_sizes = sizes if sizes is not None else get_train_sizes(dataset)

        dataset_dir  = os.path.join(args.results_root, DATASET_DIRS.get(dataset, dataset))
        fusion_dir   = os.path.join(dataset_dir, 'fusion_study')
        fusion_raw   = os.path.join(fusion_dir, 'raw_data')
        os.makedirs(fusion_raw, exist_ok=True)

        for target in targets:
            print(f"\n{'='*60}")
            print(f" Dataset={dataset} | Target={target} | depth={depth}")
            print(f"{'='*60}")

            all_rows = []

            for model_tag in fusion_models:
                print(f"\n  --- {model_tag} ---")

                for train_size in ds_sizes:
                    seeds = get_seed_schedule(train_size)

                    for seed in seeds:
                        if args.resume and run_already_done(
                                fusion_raw, model_tag, depth, seed, target, train_size):
                            print(f"    [resume] {model_tag} size={train_size} seed={seed} — skip")
                            continue

                        print(f"    size={train_size:4d} | seed={seed}")
                        try:
                            (train_pyg, val_pyg, test_pyg,
                             train_smi, val_smi, test_smi,
                             train_y, val_y, test_y, n_test) = load_data(
                                dataset, args.data_dir, train_size, seed, target)

                            node_feat_dim = train_pyg[0].x.shape[1] if train_pyg else 30

                            if model_tag == 'gtca_cat':
                                res = train_gtca(
                                    train_pyg, val_pyg, test_pyg,
                                    train_smi, val_smi, test_smi,
                                    target_name=target,
                                    node_feat_dim=node_feat_dim,
                                    gcn_layers=args.gcn_layers,
                                    bert_depth=depth,
                                    epochs=args.epochs,
                                    device=args.device,
                                    seed=seed,
                                )
                            else:  # gtca_ca
                                res = train_gtca_ca(
                                    train_pyg, val_pyg, test_pyg,
                                    train_smi, val_smi, test_smi,
                                    target_name=target,
                                    node_feat_dim=node_feat_dim,
                                    gcn_layers=args.gcn_layers,
                                    bert_depth=depth,
                                    ca_dim=args.ca_dim,
                                    ca_heads=args.ca_heads,
                                    epochs=args.epochs,
                                    device=args.device,
                                    seed=seed,
                                )

                            m = res['metrics']
                            save_fusion_csv(fusion_raw, model_tag, depth,
                                            seed, target, train_size, m, n_test)
                            all_rows.append({
                                'model':      model_tag,
                                'train_size': train_size,
                                'seed':       seed,
                                'RMSE':       m.get('RMSE'),
                                'MAE':        m.get('MAE'),
                                'Pearson_R':  m.get('Pearson_R'),
                                'R2':         m.get('R2'),
                            })

                        except Exception as e:
                            print(f"    [ERROR] {model_tag} size={train_size} seed={seed}: {e}")
                            traceback.print_exc()

            # Save fusion_comparison_{target}.csv
            if all_rows:
                comp_path = os.path.join(fusion_dir, f"fusion_comparison_{target}.csv")
                new_df = pd.DataFrame(all_rows)
                if os.path.exists(comp_path):
                    existing = pd.read_csv(comp_path)
                    new_df = pd.concat([existing, new_df], ignore_index=True)
                    new_df = new_df.drop_duplicates(
                        subset=['model', 'train_size', 'seed'], keep='last')
                new_df.to_csv(comp_path, index=False)
                print(f"\n  [saved] {comp_path}")

    print("\nDone! Fusion study complete.")


if __name__ == '__main__':
    main()
