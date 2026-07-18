# When Do Models Win?
### A Learning Curve Benchmark for Molecular Property Prediction in Low-Data Regimes

Code and aggregated results for the paper:

> **"When Do Models Win? A Learning Curve Benchmark for Molecular Property Prediction in Low-Data Regimes"**
> *Journal of Cheminformatics* (under review)

We systematically compare **20 models** — from traditional ML to graph neural networks, equivariant 3D networks, and large pretrained transformers — across **4 molecular datasets** (QM9, ESOL, Lipophilicity, BACE), tracking performance as a function of training set size (N = 50–3000, up to 10 seeds) to identify which model families excel under data scarcity.

---

## Models

| Family | Models |
|--------|--------|
| **Graph Neural Networks** | GCN, AttentiveFP, GPS, KROVEX, GTCA-Cat |
| **Sequence Transformers** | ChemBERTa (v1/v2), MoLFormer-XL, SELFormer, Chemprop (D-MPNN) |
| **3D / Equivariant** | PaiNN (SE(3)-equivariant), Uni-Mol-Scratch (SE(3)-invariant), Uni-Mol-PT (pretrained) |
| **Traditional ML** | Random Forest, XGBoost, LightGBM, SVR, GPR |
| **Multi-task (QM9)** | AttentiveFP-MTL, GCN-MTL |

---

## Datasets

| Dataset | Task | Size | Targets |
|---------|------|------|---------|
| **QM9** | Quantum properties | ~130k | HOMO, LUMO, Gap (eV) |
| **ESOL** | Aqueous solubility | ~1.1k | log mol/L |
| **Lipophilicity** | Lipophilicity | ~4.2k | log D |
| **BACE** | Binding affinity | ~1.5k | pIC50 |

Train size schedule: 50–500 (Δ25, 10 seeds) · 600–1000 (Δ100, 3 seeds) · 1500–3000 (Δ500, QM9 only, 3 seeds).
All splits use RDKit Murcko scaffold splitting; split keys are cached in `data/*_scaffold_groups.pkl`.

---

## Installation

### Option A — pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate smiles
```

> **Note**: DeepChem requires a dev build (`2.8.1.dev*`); stable PyPI releases only go up to 2.5.0.
> The `environment.yml` pins the exact versions used in the paper experiments.
> See `env_lock/` for per-machine/per-session environment snapshots.

---

## Data Preparation

### 1. MoleculeNet datasets (ESOL, Lipophilicity, BACE)
Loaded automatically via DeepChem on first run. No manual download needed.

### 2. QM9 SDF (required for PaiNN and Uni-Mol-Scratch on QM9)
```bash
# Download from the original QM9 repository (~241 MB)
wget "https://figshare.com/ndownloader/files/3195389" -O data/qm9.sdf
```
First run with QM9 will cache featurized data at `data/qm9-featurized/` (~2.1 GB).

### 3. Conformer cache (required for PaiNN / Uni-Mol-Scratch on ESOL, Lipo, BACE)
Pre-build offline ETKDG 3D coordinates (one-time, ~5 min per dataset):
```bash
python scripts/build_conformer_cache.py --dataset esol
python scripts/build_conformer_cache.py --dataset lipo
python scripts/build_conformer_cache.py --dataset bace
```
Outputs: `data/{ds}-3d-cache.pkl`

---

## Running Experiments

### Full learning curve — all models, one dataset
```bash
python run_learning_curve.py --dataset esol --device cpu --resume
```

### QM9 (3 targets: homo, lumo, gap)
```bash
python run_learning_curve.py --dataset qm9 --target homo lumo gap --device mps --resume
```

### Skip heavy models for a quick run
```bash
python run_learning_curve.py --dataset esol \
  --skip_painn --skip_molformer --skip_selformer \
  --device cpu --resume
```

### Opt-in models (disabled by default)
```bash
# Uni-Mol from scratch
python run_learning_curve.py --dataset esol --enable_unimol --device mps

# Uni-Mol pretrained (requires CUDA)
python run_learning_curve.py --dataset esol --enable_unimol_pretrained --device cuda

# Multi-task variants (QM9 only)
python run_learning_curve.py --dataset qm9 --target homo lumo gap \
  --enable_attentivefp_mtl --enable_gcn_mtl --device mps
```

### GTCA depth ablation / fusion study (QM9)
```bash
python run_depth_study.py --dataset qm9 --device mps --resume
python run_fusion_study.py --dataset qm9 --best_depth 6 --device mps --resume
```

### Save ensemble predictions (NPZ)
```bash
python run_learning_curve.py --dataset esol \
  --train_sizes 50 100 200 375 --resume --save_predictions --device cpu
```

---

## Aggregating Results & Regenerating Figures

```bash
# Rebuild paper_csv from raw_data
python rebuild_paper_csv.py

# Regenerate all learning-curve plots
python regenerate_plots.py

# Regenerate all-model 2×2 figures (PNG + PDF)
python scripts/plot_allmodels_2x2.py

# Regenerate ensemble analysis figure
python scripts/plot_ensemble_analysis.py

# Ensemble analysis (requires --save_predictions NPZ files)
python scripts/ensemble_analysis.py
```

---

## Tests

```bash
pytest tests/ -v          # 212 tests, all pass
pytest tests/ -v -k painn # run only PaiNN tests
```

Key test coverage:
- SE(3) equivariance / invariance (PaiNN, Uni-Mol)
- Radius graph correctness (`_radius_graph`, 16 tests)
- Inline denormalization math (`_apply_denorm`)
- Seed determinism for all models
- Conformer cache builder (ETKDG pipeline)
- NPZ prediction save/load (`--save_predictions`)

---

## Results

Pre-aggregated CSVs are included in `results/paper_csv/`:

| File | Contents |
|------|----------|
| `lc_{dataset}_all_models.csv` | Learning curves: mean ± std RMSE/MAE/R/R² per (model, train_size) |
| `ablation_gtca_depth_qm9.csv` | GTCA bert_depth [2, 4, 6] ablation |
| `ablation_gtca_fusion_qm9.csv` | GTCA-Cat vs GTCA-CA |
| `stats_*_welch_qm9.csv` | Welch t-test results |
| `ensemble_{dataset}.csv` | Ensemble analysis per (train_size, seed) |
| `ensemble_diversity_{dataset}.csv` | Pairwise prediction correlation matrix |

Raw experiment outputs (per-seed CSVs, NPZ predictions) are not included due to size.
Contact the authors for access.

---

## Environment Snapshots

`env_lock/` contains `pip freeze` outputs for each machine and session used in the experiments:

| File prefix | Environment |
|-------------|-------------|
| `env_lock_macos_dad_*` | dad-MacBook (Apple M-series, MPS) |
| `env_lock_macos_junho_* / junhomac_*` | junho-MacBook (Apple M-series, MPS/CPU) |
| `env_lock_colab_*` | Google Colab (NVIDIA L4/T4, CUDA) |
| `env_lock_unimol_dev_*` | Colab + unimol-tools dev environment |

Key versions across all runs: torch 2.10–2.11, transformers 5.0–5.12, rdkit 2025.9.5–2026.3.3, deepchem 2.8.1.dev, selfies 2.1.1, chemprop ≥2.0.0.

---

## Repository Structure

```
.
├── src/                    # Core library
│   ├── models.py           # All model classes
│   ├── train.py            # All training functions
│   ├── data_loader.py      # Data loading + z-score normalize
│   ├── featurizer.py       # SMILES → PyG / ECFP4 / 3D
│   ├── descriptor_selection.py  # KROVEX: ISIS + Elastic Net
│   ├── summary.py          # CSV save/rebuild utilities
│   ├── visualization.py    # Learning curve plots
│   └── ...
├── tests/                  # 212 pytest tests
├── scripts/                # Plotting and verification scripts
│   ├── build_conformer_cache.py
│   ├── ensemble_analysis.py
│   ├── plot_allmodels_2x2.py
│   └── plot_ensemble_analysis.py
├── run_learning_curve.py   # Main experiment entry point
├── run_depth_study.py      # GTCA depth ablation
├── run_fusion_study.py     # GTCA-Cat vs GTCA-CA
├── run_final_comparison.py # Merge baselines + GTCA → paper_csv
├── rebuild_paper_csv.py    # raw_data → paper_csv aggregation
├── regenerate_plots.py     # Regenerate all plots from paper_csv
├── data/                   # Scaffold split PKLs (tracked)
│   └── *_scaffold_groups.pkl
├── env_lock/               # pip freeze snapshots per session
├── results/
│   └── paper_csv/          # Aggregated CSVs (tracked)
├── requirements.txt
├── environment.yml
└── LICENSE
```

---

## Citation

> (To be added upon publication)

---

## License

MIT — see [LICENSE](LICENSE).
