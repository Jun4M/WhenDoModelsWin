# When Do Models Win?
### A Learning Curve Benchmark for Molecular Property Prediction in Low-Data Regimes

This repository contains the code and aggregated results for the paper:

> **"When Do Models Win? A Learning Curve Benchmark for Molecular Property Prediction in Low-Data Regimes"**
> *Journal of Cheminformatics* (under review)

We systematically compare 10 model families across 4 molecular datasets, measuring performance as a function of training set size (N=50–3,000) to identify which models excel under data scarcity.

---

## Models

| Model | Type | Input |
|-------|------|-------|
| GCN | Graph Neural Network | Molecular graph |
| AttentiveFP | Graph Neural Network | Molecular graph |
| GPS | Graph Neural Network | Molecular graph + RWSE |
| ChemBERTa | Transformer | SMILES |
| **GTCA-Cat** | GNN + Transformer (proposed) | Graph + SMILES (concatenation) |
| **GTCA-CA** | GNN + Transformer (proposed) | Graph + SMILES (cross-attention) |
| Random Forest | Traditional ML | ECFP4 (radius=2, 2048 bits) |
| XGBoost | Traditional ML | ECFP4 |
| LightGBM | Traditional ML | ECFP4 |
| SVR | Traditional ML | ECFP4 + StandardScaler |
| GPR | Traditional ML | ECFP4 (N≤500 only) |

---

## Datasets

| Dataset | Task | Size | Target |
|---------|------|------|--------|
| QM9 | Quantum properties | ~130k | HOMO, LUMO, Gap (eV) |
| ESOL | Aqueous solubility | ~1.1k | log mol/L |
| Lipophilicity | Lipophilicity | ~4.2k | log D |
| BACE | Binding affinity | ~1.5k | pIC50 |

Scaffold-based splits (RDKit MurckoScaffold). Training sizes: N=50–500 (step 25, 10 seeds), N=600–1,000 (step 100, 3 seeds), N=1,500–3,000 (step 500, 3 seeds, QM9 only).

---

## Environment

```bash
conda env create -f environment.yml
conda activate smiles
```

Key dependencies: Python 3.12, PyTorch 2.10, PyTorch Geometric 2.7, DeepChem 2.8.1, Transformers 5.2, scikit-learn 1.8, RDKit 2025.9.5. All experiments were conducted on Apple Silicon MacBook Pro (MPS backend).

---

## Reproducing Results

### Step 1 — Run experiments

```bash
# All baseline models (GCN, ChemBERTa, AttentiveFP, GPS, RF, XGB, LGB, SVR, GPR)
python run_learning_curve.py --dataset qm9 esol lipo bace --device mps --resume

# GTCA depth ablation (bert_depth = 2, 4, 6) on QM9
python run_depth_study.py --dataset qm9 --device mps --resume

# GTCA fusion comparison (Cat vs CA) on QM9
# Note: do not pass --train_sizes; defaults (50–500 step 25, 600–1000 step 100, 1500–3000 step 500)
# exactly match the train sizes used in the paper.
python run_fusion_study.py --dataset qm9 --best_depth 6 --device mps --resume
```

### Step 2 — Post-processing

```bash
python collect_std.py           # collect normalization statistics
python denormalize_raw.py       # convert RMSE/MAE to original units
python rebuild_paper_csv.py     # aggregate into results/paper_csv/
python regenerate_plots.py      # reproduce all paper figures
```

---

## Pre-computed Results

Aggregated results are available in `results/paper_csv/` — figures can be reproduced without re-running experiments:

```bash
python regenerate_plots.py
```

| File | Description |
|------|-------------|
| `lc_qm9_all_models.csv` | QM9 learning curves, all models (eV) |
| `lc_esol_all_models.csv` | ESOL learning curves |
| `lc_lipo_all_models.csv` | Lipophilicity learning curves |
| `lc_bace_all_models.csv` | BACE learning curves |
| `ablation_gtca_depth_qm9.csv` | GTCA depth ablation (depth=2/4/6) |
| `ablation_gtca_fusion_qm9.csv` | GTCA-Cat vs GTCA-CA comparison |
| `stats_depth_welch_qm9.csv` | Welch t-test results (depth ablation) |
| `stats_fusion_welch_qm9.csv` | Welch t-test results (fusion ablation) |

---

## License

MIT License

---

## Citation

> To be added upon publication.
