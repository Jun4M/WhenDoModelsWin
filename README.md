# WhenDoModelsWin

Low-data benchmarking of molecular property prediction models on the QM9 dataset.
Investigates **when each model architecture wins** — Transformer vs. GNN vs. hybrid vs. classical ML.

---

## Research Question

> In low-data regimes (N=100 train), which model architecture best predicts quantum molecular properties?

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| **ChemBERTa** | Transformer | `seyonec/ChemBERTa-zinc-base-v1`, fine-tuned on SMILES |
| **GCN** | Graph Neural Network | Graph Convolutional Network on molecular graphs |
| **GTCA** | Hybrid | GCN + ChemBERTa fusion (configurable BERT depth) |
| **RF / XGB / GPR** | Classical ML | Fingerprint-based sklearn regressors |
| **AttentiveFP / PaiNN / GPS** | Advanced GNN | State-of-the-art graph architectures |

---

## Targets

QM9 quantum chemical properties:
- `homo` — Highest Occupied Molecular Orbital energy
- `lumo` — Lowest Unoccupied Molecular Orbital energy
- `gap`  — HOMO-LUMO gap

---

## Setup

```bash
pip install -r requirements.txt
# For GPU:
pip install -r requirements_gpu.txt
```

---

## Usage

```bash
# Run all models on all targets
python main.py

# Specific target and device
python main.py --target homo lumo --device cuda

# Skip specific models
python main.py --skip_transformer --skip_gcn

# Custom data size
python main.py --train_size 200 --val_size 100 --test_size 5000
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | all | `homo`, `lumo`, `gap` |
| `--device` | auto | `cuda` / `cpu` |
| `--train_size` | 100 | Training set size |
| `--val_size` | 100 | Validation set size |
| `--test_size` | 10000 | Test set size |
| `--seed` | 42 | Random seed |
| `--epochs_transformer` | 200 | Transformer training epochs |
| `--epochs_gcn` | 300 | GCN training epochs |
| `--epochs_gtca` | 200 | GTCA training epochs |

---

## Output

Results are saved to `./results/`:
- `all_metrics.json` — MAE / RMSE / R² for all models and targets
- Per-model failure analysis CSVs
- Scatter plots and XAI visualizations

---

## Stack

- **Deep Learning**: PyTorch, PyTorch Geometric, HuggingFace Transformers
- **Chemistry**: RDKit, DeepChem
- **XAI**: Captum
- **ML Baselines**: scikit-learn, XGBoost, LightGBM
