"""
train.py
Training loops for all models.
- compute_metrics: RMSE, MAE, Pearson_R, R2
- TrainingLogger: weight norms + gradient noise (every 10 epochs)
- EarlyStopping: best val loss, NO disk .pt save by default
- train_transformer, train_gcn, train_gtca
- train_attentivefp, train_painn, train_gps
- train_sklearn (RF / XGB / GPR)
"""

import os
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import (
    ChemBERTaRegressor, ChemBERTa2Regressor, MoLFormerRegressor, SELFormerRegressor,
    GCNRegressor, GCNMTLRegressor, GTCAHybrid, GTCACrossAttn,
    AttentiveFPRegressor, AttentiveFPMTLRegressor, PaiNNRegressor, GPSRegressor,
    UniMolRegressor,
    SklearnRegressorWrapper, ChempropWrapper, KROVEXNet,
    get_tokenizer, get_tokenizer_v2, get_tokenizer_molformer, get_tokenizer_selformer,
    tokenize_smiles, tokenize_smiles_v2, tokenize_smiles_molformer, tokenize_selfies_selformer,
)
from src.featurizer import smiles_to_selfies, featurize_smiles_to_krovex_graph


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    r, _ = pearsonr(y_true, y_pred)
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "Pearson_R": float(r), "R2": r2}


# ===========================================================================
# Training Logger (every log_every epochs)
# ===========================================================================

class TrainingLogger:
    """Records weight norms and gradient noise ratio every `log_every` epochs."""

    def __init__(self, log_path: str = None, log_every: int = 10):
        self.log_path  = log_path
        self.log_every = log_every
        self.rows      = []

    def maybe_log(self, epoch: int, model: nn.Module,
                  train_loss: float, val_loss: float):
        if epoch % self.log_every != 0:
            return
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            key = name.replace('.', '_')
            row[f"wnorm_{key}"] = float(param.data.norm())
            if param.grad is not None:
                pnorm = float(param.data.norm()) + 1e-8
                gnorm = float(param.grad.norm())
                row[f"gnr_{key}"] = gnorm / pnorm
        self.rows.append(row)

    def save(self):
        if self.rows and self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
            print(f"  [log] training log → {self.log_path}")


# ===========================================================================
# Early Stopping (memory-only, no .pt save)
# ===========================================================================

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def save_model(model: nn.Module, path: str):
    """Optional: manually save a model to disk."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  [checkpoint] saved → {path}")


# ===========================================================================
# SMILES Dataset (for Transformer / GTCA)
# ===========================================================================

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], float(self.labels[idx])


# ===========================================================================
# Model A: Transformer
# ===========================================================================

def train_transformer(
    train_smiles, train_y,
    val_smiles,   val_y,
    test_smiles,  test_y,
    target_name: str = 'target',
    results_dir: str = '.',
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 2e-5,
    patience: int = 25,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = get_tokenizer()
    model = ChemBERTaRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=patience)
    logger = TrainingLogger(log_path=log_path)

    def encode(smiles):
        return tokenize_smiles(smiles, tokenizer, device=device)

    def run_epoch(smiles_batch, y_batch, train=True):
        model.train() if train else model.eval()
        total_loss, preds = 0.0, []
        for i in range(0, len(smiles_batch), batch_size):
            sb = smiles_batch[i: i + batch_size]
            yb = torch.tensor(y_batch[i: i + batch_size], dtype=torch.float).to(device)
            ids, mask = encode(sb)
            with torch.set_grad_enabled(train):
                out  = model(ids, mask)
                loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(sb)
            preds.extend(out.detach().cpu().numpy())
        return total_loss / len(smiles_batch), np.array(preds)

    print(f"\n=== Transformer ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(train_smiles, train_y, train=True)
        val_loss,   _ = run_epoch(val_smiles,   val_y,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds = run_epoch(test_smiles, test_y, train=False)
    metrics = compute_metrics(test_y, test_preds)
    print(f"  [Transformer] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_y}


# ===========================================================================
# Model A2: Transformer (ChemBERTa-2)
# ===========================================================================

def train_chemberta2(
    train_smiles, train_y,
    val_smiles,   val_y,
    test_smiles,  test_y,
    target_name: str = 'target',
    results_dir: str = '.',
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 2e-5,
    patience: int = 25,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:
    """Same loop as train_transformer; uses ChemBERTa2Regressor + get_tokenizer_v2."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = get_tokenizer_v2()
    model = ChemBERTa2Regressor().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=patience)
    logger = TrainingLogger(log_path=log_path)

    def encode(smiles):
        return tokenize_smiles_v2(smiles, tokenizer, device=device)

    def run_epoch(smiles_batch, y_batch, train=True):
        model.train() if train else model.eval()
        total_loss, preds = 0.0, []
        for i in range(0, len(smiles_batch), batch_size):
            sb = smiles_batch[i: i + batch_size]
            yb = torch.tensor(y_batch[i: i + batch_size], dtype=torch.float).to(device)
            ids, mask = encode(sb)
            with torch.set_grad_enabled(train):
                out  = model(ids, mask)
                loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(sb)
            preds.extend(out.detach().cpu().numpy())
        return total_loss / len(smiles_batch), np.array(preds)

    print(f"\n=== ChemBERTa-2 ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(train_smiles, train_y, train=True)
        val_loss,   _ = run_epoch(val_smiles,   val_y,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds = run_epoch(test_smiles, test_y, train=False)
    metrics = compute_metrics(test_y, test_preds)
    print(f"  [ChemBERTa-2] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_y}


def train_molformer(
    train_smiles, train_y,
    val_smiles,   val_y,
    test_smiles,  test_y,
    target_name: str = 'target',
    results_dir: str = '.',
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 2e-5,
    patience: int = 25,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:
    """Same loop as train_chemberta2; uses MoLFormerRegressor + get_tokenizer_molformer.

    Key differences:
      - Masked mean pooling (not CLS) — done inside MoLFormerRegressor.forward()
      - trust_remote_code=True / deterministic_eval=True set in MoLFormerRegressor.__init__
      - max_length=202 passed to tokenize_smiles_molformer
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = get_tokenizer_molformer()
    model = MoLFormerRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=patience)
    logger = TrainingLogger(log_path=log_path)

    def encode(smiles):
        return tokenize_smiles_molformer(smiles, tokenizer, device=device)

    def run_epoch(smiles_batch, y_batch, train=True):
        model.train() if train else model.eval()
        total_loss, preds = 0.0, []
        for i in range(0, len(smiles_batch), batch_size):
            sb = smiles_batch[i: i + batch_size]
            yb = torch.tensor(y_batch[i: i + batch_size], dtype=torch.float).to(device)
            ids, mask = encode(sb)
            with torch.set_grad_enabled(train):
                out  = model(ids, mask)
                loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(sb)
            preds.extend(out.detach().cpu().numpy())
        return total_loss / len(smiles_batch), np.array(preds)

    print(f"\n=== MoLFormer ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(train_smiles, train_y, train=True)
        val_loss,   _ = run_epoch(val_smiles,   val_y,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds = run_epoch(test_smiles, test_y, train=False)
    metrics = compute_metrics(test_y, test_preds)
    print(f"  [MoLFormer] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_y}


def train_selformer(
    train_smiles, train_y,
    val_smiles,   val_y,
    test_smiles,  test_y,
    target_name: str = 'target',
    results_dir: str = '.',
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-5,
    patience: int = 25,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:
    """Same loop as train_molformer; uses SELFormerRegressor + SELFIES tokenizer.

    Key differences:
      - SMILES are converted to SELFIES internally before tokenization.
      - Molecules that fail SELFIES encoding are excluded from training/val/test.
      - CLS pooling (done inside SELFormerRegressor.forward).
      - lr=1e-5 (conservative for 12-layer pre-trained RoBERTa).
    """
    from src.featurizer import smiles_to_selfies

    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_selfies(smiles, y_array, split_name):
        """Convert SMILES → SELFIES, filter y to matching indices."""
        sels, valid_idx, fail_log = smiles_to_selfies(smiles)
        if len(fail_log) > 0:
            print(f"  [SELFormer/{split_name}] {len(fail_log)} SMILES failed SELFIES encoding, excluded.")
        y_filtered = y_array[valid_idx] if len(valid_idx) < len(smiles) else y_array
        return sels, y_filtered

    train_sels, train_y_f = to_selfies(train_smiles, train_y, 'train')
    val_sels,   val_y_f   = to_selfies(val_smiles,   val_y,   'val')
    test_sels,  test_y_f  = to_selfies(test_smiles,  test_y,  'test')

    tokenizer = get_tokenizer_selformer()
    model = SELFormerRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=patience)
    logger = TrainingLogger(log_path=log_path)

    def encode(selfies_batch):
        return tokenize_selfies_selformer(selfies_batch, tokenizer, device=device)

    def run_epoch(sels_batch, y_batch, train=True):
        model.train() if train else model.eval()
        total_loss, preds = 0.0, []
        for i in range(0, len(sels_batch), batch_size):
            sb = sels_batch[i: i + batch_size]
            yb = torch.tensor(y_batch[i: i + batch_size], dtype=torch.float).to(device)
            ids, mask = encode(sb)
            with torch.set_grad_enabled(train):
                out  = model(ids, mask)
                loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(sb)
            preds.extend(out.detach().cpu().numpy())
        return total_loss / len(sels_batch), np.array(preds)

    print(f"\n=== SELFormer ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(train_sels, train_y_f, train=True)
        val_loss,   _ = run_epoch(val_sels,   val_y_f,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds = run_epoch(test_sels, test_y_f, train=False)
    metrics = compute_metrics(test_y_f, test_preds)
    print(f"  [SELFormer] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_y_f}


# ===========================================================================
# Model B: GCN
# ===========================================================================

def train_gcn(
    train_pyg, val_pyg, test_pyg,
    target_name: str = 'target',
    results_dir: str = '.',
    node_feat_dim: int = 30,
    num_layers: int = 3,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = PyGDataLoader(train_pyg, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_pyg,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_pyg,  batch_size=256)

    model     = GCNRegressor(node_feat_dim=node_feat_dim, num_layers=num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, preds, labels = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            with torch.set_grad_enabled(train):
                out  = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y.squeeze(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.detach().cpu().numpy())
            labels.extend(batch.y.reshape(-1).cpu().numpy())
        return total_loss / len(loader.dataset), np.array(preds), np.array(labels)

    print(f"\n=== GCN ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss,   _, _ = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds, test_true = run_epoch(test_loader, train=False)
    metrics = compute_metrics(test_true, test_preds)
    print(f"  [GCN] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_true}


# ===========================================================================
# Model C: GTCA Hybrid
# ===========================================================================

def train_gtca(
    train_pyg, val_pyg, test_pyg,
    train_smiles, val_smiles, test_smiles,
    target_name: str = 'target',
    results_dir: str = '.',
    node_feat_dim: int = 30,
    bert_depth: int = None,
    gcn_layers: int = 3,
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 5e-5,
    patience: int = 25,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = get_tokenizer()

    def attach_smiles(pyg_list, smiles_list):
        for d, s in zip(pyg_list, smiles_list):
            d.smiles = s
        return pyg_list

    train_pyg = attach_smiles(train_pyg, train_smiles)
    val_pyg   = attach_smiles(val_pyg,   val_smiles)
    test_pyg  = attach_smiles(test_pyg,  test_smiles)

    train_loader = PyGDataLoader(train_pyg, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_pyg,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_pyg,  batch_size=64)

    model     = GTCAHybrid(node_feat_dim=node_feat_dim, bert_depth=bert_depth, gcn_layers=gcn_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, preds, labels = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            smi   = batch.smiles
            ids, mask = tokenize_smiles(smi, tokenizer, device=device)
            with torch.set_grad_enabled(train):
                out  = model(batch.x, batch.edge_index, batch.batch, ids, mask)
                loss = criterion(out.squeeze(), batch.y.squeeze(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.detach().cpu().numpy())
            labels.extend(batch.y.reshape(-1).cpu().numpy())
        return total_loss / len(loader.dataset), np.array(preds), np.array(labels)

    depth_str = f"depth={bert_depth}" if bert_depth else "depth=full"
    print(f"\n=== GTCA ({target_name}, {depth_str}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss,   _, _ = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds, test_true = run_epoch(test_loader, train=False)
    metrics = compute_metrics(test_true, test_preds)
    print(f"  [GTCA] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_true}


def train_gtca_ca(
    train_pyg, val_pyg, test_pyg,
    train_smiles, val_smiles, test_smiles,
    target_name: str = 'target',
    results_dir: str = '.',
    node_feat_dim: int = 30,
    bert_depth: int = None,
    gcn_layers: int = 3,
    ca_dim: int = 256,
    ca_heads: int = 4,
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 5e-5,
    patience: int = 25,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:
    """Train GTCACrossAttn: Q=graph, K=V=BERT tokens."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = get_tokenizer()

    def attach_smiles(pyg_list, smiles_list):
        for d, s in zip(pyg_list, smiles_list):
            d.smiles = s
        return pyg_list

    train_pyg = attach_smiles(train_pyg, train_smiles)
    val_pyg   = attach_smiles(val_pyg,   val_smiles)
    test_pyg  = attach_smiles(test_pyg,  test_smiles)

    train_loader = PyGDataLoader(train_pyg, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_pyg,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_pyg,  batch_size=64)

    model = GTCACrossAttn(
        node_feat_dim=node_feat_dim,
        bert_depth=bert_depth,
        gcn_layers=gcn_layers,
        ca_dim=ca_dim,
        ca_heads=ca_heads,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, preds, labels = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            smi   = batch.smiles
            ids, mask = tokenize_smiles(smi, tokenizer, device=device)
            with torch.set_grad_enabled(train):
                out  = model(batch.x, batch.edge_index, batch.batch, ids, mask)
                loss = criterion(out.squeeze(), batch.y.squeeze(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.detach().cpu().numpy())
            labels.extend(batch.y.reshape(-1).cpu().numpy())
        return total_loss / len(loader.dataset), np.array(preds), np.array(labels)

    depth_str = f"depth={bert_depth}" if bert_depth else "depth=full"
    print(f"\n=== GTCA-CA ({target_name}, {depth_str}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss,   _, _ = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds, test_true = run_epoch(test_loader, train=False)
    metrics = compute_metrics(test_true, test_preds)
    print(f"  [GTCA-CA] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_true}


# ===========================================================================
# AttentiveFP
# ===========================================================================

def train_attentivefp(
    train_pyg, val_pyg, test_pyg,
    target_name: str = 'target',
    node_feat_dim: int = 30,
    edge_dim: int = 11,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = PyGDataLoader(train_pyg, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_pyg,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_pyg,  batch_size=256)

    model     = AttentiveFPRegressor(in_channels=node_feat_dim, edge_dim=edge_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, preds, labels = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            ea = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            with torch.set_grad_enabled(train):
                out  = model(batch.x, batch.edge_index, ea, batch.batch)
                loss = criterion(out.squeeze(), batch.y.squeeze(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.detach().cpu().numpy())
            labels.extend(batch.y.reshape(-1).cpu().numpy())
        return total_loss / len(loader.dataset), np.array(preds), np.array(labels)

    print(f"\n=== AttentiveFP ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss,   _, _ = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds, test_true = run_epoch(test_loader, train=False)
    metrics = compute_metrics(test_true, test_preds)
    print(f"  [AttentiveFP] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_true}


# ===========================================================================
# PaiNN (3D)
# ===========================================================================

def train_painn(
    train_3d, val_3d, test_3d,
    train_y, val_y, test_y,
    target_name: str = 'target',
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:
    """
    train_3d/val_3d/test_3d: list of PyG Data with .pos attribute.
    y values attached separately (already aligned).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    for d, y in zip(train_3d, train_y):
        d.y = torch.tensor([float(y)])
    for d, y in zip(val_3d, val_y):
        d.y = torch.tensor([float(y)])
    for d, y in zip(test_3d, test_y):
        d.y = torch.tensor([float(y)])

    train_loader = PyGDataLoader(train_3d, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_3d,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_3d,  batch_size=256)

    model     = PaiNNRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, preds, labels = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            with torch.set_grad_enabled(train):
                rei  = getattr(batch, 'radius_edge_index', None)
                out  = model(batch.x, batch.pos, batch.batch, radius_edge_index=rei)
                loss = criterion(out.squeeze(), batch.y.squeeze(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.detach().cpu().numpy())
            labels.extend(batch.y.reshape(-1).cpu().numpy())
        return total_loss / len(loader.dataset), np.array(preds), np.array(labels)

    print(f"\n=== PaiNN ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss,   _, _ = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds, test_true = run_epoch(test_loader, train=False)
    metrics = compute_metrics(test_true, test_preds)
    print(f"  [PaiNN] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_true}


# ===========================================================================
# UniMol (SE(3)-invariant Gaussian pair-bias transformer)
# ===========================================================================

def train_unimol(
    train_3d, val_3d, test_3d,
    train_y, val_y, test_y,
    target_name: str = 'target',
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 30,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:
    """
    Train UniMolRegressor on 3D molecules.

    train_3d/val_3d/test_3d: list of PyG Data with .z (int64 atomic numbers)
        and .pos (3D coordinates).  y values attached separately.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    for d, y in zip(train_3d, train_y):
        d.y = torch.tensor([float(y)])
    for d, y in zip(val_3d, val_y):
        d.y = torch.tensor([float(y)])
    for d, y in zip(test_3d, test_y):
        d.y = torch.tensor([float(y)])

    train_loader = PyGDataLoader(train_3d, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_3d,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_3d,  batch_size=128)

    model     = UniMolRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, preds, labels = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            with torch.set_grad_enabled(train):
                out  = model(batch.z, batch.pos, batch.batch)
                loss = criterion(out.squeeze(), batch.y.squeeze(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.detach().cpu().numpy())
            labels.extend(batch.y.reshape(-1).cpu().numpy())
        return total_loss / len(loader.dataset), np.array(preds), np.array(labels)

    print(f"\n=== UniMol ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss,   _, _ = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds, test_true = run_epoch(test_loader, train=False)
    metrics = compute_metrics(test_true, test_preds)
    print(f"  [UniMol] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_true}


# ===========================================================================
# UniMol Pretrained (unimol-tools, fine-tuning official checkpoint)
# ===========================================================================

class UniMolPretrainedWrapper:
    """Thin wrapper around unimol-tools MolPredict for pipeline compatibility."""
    def __init__(self, model_dir: str, target_name: str):
        self.model_dir = model_dir
        self.target_name = target_name


def train_unimol_pretrained(
    train_smiles, train_y,
    val_smiles, val_y,   # ignored — unimol-tools handles its own internal split
    test_smiles, test_y,
    target_name: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 20,
    device: str = 'cuda',
    log_path=None,
    seed: int = 0,
) -> dict:
    """Fine-tune pretrained Uni-Mol (DPTechnology) on a regression task.

    Returns same dict format as other train_* functions.
    Metrics are in NORMALIZED space — _apply_denorm() in run_learning_curve.py
    handles real-unit conversion.

    Limitations (unimol-tools 0.1.x):
    - val_smiles/val_y are IGNORED. unimol-tools does its own internal random
      80/20 split for validation/early-stopping. This is a paper limitation.
    - target_normalize=False to avoid double-normalization with our z-score protocol.
    - use_amp=False for reproducibility (mixed precision introduces non-determinism).
    """
    import os
    import tempfile
    import pandas as pd

    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError(
            f"Pretrained UniMol requires CUDA; got device={device!r}, "
            f"cuda_available={torch.cuda.is_available()}."
        )

    from unimol_tools import MolTrain, MolPredict  # import after CUDA guard

    # Seed everything before MolTrain instantiation (no seed kwarg in MolTrain)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    tmp_dir = tempfile.mkdtemp(prefix=f'unimol_pt_{target_name}_s{seed}_')
    train_csv = os.path.join(tmp_dir, 'train.csv')
    test_csv  = os.path.join(tmp_dir, 'test.csv')
    save_dir  = os.path.join(tmp_dir, 'model')

    pd.DataFrame({'SMILES': list(train_smiles), target_name: train_y}).to_csv(train_csv, index=False)
    pd.DataFrame({'SMILES': list(test_smiles)}).to_csv(test_csv, index=False)

    trainer = MolTrain(
        task='regression',
        data_type='molecule',
        smiles_col='SMILES',
        target_cols=[target_name],
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        early_stopping=patience,
        kfold=1,
        split='random',
        target_normalize='none', # CRITICAL: disable internal scaling (we pre-normalize); string 'none' is the accepted API value
        save_path=save_dir,
        use_cuda=True,
        use_amp=False,           # CRITICAL: disable for reproducibility
        use_ddp=False,
        metrics='mse',           # 'rmse' not in METRICS_REGISTER; 'mse' is equivalent for early-stopping ranking
        model_name='unimolv1',
        model_size='84m',
        conf_cache_level=0,      # skip SDF write — tmpdir is ephemeral
    )
    trainer.fit(train_csv)

    predictor = MolPredict(load_model=save_dir)
    preds_raw = predictor.predict(test_csv)

    # Robustly extract 1-D predictions regardless of return type
    if isinstance(preds_raw, pd.DataFrame):
        for col in [target_name, f'predict_{target_name}', 'predict_0', 'predict']:
            if col in preds_raw.columns:
                test_preds = preds_raw[col].values.astype(np.float32)
                break
        else:
            # Fallback: last numeric column
            test_preds = preds_raw.select_dtypes(include='number').iloc[:, -1].values.astype(np.float32)
    else:
        test_preds = np.asarray(preds_raw, dtype=np.float32).flatten()

    metrics = compute_metrics(test_y, test_preds)
    print(f"  [UniMol-Pretrained] {metrics}")

    return {
        'model': UniMolPretrainedWrapper(save_dir, target_name),
        'metrics': metrics,
        'test_preds': test_preds,
        'test_true': test_y,
    }


# ===========================================================================
# GPS
# ===========================================================================

def train_gps(
    train_pyg, val_pyg, test_pyg,
    target_name: str = 'target',
    node_feat_dim: int = 30,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = PyGDataLoader(train_pyg, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_pyg,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_pyg,  batch_size=256)

    model     = GPSRegressor(in_channels=node_feat_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, preds, labels = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            ea = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            with torch.set_grad_enabled(train):
                out  = model(batch.x, batch.edge_index, ea, batch.batch)
                loss = criterion(out.squeeze(), batch.y.squeeze(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds.extend(out.detach().cpu().numpy())
            labels.extend(batch.y.reshape(-1).cpu().numpy())
        return total_loss / len(loader.dataset), np.array(preds), np.array(labels)

    print(f"\n=== GPS ({target_name}) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss,   _, _ = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds, test_true = run_epoch(test_loader, train=False)
    metrics = compute_metrics(test_true, test_preds)
    print(f"  [GPS] {metrics}")

    return {"model": model, "metrics": metrics, "test_preds": test_preds, "test_true": test_true}


# ===========================================================================
# Sklearn models (RF / XGB / GPR)
# ===========================================================================

def train_sklearn(
    train_X: np.ndarray, train_y: np.ndarray,
    val_X: np.ndarray,   val_y: np.ndarray,
    test_X: np.ndarray,  test_y: np.ndarray,
    model_type: str = 'rf',
    seed: int = 42,
    train_size_limit_gpr: int = 500,
) -> dict:
    """
    No epoch loop — fit once and predict.
    GPR is automatically skipped if train set > train_size_limit_gpr.
    Returns None if skipped.
    """
    if model_type == 'gpr' and len(train_X) > train_size_limit_gpr:
        print(f"  [GPR] Skipping: train_size={len(train_X)} > {train_size_limit_gpr} (O(N³) limit)")
        return None

    print(f"\n=== {model_type.upper()} (sklearn) ===")
    wrapper = SklearnRegressorWrapper(model_type, random_state=seed)
    wrapper.fit(train_X, train_y)
    test_preds = wrapper.predict(test_X)
    metrics = compute_metrics(test_y, test_preds)
    print(f"  [{model_type.upper()}] {metrics}")

    return {"model": wrapper.model, "metrics": metrics,
            "test_preds": test_preds, "test_true": test_y}


def train_chemprop(
    train_smiles: list, train_y: np.ndarray,
    val_smiles:   list, val_y:   np.ndarray,
    test_smiles:  list, test_y:  np.ndarray,
    target_name: str = 'target',
    epochs: int = 200,
    batch_size: int = 50,
    patience: int = 20,
    seed: int = 0,
    **_kwargs,  # absorb device/log_path passed by run_learning_curve
) -> dict:
    """
    Train chemprop D-MPNN with default hyperparameters.

    Input y is normalized (z-score) — chemprop's output_transform is left as
    nn.Identity so NO internal target scaling is applied.  run_learning_curve
    performs the inverse transform via _apply_denorm before saving CSVs.

    Adversarial guard: chemprop's RegressionFFN has output_transform=None by
    default (→ nn.Identity). The CLI would set ScaleTransform/UnscaleTransform
    here, but we never call the CLI. Guard: assert output_transform is Identity.

    Lightning stdout/stderr: silenced by redirecting sys.stdout/sys.stderr during
    trainer.fit/predict calls and suppressing the lightning loggers.
    """
    import warnings, logging, sys, io
    import torch
    from rdkit import Chem
    import lightning as L

    from chemprop import data as cp_data, featurizers, models as cp_models, nn as cp_nn

    print(f"\n=== Chemprop ({target_name}) ===")

    # ── Silence lightning / chemprop verbosity ──────────────────────────────
    warnings.filterwarnings('ignore')
    for lg_name in ('lightning', 'lightning.pytorch', 'chemprop',
                    'pytorch_lightning', 'torch'):
        logging.getLogger(lg_name).setLevel(logging.ERROR)

    # ── Build datasets ───────────────────────────────────────────────────────
    def _make_dp(smiles_list, y_arr):
        out = []
        for smi, yv in zip(smiles_list, y_arr):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            out.append(cp_data.MoleculeDatapoint(mol=mol, y=np.array([float(yv)])))
        return out

    feat      = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dp  = _make_dp(train_smiles, train_y)
    val_dp    = _make_dp(val_smiles,   val_y)
    test_dp   = _make_dp(test_smiles,  test_y)

    train_ds  = cp_data.MoleculeDataset(train_dp, feat)
    val_ds    = cp_data.MoleculeDataset(val_dp,   feat)
    test_ds   = cp_data.MoleculeDataset(test_dp,  feat)

    # Use actual train_y values aligned with dp (some smiles may be filtered)
    train_y_arr = np.array([dp.y[0] for dp in train_dp])
    val_y_arr   = np.array([dp.y[0] for dp in val_dp])
    test_y_arr  = np.array([dp.y[0] for dp in test_dp])

    train_dl  = cp_data.build_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                         num_workers=0, seed=seed)
    val_dl    = cp_data.build_dataloader(val_ds,   batch_size=batch_size, shuffle=False,
                                         num_workers=0)
    test_dl   = cp_data.build_dataloader(test_ds,  batch_size=batch_size, shuffle=False,
                                         num_workers=0)

    # ── Build model (chemprop defaults, no target scaling) ───────────────────
    torch.manual_seed(seed)
    mp   = cp_nn.BondMessagePassing()
    agg  = cp_nn.MeanAggregation()
    ffn  = cp_nn.RegressionFFN(n_tasks=1)
    mpnn = cp_models.MPNN(message_passing=mp, agg=agg, predictor=ffn)

    # Guard: ensure chemprop is NOT doing its own target scaling
    import torch.nn as _nn
    assert isinstance(mpnn.predictor.output_transform, _nn.Identity), (
        "ChempropWrapper: output_transform is not Identity — chemprop may be "
        "double-normalizing targets. Check RegressionFFN(output_transform=...)."
    )

    # ── Train ────────────────────────────────────────────────────────────────
    es = L.pytorch.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, mode='min',
    )
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[es],
    )

    # Redirect stdout/stderr to suppress lightning's fd-level output
    _buf = io.StringIO()
    _old_out, _old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _buf
    try:
        trainer.fit(mpnn, train_dl, val_dl)
        raw_preds = trainer.predict(mpnn, test_dl)
    finally:
        sys.stdout, sys.stderr = _old_out, _old_err

    # ── Extract predictions (normalized space) ───────────────────────────────
    # trainer.predict() returns list[Tensor(batch, 1)]; concat across batches
    test_preds = torch.cat(raw_preds, dim=0)[:, 0].numpy()

    metrics = compute_metrics(test_y_arr, test_preds)
    print(f"  [Chemprop] {metrics}")

    return {
        "model":      ChempropWrapper(mpnn, trainer),
        "metrics":    metrics,
        "test_preds": test_preds,
        "test_true":  test_y_arr,
    }


# ===========================================================================
# train_krovex — KROVEX (Jang et al. 2026)
# ===========================================================================

def train_krovex(
    train_smiles: list,
    train_y: np.ndarray,
    val_smiles: list,
    val_y: np.ndarray,
    test_smiles: list,
    test_y: np.ndarray,
    target_name: str = 'target',
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
    nsis: int = 20,
    max_iter_isis: int = 5,
    alpha_grid: np.ndarray = None,
    l1_ratio_grid: np.ndarray = None,
    **_kwargs,
) -> dict:
    """Train KROVEX: per-fold descriptor selection + GCN + Kronecker fusion.

    Pipeline (per training fold):
      1. Descriptor selection (ISIS + Elastic Net) on train_smiles/train_y only
      2. Apply selected descriptors to val/test (no re-fitting)
      3. Featurize SMILES to KROVEX graphs (8-dim mendeleev atom features)
      4. Attach descriptor vector to each graph Data object
      5. Train KROVEXNet(num_desc) with EarlyStopping on val loss

    Returns
    -------
    dict with keys: model, metrics, test_preds, test_true, selected_descriptors
      selected_descriptors : list[str] — descriptor names chosen for this fold
    """
    from torch_geometric.data import Data as PyGData
    from src.descriptor_selection import (
        select_descriptors_per_fold, apply_descriptor_selection,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    # ── Step 1: Descriptor selection (train only) ────────────────────────────
    selected_names, fit_stats = select_descriptors_per_fold(
        train_smiles, train_y,
        seed=seed, nsis=nsis, max_iter_isis=max_iter_isis,
        alpha_grid=alpha_grid, l1_ratio_grid=l1_ratio_grid,
    )
    num_desc = max(len(selected_names), 1)  # guard: at least 1 dim for bmm
    print(f"  [KROVEX] {len(selected_names)} descriptors selected: {selected_names[:5]}{'...' if len(selected_names)>5 else ''}")

    # ── Step 2: Apply descriptors to all splits ──────────────────────────────
    train_desc = apply_descriptor_selection(train_smiles, selected_names, fit_stats)  # (N_train, k)
    val_desc   = apply_descriptor_selection(val_smiles,   selected_names, fit_stats)  # (N_val, k)
    test_desc  = apply_descriptor_selection(test_smiles,  selected_names, fit_stats)  # (N_test, k)

    # Pad to num_desc if selected_names was empty
    if train_desc.shape[1] == 0:
        train_desc = np.zeros((len(train_smiles), 1), dtype=np.float32)
        val_desc   = np.zeros((len(val_smiles),   1), dtype=np.float32)
        test_desc  = np.zeros((len(test_smiles),  1), dtype=np.float32)

    # ── Step 3: Graph featurization ──────────────────────────────────────────
    def _build_split(smiles_list, y_arr, desc_matrix):
        graphs, g_valid = featurize_smiles_to_krovex_graph(smiles_list)
        data_list, y_out = [], []
        for local_i, global_i in enumerate(g_valid):
            g = graphs[local_i]
            dv = torch.tensor(desc_matrix[global_i], dtype=torch.float).unsqueeze(0)  # (1, k)
            data = PyGData(
                x=g.x,
                edge_index=g.edge_index,
                y=torch.tensor([float(y_arr[global_i])]),
                desc=dv,
            )
            data_list.append(data)
            y_out.append(float(y_arr[global_i]))
        return data_list, np.array(y_out, dtype=np.float32)

    train_data, train_y_clean = _build_split(train_smiles, train_y, train_desc)
    val_data,   val_y_clean   = _build_split(val_smiles,   val_y,   val_desc)
    test_data,  test_y_clean  = _build_split(test_smiles,  test_y,  test_desc)

    if not train_data:
        raise RuntimeError("KROVEX: no valid training molecules after featurization.")

    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader  = PyGDataLoader(test_data,  batch_size=batch_size, shuffle=False)

    # ── Step 4: Build model ──────────────────────────────────────────────────
    torch.manual_seed(seed)
    model = KROVEXNet(num_desc=num_desc, dim_in=8).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # ── Step 5: Training loop ────────────────────────────────────────────────
    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(dev)
            pred  = model(batch.x, batch.edge_index, batch.batch, batch.desc)
            loss  = criterion(pred, batch.y.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev)
                pred  = model(batch.x, batch.edge_index, batch.batch, batch.desc)
                val_losses.append(criterion(pred, batch.y.squeeze(-1)).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Step 6: Evaluate on test ─────────────────────────────────────────────
    model.eval()
    preds_list, true_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(dev)
            pred  = model(batch.x, batch.edge_index, batch.batch, batch.desc)
            preds_list.append(pred.cpu().numpy())
            true_list.append(batch.y.squeeze(-1).cpu().numpy())

    test_preds = np.concatenate(preds_list) if preds_list else np.array([])
    test_true  = np.concatenate(true_list)  if true_list  else test_y_clean

    metrics = compute_metrics(test_true, test_preds)
    print(f"  [KROVEX] {metrics}")

    return {
        "model":                model,
        "metrics":              metrics,
        "test_preds":           test_preds,
        "test_true":            test_true,
        "selected_descriptors": selected_names,
    }


# ===========================================================================
# MTL shared training loop (AFP-MTL and GCN-MTL reuse this)
# ===========================================================================

import torch.nn.functional as _F


def _run_mtl_training(
    model,
    forward_fn,        # callable(model, batch) -> (B, n_tasks) tensor
    train_loader,
    val_loader,
    test_loader,
    n_tasks: int,
    target_names: list,
    optimizer,
    scheduler,
    es,
    logger,
    epochs: int,
    device: str,
    label: str = 'MTL',
) -> dict:
    """Shared training loop for all MTL models.

    forward_fn captures model-specific forward signature (e.g. AFP needs edge_attr,
    GCN does not). Everything else (loss, early stopping, logging) is identical.
    """

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        for batch in loader:
            batch = batch.to(device)
            y = batch.y.view(-1, n_tasks)          # (B, n_tasks)
            if torch.any(torch.isnan(y)):
                raise ValueError(
                    f"NaN detected in batch labels for {label} MTL training "
                    f"(tasks: {target_names}). Check per-task z-score stats."
                )
            with torch.set_grad_enabled(train):
                out  = forward_fn(model, batch)    # (B, n_tasks)
                loss = _F.mse_loss(out, y)         # mean over B*n_tasks
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
        preds  = np.concatenate(all_preds,  axis=0)  # (N, n_tasks)
        labels = np.concatenate(all_labels, axis=0)  # (N, n_tasks)
        return total_loss / len(loader.dataset), preds, labels

    print(f"\n=== {label} ({n_tasks} tasks) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss,   _, _ = run_epoch(val_loader,   train=False)
        scheduler.step(val_loss)
        logger.maybe_log(epoch, model, train_loss, val_loss)
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"  Early stop @ epoch {epoch}")
            break

    es.restore(model)
    logger.save()

    _, test_preds, test_true = run_epoch(test_loader, train=False)

    metrics_per_task = {}
    for i, tname in enumerate(target_names):
        m = compute_metrics(test_true[:, i], test_preds[:, i])
        metrics_per_task[tname] = m
        print(f"  [{label}:{tname:>6s}] {m}")

    return {
        "model":            model,
        "metrics_per_task": metrics_per_task,
        "test_preds":       test_preds,
        "test_true":        test_true,
    }


# ===========================================================================
# AttentiveFP MTL (QM9 12-task multi-task learning)
# ===========================================================================

def train_attentivefp_mtl(
    train_pyg, val_pyg, test_pyg,
    train_y_normalized: np.ndarray,
    val_y_normalized: np.ndarray,
    test_y_normalized: np.ndarray,
    stats: list,
    target_names: list,
    n_tasks: int = 12,
    node_feat_dim: int = 30,
    edge_dim: int = 11,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:
    """Multi-task AttentiveFP for QM9 12 properties. See _run_mtl_training."""
    assert len(stats) == n_tasks, f"stats length {len(stats)} != n_tasks {n_tasks}"
    assert len(target_names) == n_tasks, (
        f"target_names length {len(target_names)} != n_tasks {n_tasks}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = PyGDataLoader(train_pyg, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_pyg,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_pyg,  batch_size=256)

    model     = AttentiveFPMTLRegressor(
        in_channels=node_feat_dim, edge_dim=edge_dim, n_tasks=n_tasks,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def afp_forward(m, batch):
        ea = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        return m(batch.x, batch.edge_index, ea, batch.batch)

    return _run_mtl_training(
        model, afp_forward,
        train_loader, val_loader, test_loader,
        n_tasks, target_names,
        optimizer, scheduler, es, logger,
        epochs, device, label='AFP-MTL',
    )


# ===========================================================================
# GCN MTL (QM9 12-task multi-task learning, robustness check vs AFP-MTL)
# ===========================================================================

def train_gcn_mtl(
    train_pyg, val_pyg, test_pyg,
    train_y_normalized: np.ndarray,
    val_y_normalized: np.ndarray,
    test_y_normalized: np.ndarray,
    stats: list,
    target_names: list,
    n_tasks: int = 12,
    node_feat_dim: int = 30,
    edge_dim: int = 11,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 30,
    device: str = 'cpu',
    log_path: str = None,
    seed: int = 0,
) -> dict:
    """Multi-task GCN for QM9 12 properties.

    Identical training protocol to train_attentivefp_mtl; encoder is GCNMTLRegressor
    (3-layer GCNConv + global mean pool + multi-output head) instead of AttentiveFP.

    Returns:
        {
          "model": GCNMTLRegressor,
          "metrics_per_task": {task_name: {"RMSE":..., "MAE":..., "Pearson_R":..., "R2":...}},
          "test_preds": np.ndarray (N, n_tasks),
          "test_true":  np.ndarray (N, n_tasks),
        }
    """
    assert len(stats) == n_tasks, f"stats length {len(stats)} != n_tasks {n_tasks}"
    assert len(target_names) == n_tasks, (
        f"target_names length {len(target_names)} != n_tasks {n_tasks}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = PyGDataLoader(train_pyg, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_pyg,   batch_size=batch_size)
    test_loader  = PyGDataLoader(test_pyg,  batch_size=256)

    model     = GCNMTLRegressor(
        node_feat_dim=node_feat_dim, n_tasks=n_tasks,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    es        = EarlyStopping(patience=patience)
    logger    = TrainingLogger(log_path=log_path)

    def gcn_forward(m, batch):
        return m(batch.x, batch.edge_index, batch.batch)

    return _run_mtl_training(
        model, gcn_forward,
        train_loader, val_loader, test_loader,
        n_tasks, target_names,
        optimizer, scheduler, es, logger,
        epochs, device, label='GCN-MTL',
    )
