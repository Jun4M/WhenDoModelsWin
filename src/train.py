"""
train.py
Training loops for all models.
- compute_metrics: RMSE, MAE, Pearson_R, R2
- TrainingLogger: weight norms + gradient noise (every 10 epochs)
- EarlyStopping: best val loss, NO disk .pt save by default
- train_transformer, train_gcn, train_gtca
- train_attentivefp, train_gps
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
    ChemBERTaRegressor, GCNRegressor, GTCAHybrid, GTCACrossAttn,
    AttentiveFPRegressor, GPSRegressor,
    SklearnRegressorWrapper,
    get_tokenizer, tokenize_smiles,
)


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
