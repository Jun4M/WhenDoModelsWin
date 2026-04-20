"""
models.py
Model architectures:
  A: ChemBERTaRegressor (Transformer, seyonec/ChemBERTa-zinc-base-v1, 6 layers)
  B: GCNRegressor
  C: GTCAHybrid (GCN + ChemBERTa, configurable bert_depth)
  ML: SklearnRegressorWrapper (RF, XGB, GPR)
  GNN+: AttentiveFPRegressor, GPSRegressor
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoTokenizer, AutoModel, AutoConfig


CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"   # 6 layers, hidden=768


# ===========================================================================
# Model A: Transformer (ChemBERTa)
# ===========================================================================

class ChemBERTaRegressor(nn.Module):
    def __init__(self, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CHEMBERTA_MODEL)
        hidden = self.encoder.config.hidden_size  # 768

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.regressor(cls).squeeze(-1)


# ===========================================================================
# Model B: GCN
# ===========================================================================

class GCNRegressor(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)

    def get_graph_embedding(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return global_mean_pool(x, batch)


# ===========================================================================
# Model C: GTCA Hybrid (configurable BERT depth)
# ===========================================================================

class GTCAHybrid(nn.Module):
    """
    GNN-Transformer Cooperative Architecture.
    bert_depth: number of ChemBERTa encoder layers to use (1 ~ max_layers).
                None = use all layers.
    """

    def __init__(
        self,
        node_feat_dim: int = 30,
        gcn_hidden: int = 128,
        gcn_layers: int = 3,
        dropout: float = 0.1,
        freeze_bert: bool = False,
        bert_depth: int = None,
    ):
        super().__init__()

        # GCN encoder
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns   = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(gcn_layers):
            self.gcn_convs.append(GCNConv(in_dim, gcn_hidden))
            self.gcn_bns.append(nn.BatchNorm1d(gcn_hidden))
            in_dim = gcn_hidden

        # BERT encoder
        self.bert = AutoModel.from_pretrained(CHEMBERTA_MODEL)
        bert_hidden = self.bert.config.hidden_size  # 768
        total_layers = len(self.bert.encoder.layer)

        # Handle bert_depth
        if bert_depth is None:
            self._bert_depth = total_layers
        elif bert_depth > total_layers:
            warnings.warn(
                f"bert_depth={bert_depth} > total_layers={total_layers}. Capping to {total_layers}."
            )
            self._bert_depth = total_layers
        else:
            self._bert_depth = bert_depth

        # Freeze unused BERT layers (save compute)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i >= self._bert_depth:
                    for p in layer.parameters():
                        p.requires_grad = False

        self.dropout = dropout
        fused_dim = gcn_hidden + bert_hidden  # 128 + 768 = 896

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _gcn_embed(self, x, edge_index, batch):
        for conv, bn in zip(self.gcn_convs, self.gcn_bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def _bert_embed(self, input_ids, attention_mask):
        """Forward through BERT, return CLS token from encoder layer _bert_depth.
        Layers beyond _bert_depth are frozen so don't update, and we pick
        the hidden state before those frozen layers."""
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[0] = embedding output
        # hidden_states[i] = output after encoder layer i  (i = 1..num_layers)
        return out.hidden_states[self._bert_depth][:, 0, :]  # CLS token

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        graph_emb = self._gcn_embed(x, edge_index, batch)
        bert_emb  = self._bert_embed(input_ids, attention_mask)
        fused     = torch.cat([graph_emb, bert_emb], dim=-1)
        return self.fusion_head(fused).squeeze(-1)


# ===========================================================================
# Model D: GTCACrossAttn  (Cross-Attention fusion)
#   Q = h_graph (GCN global mean pool)
#   K = V = BERT token hidden states
#   "From the graph's perspective, which BERT tokens matter?"
# ===========================================================================

class GTCACrossAttn(nn.Module):
    """
    GTCA with Cross-Attention fusion.
    Ablation counterpart to GTCAHybrid (Cat fusion).

    Architecture:
        GCN → mean_pool  → h_graph (B, gcn_hidden)
        BERT → tokens    → h_tokens (B, seq_len, bert_hidden)

        Q = q_proj(h_graph).unsqueeze(1)   (B, 1, ca_dim)
        K = k_proj(h_tokens)               (B, seq_len, ca_dim)
        V = v_proj(h_tokens)               (B, seq_len, ca_dim)
        attn_out = MHA(Q, K, V)            (B, 1, ca_dim) → (B, ca_dim)

        fused = cat([h_graph, attn_out])   (B, gcn_hidden + ca_dim)
        pred  = MLP(fused)                 scalar
    """

    def __init__(
        self,
        node_feat_dim: int = 30,
        gcn_hidden: int = 128,
        gcn_layers: int = 3,
        dropout: float = 0.1,
        bert_depth: int = None,
        ca_dim: int = 256,
        ca_heads: int = 4,
    ):
        super().__init__()

        # GCN encoder (identical to GTCAHybrid)
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns   = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(gcn_layers):
            self.gcn_convs.append(GCNConv(in_dim, gcn_hidden))
            self.gcn_bns.append(nn.BatchNorm1d(gcn_hidden))
            in_dim = gcn_hidden

        # BERT encoder (identical to GTCAHybrid)
        self.bert = AutoModel.from_pretrained(CHEMBERTA_MODEL)
        bert_hidden = self.bert.config.hidden_size  # 768
        total_layers = len(self.bert.encoder.layer)

        if bert_depth is None:
            self._bert_depth = total_layers
        elif bert_depth > total_layers:
            warnings.warn(f"bert_depth={bert_depth} > total_layers={total_layers}. Capping.")
            self._bert_depth = total_layers
        else:
            self._bert_depth = bert_depth

        for i, layer in enumerate(self.bert.encoder.layer):
            if i >= self._bert_depth:
                for p in layer.parameters():
                    p.requires_grad = False

        self.dropout = dropout

        # Cross-Attention projections
        self.q_proj = nn.Linear(gcn_hidden, ca_dim)
        self.k_proj = nn.Linear(bert_hidden, ca_dim)
        self.v_proj = nn.Linear(bert_hidden, ca_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ca_dim, num_heads=ca_heads,
            dropout=dropout, batch_first=True,
        )

        # Fusion head
        fused_dim = gcn_hidden + ca_dim  # 128 + 256 = 384
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _gcn_embed(self, x, edge_index, batch):
        for conv, bn in zip(self.gcn_convs, self.gcn_bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)  # (B, gcn_hidden)

    def _bert_tokens(self, input_ids, attention_mask):
        """Return all token hidden states at bert_depth layer. (B, seq_len, bert_hidden)"""
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[self._bert_depth]  # (B, seq_len, bert_hidden)

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        # Graph embedding: (B, gcn_hidden)
        h_graph = self._gcn_embed(x, edge_index, batch)

        # BERT token hidden states: (B, seq_len, bert_hidden)
        h_tokens = self._bert_tokens(input_ids, attention_mask)

        # Project to cross-attention space
        Q = self.q_proj(h_graph).unsqueeze(1)   # (B, 1, ca_dim)
        K = self.k_proj(h_tokens)               # (B, seq_len, ca_dim)
        V = self.v_proj(h_tokens)               # (B, seq_len, ca_dim)

        # key_padding_mask: True where padding (attention_mask=0)
        key_padding_mask = (attention_mask == 0)  # (B, seq_len)

        attn_out, _ = self.cross_attn(Q, K, V, key_padding_mask=key_padding_mask)
        attn_out = attn_out.squeeze(1)           # (B, ca_dim)

        # Fuse graph + cross-attn output
        fused = torch.cat([h_graph, attn_out], dim=-1)  # (B, gcn_hidden + ca_dim)
        return self.fusion_head(fused).squeeze(-1)


# ===========================================================================
# ML Baselines: Sklearn wrapper
# ===========================================================================

class SklearnRegressorWrapper:
    """Thin wrapper so RF/XGB/GPR share the same train/predict API."""

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=500, n_jobs=-1,
                random_state=kwargs.pop('random_state', 42), **kwargs
            )
        elif model_type == 'xgb':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=500, n_jobs=-1,
                random_state=kwargs.pop('random_state', 42),
                verbosity=0, **kwargs
            )
        elif model_type == 'gpr':
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            self.model = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=0,
                normalize_y=False, **kwargs
            )
        elif model_type == 'svr':
            from sklearn.svm import SVR
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1)),
            ])
        elif model_type == 'lgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=500, n_jobs=-1,
                random_state=kwargs.pop('random_state', 42),
                verbosity=-1, **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).astype(float)


# ===========================================================================
# AttentiveFP
# ===========================================================================

class AttentiveFPRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 30,
        edge_dim: int = 11,
        hidden_channels: int = 200,
        num_layers: int = 2,
        num_timesteps: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        from torch_geometric.nn.models import AttentiveFP
        self.afp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr may be None (fallback to zeros)
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.shape[1], 1, device=x.device)
        assert edge_attr.shape[1] == self.afp.edge_dim, \
            f"edge_attr dim {edge_attr.shape[1]} != AttentiveFP edge_dim {self.afp.edge_dim}"
        return self.afp(x, edge_index, edge_attr, batch).squeeze(-1)

# ===========================================================================
# GPS (Graph Positional-Structural = Graphormer substitute)
# ===========================================================================

class GPSRegressor(nn.Module):
    """
    GPS: General Powerful Scalable Graph Transformer (Rampasek et al., NeurIPS 2022).
    Uses GPSConv (local MPNN + global attention) from PyG.
    """

    def __init__(
        self,
        in_channels: int = 30,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        walk_length: int = 20,
    ):
        super().__init__()
        from torch_geometric.nn import GPSConv, GINEConv, global_mean_pool

        self.walk_length = walk_length

        # Input projection
        self.input_proj = nn.Linear(in_channels + walk_length, hidden_channels)

        # GPS layers (each = local GINEConv + global MultiheadAttention)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            # GINEConv needs edge_dim; we'll use 1 (dummy) if no edge features
            gine = GINEConv(local_nn, edge_dim=hidden_channels)
            self.layers.append(
                GPSConv(
                    channels=hidden_channels,
                    conv=gine,
                    heads=4,
                    dropout=attn_dropout,
                )
            )

        self.edge_proj = nn.Linear(11, hidden_channels)  # MolGraphConv edge_dim=11
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _compute_rwse_single(self, edge_index, n):
        """RWSE for a single molecule (n atoms) on CPU."""
        from torch_geometric.utils import add_self_loops, degree
        ei_sl, _ = add_self_loops(edge_index, num_nodes=n)
        deg = degree(ei_sl[0], num_nodes=n).clamp(min=1)
        row, col = ei_sl
        pe = torch.zeros(n, self.walk_length)
        p = torch.eye(n)
        for k in range(self.walk_length):
            p_new = torch.zeros_like(p)
            p_new[col] += p[row] / deg[row].unsqueeze(-1)
            p = p_new
            pe[:, k] = p.diagonal()
        return pe

    def _compute_rwse(self, edge_index, num_nodes, device, batch=None):
        """Compute RWSE on CPU per molecule, return tensor on original device."""
        orig_device = device
        edge_index = edge_index.cpu()
        pe = torch.zeros(num_nodes, self.walk_length)  # CPU
        if batch is None:
            pe = self._compute_rwse_single(edge_index, num_nodes)
        else:
            batch_cpu = batch.cpu()
            for b in batch_cpu.unique():
                mask = (batch_cpu == b).nonzero(as_tuple=True)[0]
                n = mask.shape[0]
                src_in = torch.isin(edge_index[0], mask)
                local_ei = edge_index[:, src_in]
                idx_map = torch.zeros(num_nodes, dtype=torch.long)
                idx_map[mask] = torch.arange(n)
                local_ei = idx_map[local_ei]
                pe[mask] = self._compute_rwse_single(local_ei, n)
        return pe.to(orig_device)

    def forward(self, x, edge_index, edge_attr, batch):
        num_nodes = x.size(0)
        # RWSE per molecule (avoids N_batch×N_batch OOM on large batches)
        pe = self._compute_rwse(edge_index, num_nodes, x.device, batch)
        x = torch.cat([x, pe], dim=-1)
        x = self.input_proj(x)

        # Edge features
        if edge_attr is not None and edge_attr.shape[1] == 11:
            edge_emb = self.edge_proj(edge_attr)
        else:
            edge_emb = torch.zeros(edge_index.shape[1], x.shape[1], device=x.device)

        for layer in self.layers:
            x = layer(x, edge_index, batch, edge_attr=edge_emb)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


# ===========================================================================
# Tokeniser helpers
# ===========================================================================

def get_tokenizer(max_length: int = 128):
    return AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)


def tokenize_smiles(smiles_list: list, tokenizer, max_length: int = 128, device='cpu'):
    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)
