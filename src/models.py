"""
models.py
Model architectures:
  A:  ChemBERTaRegressor  (seyonec/ChemBERTa-zinc-base-v1, 6 layers, hidden=768)
  A2: ChemBERTa2Regressor (DeepChem/ChemBERTa-77M-MTR, 3 layers, hidden=384)
  A3: MoLFormerRegressor  (ibm/MoLFormer-XL-both-10pct, 12 layers, hidden=768, linear attn)
  A4: SELFormerRegressor  (HUBioDataLab/SELFormer, 12 layers, hidden=768, SELFIES input)
  B:  GCNRegressor
  B2: GCNMTLRegressor (MTL variant of GCNRegressor)
  C:  GTCAHybrid (GCN + ChemBERTa concat fusion)
  D:  ChempropWrapper (chemprop D-MPNN, lightweight wrapper for train_chemprop())
  E:  KROVEXNet (2-layer mean-agg GCN + Kronecker descriptor fusion, Jang et al. 2026)
  F:  UniMolRegressor (SE(3)-invariant transformer, from-scratch, Gaussian pair-bias attention)
  ML: SklearnRegressorWrapper (RF, XGB, GPR, SVR, LGBM)
  GNN+: AttentiveFPRegressor, AttentiveFPMTLRegressor, PaiNNRegressor, GPSRegressor
"""

from src import _transformers_compat  # noqa: F401  # MUST precede MoLFormer load

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import MessagePassing
from transformers import AutoTokenizer, AutoModel, AutoConfig


CHEMBERTA_MODEL  = "seyonec/ChemBERTa-zinc-base-v1"    # 6 layers, hidden=768
CHEMBERTA2_MODEL = "DeepChem/ChemBERTa-77M-MTR"        # 3 layers, hidden=384, MTR pretrained
MOLFORMER_MODEL  = "ibm/MoLFormer-XL-both-10pct"       # 12 layers, hidden=768, linear attn


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
# Model A2: Transformer (ChemBERTa-2, DeepChem/ChemBERTa-77M-MTR)
# ===========================================================================

class ChemBERTa2Regressor(nn.Module):
    """
    ChemBERTa-2 (77M-MTR) regression head.

    Key differences from ChemBERTaRegressor:
      - hidden_size = 384 (not 768) — must NOT be hardcoded; auto-inferred
      - 3 encoder layers (not 6)
      - atom-level tokenizer with vocab_size=591 (not BPE)
      - MTR-pretrained on 77M-molecule PubChem corpus (property prediction)
    """
    def __init__(self, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CHEMBERTA2_MODEL)
        hidden = self.encoder.config.hidden_size  # 384 — auto-inferred, not hardcoded

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
# Model A3: MoLFormer (ibm/MoLFormer-XL-both-10pct)
# ===========================================================================

class MoLFormerRegressor(nn.Module):
    """
    MoLFormer-XL regression head with masked mean pooling.

    Key differences from ChemBERTaRegressor:
      - Pooling: masked mean over all real tokens (NOT CLS at position 0)
      - trust_remote_code=True required for custom linear attention code
      - deterministic_eval=True freezes random feature weights at inference
      - max_length=202 (MoLFormer's max_position_embeddings)
      - hidden_size=768 auto-inferred from config
      - Special tokens: cls=0, pad=2, sep=1, unk=2361
    """
    def __init__(self, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            MOLFORMER_MODEL,
            trust_remote_code=True,
            deterministic_eval=True,
        )
        # The MoLFormer checkpoint stores inv_freq=0 in its state_dict, making
        # cos/sin caches NaN. _transformers_compat.reinit_molformer_rotary()
        # recomputes them from dim/base. Must run after from_pretrained().
        _transformers_compat.reinit_molformer_rotary(self.encoder)

        hidden = self.encoder.config.hidden_size  # 768 — auto-inferred, not hardcoded

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
        hs = out.last_hidden_state                          # (B, L, H)
        mask = attention_mask.float().unsqueeze(-1)         # (B, L, 1)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, H)
        return self.regressor(pooled).squeeze(-1)


# ===========================================================================
# Model A4: SELFormer (HUBioDataLab/SELFormer)
# ===========================================================================

SELFORMER_MODEL = "HUBioDataLab/SELFormer"   # RoBERTa-based, 12 layers, hidden=768

class SELFormerRegressor(nn.Module):
    """
    SELFormer regression head using CLS pooling.

    Key properties:
      - Input: SELFIES strings (NOT SMILES) — caller must convert via smiles_to_selfies()
      - Pooling: CLS token (last_hidden_state[:, 0, :]) — standard for RoBERTa
      - No trust_remote_code needed (standard RoBERTa architecture)
      - Special tokens: cls=1, pad=3, sep=2, unk=0  (differs from all CB variants)
      - hidden_size=768 auto-inferred from config
      - max_length=128 (SELFIES tokens are ~1.5–2× longer than SMILES characters;
        128 covers most drug-like molecules; imatinib ~100 tokens)
    """
    def __init__(self, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(SELFORMER_MODEL)

        # Checkpoint sanity check — guard against MoLFormer-style buffer corruption.
        # SELFormer has no rotary embeddings, so the loop should find nothing to fail.
        # If a future checkpoint ships with corrupted buffers, this raises immediately.
        for name, buf in self.encoder.named_buffers():
            if any(k in name for k in ['inv_freq', 'cos_cached', 'sin_cached']):
                if not torch.isfinite(buf).all() or buf.abs().sum() == 0:
                    raise RuntimeError(
                        f"SELFormer checkpoint buffer corruption: {name} is zero or non-finite. "
                        f"Likely rotary embedding de-persisted (same class as MoLFormer issue). "
                        f"Add a reinit helper to src/_transformers_compat following "
                        f"reinit_molformer_rotary pattern."
                    )

        hidden = self.encoder.config.hidden_size  # 768 — auto-inferred, not hardcoded

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
        cls = out.last_hidden_state[:, 0, :]   # CLS token — RoBERTa convention
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


class GCNMTLRegressor(nn.Module):
    """Multi-task GCN: identical encoder to GCNRegressor, multi-output head.

    Encoder: num_layers GCNConv + BN + ReLU + Dropout → global_mean_pool → (B, hidden_dim).
    Head:    Linear(hidden_dim, 64) → ReLU → Dropout → Linear(64, n_tasks) → (B, n_tasks).

    forward output: (B, n_tasks) — no squeeze.
    Adversarial: NaN guard and per-task metric extraction in train_gcn_mtl (not here).
    """

    def __init__(
        self,
        node_feat_dim: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_tasks: int = 12,
    ):
        super().__init__()
        self.n_tasks = n_tasks
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
            nn.Linear(64, n_tasks),   # (B, n_tasks) — no squeeze
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)   # (B, n_tasks)


# ===========================================================================
# Model F: UniMol — SE(3)-invariant molecular transformer
#   From-scratch reimplementation inspired by Uni-Mol (Zhou et al. 2023).
#   Uses Gaussian RBF pairwise-distance bias in multi-head attention.
#   SE(3)-invariance: pairwise distances are rotation/translation invariant.
# ===========================================================================

class _GaussianLayer(nn.Module):
    """Gaussian RBF expansion of scalar distances.

    d → exp(-((d - mu_k) / sigma_k)^2), k = 0 .. n_rbf-1
    mu evenly spaced in [0, cutoff]; sigma = cutoff / n_rbf.
    """

    def __init__(self, n_rbf: int = 64, cutoff: float = 10.0):
        super().__init__()
        self.register_buffer('mu',    torch.linspace(0.0, cutoff, n_rbf))
        self.register_buffer('sigma', torch.full((n_rbf,), cutoff / n_rbf))

    def forward(self, d):
        # d: arbitrary shape (...) → output (..., n_rbf)
        return torch.exp(-((d.unsqueeze(-1) - self.mu) / self.sigma) ** 2)


class _UniMolLayer(nn.Module):
    """Pre-norm transformer layer with Gaussian pair-bias attention.

    pair_rbf (B, N, N, n_rbf) is projected to (n_heads,) and added to
    the attention logits before softmax — this injects 3D geometry without
    breaking permutation equivariance over atoms.
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int,
                 dropout: float, n_rbf: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # RBF → per-head scalar pair bias
        self.pair_bias = nn.Linear(n_rbf, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pair_rbf, pad_mask=None):
        # x:        (B, N, d_model)
        # pair_rbf: (B, N, N, n_rbf)
        # pad_mask: (B, N) bool, True = padding position
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # --- Self-attention with pair bias (pre-norm) ---
        residual = x
        x = self.norm1(x)
        Q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        K = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale           # (B, H, N, N)

        # Pair bias: (B, N, N, n_rbf) → linear → (B, N, N, H) → (B, H, N, N)
        bias = self.pair_bias(pair_rbf).permute(0, 3, 1, 2)
        attn = attn + bias

        # Mask out key positions that are padding (columns)
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask[:, None, None, :], float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard: all-padding row → 0
        attn = self.drop(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        x = residual + self.drop(out)

        # --- FFN (pre-norm) ---
        residual = x
        x = self.norm2(x)
        x = residual + self.drop(self.ffn(x))
        return x


class UniMolRegressor(nn.Module):
    """SE(3)-invariant molecular transformer with Gaussian pair-bias attention.

    From-scratch reimplementation inspired by Uni-Mol (Zhou et al. 2023, DP Technology).
    Trained from scratch on task data — no pretrained weights loaded.

    Architecture:
        atom_embed(z) →
        to_dense_batch →
        Gaussian RBF(pairwise distances) →
        _UniMolLayer × n_layers →
        LayerNorm → masked mean pool → MLP head → scalar

    SE(3)-invariance: only pairwise Euclidean distances enter the pair bias;
    these are invariant to rotation and translation of the entire molecule.

    forward(z, pos, batch) where:
      z:     (N_total,) int64 atomic numbers (1=H, 6=C, 7=N, 8=O, ...; 0=padding)
      pos:   (N_total, 3) float32 3D coordinates in Å
      batch: (N_total,) int64 molecule index per atom (from PyG DataLoader)
    """

    MAX_ATOM_NUM = 118   # full periodic table (H=1 .. Og=118); 0 = padding

    def __init__(
        self,
        d_model:  int   = 128,
        n_heads:  int   = 8,
        n_layers: int   = 4,
        d_ffn:    int   = 256,
        n_rbf:    int   = 64,
        cutoff:   float = 10.0,
        dropout:  float = 0.1,
    ):
        super().__init__()
        # z=0 → padding (embedding row is always 0 due to padding_idx)
        self.atom_embed = nn.Embedding(
            self.MAX_ATOM_NUM + 1, d_model, padding_idx=0,
        )
        self.gaussian = _GaussianLayer(n_rbf=n_rbf, cutoff=cutoff)
        self.layers   = nn.ModuleList([
            _UniMolLayer(d_model, n_heads, d_ffn, dropout, n_rbf)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z, pos, batch):
        from torch_geometric.utils import to_dense_batch

        # Atom embeddings (N_total, d_model)
        x = self.atom_embed(z.clamp(0, self.MAX_ATOM_NUM))

        # Dense batching: pad variable-length to (B, N_max, ...)
        x_dense,   mask = to_dense_batch(x,   batch)  # (B,N_max,D), (B,N_max) bool
        pos_dense, _    = to_dense_batch(pos, batch)  # (B, N_max, 3)

        # Pairwise SE(3)-invariant distances: (B, N_max, N_max)
        diff = pos_dense.unsqueeze(2) - pos_dense.unsqueeze(1)   # (B,N,N,3)
        dist = diff.norm(dim=-1)                                  # (B,N,N)

        # Gaussian RBF expansion: (B, N_max, N_max, n_rbf)
        pair_rbf = self.gaussian(dist)

        # pad_mask: True for positions that are padding (not real atoms)
        pad_mask = ~mask   # (B, N_max)

        for layer in self.layers:
            x_dense = layer(x_dense, pair_rbf, pad_mask)

        x_dense = self.norm(x_dense)

        # Masked mean pool over real atoms only
        x_dense = x_dense * mask.unsqueeze(-1).float()           # zero padding
        n_atoms = mask.sum(dim=1, keepdim=True).float()          # (B, 1)
        mol_emb = x_dense.sum(dim=1) / n_atoms.clamp(min=1.0)   # (B, d_model)

        return self.head(mol_emb).squeeze(-1)                    # (B,)


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

class ChempropWrapper:
    """
    Thin container for a trained chemprop MPNN + its lightning Trainer.
    Created by train_chemprop(); holds state needed for post-hoc predict().

    Not a nn.Module — chemprop's MPNN is a LightningModule and is managed
    by pytorch-lightning's Trainer. This wrapper keeps them together so
    the rest of the pipeline (tests, adversarial guards) has a single handle.
    """

    def __init__(self, mpnn, trainer):
        self.mpnn    = mpnn     # chemprop.models.MPNN (LightningModule)
        self.trainer = trainer  # lightning.Trainer


# ===========================================================================
# Model E: KROVEX (Jang et al. 2026)
#   2-layer mean-aggregation GCN (no self-loop, no root weight) +
#   Kronecker outer-product fusion with RDKit descriptors + MLP head.
#
#   Reference architecture from KROVEX_ref/model/KROVEX.py:
#     GCNLayer: mean(neighbours) → Linear  (NOT PyG GCNConv — no self-loop)
#     Kronecker: outer(h_graph, desc) → flatten to (B, 20 * num_desc)
#     MLP: 20k → 128 (BN+ReLU+Drop0.3) → 32 (BN+ReLU) → 1
# ===========================================================================

class _KROVEXConv(MessagePassing):
    """Pure mean-aggregation GCN layer: h_i = W · mean_{j ∈ N(i)} h_j.

    No self-loop (root weight). Matches the original DGL GCNLayer that
    uses fn.copy_u('h', 'm') + mean-reduce, then applies a Linear.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='mean')
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Aggregate: h_i = mean_{j∈N(i)} h_j
        agg = self.propagate(edge_index, x=x)  # (N, in_channels)
        return self.linear(agg)                  # (N, out_channels)

    def message(self, x_j):
        return x_j                               # copy source feature


class KROVEXNet(nn.Module):
    """KROVEX graph network (Jang et al. 2026).

    Parameters
    ----------
    num_desc : int — number of RDKit descriptor features (varies per fold)
    dim_in   : int — atom feature dimension (8 for mendeleev features)
    dim_out  : int — output dimension (1 for regression)

    Forward inputs
    --------------
    g_x          : (N_total_atoms, dim_in)   — node features (batch)
    edge_index   : (2, E_total)              — graph connectivity (batch)
    batch        : (N_total_atoms,)          — molecule index per atom
    desc         : (B, num_desc)             — z-scored descriptor vector
    """

    def __init__(self, num_desc: int, dim_in: int = 8, dim_out: int = 1):
        super().__init__()
        self.gc1 = _KROVEXConv(dim_in, 100)
        self.gc2 = _KROVEXConv(100, 20)

        self.fc1 = nn.Linear(20 * num_desc, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, dim_out)

        self.bn1     = nn.BatchNorm1d(128)
        self.bn2     = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g_x, edge_index, batch, desc):
        h = F.relu(self.gc1(g_x, edge_index))
        h = F.relu(self.gc2(h, edge_index))

        # Global mean pool: (B, 20)
        hg = global_mean_pool(h, batch)

        # Kronecker (outer product) fusion:
        #   hg  : (B, 20) → (B, 20, 1)
        #   desc: (B, k)  → (B, 1, k)
        #   bmm → (B, 20, k) → flatten → (B, 20*k)
        hg   = hg.unsqueeze(2)       # (B, 20, 1)
        desc = desc.unsqueeze(1)     # (B, 1, k)
        fused = torch.bmm(hg, desc)  # (B, 20, k)
        fused = fused.view(fused.size(0), -1)  # (B, 20*k)

        out = F.relu(self.bn1(self.fc1(fused)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)
        return out.squeeze(-1)


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
# AttentiveFP Multi-Task (QM9 MTL variant)
# ===========================================================================

class AttentiveFPMTLRegressor(nn.Module):
    """Multi-task AttentiveFP for simultaneous regression on n_tasks targets.

    Encoder: identical to AttentiveFPRegressor (PyG AttentiveFP backbone).
    Head: shared encoder → (B, n_tasks) via out_channels=n_tasks.

    forward output: (B, n_tasks) — one scalar per task per molecule.

    Adversarial guard:
      - output shape assertion in tests (not runtime) to avoid perf overhead.
      - Per-task NaN guard in train_attentivefp_mtl (not here).
    """

    def __init__(
        self,
        in_channels: int = 30,
        edge_dim: int = 11,
        hidden_channels: int = 200,
        num_layers: int = 2,
        num_timesteps: int = 2,
        dropout: float = 0.2,
        n_tasks: int = 12,
    ):
        super().__init__()
        from torch_geometric.nn.models import AttentiveFP
        self.n_tasks = n_tasks
        self.afp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=n_tasks,    # multi-output: (B, n_tasks)
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.shape[1], self.afp.edge_dim, device=x.device)
        assert edge_attr.shape[1] == self.afp.edge_dim, \
            f"edge_attr dim {edge_attr.shape[1]} != AttentiveFP edge_dim {self.afp.edge_dim}"
        return self.afp(x, edge_index, edge_attr, batch)  # (B, n_tasks) — no squeeze


# ===========================================================================
# PaiNN (pure PyTorch, no PaiNNConv dependency)
# Schütt et al. (2021) "Equivariant message passing for the prediction of
# tensorial properties and molecular spectra", ICML 2021.
# ===========================================================================

class _PaiNNMessage(nn.Module):
    """Message step: propagate scalar s and vector v separately."""

    def __init__(self, hidden_channels: int, num_rbf: int):
        super().__init__()
        F = hidden_channels
        # phi_s: maps source scalar → 3F channels (split: ds, v_vv, v_vr)
        self.phi_s = nn.Sequential(
            nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 3 * F)
        )
        # W_filter: RBF → 3F (same split)
        self.W_filter = nn.Sequential(
            nn.Linear(num_rbf, F), nn.SiLU(), nn.Linear(F, 3 * F)
        )

    def forward(self, s, v, edge_index, rbf, unit):
        # s: (N, F)  v: (N, F, 3)
        # rbf: (E, num_rbf)  unit: (E, 3)
        src, dst = edge_index
        F = s.shape[1]

        phi = self.phi_s(s[src])          # (E, 3F)
        W   = self.W_filter(rbf)          # (E, 3F)
        x   = phi * W                     # (E, 3F)

        x_s, x_vv, x_vr = x.split(F, dim=-1)  # each (E, F)

        # Scalar aggregation
        ds = torch.zeros_like(s)
        ds.scatter_add_(0, dst.unsqueeze(-1).expand_as(x_s), x_s)

        # Vector aggregation: contribution from neighbor v + direction r_ij
        # x_vv * v[src]: (E, F, 3)
        msg_vv = x_vv.unsqueeze(-1) * v[src]              # (E, F, 3)
        # x_vr * unit_ij: (E, F, 3)
        msg_vr = x_vr.unsqueeze(-1) * unit.unsqueeze(-2)  # (E, F, 3)

        dv  = torch.zeros_like(v)
        idx = dst.view(-1, 1, 1).expand_as(msg_vv)
        dv.scatter_add_(0, idx, msg_vv + msg_vr)

        return s + ds, v + dv


class _PaiNNUpdate(nn.Module):
    """Update step: equivariant gated residual (scalar + vector)."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        F = hidden_channels
        # Channel-wise linear transforms on v (no bias for equivariance)
        self.U = nn.Linear(F, F, bias=False)
        self.V = nn.Linear(F, F, bias=False)
        # Gate network: input = [s, ||V(v)||]  (2F → 3F)
        self.gate = nn.Sequential(
            nn.Linear(2 * F, F), nn.SiLU(), nn.Linear(F, 3 * F)
        )

    def forward(self, s, v):
        # v: (N, F, 3) — apply U, V over the F dimension
        # Permute to (N, 3, F), apply linear, permute back
        Uv = self.U(v.permute(0, 2, 1)).permute(0, 2, 1)  # (N, F, 3)
        Vv = self.V(v.permute(0, 2, 1)).permute(0, 2, 1)  # (N, F, 3)

        Vv_norm = Vv.norm(dim=-1)                           # (N, F)
        inner   = (Uv * Vv).sum(dim=-1)                    # (N, F)

        a = self.gate(torch.cat([s, Vv_norm], dim=-1))     # (N, 3F)
        F = s.shape[1]
        a_ss, a_sv, a_vv = a.split(F, dim=-1)

        s_new = s + a_ss + a_sv * inner
        v_new = v + a_vv.unsqueeze(-1) * Uv

        return s_new, v_new


class PaiNNRegressor(nn.Module):
    """
    PaiNN for 3D molecular property prediction (pure PyTorch, no PaiNNConv).
    Requires PyG Data with .pos (N_atoms, 3) attribute.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        cutoff: float = 5.0,
        num_rbf: int = 20,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.num_rbf = num_rbf

        self.atom_emb = nn.Embedding(100, hidden_channels)

        self.msg_layers = nn.ModuleList([
            _PaiNNMessage(hidden_channels, num_rbf) for _ in range(num_layers)
        ])
        self.upd_layers = nn.ModuleList([
            _PaiNNUpdate(hidden_channels) for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _rbf(self, dist):
        """Gaussian RBF: (E,) → (E, num_rbf)."""
        centers = torch.linspace(0.0, self.cutoff, self.num_rbf, device=dist.device, dtype=dist.dtype)
        width   = (self.cutoff / self.num_rbf) ** 2
        return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / width)

    @staticmethod
    def _radius_graph(pos, r, batch, max_num_neighbors=32):
        """Pure-PyTorch radius graph (avoids torch-cluster dependency)."""
        device = pos.device
        num_nodes = pos.shape[0]
        rows, cols = [], []
        for b in batch.unique():
            mask = (batch == b).nonzero(as_tuple=True)[0]
            p = pos[mask]                                    # (n, 3)
            dist = torch.cdist(p, p)                        # (n, n) Euclidean distances
            # exclude self-loops
            dist.fill_diagonal_(float('inf'))
            adj = dist < r
            local_src, local_dst = adj.nonzero(as_tuple=True)
            # cap neighbors
            if max_num_neighbors < p.shape[0] - 1:
                for node in range(p.shape[0]):
                    nbrs = (local_dst == node).nonzero(as_tuple=True)[0]
                    if len(nbrs) > max_num_neighbors:
                        keep = nbrs[dist[local_src[nbrs], node].argsort()[:max_num_neighbors]]
                        drop = nbrs[~torch.isin(nbrs, keep)]
                        adj[local_src[drop], node] = False
                local_src, local_dst = adj.nonzero(as_tuple=True)
            global_idx = mask
            rows.append(global_idx[local_src])
            cols.append(global_idx[local_dst])
        if len(rows) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)

    def forward(self, x, pos, batch, radius_edge_index=None):
        if radius_edge_index is not None:
            edge_index = radius_edge_index
        else:
            edge_index = self._radius_graph(pos, self.cutoff, batch, max_num_neighbors=32)
        src, dst   = edge_index
        diff       = pos[src] - pos[dst]                    # (E, 3)
        dist       = diff.norm(dim=-1)                      # (E,)
        unit       = diff / (dist.unsqueeze(-1) + 1e-8)     # (E, 3)
        rbf        = self._rbf(dist)                        # (E, num_rbf)

        atom_idx = x[:, 0].long().clamp(0, 99)
        s = self.atom_emb(atom_idx)                         # (N, F)
        v = torch.zeros(s.shape[0], s.shape[1], 3, device=s.device, dtype=s.dtype)  # (N, F, 3)

        for msg, upd in zip(self.msg_layers, self.upd_layers):
            s, v = msg(s, v, edge_index, rbf, unit)
            s, v = upd(s, v)

        # global sum pool (pure PyTorch, no torch_geometric dependency)
        num_graphs = int(batch.max().item()) + 1
        out = torch.zeros(num_graphs, s.shape[1], device=s.device, dtype=s.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(s), s)
        return self.head(out).squeeze(-1)


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


def get_tokenizer_v2(max_length: int = 128):
    """Tokenizer for ChemBERTa-2 (DeepChem/ChemBERTa-77M-MTR).

    Uses atom-level SMILES tokenization (vocab_size=591, not BPE).
    Special token IDs differ from ChemBERTa-1: cls=12, pad=0, sep=13, unk=11.
    """
    return AutoTokenizer.from_pretrained(CHEMBERTA2_MODEL)


def tokenize_smiles(smiles_list: list, tokenizer, max_length: int = 128, device='cpu'):
    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)


def tokenize_smiles_v2(smiles_list: list, tokenizer, max_length: int = 128, device='cpu'):
    """Tokenize for ChemBERTa-2. Same call signature as tokenize_smiles.

    Kept as a separate function (not aliased) to prevent accidental cross-use:
    applying get_tokenizer() output to tokenize_smiles_v2 (or vice versa)
    would silently produce wrong token IDs.
    """
    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)


def get_tokenizer_molformer(max_length: int = 202):
    """Tokenizer for MoLFormer-XL (ibm/MoLFormer-XL-both-10pct).

    Uses SMILES character-level tokenization (vocab_size=2362).
    Special token IDs: cls=0, pad=2, sep=1, unk=2361.
    Default max_length=202 matches MoLFormer's max_position_embeddings.
    trust_remote_code required for the custom tokenizer.
    """
    return AutoTokenizer.from_pretrained(MOLFORMER_MODEL, trust_remote_code=True)


def tokenize_smiles_molformer(smiles_list: list, tokenizer, max_length: int = 202, device='cpu'):
    """Tokenize for MoLFormer. Same call signature as tokenize_smiles.

    Kept as a separate function to prevent cross-tokenizer mistakes:
    MoLFormer (cls=0, pad=2) vs ChemBERTa (cls=0, pad=1) would silently
    produce wrong padding behavior if tokenizers are mixed.
    """
    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)


def get_tokenizer_selformer(max_length: int = 128):
    """Tokenizer for SELFormer (HUBioDataLab/SELFormer).

    Uses SELFIES-aware tokenization (vocab_size=428, RoBERTa-based).
    Special token IDs: cls=1, pad=3, sep=2, unk=0.
    Default max_length=128 covers most drug-like molecules (SELFIES tokens
    are ~1.5–2× longer than SMILES; imatinib produces ~100 tokens).
    No trust_remote_code needed.
    """
    return AutoTokenizer.from_pretrained(SELFORMER_MODEL)


def tokenize_selfies_selformer(selfies_list: list, tokenizer, max_length: int = 128, device='cpu'):
    """Tokenize SELFIES strings for SELFormer.

    Input is SELFIES (not SMILES) — caller must first call smiles_to_selfies().
    Kept as a separate function to prevent cross-tokenizer mistakes:
    SELFormer (cls=1, pad=3) vs MoLFormer (cls=0, pad=2) vs ChemBERTa (cls=0, pad=1)
    all differ — mixing tokenizers silently produces wrong encodings.
    """
    enc = tokenizer(
        selfies_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)
