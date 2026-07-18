"""
tests/test_hybrid_train.py
===========================
Gradient flow, branch contribution, and tiny-overfit tests for
GTCAHybrid (Cat) and GTCACrossAttn (CA).

(d) Gradient flow — GCN branch, BERT branch (bert_depth=1), fusion layer
(e) Both branches contribute — zeroing either branch changes the output
(f) Tiny overfit — 16 molecules, 500 steps, final < 5% initial loss

6 tests total (2 models × 3 test types).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch
from torch_geometric.data import Batch


# ── SMILES ───────────────────────────────────────────────────────────────────

_TRAIN_SMILES = [
    'c1ccccc1',
    'CCO',
    'CC(=O)O',
    'c1cccnc1',
    'CC(=O)Nc1ccc(O)cc1',
    'c1ccc2ccccc2c1',
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'OC(=O)c1ccccc1',
    'c1ccsc1',
    'Cc1ccccc1',
    'CC(C)O',
    'CCCC',
    'CC1CCCCC1',
    'c1ccc(O)cc1',
    'CC(=O)c1ccccc1',
    'c1cncnc1',
]


# ── Data helpers ─────────────────────────────────────────────────────────────

def _build_hybrid_batch(smiles, seed=0):
    """Returns (Batch, y_tensor, input_ids, attention_mask)."""
    from src.featurizer import featurize_smiles_to_graphs, dcgraph_to_pyg
    from src.models import get_tokenizer, tokenize_smiles

    torch.manual_seed(seed)
    y_vals = torch.randn(len(smiles))

    graphs, valid = featurize_smiles_to_graphs(smiles)
    valid_smiles  = [smiles[i] for i in valid]
    data_list = [dcgraph_to_pyg(g, float(y_vals[i]))
                 for i, g in zip(valid, graphs)]

    y       = torch.tensor([d.y.item() for d in data_list])
    batched = Batch.from_data_list(data_list)
    tok     = get_tokenizer()
    ids, mask = tokenize_smiles(valid_smiles, tok)

    return batched, y, ids, mask


def _make_single(smi):
    """Returns (Data, batch_vec, input_ids, attention_mask) for one molecule."""
    from src.featurizer import featurize_smiles_to_graphs, dcgraph_to_pyg
    from src.models import get_tokenizer, tokenize_smiles

    graphs, valid = featurize_smiles_to_graphs([smi])
    assert valid
    data  = dcgraph_to_pyg(graphs[0], y_val=0.0)
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    tok   = get_tokenizer()
    ids, mask = tokenize_smiles([smi], tok)
    return data, batch, ids, mask


# ── Model factories ───────────────────────────────────────────────────────────

def _cat(dropout=0.1, freeze_bert=False):
    from src.models import GTCAHybrid
    return GTCAHybrid(
        node_feat_dim=30, gcn_hidden=64, gcn_layers=2,
        bert_depth=1, dropout=dropout, freeze_bert=freeze_bert,
    )


def _ca(dropout=0.1, freeze_bert=False):
    from src.models import GTCACrossAttn
    return GTCACrossAttn(
        node_feat_dim=30, gcn_hidden=64, gcn_layers=2,
        bert_depth=1, dropout=dropout, ca_dim=128, ca_heads=2,
    )


_MODELS = [
    pytest.param(_cat, id='Cat'),
    pytest.param(_ca,  id='CA'),
]


# ── (d) Gradient flow ────────────────────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_gradient_flow(make_model):
    """
    Every key parameter group receives a non-zero gradient:
      - GCN branch (gcn_convs, gcn_bns)
      - BERT branch trainable layer (bert.encoder.layer.0)
      - Fusion/head layer

    The model uses bert_depth=1 with freeze_bert=False so BERT encoder layer 0
    is trainable; layers 1+ are intentionally frozen (requires_grad=False) and
    are skipped in the check.  bert.pooler is not in the computation path
    (GTCAHybrid reads hidden_states[k], not pooler_output) so it is also skipped.
    """
    torch.manual_seed(0)
    batched, y, ids, mask = _build_hybrid_batch(_TRAIN_SMILES[:8], seed=0)

    model = make_model(dropout=0.1, freeze_bert=False)
    model.train()

    out  = model(batched.x, batched.edge_index, batched.batch, ids, mask)
    loss = F.mse_loss(out, y)

    # Integrity: non-trivial loss before backward (zero loss → zero grads everywhere)
    assert loss.item() > 1e-6, (
        f"[{make_model.__name__}] loss={loss.item():.2e} is near zero before backward "
        "— targets may all be zero; gradient-flow check is vacuous."
    )

    loss.backward()

    # Key groups that MUST receive gradients
    required_groups = ['gcn_convs', 'gcn_bns', 'bert.encoder.layer.0', 'fusion_head']
    if hasattr(model, 'q_proj'):     # CA only
        required_groups += ['q_proj', 'k_proj', 'v_proj', 'cross_attn']

    group_active = {g: False for g in required_groups}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue                  # frozen by design — skip
        if 'pooler' in name:
            continue                  # not in computation path — skip
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            for g in required_groups:
                if name.startswith(g):
                    group_active[g] = True

    dead_groups = [g for g, active in group_active.items() if not active]
    assert not dead_groups, (
        f"[{make_model.__name__}] no non-zero grad in group(s): {dead_groups}\n"
        "One branch may be disconnected from the computation graph."
    )

    # Sanity: frozen layers must have no gradients
    frozen_with_grad = [
        name for name, p in model.named_parameters()
        if not p.requires_grad and p.grad is not None
        and p.grad.abs().sum().item() > 0.0
    ]
    assert not frozen_with_grad, (
        f"Frozen params received gradients: {frozen_with_grad}"
    )


# ── (e) Both branches contribute ─────────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_both_branches_contribute(make_model):
    """
    Behavioural evidence that neither branch is effectively dead:

    1. pred_full — normal forward
    2. pred_no_bert — BERT branch zeroed; output must differ from pred_full
    3. pred_no_graph — GCN branch zeroed; output must differ from pred_full

    If pred_full == pred_no_bert: BERT contribution is zero (fusion head
    ignores the BERT portion, or BERT always outputs the same value).
    If pred_full == pred_no_graph: GCN contribution is zero.

    Zeroing is done via patch.object on _bert_embed/_bert_tokens and
    _gcn_embed so that the respective branch's downstream path sees zeros.
    """
    from src.models import GTCAHybrid

    smi = 'CC(=O)Oc1ccccc1C(=O)O'   # aspirin
    data, batch, ids, mask = _make_single(smi)

    model = make_model(dropout=0.0, freeze_bert=False)
    model.eval()

    gcn_hidden  = model.gcn_bns[0].num_features    # 64
    bert_hidden = model.bert.config.hidden_size     # 768

    with torch.no_grad():
        pred_full = model(data.x, data.edge_index, batch, ids, mask)

    # Zero out BERT branch
    if isinstance(model, GTCAHybrid):
        # _bert_embed returns (B, bert_hidden)
        zero_bert = torch.zeros(1, bert_hidden)
        bert_patch_name = '_bert_embed'
        bert_zero_val   = zero_bert
    else:
        # _bert_tokens returns (B, seq_len, bert_hidden)
        seq_len = ids.shape[1]
        zero_bert = torch.zeros(1, seq_len, bert_hidden)
        bert_patch_name = '_bert_tokens'
        bert_zero_val   = zero_bert

    with patch.object(model, bert_patch_name, return_value=bert_zero_val):
        with torch.no_grad():
            pred_no_bert = model(data.x, data.edge_index, batch, ids, mask)

    delta_bert = (pred_full - pred_no_bert).abs().item()
    assert delta_bert > 1e-4, (
        f"[{make_model.__name__}] zeroing BERT branch changed output by only "
        f"Δ={delta_bert:.2e} — too small to be a genuine BERT contribution "
        "(threshold 1e-4; ~1000× above float32 arithmetic noise ~1e-7)."
    )

    # Zero out GCN branch
    zero_graph = torch.zeros(1, gcn_hidden)
    with patch.object(model, '_gcn_embed', return_value=zero_graph):
        with torch.no_grad():
            pred_no_graph = model(data.x, data.edge_index, batch, ids, mask)

    delta_graph = (pred_full - pred_no_graph).abs().item()
    assert delta_graph > 1e-4, (
        f"[{make_model.__name__}] zeroing GCN branch changed output by only "
        f"Δ={delta_graph:.2e} — too small to be a genuine GCN contribution "
        "(threshold 1e-4; ~1000× above float32 arithmetic noise ~1e-7)."
    )


# ── (f) Tiny overfit ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_tiny_overfit(make_model):
    """
    Hybrid model memorises 16 molecules in 500 Adam steps (lr=1e-3).
    Criterion: final_loss < initial_loss × 0.05.

    Uses freeze_bert=True so only the GCN branch and fusion head are updated.
    This keeps the test fast (no BERT backward) while still testing that the
    fusion architecture has enough capacity to overfit and that gradients
    reach the fusion head correctly.

    Note: with freeze_bert=False the model can also overfit but requires
    smaller lr (1e-4) and more steps (~1000) due to BERT's large param space.
    """
    torch.manual_seed(42)
    batched, y, ids, mask = _build_hybrid_batch(_TRAIN_SMILES, seed=42)

    # Integrity: targets must be varied so overfit is meaningful
    assert y.std().item() > 0.3, (
        f"Target std={y.std().item():.4f} — targets are nearly identical; "
        "any constant predictor would appear to overfit."
    )

    model = make_model(dropout=0.0, freeze_bert=True)
    model.train()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    initial_loss = None
    for step in range(500):
        opt.zero_grad()
        out  = model(batched.x, batched.edge_index, batched.batch, ids, mask)
        loss = F.mse_loss(out, y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()

    final_loss = loss.item()
    ratio = final_loss / initial_loss if initial_loss else float('inf')
    assert ratio < 0.05, (
        f"[{make_model.__name__}] failed to overfit in 500 steps: "
        f"initial={initial_loss:.4f}, final={final_loss:.4f}, "
        f"ratio={ratio:.3f}  (target <0.05)"
    )

    # Integrity: final predictions must be non-constant (model learned variation)
    with torch.no_grad():
        out_final = model(batched.x, batched.edge_index, batched.batch, ids, mask)
    assert out_final.std().item() > 0.01, (
        f"[{make_model.__name__}] final predictions are constant "
        f"(std={out_final.std().item():.4f}) — model collapsed despite low loss."
    )
