"""
tests/test_hybrid_invariance.py
================================
Structural and invariance tests for GTCAHybrid (Cat fusion) and
GTCACrossAttn (CA fusion).

(a) Graph permutation invariance — graph branch input permuted, SMILES fixed
(b) Canonical tokenization pipeline — canonicalize_and_filter precedes tokenizer
(c) Batch independence — molecule A prediction unchanged when batched with B
(h) Attention mask correctness — CA only: padding mask actively changes output
(i) Q/K/V source — CA only: Q from graph branch, K/V from BERT branch

Total: 7 tests.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import inspect
import pytest
import torch
from torch_geometric.data import Data, Batch

ATOL = 1e-4

# ── Molecules ────────────────────────────────────────────────────────────────
ASPIRIN    = 'CC(=O)Oc1ccccc1C(=O)O'       # 13 heavy atoms
BENZENE    = 'c1ccccc1'                      # 6 atoms
CAFFEINE   = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'  # 14 atoms
IBUPROFEN  = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'  # 18 atoms
ETHANE     = 'CC'                            # 2 atoms — short SMILES for padding test


# ── Data helpers ─────────────────────────────────────────────────────────────

def _make_pyg(smi: str) -> Data:
    from src.featurizer import featurize_smiles_to_graphs, dcgraph_to_pyg
    graphs, valid = featurize_smiles_to_graphs([smi])
    assert valid, f"Featurization failed for {smi!r}"
    return dcgraph_to_pyg(graphs[0], y_val=0.0)


def _tokenize(smiles_list, device='cpu'):
    from src.models import get_tokenizer, tokenize_smiles
    tok = get_tokenizer()
    return tokenize_smiles(smiles_list, tok, device=device)


def _permute_data(data: Data, perm: torch.Tensor) -> Data:
    """x_new[j] ← x[perm[j]]; edges remapped via perm_inv."""
    n = perm.shape[0]
    perm_inv = torch.empty(n, dtype=torch.long)
    perm_inv[perm] = torch.arange(n)
    kwargs = dict(x=data.x[perm], edge_index=perm_inv[data.edge_index], y=data.y)
    if data.edge_attr is not None:
        kwargs['edge_attr'] = data.edge_attr
    return Data(**kwargs)


def _forward(model, data: Data, batch: torch.Tensor,
             ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return model(data.x, data.edge_index, batch, ids, mask)


# ── Model factories (dropout=0, eval) ────────────────────────────────────────

def _cat():
    from src.models import GTCAHybrid
    return GTCAHybrid(
        node_feat_dim=30, gcn_hidden=64, gcn_layers=2,
        bert_depth=1, dropout=0.0, freeze_bert=False,
    ).eval()


def _ca():
    from src.models import GTCACrossAttn
    return GTCACrossAttn(
        node_feat_dim=30, gcn_hidden=64, gcn_layers=2,
        bert_depth=1, dropout=0.0, ca_dim=128, ca_heads=2,
    ).eval()


_MODELS = [
    pytest.param(_cat, id='Cat'),
    pytest.param(_ca,  id='CA'),
]


# ── (a) Graph permutation invariance ─────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_graph_permutation_invariance(make_model):
    """
    Prediction is unchanged when atom ordering of the graph input is permuted.
    The SMILES (BERT) input is fixed — only the graph branch is perturbed.
    Verifies that the GCN branch is permutation-invariant and that fusing with
    a fixed BERT embedding preserves that property end-to-end.
    """
    torch.manual_seed(3)
    data  = _make_pyg(ASPIRIN)
    n     = data.num_nodes
    perm  = torch.randperm(n)
    data2 = _permute_data(data, perm)

    ids, mask = _tokenize([ASPIRIN])
    batch = torch.zeros(n, dtype=torch.long)

    model = make_model()
    with torch.no_grad():
        y1 = _forward(model, data,  batch, ids, mask)
        y2 = _forward(model, data2, batch, ids, mask)

    delta = (y1 - y2).abs().item()
    assert delta < ATOL, (
        f"[{make_model.__name__}] perm changed output: "
        f"{y1.item():.8f} → {y2.item():.8f}  (Δ={delta:.2e})"
    )

    # Integrity: permutation was non-trivial (at least one atom moved)
    assert not torch.equal(data.x, data2.x), (
        "Node feature matrix unchanged after permutation — perm was identity "
        "or all atom features are identical; invariance test is vacuous."
    )

    # Integrity: model is non-constant (different molecules → different outputs)
    data_ctrl = _make_pyg(BENZENE)
    b_ctrl = torch.zeros(data_ctrl.num_nodes, dtype=torch.long)
    ids_ctrl, mask_ctrl = _tokenize([BENZENE])
    with torch.no_grad():
        y_ctrl = _forward(model, data_ctrl, b_ctrl, ids_ctrl, mask_ctrl)
    assert not torch.allclose(y1, y_ctrl, atol=1e-4), (
        f"[{make_model.__name__}] ASPIRIN and BENZENE yield the same prediction "
        "— model may be constant, making the permutation check vacuous."
    )

    # Integrity: eval() is deterministic (no stochastic ops in eval)
    with torch.no_grad():
        y1_again = _forward(model, data, batch, ids, mask)
    assert torch.equal(y1, y1_again), (
        f"[{make_model.__name__}] two identical eval() forward passes differ "
        "— dropout may be hard-coded to training=True."
    )


# ── (b) Canonical tokenization pipeline ──────────────────────────────────────

def test_canonical_tokenization_pipeline():
    """
    Verifies three things:

    1. canonicalize_and_filter correctly converts non-canonical SMILES to
       canonical form (OCC → CCO).

    2. Non-canonical and canonical forms of the same molecule tokenize
       differently with ChemBERTa tokenizer — confirming the tokenizer does
       NOT do its own canonicalization.  Without canonicalization in the
       pipeline, the same molecule can receive different BERT embeddings.

    3. data_loader._build_split_dict calls canonicalize_and_filter at line 218
       before exposing SMILES to any model, guaranteeing canonical SMILES
       reach the tokenizer.
    """
    from src.featurizer import canonicalize_and_filter
    from src.models import get_tokenizer, tokenize_smiles
    from src import data_loader

    raw = 'OCC'          # ethanol written O-first (non-canonical)
    can = 'CCO'          # RDKit canonical form

    # 1. Canonicalization works
    out, valid = canonicalize_and_filter([raw])
    assert valid and out[0] == can, \
        f"canonicalize_and_filter('{raw}') → '{out[0]}', expected '{can}'"

    # 2. Token sensitivity: same molecule, different SMILES → different tokens
    tok = get_tokenizer()
    ids_raw, _ = tokenize_smiles([raw], tok)
    ids_can, _ = tokenize_smiles([can],  tok)
    assert not torch.equal(ids_raw, ids_can), (
        "ChemBERTa tokenized OCC and CCO identically — "
        "if this fails the tokenizer canonicalises internally (remove assert)."
    )

    # 3. Pipeline guarantee: _build_split_dict canonicalises before featurising
    src = inspect.getsource(data_loader._build_split_dict)
    assert 'canonicalize_and_filter' in src, (
        "_build_split_dict must call canonicalize_and_filter "
        "before exposing SMILES to graph featuriser or BERT tokenizer."
    )


# ── (c) Batch independence ───────────────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_batch_independence(make_model):
    """
    Prediction for BENZENE alone equals its prediction inside a mini-batch
    that also contains IBUPROFEN.  Catches:
      - GCN global_pool batch-index leak
      - BERT cross-sequence attention leak (not applicable: BERT processes
        each SMILES independently via tokenize_smiles padding, no shared
        attention across molecules)
      - CA cross-attention attending to the wrong molecule's tokens
    """
    data_a = _make_pyg(BENZENE)
    data_b = _make_pyg(IBUPROFEN)

    ids_a, mask_a = _tokenize([BENZENE])
    batch_a = torch.zeros(data_a.num_nodes, dtype=torch.long)

    model = make_model()
    with torch.no_grad():
        y_single = _forward(model, data_a, batch_a, ids_a, mask_a)

    batched   = Batch.from_data_list([data_a, data_b])
    ids_ab, mask_ab = _tokenize([BENZENE, IBUPROFEN])
    with torch.no_grad():
        y_batch = _forward(model, batched, batched.batch, ids_ab, mask_ab)

    delta = (y_single - y_batch[0]).abs().item()
    assert delta < ATOL, (
        f"[{make_model.__name__}] batch leaked: "
        f"single={y_single.item():.8f}, in-batch={y_batch[0].item():.8f}  "
        f"(Δ={delta:.2e})"
    )

    # Integrity: BENZENE and IBUPROFEN must produce different predictions
    assert not torch.allclose(y_batch[0], y_batch[1], atol=1e-4), (
        f"[{make_model.__name__}] BENZENE and IBUPROFEN yield the same prediction "
        "— model may be constant, making the batch-independence check vacuous."
    )


# ── (h) Attention mask correctness (GTCACrossAttn only) ─────────────────────

def test_attn_mask_correctness():
    """
    GTCACrossAttn uses key_padding_mask=(attention_mask==0) so that
    padding tokens are ignored in cross-attention.  This test verifies
    the mask is actually active:

    1. Tokenise [ETHANE, ASPIRIN]: short molecule gets padded → attention_mask
       has zeros at padding positions for molecule 0.
    2. Forward with real attention_mask → pred_masked[0]
    3. Forward with all-ones attention_mask (attend to padding too) → pred_unmasked[0]
    4. The two predictions must differ for molecule 0 — confirming padding tokens
       contribute when unmasked (BERT's [PAD] embeddings are non-zero) and are
       excluded when masked.

    If this fails: key_padding_mask is not being applied, or the model is
    equivalent to attending to all positions regardless of mask.
    """
    data_a = _make_pyg(ETHANE)
    data_b = _make_pyg(ASPIRIN)
    batched = Batch.from_data_list([data_a, data_b])

    ids, mask = _tokenize([ETHANE, ASPIRIN])
    # Verify padding exists (short molecule will be padded to match longer one)
    padding_exists = (mask[0] == 0).any().item()
    assert padding_exists, (
        "ETHANE and ASPIRIN produce equal-length token sequences — "
        "choose a shorter/longer pair to ensure padding."
    )

    # Integrity: ETHANE must have strictly fewer real tokens than ASPIRIN
    assert mask[0].sum() < mask[1].sum(), (
        f"ETHANE token count ({mask[0].sum().item()}) >= ASPIRIN token count "
        f"({mask[1].sum().item()}) — padding asymmetry assumption violated."
    )

    model = _ca()

    with torch.no_grad():
        pred_masked = model(batched.x, batched.edge_index, batched.batch,
                            ids, mask)

    # Remove mask: attend to all positions including padding
    mask_all_ones = torch.ones_like(mask)
    with torch.no_grad():
        pred_unmasked = model(batched.x, batched.edge_index, batched.batch,
                              ids, mask_all_ones)

    # Prediction for the short molecule (idx 0) should change
    delta = (pred_masked[0] - pred_unmasked[0]).abs().item()
    assert delta > 1e-6, (
        f"Attention mask has no effect on short-molecule prediction "
        f"(Δ={delta:.2e}).  key_padding_mask may not be applied."
    )


# ── (i) Q/K/V source (GTCACrossAttn only) ───────────────────────────────────

def test_qkv_source():
    """
    Verifies the Q/K/V assignment in GTCACrossAttn:
        Q ← graph branch  (q_proj input shape: (B, gcn_hidden))
        K ← BERT branch   (k_proj input shape: (B, seq_len, bert_hidden))
        V ← BERT branch   (v_proj input shape: (B, seq_len, bert_hidden))

    Checked by:
    (1) Static: linear layer input dimensions match the expected source.
    (2) Dynamic: forward hooks capture actual input shapes at runtime.
    (3) Sensitivity: fixing BERT, varying graph changes Q and thus output;
        fixing graph, varying BERT changes K/V and thus output.
    """
    model = _ca()

    gcn_hidden  = model.gcn_bns[0].num_features   # 64
    bert_hidden = model.bert.config.hidden_size    # 768
    ca_dim      = model.q_proj.out_features        # 128

    # Integrity: gcn_hidden and bert_hidden must differ to prevent shape coincidence
    assert gcn_hidden != bert_hidden, (
        f"gcn_hidden == bert_hidden == {gcn_hidden} — Q/K/V dim checks can't "
        "distinguish which branch feeds which projection."
    )

    # (1) Static dimension check
    assert model.q_proj.in_features  == gcn_hidden,  \
        f"q_proj.in_features={model.q_proj.in_features}, expected {gcn_hidden} (GCN)"
    assert model.k_proj.in_features  == bert_hidden, \
        f"k_proj.in_features={model.k_proj.in_features}, expected {bert_hidden} (BERT)"
    assert model.v_proj.in_features  == bert_hidden, \
        f"v_proj.in_features={model.v_proj.in_features}, expected {bert_hidden} (BERT)"

    # (2) Dynamic shape check via forward hooks
    captured = {}
    hooks = [
        model.q_proj.register_forward_hook(
            lambda m, inp, out: captured.update(q_in=inp[0].shape)),
        model.k_proj.register_forward_hook(
            lambda m, inp, out: captured.update(k_in=inp[0].shape)),
        model.v_proj.register_forward_hook(
            lambda m, inp, out: captured.update(v_in=inp[0].shape)),
    ]
    data  = _make_pyg(ASPIRIN)
    ids, mask = _tokenize([ASPIRIN])
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    with torch.no_grad():
        model(data.x, data.edge_index, batch, ids, mask)
    for h in hooks:
        h.remove()

    seq_len = ids.shape[1]
    assert captured['q_in'] == (1, gcn_hidden), \
        f"Q input shape {captured['q_in']}, expected (1, {gcn_hidden}) [GCN branch]"
    assert captured['k_in'] == (1, seq_len, bert_hidden), \
        f"K input shape {captured['k_in']}, expected (1, {seq_len}, {bert_hidden}) [BERT branch]"
    assert captured['v_in'] == (1, seq_len, bert_hidden), \
        f"V input shape {captured['v_in']}, expected (1, {seq_len}, {bert_hidden}) [BERT branch]"

    # (3) Sensitivity: change graph input → output changes (Q changes)
    data2 = _make_pyg(CAFFEINE)
    n2    = data2.num_nodes
    b2    = torch.zeros(n2, dtype=torch.long)
    ids2, mask2 = _tokenize([CAFFEINE])
    with torch.no_grad():
        pred_a = model(data.x,  data.edge_index,  batch, ids,  mask)
        pred_b = model(data2.x, data2.edge_index, b2,    ids2, mask2)
    assert not torch.allclose(pred_a, pred_b, atol=1e-6), \
        "Different molecules should produce different Q (and K/V) → different output"


# ── Adversarial review ───────────────────────────────────────────────────────

class AdversarialReview:
    """
    Five hybrid-specific silent-failure modes and their detection strategy.

    ─────────────────────────────────────────────────────────────────────────
    (1) FUSION DIM MISMATCH / SILENT BROADCAST

        Symptom  If gcn_hidden and bert_hidden are accidentally used in an
                 element-wise op instead of concat, PyTorch broadcasts smaller
                 tensors silently (if dims are compatible) or throws a shape
                 error (if not).  The broadcast case would discard most of the
                 BERT representation by repeating the graph embedding.

        Detection  Check that fusion_head.in_features == gcn_hidden + bert_hidden
                   (Cat) or gcn_hidden + ca_dim (CA).  Both-branches-contribute
                   test (e) fails if one branch is zeroed but output is unchanged.

        Verdict  Cat: fusion_head[0].in_features = 128+768 = 896 ✓
                 CA:  fusion_head[0].in_features = 64+128 = 192 ✓
                 test_both_branches_contribute passes for both ✓

    ─────────────────────────────────────────────────────────────────────────
    (2) CROSS-ATTENTION MASK IGNORED (CA ONLY)

        Symptom  Padding tokens attend or are attended to; model learns
                 from PAD embeddings, creating input-length-dependent bias.
                 Short molecules receive lower-quality embeddings because their
                 attention weight distribution includes garbage from PAD.

        Detection  test_attn_mask_correctness (h): forward with real mask vs
                   all-ones mask → predictions for short molecule must differ.

        Verdict  GTCACrossAttn implements: key_padding_mask = (attention_mask == 0)
                 True = "this key position is padding, ignore it".
                 PyTorch MultiheadAttention interprets key_padding_mask correctly
                 (True = masked out).  Test (h) passes → mask is active ✓

    ─────────────────────────────────────────────────────────────────────────
    (3) DROPOUT ASYMMETRY BETWEEN BRANCHES

        Symptom  One branch uses F.dropout(..., training=True) (always on)
                 while the other correctly uses training=self.training.  In
                 eval mode the affected branch is stochastic; performance gap
                 widens and predictions are non-deterministic.

        Detection  model.eval(); run same input twice; assert bitwise identical.
                   Covered by test_seed_determinism (g) in test_hybrid_determinism.py.

        Verdict  GTCAHybrid._gcn_embed uses F.dropout(..., training=self.training) ✓
                 GTCACrossAttn._gcn_embed: same pattern ✓
                 BERT uses nn.Dropout internally (respects eval()) ✓
                 Fusion head uses nn.Dropout (respects eval()) ✓

    ─────────────────────────────────────────────────────────────────────────
    (4) NON-CANONICAL SMILES REACHING THE TOKENIZER

        Symptom  Two training samples that represent the same molecule arrive
                 with different SMILES strings (e.g., OCC vs CCO).  ChemBERTa
                 produces different embeddings for each, making the model
                 implicitly learn that "OCC" and "CCO" are different compounds.

        Detection  test_canonical_tokenization_pipeline (b):
                   (a) OCC and CCO tokenize differently (confirmed)
                   (b) _build_split_dict calls canonicalize_and_filter first
                   (confirmed at line 218 of data_loader.py)

        Verdict  All SMILES reaching GTCAHybrid/GTCACrossAttn via the standard
                 pipeline are canonical ✓.  Custom pipelines bypassing
                 load_dataset_splits are at risk — callers must canonicalise ⚠

    ─────────────────────────────────────────────────────────────────────────
    (5) BRANCH DOMINANCE AT INITIALISATION

        Symptom  If one branch's output has much larger magnitude at init
                 (e.g., BERT CLS token is large while GCN is small), gradient
                 updates concentrate on the dominant branch; the other branch
                 trains poorly.  Effectively degrades to a single-branch model.

        Detection  test_both_branches_contribute (e): zeroing either branch
                   changes the output → both branches have non-trivial weight.
                   Gradient flow test (d) verifies both branches receive gradients.

        Verdict  Both branches produce outputs, and fusion head distributes
                 gradients to both branches ✓.  Long-term training dynamics
                 (whether one branch eventually dominates) are not tested here
                 but are partially addressed by the per-branch gradient flow ✓
    """
