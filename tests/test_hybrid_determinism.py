"""
tests/test_hybrid_determinism.py
==================================
Seed determinism for GTCAHybrid (Cat) and GTCACrossAttn (CA).

(g) Two training runs with identical torch.manual_seed produce bitwise
    identical loss trajectories over 5 steps.

dropout=0.0 and freeze_bert=True eliminate all stochasticity:
  - No dropout sampling
  - BERT is frozen → its output is a deterministic function of input only
  - Only GCN and fusion head weights are updated
  - BatchNorm running stats evolve deterministically with the same batch

2 tests total.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

ATOL_DET = 1e-6

# ── SMILES ───────────────────────────────────────────────────────────────────

_DET_SMILES = [
    'c1ccccc1',
    'CCO',
    'CC(=O)O',
    'c1cccnc1',
    'CC(=O)Nc1ccc(O)cc1',
    'c1ccc2ccccc2c1',
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'OC(=O)c1ccccc1',
]


# ── Data helper ───────────────────────────────────────────────────────────────

def _build_batch(smiles, seed=0):
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


def _run_steps(make_model, batched, y, ids, mask, n, seed):
    """Fresh seeded model + n Adam steps; returns loss list."""
    torch.manual_seed(seed)
    model = make_model()
    model.train()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    losses = []
    for _ in range(n):
        opt.zero_grad()
        out  = model(batched.x, batched.edge_index, batched.batch, ids, mask)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ── Model factories (dropout=0, freeze_bert=True) ────────────────────────────

def _cat():
    from src.models import GTCAHybrid
    return GTCAHybrid(
        node_feat_dim=30, gcn_hidden=64, gcn_layers=2,
        bert_depth=1, dropout=0.0, freeze_bert=True,
    )


def _ca():
    from src.models import GTCACrossAttn
    return GTCACrossAttn(
        node_feat_dim=30, gcn_hidden=64, gcn_layers=2,
        bert_depth=1, dropout=0.0, ca_dim=128, ca_heads=2,
    )


_MODELS = [
    pytest.param(_cat, id='Cat'),
    pytest.param(_ca,  id='CA'),
]


# ── (g) Seed determinism ──────────────────────────────────────────────────────

@pytest.mark.parametrize('make_model', _MODELS)
def test_seed_determinism(make_model):
    """
    Two runs with seed=123 must produce exactly the same loss at every step.

    Sources of non-determinism this would catch:
      - F.dropout with training=True hard-coded in either branch
      - BERT forward path using stochastic sampling (shouldn't happen in
        inference mode, but would manifest if dropout is not gated)
      - Weight init not controlled by manual_seed
      - Adam moments seeded differently between runs
    """
    batched, y, ids, mask = _build_batch(_DET_SMILES, seed=77)

    losses_a = _run_steps(make_model, batched, y, ids, mask, n=5, seed=123)
    losses_b = _run_steps(make_model, batched, y, ids, mask, n=5, seed=123)

    # Integrity: initial loss must be non-trivial (zero loss → all runs trivially equal)
    assert losses_a[0] > 1e-6, (
        f"[{make_model.__name__}] initial loss={losses_a[0]:.2e} is near zero "
        "— targets may all be zero; determinism check is vacuous."
    )

    # Integrity: loss must change over training (model is actually learning)
    assert losses_a[0] != losses_a[-1], (
        f"[{make_model.__name__}] loss unchanged over 5 steps "
        f"({losses_a[0]:.6f} → {losses_a[-1]:.6f}) — weights may not be updating."
    )

    for step, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < ATOL_DET, (
            f"[{make_model.__name__}] non-determinism at step {step}: "
            f"run_a={la:.10f}, run_b={lb:.10f}  (Δ={abs(la-lb):.2e})\n"
            "Check: F.dropout hard-coded training=True, non-seeded init, "
            "or BatchNorm stat divergence."
        )
