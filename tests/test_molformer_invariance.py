"""
tests/test_molformer_invariance.py
=====================================
Structural and invariance tests for MoLFormerRegressor.

(a) Tokenizer round-trip     — tokenize → decode → re-tokenize → identical IDs
(b) Padding mask correctness — masking pad tokens changes pooled output
(c) Length truncation        — ASPIRIN truncated to 10 → SEP at boundary
(d) Special token IDs pinned — pin cls/pad/sep/unk for ibm/MoLFormer-XL-both-10pct
(e) Pooling uses masked mean  — NOT CLS: masked average ≠ cls_token output

5 tests total.

MoLFormer tokenizer notes:
  - Character-level SMILES tokenization (vocab_size=2362)
  - cls_token_id=0, pad_token_id=2, sep_token_id=1, unk_token_id=2361
  - ASPIRIN = 23 tokens (character-level); METHANE = 3 tokens (CLS+C+SEP)
  - max_position_embeddings=202
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

METHANE  = 'C'                              # 3 tokens: CLS + C + SEP
ASPIRIN  = 'CC(=O)Oc1ccccc1C(=O)O'         # 23 tokens character-level

_ROUND_TRIP_SMILES = [
    'c1ccccc1',
    'CC(=O)Oc1ccccc1C(=O)O',
    'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
    'C1CCCCC1',
    'c1ccc2ccccc2c1',
]

_TRUNC_MAX_LEN = 10   # ASPIRIN has 23 tokens → genuinely truncated


def _tok():
    from src.models import get_tokenizer_molformer
    return get_tokenizer_molformer()


def _model(dropout=0.0, freeze_encoder=False):
    from src.models import MoLFormerRegressor
    return MoLFormerRegressor(dropout=dropout, freeze_encoder=freeze_encoder)


# ── (d) Special token IDs pinned ─────────────────────────────────────────────

def test_special_token_ids_pinned():
    """
    Pin special token IDs for ibm/MoLFormer-XL-both-10pct tokenizer.

    MoLFormer uses character-level SMILES tokenization with a 2362-token vocabulary.
    Token IDs differ from both ChemBERTa-1 (cls=0, pad=1, sep=2) and CB-2
    (cls=12, pad=0, sep=13) — mixing tokenizers would produce silently wrong encodings.

    Catches HuggingFace version drift or accidental tokenizer swap.
    """
    from src.models import get_tokenizer_molformer
    tok = get_tokenizer_molformer()
    assert tok.cls_token_id ==    0, f"cls_token_id={tok.cls_token_id}, expected 0"
    assert tok.pad_token_id ==    2, f"pad_token_id={tok.pad_token_id}, expected 2"
    assert tok.sep_token_id ==    1, f"sep_token_id={tok.sep_token_id}, expected 1"
    assert tok.unk_token_id == 2361, f"unk_token_id={tok.unk_token_id}, expected 2361"
    assert tok.vocab_size > 2000,    f"vocab_size={tok.vocab_size} suspiciously small"

    # Guard: MoLFormer pad=2 differs from CB-1 pad=1 and CB-2 pad=0
    assert tok.pad_token_id != 1, (
        "MoLFormer pad_token_id should be 2, not 1 (1 is CB-1's PAD). "
        "Possible: get_tokenizer() used instead of get_tokenizer_molformer()."
    )
    assert tok.pad_token_id != 0, (
        "MoLFormer pad_token_id should be 2, not 0 (0 is CB-2's PAD). "
        "Possible: get_tokenizer_v2() used instead of get_tokenizer_molformer()."
    )


# ── (a) Tokenizer round-trip ──────────────────────────────────────────────────

def test_tokenizer_round_trip():
    """
    For every test SMILES: encode → decode (skip_special_tokens) → re-encode.
    The two ID sequences must be bitwise identical.

    MoLFormer character-level tokenizer decode produces space-separated characters
    which re-encode identically.

    Integrity: no UNK token in original encoding (UNK round-trip is vacuous).
    """
    tok = _tok()
    unk_id = tok.unk_token_id

    for smi in _ROUND_TRIP_SMILES:
        enc  = tok(smi, return_tensors='pt')
        ids1 = enc['input_ids'][0]

        assert (ids1 != unk_id).all(), (
            f"UNK token in encoding of {smi!r}. "
            "Round-trip is vacuous through [UNK]. Use a standard organic SMILES."
        )

        decoded = tok.decode(ids1, skip_special_tokens=True)
        enc2    = tok(decoded, return_tensors='pt')
        ids2    = enc2['input_ids'][0]

        assert torch.equal(ids1, ids2), (
            f"Round-trip failed for {smi!r}:\n"
            f"  original  : {ids1.tolist()}\n"
            f"  decoded   : {decoded!r}\n"
            f"  re-encoded: {ids2.tolist()}\n"
            "MoLFormer character-level tokenizer should round-trip cleanly."
        )


# ── (b) Padding mask correctness ─────────────────────────────────────────────

def test_padding_mask_correctness():
    """
    Real attention_mask (0 at pad positions) must change the masked-mean pooled output
    vs all-ones mask.

    METHANE (3 tokens) is padded to match ASPIRIN (23 tokens) in a joint batch.
    With all-ones mask, the encoder attends to [PAD] tokens, shifting the pooled vector.

    Integrity:
      (1) Padding exists: mask[0] has at least one 0.
      (2) METHANE token count < ASPIRIN token count (genuine asymmetry).
      (3) Position 0 is [CLS] (cls_token_id=0 for MoLFormer).
    """
    from src.models import get_tokenizer_molformer, tokenize_smiles_molformer

    tok = _tok()
    ids, mask = tokenize_smiles_molformer([METHANE, ASPIRIN], tok)

    # Integrity (1): padding must exist
    assert (mask[0] == 0).any(), (
        "No padding in METHANE's mask — METHANE and ASPIRIN produce equal-length "
        "token sequences. Choose a shorter/longer pair."
    )
    # Integrity (2): asymmetry
    assert mask[0].sum() < mask[1].sum(), (
        f"METHANE real-token count ({mask[0].sum().item()}) >= "
        f"ASPIRIN real-token count ({mask[1].sum().item()})."
    )
    # MoLFormer CLS position check
    assert ids[0, 0].item() == tok.cls_token_id, (
        f"Position 0 is not [CLS] (id={ids[0, 0].item()}, expected {tok.cls_token_id}). "
        "MoLFormerRegressor uses masked mean pooling over all real tokens."
    )

    model = _model(dropout=0.0).eval()

    with torch.no_grad():
        pred_masked   = model(ids, mask)

    mask_all_ones = torch.ones_like(mask)
    with torch.no_grad():
        pred_unmasked = model(ids, mask_all_ones)

    delta = (pred_masked[0] - pred_unmasked[0]).abs().item()
    assert delta > 1e-6, (
        f"Attention mask has no effect on METHANE prediction (Δ={delta:.2e}). "
        "Encoder may be ignoring key_padding_mask, or [PAD] embeddings are all-zero."
    )


# ── (c) Length truncation ─────────────────────────────────────────────────────

def test_length_truncation():
    """
    ASPIRIN truncated to max_length=10 must:
      - produce ids.shape[1] == 10
      - have sep_token_id=1 at the last position
      - produce finite model output (no NaN/Inf)

    Integrity: verify ASPIRIN genuinely exceeds _TRUNC_MAX_LEN before truncation.
    """
    from src.models import get_tokenizer_molformer, tokenize_smiles_molformer

    tok = _tok()

    enc_full = tok(ASPIRIN, truncation=False, return_tensors='pt')
    n_full   = enc_full['input_ids'].shape[1]
    assert n_full > _TRUNC_MAX_LEN, (
        f"ASPIRIN produces {n_full} tokens ≤ {_TRUNC_MAX_LEN} without truncation. "
        "Decrease _TRUNC_MAX_LEN or use a longer molecule."
    )

    ids, mask = tokenize_smiles_molformer([ASPIRIN], tok, max_length=_TRUNC_MAX_LEN)

    assert ids.shape[1] == _TRUNC_MAX_LEN, (
        f"ids.shape[1] = {ids.shape[1]}, expected {_TRUNC_MAX_LEN}."
    )

    # CLS at position 0
    assert ids[0, 0].item() == tok.cls_token_id, (
        f"Position 0 is not [CLS] (id={ids[0, 0].item()}, expected {tok.cls_token_id})."
    )

    sep_id = tok.sep_token_id  # 1 for MoLFormer
    assert ids[0, -1].item() == sep_id, (
        f"Last token after truncation = {ids[0, -1].item()}, "
        f"expected sep_token_id = {sep_id}."
    )

    model = _model(dropout=0.0).eval()
    with torch.no_grad():
        out = model(ids, mask)

    assert not torch.isnan(out).any(), "NaN in output for truncated SMILES"
    assert not torch.isinf(out).any(), "Inf in output for truncated SMILES"


# ── (e) Pooling uses masked mean (NOT CLS) ────────────────────────────────────

def test_pooling_uses_masked_mean():
    """
    MoLFormerRegressor must use masked mean pooling over all real tokens,
    NOT the CLS token at position 0.

    Strategy: compare the actual model prediction against two alternative poolings
    computed from the encoder's last_hidden_state directly:
      - cls_pool: last_hidden_state[:, 0, :] fed through model.regressor
      - mean_pool: masked mean fed through model.regressor

    The model's prediction must match mean_pool (within 1e-5) and differ from
    cls_pool by more than 1e-6 on a molecule with >= 5 real tokens.

    Integrity: choose ASPIRIN (23 tokens) so CLS and masked-mean diverge.
    """
    from src.models import get_tokenizer_molformer, tokenize_smiles_molformer

    tok = _tok()
    ids, mask = tokenize_smiles_molformer([ASPIRIN], tok)

    model = _model(dropout=0.0).eval()

    with torch.no_grad():
        # Full forward pass
        pred = model(ids, mask)

        # Extract last_hidden_state manually
        enc_out = model.encoder(input_ids=ids, attention_mask=mask)
        hs = enc_out.last_hidden_state  # (1, L, H)

        # CLS pooling alternative
        cls_vec  = hs[:, 0, :]
        cls_pred = model.regressor(cls_vec).squeeze(-1)

        # Masked mean (what MoLFormerRegressor.forward does)
        m = mask.float().unsqueeze(-1)
        mean_vec  = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
        mean_pred = model.regressor(mean_vec).squeeze(-1)

    # Prediction must match masked mean
    delta_mean = (pred - mean_pred).abs().item()
    assert delta_mean < 1e-5, (
        f"Model prediction differs from masked-mean prediction by {delta_mean:.2e}. "
        "MoLFormerRegressor.forward() should use masked mean pooling."
    )

    # Prediction must differ from CLS pooling (ASPIRIN is long enough for them to diverge)
    delta_cls = (pred - cls_pred).abs().item()
    assert delta_cls > 1e-6, (
        f"Model prediction is identical to CLS-pooled prediction (Δ={delta_cls:.2e}). "
        "MoLFormerRegressor may accidentally be using CLS instead of masked mean. "
        "Use a longer molecule (ASPIRIN, 23 tokens) so the two poolings diverge."
    )
