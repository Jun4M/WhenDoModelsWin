"""
tests/test_chemberta2_invariance.py
=====================================
Structural and invariance tests for ChemBERTa2Regressor.

(a) Tokenizer round-trip     — tokenize → decode → re-tokenize → identical IDs
(b) Padding mask correctness — masking pad tokens changes CLS output
(c) Length truncation        — ASPIRIN (23 tokens) truncated to 10 → SEP at boundary
(d) Special token IDs pinned — pin cls/pad/sep/unk IDs for DeepChem/ChemBERTa-77M-MTR

4 tests total.

CB-2 tokenizer differs from CB-1:
  - Atom-level tokenization (vocab_size=591, not BPE)
  - cls_token_id=12, pad_token_id=0, sep_token_id=13, unk_token_id=11
  - ASPIRIN = 23 tokens (vs 15 for CB-1) — each heavy atom is a separate token
  - METHANE = 3 tokens (CLS, C, SEP)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

METHANE  = 'C'                              # 3 tokens: CLS + C + SEP
ASPIRIN  = 'CC(=O)Oc1ccccc1C(=O)O'         # 23 tokens (atom-level) — good for trunc + padding

_ROUND_TRIP_SMILES = [
    'c1ccccc1',
    'CC(=O)Oc1ccccc1C(=O)O',
    'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
    'C1CCCCC1',
    'c1ccc2ccccc2c1',
]

_TRUNC_MAX_LEN = 10   # ASPIRIN has 23 tokens without truncation → genuinely truncated


def _tok():
    from src.models import get_tokenizer_v2
    return get_tokenizer_v2()


def _model(dropout=0.0, freeze_encoder=False):
    from src.models import ChemBERTa2Regressor
    return ChemBERTa2Regressor(dropout=dropout, freeze_encoder=freeze_encoder)


# ── (d) Special token IDs pinned ─────────────────────────────────────────────

def test_special_token_ids_pinned():
    """
    Pin special token IDs for DeepChem/ChemBERTa-77M-MTR tokenizer.

    CB-2 uses atom-level SMILES tokenization with a 591-token vocabulary.
    Special token IDs differ from ChemBERTa-1 (cls=0, sep=2) — using the
    wrong tokenizer would produce silently wrong encodings.

    Catches HuggingFace version drift or accidental tokenizer swap.
    """
    from src.models import get_tokenizer_v2
    tok = get_tokenizer_v2()
    assert tok.cls_token_id == 12, f"cls_token_id={tok.cls_token_id}, expected 12"
    assert tok.pad_token_id ==  0, f"pad_token_id={tok.pad_token_id}, expected 0"
    assert tok.sep_token_id == 13, f"sep_token_id={tok.sep_token_id}, expected 13"
    assert tok.unk_token_id == 11, f"unk_token_id={tok.unk_token_id}, expected 11"
    assert tok.vocab_size > 500,   f"vocab_size={tok.vocab_size} suspiciously small"

    # Guard: CB-2 token IDs are different from CB-1 (cls=0, sep=2)
    # If CB-1 tokenizer is accidentally used here, the assertion above will fire.
    assert tok.cls_token_id != 0, (
        "CB-2 cls_token_id should be 12, not 0 (0 is CB-1's CLS). "
        "Possible: get_tokenizer() used instead of get_tokenizer_v2()."
    )


# ── (a) Tokenizer round-trip ──────────────────────────────────────────────────

def test_tokenizer_round_trip():
    """
    For every test SMILES: encode → decode (skip_special_tokens) → re-encode.
    The two ID sequences must be bitwise identical.

    CB-2 atom-level tokenizer decode restores the canonical SMILES exactly
    (no BPE whitespace or Ġ prefix issues that affect CB-1's byte-level BPE).

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
            "CB-2 atom-level tokenizer should round-trip cleanly."
        )


# ── (b) Padding mask correctness ─────────────────────────────────────────────

def test_padding_mask_correctness():
    """
    Real attention_mask (0 at pad positions) must change CLS output vs all-ones mask.

    METHANE (3 tokens) is padded to match ASPIRIN (23 tokens) in a joint batch.
    With all-ones mask, the encoder attends to [PAD] tokens, shifting CLS.

    Integrity:
      (1) Padding exists: mask[0] has at least one 0.
      (2) METHANE token count < ASPIRIN token count (genuine asymmetry).
      (3) Position 0 is [CLS] (cls_token_id=12 for CB-2).
    """
    from src.models import get_tokenizer_v2, tokenize_smiles_v2

    tok = _tok()
    ids, mask = tokenize_smiles_v2([METHANE, ASPIRIN], tok)

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
    # CB-2 CLS position check
    assert ids[0, 0].item() == tok.cls_token_id, (
        f"Position 0 is not [CLS] (id={ids[0, 0].item()}, expected {tok.cls_token_id}). "
        "ChemBERTa2Regressor uses last_hidden_state[:, 0, :]."
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
    ASPIRIN (23 tokens in CB-2) truncated to max_length=10 must:
      - produce ids.shape[1] == 10
      - have sep_token_id=13 at the last position
      - produce finite model output (no NaN/Inf)

    Integrity: verify ASPIRIN genuinely exceeds _TRUNC_MAX_LEN before truncation.

    Note: CB-2 uses atom-level tokens so ASPIRIN has 23 tokens (one per heavy atom
    + ring/branch notation), vs 15 for the BPE-based CB-1 tokenizer.
    """
    from src.models import get_tokenizer_v2, tokenize_smiles_v2

    tok = _tok()

    enc_full = tok(ASPIRIN, truncation=False, return_tensors='pt')
    n_full   = enc_full['input_ids'].shape[1]
    assert n_full > _TRUNC_MAX_LEN, (
        f"ASPIRIN produces {n_full} tokens ≤ {_TRUNC_MAX_LEN} without truncation. "
        "Decrease _TRUNC_MAX_LEN or use a longer molecule."
    )

    ids, mask = tokenize_smiles_v2([ASPIRIN], tok, max_length=_TRUNC_MAX_LEN)

    assert ids.shape[1] == _TRUNC_MAX_LEN, (
        f"ids.shape[1] = {ids.shape[1]}, expected {_TRUNC_MAX_LEN}."
    )

    # CB-2 CLS position check
    assert ids[0, 0].item() == tok.cls_token_id, (
        f"Position 0 is not [CLS] (id={ids[0, 0].item()}, expected {tok.cls_token_id})."
    )

    sep_id = tok.sep_token_id  # 13 for CB-2
    assert ids[0, -1].item() == sep_id, (
        f"Last token after truncation = {ids[0, -1].item()}, "
        f"expected sep_token_id = {sep_id}."
    )

    model = _model(dropout=0.0).eval()
    with torch.no_grad():
        out = model(ids, mask)

    assert not torch.isnan(out).any(), "NaN in output for truncated SMILES"
    assert not torch.isinf(out).any(), "Inf in output for truncated SMILES"
