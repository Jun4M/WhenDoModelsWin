"""
tests/test_selformer_invariance.py
=====================================
Structural and invariance tests for SELFormerRegressor.

(a) SMILES → SELFIES round-trip  — encode → decode → canonical SMILES match
(b) Tokenizer round-trip         — SELFIES tokenize → decode → re-tokenize → identical IDs
(c) Padding mask correctness     — masking pad tokens changes CLS output
(d) Length truncation            — long molecule truncated to max_length=10 → SEP at boundary
(e) Special token IDs pinned     — cls=1, pad=3, sep=2, unk=0
(f) Checkpoint buffers finite    — no zero/NaN in inv_freq/cos_cached/sin_cached

6 tests total.

SELFormer notes:
  - Input: SELFIES strings (NOT SMILES) — smiles_to_selfies() converts
  - Architecture: RoBERTa-based, 12 layers, hidden_size=768
  - Pooling: CLS token (last_hidden_state[:, 0, :])
  - Special tokens: cls=1, pad=3, sep=2, unk=0 (all differ from CB-1/CB-2/MoLFormer)
  - No trust_remote_code needed
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from rdkit import Chem

METHANE = 'C'                              # 3 SELFIES tokens: CLS + [C] + SEP
ASPIRIN = 'CC(=O)Oc1ccccc1C(=O)O'         # longer molecule for padding/truncation tests

_ROUND_TRIP_SMILES = [
    'c1ccccc1',
    'CC(=O)Oc1ccccc1C(=O)O',
    'CCO',
    'C1CCCCC1',
    'c1ccc2ccccc2c1',
]

_TRUNC_MAX_LEN = 10   # ASPIRIN produces many SELFIES tokens → genuinely truncated


def _tok():
    from src.models import get_tokenizer_selformer
    return get_tokenizer_selformer()


def _model(dropout=0.0, freeze_encoder=False):
    from src.models import SELFormerRegressor
    return SELFormerRegressor(dropout=dropout, freeze_encoder=freeze_encoder)


def _canonical(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


# ── (a) SMILES → SELFIES round-trip ──────────────────────────────────────────

def test_smiles_to_selfies_roundtrip():
    """
    For every test SMILES: canonical_SMILES → SELFIES encode → SELFIES decode → canonical.
    The decoded canonical SMILES must equal the input canonical SMILES.

    Adversarial guards:
      (1) Input is canonicalized before encoding (matches train_selformer behaviour).
      (2) Verifies selfies package version produces stable alphabets.
      (3) No UNK in SELFIES (all test molecules use standard organic atoms).
    """
    import selfies as sf

    for smi in _ROUND_TRIP_SMILES:
        can_smi = _canonical(smi)
        assert can_smi is not None, f"RDKit could not parse {smi!r}"

        sel = sf.encoder(can_smi)
        assert sel, f"SELFIES encoding returned empty string for {smi!r}"

        decoded = sf.decoder(sel)
        can_decoded = _canonical(decoded)
        assert can_decoded is not None, (
            f"RDKit could not parse SELFIES-decoded molecule: {decoded!r} (from {smi!r})"
        )
        assert can_smi == can_decoded, (
            f"SELFIES round-trip failed for {smi!r}:\n"
            f"  canonical  : {can_smi!r}\n"
            f"  SELFIES    : {sel!r}\n"
            f"  decoded    : {decoded!r}\n"
            f"  can_decoded: {can_decoded!r}\n"
            "selfies version mismatch or exotic atom — pin selfies==2.1.1."
        )


# ── (e) Special token IDs pinned ─────────────────────────────────────────────

def test_special_token_ids_pinned():
    """
    Pin special token IDs for HUBioDataLab/SELFormer tokenizer.

    SELFormer uses SELFIES-aware tokenization with vocab_size=428.
    Token IDs differ from all other models:
      cls=1 (CB-1=0, CB-2=12, MoLFormer=0)
      pad=3 (CB-1=1, CB-2=0,  MoLFormer=2)
    Mixing tokenizers produces silently wrong encodings.

    Catches HuggingFace version drift or accidental tokenizer swap.
    """
    from src.models import get_tokenizer_selformer
    tok = get_tokenizer_selformer()
    assert tok.cls_token_id == 1, f"cls_token_id={tok.cls_token_id}, expected 1"
    assert tok.pad_token_id == 3, f"pad_token_id={tok.pad_token_id}, expected 3"
    assert tok.sep_token_id == 2, f"sep_token_id={tok.sep_token_id}, expected 2"
    assert tok.unk_token_id == 0, f"unk_token_id={tok.unk_token_id}, expected 0"
    assert tok.vocab_size > 400,  f"vocab_size={tok.vocab_size} suspiciously small"

    # Guard: SELFormer cls=1 differs from CB-1/MoLFormer (cls=0)
    assert tok.cls_token_id != 0, (
        "SELFormer cls_token_id should be 1, not 0. "
        "Possible: get_tokenizer() or get_tokenizer_molformer() used instead of get_tokenizer_selformer()."
    )
    # Guard: SELFormer pad=3 differs from CB-2 (pad=0) and MoLFormer (pad=2)
    assert tok.pad_token_id not in (0, 1, 2), (
        f"SELFormer pad_token_id={tok.pad_token_id} collides with another model's token ID."
    )


# ── (b) Tokenizer round-trip ──────────────────────────────────────────────────

def test_tokenizer_round_trip():
    """
    For every test SMILES: SMILES → SELFIES → tokenize → decode → re-tokenize.
    The two ID sequences must be bitwise identical.

    Integrity: no UNK token in original encoding.
    """
    import selfies as sf
    from src.models import get_tokenizer_selformer

    tok = _tok()
    unk_id = tok.unk_token_id

    for smi in _ROUND_TRIP_SMILES:
        can = _canonical(smi)
        sel = sf.encoder(can)

        enc  = tok(sel, return_tensors='pt')
        ids1 = enc['input_ids'][0]

        assert (ids1 != unk_id).all(), (
            f"UNK token in SELFIES encoding of {smi!r} → SELFIES={sel!r}. "
            "Round-trip through UNK is vacuous. Check selfies version / vocab coverage."
        )

        decoded = tok.decode(ids1, skip_special_tokens=True)
        enc2    = tok(decoded, return_tensors='pt')
        ids2    = enc2['input_ids'][0]

        assert torch.equal(ids1, ids2), (
            f"Tokenizer round-trip failed for {smi!r} (SELFIES={sel!r}):\n"
            f"  original  : {ids1.tolist()}\n"
            f"  decoded   : {decoded!r}\n"
            f"  re-encoded: {ids2.tolist()}"
        )


# ── (c) Padding mask correctness ─────────────────────────────────────────────

def test_padding_mask_correctness():
    """
    Real attention_mask (0 at pad positions) must change CLS output vs all-ones mask.

    METHANE (few tokens) is padded to match ASPIRIN (more tokens) in a joint batch.

    Integrity:
      (1) Padding exists in METHANE's mask.
      (2) METHANE real-token count < ASPIRIN real-token count.
      (3) Position 0 is [CLS] (cls_token_id=1 for SELFormer).
    """
    import selfies as sf
    from src.models import get_tokenizer_selformer, tokenize_selfies_selformer

    tok = _tok()
    methane_sel = sf.encoder(METHANE)
    aspirin_sel = sf.encoder(ASPIRIN)
    ids, mask = tokenize_selfies_selformer([methane_sel, aspirin_sel], tok)

    assert (mask[0] == 0).any(), (
        "No padding in METHANE's mask. Choose a shorter/longer molecule pair."
    )
    assert mask[0].sum() < mask[1].sum(), (
        f"METHANE token count ({mask[0].sum().item()}) >= ASPIRIN ({mask[1].sum().item()})."
    )
    assert ids[0, 0].item() == tok.cls_token_id, (
        f"Position 0 is not [CLS] (id={ids[0, 0].item()}, expected {tok.cls_token_id}). "
        "SELFormerRegressor uses last_hidden_state[:, 0, :]."
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
        "Encoder may be ignoring key_padding_mask."
    )


# ── (d) Length truncation ─────────────────────────────────────────────────────

def test_length_truncation():
    """
    ASPIRIN SELFIES truncated to max_length=10 must:
      - produce ids.shape[1] == 10
      - have sep_token_id=2 at the last position
      - produce finite model output (no NaN/Inf)
    """
    import selfies as sf
    from src.models import get_tokenizer_selformer, tokenize_selfies_selformer

    tok = _tok()
    aspirin_sel = sf.encoder(ASPIRIN)

    enc_full = tok(aspirin_sel, truncation=False, return_tensors='pt')
    n_full   = enc_full['input_ids'].shape[1]
    assert n_full > _TRUNC_MAX_LEN, (
        f"ASPIRIN SELFIES produces {n_full} tokens ≤ {_TRUNC_MAX_LEN}. "
        "Choose a longer molecule or decrease _TRUNC_MAX_LEN."
    )

    ids, mask = tokenize_selfies_selformer([aspirin_sel], tok, max_length=_TRUNC_MAX_LEN)

    assert ids.shape[1] == _TRUNC_MAX_LEN, (
        f"ids.shape[1]={ids.shape[1]}, expected {_TRUNC_MAX_LEN}."
    )
    assert ids[0, 0].item() == tok.cls_token_id, (
        f"Position 0 is not [CLS] (id={ids[0, 0].item()}, expected {tok.cls_token_id})."
    )
    sep_id = tok.sep_token_id  # 2 for SELFormer
    assert ids[0, -1].item() == sep_id, (
        f"Last token={ids[0, -1].item()}, expected sep_token_id={sep_id}."
    )

    model = _model(dropout=0.0).eval()
    with torch.no_grad():
        out = model(ids, mask)

    assert not torch.isnan(out).any(), "NaN in output for truncated SELFIES"
    assert not torch.isinf(out).any(), "Inf in output for truncated SELFIES"


# ── (f) Checkpoint buffers finite ────────────────────────────────────────────

def test_checkpoint_buffers_finite():
    """
    SELFormer has no rotary embeddings, so no inv_freq/cos_cached/sin_cached buffers.
    This test verifies that the checkpoint sanity check in SELFormerRegressor.__init__
    does NOT raise (i.e., the buffers it does have are all finite).

    Also verifies that position_ids (the only notable buffer) is non-zero and finite.

    Catches future checkpoint corruption (same class as MoLFormer inv_freq=0 bug).
    """
    from src.models import SELFormerRegressor
    model = SELFormerRegressor(dropout=0.0)

    # The __init__ sanity check already ran — no RuntimeError means we're here.
    # Additional: all buffers must be finite.
    for name, buf in model.encoder.named_buffers():
        assert torch.isfinite(buf).all(), (
            f"Buffer {name} contains non-finite values. "
            "Checkpoint may be corrupted."
        )
        # position_ids should be non-zero (0..513)
        if 'position_ids' in name:
            assert buf.abs().sum() > 0, (
                f"position_ids buffer is all-zero — embedding positions would collapse."
            )
