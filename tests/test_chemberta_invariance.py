"""
tests/test_chemberta_invariance.py
=====================================
Structural and invariance tests for ChemBERTaRegressor.

(a) Tokenizer round-trip     — tokenize → decode → re-tokenize → identical sequence
(b) Canonical SMILES         — same molecule, different notation → same tokens after pipeline
(c) Padding mask correctness — masking pad tokens actively changes the CLS output
(d) Length truncation        — sequences > max_length are cleanly truncated, no NaN/Inf

4 tests total.

Adversarial review (2 sections):
  [Model-specific silent bugs]  — 5 transformer-specific failure modes
  [Test integrity silent passes] — 5 ways this test suite could silently pass despite bugs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

# ── Molecules ─────────────────────────────────────────────────────────────────
METHANE   = 'C'                              # 3 tokens (CLS + C + SEP) — very short
ASPIRIN   = 'CC(=O)Oc1ccccc1C(=O)O'         # 15 tokens — used for padding + truncation
CAFFEINE  = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'   # complex heteroatom ring

_ROUND_TRIP_SMILES = [
    'c1ccccc1',                          # benzene
    'CC(=O)Oc1ccccc1C(=O)O',            # aspirin
    'Cn1cnc2c1c(=O)n(c(=O)n2C)C',      # caffeine
    'C1CCCCC1',                          # cyclohexane
    'c1ccc2ccccc2c1',                    # naphthalene
]

_TRUNC_MAX_LEN = 10   # ASPIRIN has 15 tokens → truncated to 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tok():
    from src.models import get_tokenizer
    return get_tokenizer()


def _model(dropout=0.0, freeze_encoder=False):
    from src.models import ChemBERTaRegressor
    return ChemBERTaRegressor(dropout=dropout, freeze_encoder=freeze_encoder)


# ── (CB#5) Special token ID pin ──────────────────────────────────────────────

def test_special_token_ids_pinned():
    """
    Pin special token IDs for the seyonec/ChemBERTa-zinc-base-v1 tokenizer.
    Catches HuggingFace version drift that silently shifts ID assignments.
    """
    from src.models import get_tokenizer
    tok = get_tokenizer()
    # Expected IDs for seyonec/ChemBERTa-zinc-base-v1 (RoBERTa-based)
    assert tok.cls_token_id == 0,  f"cls_token_id={tok.cls_token_id}, expected 0"
    assert tok.pad_token_id == 1,  f"pad_token_id={tok.pad_token_id}, expected 1"
    assert tok.sep_token_id == 2,  f"sep_token_id={tok.sep_token_id}, expected 2"
    assert tok.unk_token_id == 3,  f"unk_token_id={tok.unk_token_id}, expected 3"
    # vocab_size sanity check (catches model swap to different vocab)
    assert tok.vocab_size > 500, f"vocab_size={tok.vocab_size} suspiciously small"


# ── (a) Tokenizer round-trip ──────────────────────────────────────────────────

def test_tokenizer_round_trip():
    """
    For every test SMILES: encode → decode (skip special tokens) → re-encode.
    The two ID sequences must be bitwise identical.

    Catches:
      - Tokenizer inserts whitespace during decode that shifts BPE merges on
        re-encode (e.g., RoBERTa byte-level BPE adds Ġ space prefixes)
      - Special-token IDs differ between encode and decode (HuggingFace version drift)
      - Non-deterministic BPE fallback for ambiguous merge boundaries

    Integrity guard:
      Assert no UNK token in the original encoding.  If [UNK] is present,
      decode([unk_id]) → '[UNK]', which re-encodes to [unk_id] — round-trip
      passes trivially while masking a vocabulary-coverage problem.
    """
    tok = _tok()
    unk_id = tok.unk_token_id

    for smi in _ROUND_TRIP_SMILES:
        enc  = tok(smi, return_tensors='pt')
        ids1 = enc['input_ids'][0]                           # includes [CLS], [SEP]

        # Integrity: no UNK — round-trip is vacuous through unknown tokens
        assert (ids1 != unk_id).all(), (
            f"UNK token in encoding of {smi!r} — the round-trip trivially passes "
            "for UNK positions. Use a SMILES with fully-covered vocabulary."
        )

        decoded = tok.decode(ids1, skip_special_tokens=True)
        enc2    = tok(decoded, return_tensors='pt')
        ids2    = enc2['input_ids'][0]

        assert torch.equal(ids1, ids2), (
            f"Round-trip failed for {smi!r}:\n"
            f"  original  : {ids1.tolist()}\n"
            f"  decoded   : {decoded!r}\n"
            f"  re-encoded: {ids2.tolist()}\n"
            "Tokenizer decode may add whitespace or normalise characters that "
            "shift BPE boundaries on re-encoding."
        )


# ── (b) Canonical SMILES consistency ─────────────────────────────────────────

def test_canonical_smiles_consistency():
    """
    Part (i): after canonicalize_and_filter, different SMILES of the same molecule
    produce the same canonical form, hence the same token sequence.

    Part (ii) [integrity]: without canonicalization, 'OCC' and 'CCO' tokenize
    differently — confirming the tokenizer IS sensitive to SMILES string order.
    This proves that canonicalization IS a necessary step (not redundant pre-processing),
    and that this test would catch a pipeline where canonicalize_and_filter was absent.

    Catches:
      - canonicalize_and_filter producing wrong canonical form
      - Pipeline that exposes non-canonical SMILES to the tokenizer (silent in the
        absence of this check because both strings featurize to a valid molecule)
    """
    from src.featurizer import canonicalize_and_filter
    from src.models import get_tokenizer, tokenize_smiles

    tok = get_tokenizer()

    noncan_pairs = [
        ('OCC',           'CCO'),           # ethanol O-first vs canonical C-first
        ('C(O)C',         'CCO'),           # ethanol branch-notation vs canonical
        ('c1cccc(c1)O',   'Oc1ccccc1'),     # phenol ring-start vs canonical
    ]

    for raw, expected_can in noncan_pairs:
        cans, valid = canonicalize_and_filter([raw])
        assert valid, f"canonicalize_and_filter rejected {raw!r}"
        assert cans[0] == expected_can, (
            f"canonicalize_and_filter({raw!r}) → {cans[0]!r}, expected {expected_can!r}"
        )

        # (i) Both routes to the canonical form give the same tokens
        ids_a, _ = tokenize_smiles([expected_can], tok)
        ids_b, _ = tokenize_smiles([cans[0]],      tok)
        assert torch.equal(ids_a, ids_b), (
            f"canonical forms {expected_can!r} and {cans[0]!r} produce different tokens"
        )

        # (ii) Integrity: without canonicalization the two SMILES differ
        ids_raw, _ = tokenize_smiles([raw],          tok)
        ids_can, _ = tokenize_smiles([expected_can], tok)
        same_len  = ids_raw.shape[1] == ids_can.shape[1]
        same_toks = same_len and torch.equal(ids_raw, ids_can)
        assert not same_toks, (
            f"ChemBERTa tokenizer produces identical tokens for non-canonical {raw!r} "
            f"and canonical {expected_can!r} — the tokenizer self-canonicalises. "
            "In that case, canonicalize_and_filter is redundant (not a bug), "
            "but the test cannot distinguish between the two SMILES representations."
        )


# ── (c) Padding mask correctness ─────────────────────────────────────────────

def test_padding_mask_correctness():
    """
    When a short molecule is padded to match a longer one in a batch, the
    real attention_mask (0 at pad positions) must actively change the CLS
    representation compared to attending to all positions (all-ones mask).

    Why this must hold:
      [PAD] token embeddings are non-zero (they are learned);  when the
      attention_mask is set to all-ones, the encoder's self-attention
      computes cross-token interactions including [PAD] tokens.  The
      CLS representation — and thus the prediction — should change.

    Integrity guards:
      (1) Assert padding actually exists in mask[0] (otherwise the test
          is comparing two identical masks, guaranteed to agree).
      (2) Assert mask[0].sum() < mask[1].sum() (explicit asymmetry:
          METHANE is shorter than ASPIRIN, so they actually differ in padding).
    """
    from src.models import get_tokenizer, tokenize_smiles

    tok = _tok()
    ids, mask = tokenize_smiles([METHANE, ASPIRIN], tok)

    # Integrity (1): padding must exist for the short molecule
    assert (mask[0] == 0).any(), (
        "No padding in METHANE's attention_mask — the pair produces equal-length "
        "token sequences.  Choose a shorter/longer pair to create genuine padding."
    )
    # Integrity (2): explicit asymmetry
    assert mask[0].sum() < mask[1].sum(), (
        f"METHANE real-token count ({mask[0].sum().item()}) >= "
        f"ASPIRIN real-token count ({mask[1].sum().item()}) — "
        "padding asymmetry assumption violated."
    )

    # CB#4: CLS pooling position check
    assert ids[0, 0].item() == tok.cls_token_id, (
        f"Position 0 is not [CLS] (id={ids[0, 0].item()}, expected {tok.cls_token_id}). "
        "ChemBERTaRegressor uses last_hidden_state[:, 0, :] — non-CLS at position 0 "
        "would silently use the wrong representation."
    )

    model = _model(dropout=0.0).eval()

    with torch.no_grad():
        pred_masked   = model(ids, mask)

    mask_all_ones = torch.ones_like(mask)
    with torch.no_grad():
        pred_unmasked = model(ids, mask_all_ones)

    delta = (pred_masked[0] - pred_unmasked[0]).abs().item()
    assert delta > 1e-6, (
        f"Attention mask has no effect on the short-molecule prediction "
        f"(Δ={delta:.2e}).  The encoder may be ignoring key_padding_mask, "
        "or [PAD] token embeddings are all-zero."
    )


# ── (d) Length truncation ─────────────────────────────────────────────────────

def test_length_truncation():
    """
    Sequences whose untrimmed length exceeds max_length are silently truncated
    by the HuggingFace tokenizer.  The model must still produce finite output.

    Why this matters:
      max_length truncation is *silent* in production — no warning, no error.
      If the last few tokens contain critical structural information, the
      embedding is incomplete and the model produces a prediction without
      indicating that part of the molecule was discarded.

    Structural assertions (not just 'no crash'):
      - Truncation actually occurred: ids.shape[1] == max_length.
      - [SEP] is at the last position after truncation: the tokenizer must
        re-insert [SEP] at the boundary (not leave a mid-molecule token there).
        A missing [SEP] would alter the encoder's position-embedding alignment.

    Integrity:
      Assert that ASPIRIN's un-truncated token count exceeds _TRUNC_MAX_LEN,
      so this test exercises genuine truncation, not a molecule that fits.
    """
    from src.models import get_tokenizer, tokenize_smiles

    tok = _tok()

    # Integrity: verify ASPIRIN is actually longer than the truncation limit
    enc_full  = tok(ASPIRIN, truncation=False, return_tensors='pt')
    n_full    = enc_full['input_ids'].shape[1]
    assert n_full > _TRUNC_MAX_LEN, (
        f"ASPIRIN produces {n_full} tokens ≤ {_TRUNC_MAX_LEN} without truncation. "
        "Choose a longer molecule or decrease _TRUNC_MAX_LEN."
    )

    # Tokenize with truncation
    ids, mask = tokenize_smiles([ASPIRIN], tok, max_length=_TRUNC_MAX_LEN)

    # Truncation must have fired
    assert ids.shape[1] == _TRUNC_MAX_LEN, (
        f"ids.shape[1] = {ids.shape[1]}, expected {_TRUNC_MAX_LEN}."
    )

    # CB#4: CLS pooling position check
    assert ids[0, 0].item() == tok.cls_token_id, (
        f"Position 0 is not [CLS] (id={ids[0, 0].item()}, expected {tok.cls_token_id}). "
        "ChemBERTaRegressor uses last_hidden_state[:, 0, :] — non-CLS at position 0 "
        "would silently use the wrong representation."
    )

    # [SEP] must be at the last position
    sep_id = tok.sep_token_id
    assert ids[0, -1].item() == sep_id, (
        f"Last token after truncation = {ids[0, -1].item()}, "
        f"expected sep_token_id = {sep_id}.  "
        "The tokenizer did not re-insert [SEP] at the truncation boundary."
    )

    # Model forward must be finite
    model = _model(dropout=0.0).eval()
    with torch.no_grad():
        out = model(ids, mask)

    assert not torch.isnan(out).any(),  f"NaN in output for truncated SMILES"
    assert not torch.isinf(out).any(),  f"Inf in output for truncated SMILES"


# ── Adversarial review ────────────────────────────────────────────────────────

class AdversarialReview:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SECTION I — MODEL-SPECIFIC SILENT BUGS (5 failure modes)
    ═══════════════════════════════════════════════════════════════════════════

    (1) UNKNOWN-ATOM SILENT [UNK] DROP
        Symptom  SMILES containing rare atoms (Xe, As, Se metals) tokenize
                 to one or more [UNK] tokens.  The model produces a prediction
                 with no warning.  The embedding silently conflates all rare
                 atoms via the shared [UNK] embedding, making predictions
                 indistinguishable for chemically different elements.

        Detection  After tokenize_smiles, assert (ids != unk_token_id).all().
                   Log a warning listing which molecules triggered UNK.
                   test (a) enforces this on 5 standard molecules; production
                   code should apply it to every batch before model forward.

        Verdict  test_tokenizer_round_trip enforces no-UNK on all test SMILES.
                 Production molecules with rare atoms are NOT covered — add a
                 validation pass in train.py before calling tokenize_smiles ✗

    ──────────────────────────────────────────────────────────────────────────
    (2) PRETRAINED WEIGHT LOAD FAILURE → SILENT RANDOM INIT
        Symptom  If AutoModel.from_pretrained() fails due to a network timeout
                 or a local cache corruption, HuggingFace (depending on version
                 and local_files_only setting) may fall back to random
                 initialisation without raising an exception — or raise a
                 warning that is swallowed by the logging level.  Training
                 continues on a randomly initialised BERT; performance degrades
                 silently compared to fine-tuned expectations.

        Detection  After model construction, spot-check a specific weight norm:
                   embed_norm = model.encoder.embeddings.word_embeddings.weight.norm()
                   assert embed_norm > 10, "embedding norm too low for pretrained weights"
                   (Pretrained RoBERTa embeddings have L2 norm ~20-40; random
                   Xavier init gives ~sqrt(vocab_size/hidden) ≈ 2-4.)
                   Also: check model.encoder.config.model_type == 'roberta'.

        Verdict  Not tested directly.  Recommend adding a one-time sanity check
                 in the test suite or a post-init assertion in ChemBERTaRegressor.

    ──────────────────────────────────────────────────────────────────────────
    (3) SILENT TRUNCATION IN PRODUCTION DATA
        Symptom  tokenize_smiles uses truncation=True, max_length=128.  Any
                 molecule whose SMILES tokenises to >128 subword units is
                 silently truncated.  The tail of the molecule (last atoms/bonds)
                 is discarded; the model produces a prediction without warning.
                 For drug-like molecules with max_length=128, typical coverage
                 is >99%, but polymers, oligonucleotides, or SMILES with many
                 ring-closure digits can exceed this limit.

        Detection  test (d) verifies truncation is clean (SEP at boundary, no NaN).
                   To detect truncation in production: after tokenize_smiles,
                   check if ids[:, -1] != sep_token_id for any sample — if so, the
                   sequence was truncated mid-molecule (the true [SEP] was removed).
                   Wait, actually [SEP] IS re-inserted at max_length-1, so this
                   check doesn't work.  A better approach:
                     len_check = tok(smiles_list, truncation=False)['input_ids']
                     truncated = [len(x) > max_length for x in len_check]
                   Log a warning for each truncated molecule.

        Verdict  test (d) covers the clean-truncation case.  Production data
                 truncation detection must be added to the data pipeline ✗

    ──────────────────────────────────────────────────────────────────────────
    (4) CLS POOLING ASSUMPTION NOT VERIFIED
        Symptom  ChemBERTaRegressor uses out.last_hidden_state[:, 0, :] as the
                 molecular representation, assuming position 0 is the [CLS] token.
                 If the tokenizer is swapped to one that does not prepend [CLS]
                 (e.g., some sequence-classification tokenisers use BOS ≠ CLS),
                 position 0 would be the first SMILES atom, not the aggregate
                 representation.  The model would still train and predict, just
                 on a worse representation.

        Detection  assert ids[0, 0].item() == tok.cls_token_id, "Position 0 is not [CLS]"
                   This should be checked in test (c)/(d) and anywhere the model is used.
                   test (a) implicitly verifies CLS is present (round-trip includes it),
                   but does not assert its position.

        Verdict  Partially covered: [CLS] presence is verified via round-trip (a).
                 Explicit position check missing — add to test (c). ⚠

    ──────────────────────────────────────────────────────────────────────────
    (5) HUGGINGFACE VERSION DRIFT IN SPECIAL TOKEN IDs
        Symptom  Between HuggingFace transformers versions, the special token
                 IDs for [CLS], [SEP], [PAD], [UNK] can shift (e.g., if the
                 vocabulary file or config is updated).  If the serialised
                 model checkpoint used sep_id=2 but the current tokeniser uses
                 sep_id=3, the model silently attends to the wrong position.
                 The performance degradation is gradual and hard to attribute.

        Detection  Pin the exact versions of transformers and tokenizers in
                   requirements.txt.  Add a one-time assertion:
                   assert tok.cls_token_id == 0 and tok.sep_token_id == 2
                   (for seyonec/ChemBERTa-zinc-base-v1 as of 2024).
                   Ideally, lock the tokenizer config to a specific HuggingFace
                   commit hash.

        Verdict  Not tested.  Recommend pinning transformers version and adding
                 special-token-ID assertions in a setup fixture. ✗


    ═══════════════════════════════════════════════════════════════════════════
    SECTION II — TEST INTEGRITY: WAYS THIS SUITE COULD SILENTLY PASS DESPITE BUGS
    ═══════════════════════════════════════════════════════════════════════════

    (1) ROUND-TRIP TRIVIALLY PASSES THROUGH [UNK]
        Scenario  All rare atoms are mapped to [UNK].  decode([unk_id]) returns
                  the literal string '[UNK]'.  tok('[UNK]') re-encodes to [unk_id].
                  The round-trip is self-consistent even if the original SMILES
                  contained meaningful atom information that was discarded.

        Guard     The integrity assertion `(ids != unk_id).all()` in test (a)
                  ensures no [UNK] in the test molecules.  Standard organic atoms
                  (C, N, O, S, F, Cl, Br, I, P) are all in ChemBERTa's vocab.

    ──────────────────────────────────────────────────────────────────────────
    (2) CANONICAL CONSISTENCY PROVES PIPELINE CORRECTNESS, NOT ChemBERTa ITSELF
        Scenario  test (b) verifies that canonicalize_and_filter works and that
                  two different SMILES of the same molecule reach the tokenizer
                  in the same canonical form.  If ChemBERTa itself had a bug
                  (e.g., wrong positional embeddings), test (b) would still pass —
                  it never calls the full model forward, only the tokenizer.

        Guard     The integrity check in part (ii) of test (b) verifies that the
                  tokenizer IS sensitive to non-canonical SMILES strings (OCC ≠ CCO
                  in token space), proving that pipeline canonicalization is load-
                  bearing and not redundant.  Model-level bugs require test (e)–(h).

    ──────────────────────────────────────────────────────────────────────────
    (3) PADDING MASK TEST PASSES IF PADDING NEVER OCCURS
        Scenario  If METHANE and ASPIRIN produce equal-length token sequences
                  after HuggingFace padding (e.g., both truncated to the same
                  length), mask is all-ones for both → `mask == mask_all_ones`
                  → the assertion `delta > 1e-6` trivially holds (both are
                  identical, so delta = 0, which should FAIL the test — but
                  if padding genuinely does not occur, delta = 0 and the test
                  correctly fails; however the error message might be misleading).

        Guard     Integrity assertions (1) and (2) in test (c) check that padding
                  genuinely exists and is asymmetric between the two molecules.

    ──────────────────────────────────────────────────────────────────────────
    (4) [SEP] RE-INSERTION AFTER TRUNCATION IS TOKENIZER BEHAVIOUR, NOT MODEL BEHAVIOUR
        Scenario  test (d) verifies that `ids[0, -1] == sep_id` after truncation.
                  This tests HuggingFace tokenizer behaviour, not ChemBERTaRegressor.
                  If the tokenizer is changed to NOT re-insert [SEP], the model
                  might still work (CLS is still at position 0; the missing [SEP]
                  at the boundary changes the attention but does not crash the model).
                  The test would catch the tokenizer change, but the model might
                  silently degrade on truncated sequences.

        Guard     The no-NaN/no-Inf assertions in test (d) cover model correctness.
                  The [SEP] assertion flags the tokenizer boundary contract.
                  Together, they verify both layers.

    ──────────────────────────────────────────────────────────────────────────
    (5) ALL INVARIANCE TESTS PASS BECAUSE THE MODEL IS CONSTANT
        Scenario  A buggy ChemBERTaRegressor that always outputs 0.0 (e.g.,
                  the regressor head initialised with zero weights) would pass
                  tests (c) and (d) in some edge cases — but not (c), because
                  pred_masked[0] == pred_unmasked[0] == 0.0 → delta = 0 → FAIL.
                  However, tests (a) and (b) don't call the model at all.

        Guard     test (c) directly calls model forward and asserts `delta > 1e-6`.
                  test (g) in test_chemberta_train.py verifies train-mode
                  non-determinism (a constant model has no stochasticity → FAIL).
                  The gradient flow test (e) requires non-zero gradients (a
                  constant model with zero output and zero targets has zero loss
                  → zero grads → FAIL on integrity check `loss > 1e-6`).
    """
