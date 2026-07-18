"""
transformers >= 5.x compatibility shims for MoLFormer and future custom-code models.
Idempotent (hasattr-guarded). Import before any AutoModel.from_pretrained call.

Why this file exists
--------------------
MoLFormer (ibm/MoLFormer-XL-both-10pct, trust_remote_code=True) uses three symbols
that were removed in transformers >= 5.x:
  - transformers.pytorch_utils.find_pruneable_heads_and_indices
  - transformers.onnx.OnnxConfig
  - PreTrainedModel.get_head_mask

The HuggingFace module cache (~/.cache/huggingface/modules/) is recompiled on every
fresh environment (Colab, CI, new machine). Patching the cached files directly does
not survive those recompilations. This module monkey-patches the live transformers
package objects so the remote code can import the symbols from any environment.

Usage
-----
Import this module before any `AutoModel.from_pretrained(trust_remote_code=True)` call.
It is already imported at the top of src/models.py and src/__init__.py.

Adversarial failure modes
-------------------------
(1) Wrong import order: compat not imported before MoLFormer load
    Guard: imported in both src/__init__.py (package-level) and src/models.py (top).
    Both places must be maintained together.

(2) transformers 6.x re-removes or renames these symbols differently
    Guard: all three patches are hasattr-guarded, so they skip gracefully if the
    symbol already exists. New removals in 6.x would need new stubs here.

(3) get_head_mask stub is too permissive, masking real attention-mask bugs
    Guard: the stub matches the original transformers 4.x signature exactly.
    In practice, MoLFormer always calls get_head_mask(None, num_layers), so the stub
    path is always the [None]*n branch — the mask path is never exercised at runtime.

(4) transformers.onnx stub collides with other code that genuinely uses onnx export
    Guard: OnnxConfig.__init__ raises NotImplementedError so any accidental
    instantiation fails loudly rather than silently returning a broken object.

(5) Fresh cache load surfaces additional 5.x-removed symbols beyond the known three
    Guard: delete the modules cache, run tests, and catch new ImportError messages.
    Each new symbol gets its own hasattr-guarded stub in this file.
"""

import sys
import types

import torch
import transformers
from transformers import pytorch_utils
from transformers.modeling_utils import PreTrainedModel


# ---------------------------------------------------------------------------
# 1. find_pruneable_heads_and_indices
#    Used in: modeling_molformer.py  (imported from transformers.pytorch_utils)
#    Removed: transformers >= 5.0
#    Behaviour: MoLFormer references the symbol but never calls prune_heads at
#    runtime; returning an empty set + empty index tensor is always correct.
# ---------------------------------------------------------------------------
if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        """Stub: returns empty set + empty index tensor.

        MoLFormer imports this symbol but never calls prune_heads at runtime,
        so the stub body is never exercised in normal inference/training.
        """
        return set(), torch.tensor([], dtype=torch.long)

    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


# ---------------------------------------------------------------------------
# 2. transformers.onnx module + OnnxConfig class
#    Used in: configuration_molformer.py  (from transformers.onnx import OnnxConfig)
#    Removed: transformers >= 5.0 (entire transformers.onnx subpackage deleted)
#    Behaviour: MolformerOnnxConfig inherits OnnxConfig for ONNX export support.
#    We never perform ONNX export; the stub is never instantiated.
# ---------------------------------------------------------------------------
if not hasattr(transformers, "onnx"):
    transformers.onnx = types.ModuleType("transformers.onnx")
    sys.modules["transformers.onnx"] = transformers.onnx

if not hasattr(transformers.onnx, "OnnxConfig"):
    class OnnxConfig:
        """Stub for transformers >= 5.x compatibility.

        MolformerOnnxConfig inherits this class; instantiation would require
        the full ONNX export infrastructure that was removed. If you need
        actual ONNX export, pin transformers < 5.0.
        """
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "OnnxConfig is a compatibility stub only. "
                "ONNX export requires transformers < 5.0. "
                "MoLFormer inference/training does not use ONNX."
            )

    transformers.onnx.OnnxConfig = OnnxConfig


# ---------------------------------------------------------------------------
# 3. PreTrainedModel.get_head_mask method
#    Used in: modeling_molformer.py MolformerModel.forward()
#    Removed: transformers >= 5.0
#    Behaviour: MoLFormer always calls get_head_mask(None, num_hidden_layers),
#    so only the [None]*n branch is ever reached in practice.
# ---------------------------------------------------------------------------
if not hasattr(PreTrainedModel, "get_head_mask"):
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """Stub: identity-mask. Returns [None]*num_hidden_layers when head_mask is None.

        Matches the original transformers 4.x signature. The non-None branches
        handle explicit head masks for attention head pruning experiments.
        """
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return head_mask

    PreTrainedModel.get_head_mask = get_head_mask


# ---------------------------------------------------------------------------
# Rotary embedding re-initialiser for MoLFormer
# ---------------------------------------------------------------------------
# The ibm/MoLFormer-XL-both-10pct checkpoint stores inv_freq as all-zeros in
# its state_dict (the buffer was apparently serialised with zero values during
# the original training save). Because inv_freq is declared persistent=False in
# the current remote code, it should NOT be loaded from the checkpoint — but
# PyTorch restores any key whose name matches a buffer, regardless of the
# persistent flag. The zero inv_freq propagates to cos/sin caches, causing
# every attention output to be NaN.
#
# reinit_molformer_rotary(model) recomputes inv_freq from base/dim and
# rebuilds the cos/sin cache. Call it immediately after from_pretrained().
# MoLFormerRegressor.__init__ calls this automatically.

def reinit_molformer_rotary(model) -> None:
    """Recompute MolformerRotaryEmbedding inv_freq/cos/sin caches.

    The MoLFormer checkpoint stores inv_freq=0 in its state_dict, causing NaN
    outputs. This function restores the correct values from the embedding's
    dim/base attributes. Idempotent — safe to call multiple times.

    Args:
        model: a MolformerModel (or any nn.Module containing
               MolformerRotaryEmbedding submodules).
    """
    import torch

    for module in model.modules():
        if type(module).__name__ == "MolformerRotaryEmbedding":
            dim  = module.dim
            base = module.base
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module._set_cos_sin_cache(
                seq_len=module.max_position_embeddings,
                device=inv_freq.device,
                dtype=torch.get_default_dtype(),
            )
