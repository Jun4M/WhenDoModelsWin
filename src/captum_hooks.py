"""
captum_hooks.py
Forward hook utilities for XAI / Captum attribution analysis.
Use after training to capture activations without modifying model code.
"""

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Activation Store
# ---------------------------------------------------------------------------

class ActivationStore:
    """
    Registers forward hooks on named layers to capture activations.

    Usage:
        store = ActivationStore(model, layer_names=['gcn_convs.0', 'fusion_head.0'])
        out = model(...)
        acts = store.get()     # dict: {layer_name: tensor}
        store.clear()          # clear stored activations
        store.remove()         # remove all hooks
    """

    def __init__(self, model: nn.Module, layer_names: list):
        self._hooks      = []
        self._activations = {}

        found = set()
        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)
                found.add(name)

        missing = set(layer_names) - found
        if missing:
            print(f"  [captum_hooks] Warning: layers not found: {missing}")
            print(f"  Available layers: {[n for n, _ in model.named_modules() if n]}")

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self._activations[name] = output[0].detach().cpu()
            else:
                self._activations[name] = output.detach().cpu()
        return hook

    def get(self) -> dict:
        return dict(self._activations)

    def clear(self):
        self._activations.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# List available layer names
# ---------------------------------------------------------------------------

def list_layer_names(model: nn.Module) -> list:
    """Print and return all named module paths in a model."""
    names = [name for name, _ in model.named_modules() if name]
    for n in names:
        print(f"  {n}")
    return names


# ---------------------------------------------------------------------------
# Captum model wrapper (for PyG GNN attribution)
# ---------------------------------------------------------------------------

def get_captum_model(model: nn.Module, model_type: str = 'graph'):
    """
    Returns a Captum-compatible wrapper for PyG models.
    Requires: pip install captum

    model_type: 'node' | 'edge' | 'graph'

    Example usage with Integrated Gradients:
        from captum.attr import IntegratedGradients
        captum_model = get_captum_model(gcn_model, 'graph')
        ig = IntegratedGradients(captum_model)
        attribution = ig.attribute(
            inputs=batch.x,
            additional_forward_args=(batch.edge_index, batch.batch),
            target=0,
        )
    """
    try:
        from torch_geometric.nn import to_captum_model
        return to_captum_model(model, task=model_type)
    except ImportError:
        raise ImportError("captum not installed. Run: pip install captum")
    except Exception as e:
        raise RuntimeError(f"Failed to wrap model for Captum: {e}")


# ---------------------------------------------------------------------------
# Gradient-based saliency (lightweight, no Captum required)
# ---------------------------------------------------------------------------

def compute_grad_saliency(model: nn.Module, batch, device: str = 'cpu') -> torch.Tensor:
    """
    Compute |d(output)/d(node_features)| for a PyG batch.
    Returns saliency tensor of shape (N_nodes,).
    Does not require Captum.
    """
    model.eval()
    batch = batch.to(device)
    batch.x = batch.x.requires_grad_(True)

    out = model(batch.x, batch.edge_index, batch.batch)
    out.sum().backward()

    saliency = batch.x.grad.norm(dim=-1)
    return saliency.detach().cpu()


def compute_attention_saliency(model, tokenizer, smiles_list: list,
                                device: str = 'cpu') -> list:
    """
    Extract CLS→token attention from ChemBERTa last layer.
    Returns list of attention score lists (one per molecule).
    """
    from src.models import tokenize_smiles
    model.eval()
    results = []
    for smi in smiles_list:
        try:
            ids, mask = tokenize_smiles([smi], tokenizer, device=device)
            with torch.no_grad():
                out = model.encoder(
                    input_ids=ids, attention_mask=mask,
                    output_attentions=True,
                )
            attn = out.attentions[-1][0].mean(0)[0]  # (seq_len,)
            results.append(attn.cpu().tolist())
        except Exception:
            results.append([])
    return results
