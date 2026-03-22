"""
featurizer.py
Centralized featurization utilities for all models.
  - Graph features (MolGraphConvFeaturizer) for GNN models
  - ECFP4 fingerprints (numpy) for sklearn models
  - 3D coordinates (RDKit ETKDG) for PaiNN
"""

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import warnings
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Canonicalize & Filter
# ---------------------------------------------------------------------------

def canonicalize_and_filter(smiles_list: list) -> tuple:
    """
    Canonicalize SMILES (RemoveHs) and drop invalid ones.
    Returns (canonical_smiles, valid_indices).
    """
    canonical, valid = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.RemoveHs(mol)
            can = Chem.MolToSmiles(mol, canonical=True)
            canonical.append(can)
            valid.append(i)
        except Exception:
            continue
    return canonical, valid


# ---------------------------------------------------------------------------
# Graph Features (MolGraphConvFeaturizer → PyG)
# ---------------------------------------------------------------------------

def _featurize_one_graph(feat, smi: str):
    """Featurize one SMILES with MolGraphConvFeaturizer, return GraphData or None."""
    try:
        results = feat.featurize([smi])
        # results is a numpy array; try to get the first item
        g = results[0] if len(results) > 0 else None
        if (g is not None
                and hasattr(g, 'node_features')
                and g.node_features is not None
                and len(g.node_features) > 0):
            return g
    except Exception:
        pass
    return None


def featurize_smiles_to_graphs(smiles_list: list, use_edges: bool = True,
                               batch_size: int = 500) -> tuple:
    """
    Returns (dc_graph_list, valid_indices) using MolGraphConvFeaturizer.
    Falls back to per-molecule featurization when batch numpy stacking fails.
    Drops molecules that fail featurization.
    """
    import deepchem as dc
    feat = dc.feat.MolGraphConvFeaturizer(use_edges=use_edges)

    valid_graphs, valid = [], []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start:start + batch_size]
        try:
            graphs = feat.featurize(batch)
            # graphs is a numpy object array; iterate safely
            for local_i, g in enumerate(graphs):
                global_i = start + local_i
                try:
                    if (g is not None
                            and hasattr(g, 'node_features')
                            and g.node_features is not None
                            and len(g.node_features) > 0):
                        valid_graphs.append(g)
                        valid.append(global_i)
                except Exception:
                    continue
        except Exception:
            # Batch failed (numpy inhomogeneous shape error); fall back to per-molecule
            for local_i, smi in enumerate(batch):
                global_i = start + local_i
                g = _featurize_one_graph(feat, smi)
                if g is not None:
                    valid_graphs.append(g)
                    valid.append(global_i)

    return valid_graphs, valid


def dcgraph_to_pyg(dc_graph, y_val: float = 0.0):
    """Convert a single DeepChem GraphData to PyG Data object."""
    import torch
    from torch_geometric.data import Data
    x = torch.tensor(dc_graph.node_features, dtype=torch.float)
    if dc_graph.edge_index is not None and len(dc_graph.edge_index) > 0:
        ei = np.array(dc_graph.edge_index)
        # DeepChem returns edge_index in (2, num_edges) format already
        if ei.ndim == 2 and ei.shape[0] == 2:
            edge_index = torch.tensor(ei, dtype=torch.long)
        else:
            # Fallback: (num_edges, 2) format → transpose
            edge_index = torch.tensor(ei.T, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = None
    if dc_graph.edge_features is not None and len(dc_graph.edge_features) > 0:
        edge_attr = torch.tensor(dc_graph.edge_features, dtype=torch.float)
    y = torch.tensor([y_val], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_pyg_list(dc_graphs: list, y_array: np.ndarray) -> list:
    """Build list of PyG Data objects from DeepChem graphs + labels."""
    return [dcgraph_to_pyg(g, float(y)) for g, y in zip(dc_graphs, y_array)]


# ---------------------------------------------------------------------------
# ECFP4 Fingerprints (for sklearn models)
# ---------------------------------------------------------------------------

def featurize_smiles_to_ecfp(
    smiles_list: list,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple:
    """
    Returns (fingerprint_matrix [N, n_bits], valid_indices).
    Used for RF, XGBoost, GPR.
    """
    fps, valid = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            fps.append(np.array(fp))
            valid.append(i)
        except Exception:
            continue
    if not fps:
        return np.zeros((0, n_bits)), []
    return np.stack(fps, axis=0), valid


# ---------------------------------------------------------------------------
# 3D Coordinates (RDKit ETKDG, for PaiNN)
# ---------------------------------------------------------------------------

def featurize_smiles_to_3d(smiles_list: list) -> tuple:
    """
    Generate 3D conformers via RDKit ETKDGv3.
    Returns (pyg_data_list_with_pos, valid_indices).
    Each PyG Data has .pos attribute (N_atoms, 3).
    """
    import torch
    from torch_geometric.data import Data
    import deepchem as dc

    feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    pyg_list, valid = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if result != 0:
                continue
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            can_smi = Chem.MolToSmiles(mol, canonical=True)

            # Graph features
            graphs = feat.featurize([can_smi])
            g = graphs[0]
            if g is None or g.node_features is None or len(g.node_features) == 0:
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()  # (n_atoms, 3)

            x = torch.tensor(g.node_features, dtype=torch.float)
            if len(pos) != x.shape[0]:
                continue  # atom count mismatch after H removal

            if g.edge_index is not None and len(g.edge_index) > 0:
                ei3d = np.array(g.edge_index)
                edge_index = torch.tensor(ei3d if ei3d.shape[0] == 2 else ei3d.T, dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(g.edge_features, dtype=torch.float) if g.edge_features is not None else None
            pos_t = torch.tensor(pos, dtype=torch.float)

            pyg_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                 pos=pos_t, y=torch.tensor([0.0])))
            valid.append(i)
        except Exception:
            continue

    return pyg_list, valid
