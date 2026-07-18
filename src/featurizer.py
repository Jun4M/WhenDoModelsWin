"""
featurizer.py
Centralized featurization utilities for all models.
  - Graph features (MolGraphConvFeaturizer) for GNN models
  - ECFP4 fingerprints (numpy) for sklearn models
  - 3D coordinates: ETKDG for ESOL/Lipo/BACE, DFT SDF for QM9 (PaiNN)
"""

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import pickle
import warnings
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# In-process cache for QM9 SDF lookup (populated on first call, reused across seeds)
# dict[canonical_smiles] = np.ndarray (N_heavy, 3) float32
_QM9_3D_DISK_CACHE: dict | None = None

# In-process cache: dataset name → loaded offline ETKDG conformer dict
# Avoids re-loading the pickle file on every seed/train_size call
_ETKDG_DISK_CACHE: dict = {}

# Tensor-level featurization cache.
# Key: ('qm9_painn', can_smi) | ('painn', dataset, can_smi)
#      ('unimol', dataset, can_smi) | ('qm9_unimol', can_smi)
# Value: dict of pre-built tensors (x, edge_index, edge_attr, pos, radius_edge_index)
#        or (z, pos) for UniMol variants.
# PaiNNData / PyGData wrappers are NOT cached — only the underlying tensors.
# .y is always freshly set per call (torch.tensor([0.0])) to avoid mutation bugs.
_FEAT_TENSOR_CACHE: dict = {}


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
# SMILES → SELFIES (SELFormer input)
# ---------------------------------------------------------------------------

def smiles_to_selfies(smiles_list: list) -> tuple:
    """Convert canonical SMILES list to SELFIES strings.

    Call this AFTER canonicalize_and_filter so that atom ordering is stable.
    Molecules that fail SELFIES encoding (rare atoms, exotic stereo) are
    silently excluded and their original-list indices are NOT included in
    valid_indices.

    Adversarial guards:
      - Any Exception from sf.encoder is caught and logged rather than raised,
        so a single exotic molecule does not abort the whole batch.
      - Returns both lists so callers can align y-arrays with valid indices.
      - Empty SELFIES (length 0) treated as failure.

    Args:
        smiles_list: list of canonical SMILES strings (post canonicalize_and_filter).

    Returns:
        selfies_list (list[str]): SELFIES for successfully encoded molecules.
        valid_indices (list[int]): positions in smiles_list that succeeded.
        fail_log (list[tuple[int, str, str]]): (index, smiles, error) for failures.
    """
    try:
        import selfies as sf
    except ImportError as exc:
        raise ImportError(
            "selfies package not found. Install with: pip install selfies==2.1.1"
        ) from exc

    selfies_list, valid_indices, fail_log = [], [], []
    for i, smi in enumerate(smiles_list):
        try:
            sel = sf.encoder(smi)
            if not sel:
                fail_log.append((i, smi, "empty SELFIES output"))
                continue
            selfies_list.append(sel)
            valid_indices.append(i)
        except Exception as exc:
            fail_log.append((i, smi, str(exc)))
    if fail_log:
        print(f"  [smiles_to_selfies] {len(fail_log)} SMILES failed SELFIES encoding "
              f"(exotic atoms / stereo). First failure: {fail_log[0]}")
    return selfies_list, valid_indices, fail_log


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


def build_pyg_list_mtl(dc_graphs: list, y_matrix: np.ndarray) -> list:
    """Build PyG Data list for multi-task regression.

    y_matrix : (N, n_tasks) float32 — already normalized.
    Each Data gets .y of shape (1, n_tasks) so that PyGDataLoader batching
    produces batch.y of shape (B, n_tasks) via dim-0 concatenation.
    """
    import torch
    result = []
    for g, y_row in zip(dc_graphs, y_matrix):
        d = dcgraph_to_pyg(g, y_val=0.0)
        d.y = torch.tensor(y_row, dtype=torch.float).unsqueeze(0)  # (1, n_tasks)
        result.append(d)
    return result


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

class PaiNNData:
    """PyG Data subclass that correctly increments radius_edge_index during batching."""
    pass

try:
    from torch_geometric.data import Data as _PyGData

    class PaiNNData(_PyGData):
        def __inc__(self, key, value, *args, **kwargs):
            if key == 'radius_edge_index':
                return self.num_nodes
            return super().__inc__(key, value, *args, **kwargs)
except Exception:
    from torch_geometric.data import Data as PaiNNData


def _precompute_radius_graph(pos: 'torch.Tensor', cutoff: float = 5.0,
                              max_num_neighbors: int = 32) -> 'torch.Tensor':
    """
    Compute radius graph for a single molecule (no batch dimension).
    pos: (N, 3) tensor. Returns edge_index (2, E).
    """
    import torch
    n = pos.shape[0]
    if n <= 1:
        return torch.zeros(2, 0, dtype=torch.long)
    dist = torch.cdist(pos, pos)            # (N, N) Euclidean distances
    dist.fill_diagonal_(float('inf'))
    adj = dist < cutoff                     # exclude self-loops
    if max_num_neighbors < n - 1:
        # cap per destination node
        for dst in range(n):
            nbrs = adj[:, dst].nonzero(as_tuple=True)[0]
            if len(nbrs) > max_num_neighbors:
                dists = dist[nbrs, dst]
                keep = nbrs[dists.argsort()[:max_num_neighbors]]
                mask = torch.ones(n, dtype=torch.bool)
                mask[keep] = False
                mask[dst]  = False          # keep dst itself masked
                adj[mask, dst] = False
    src, dst = adj.nonzero(as_tuple=True)  # noqa: F841 (dst reused as loop var above)
    return torch.stack([src, dst], dim=0)

def _load_or_build_qm9_3d_cache(sdf_path: str, cache_path: str) -> dict:
    """
    Return dict[canonical_smiles] = np.ndarray(N_heavy, 3) float32.

    Cache policy:
      - cache exists AND sdf_mtime <= cache_mtime → load pickle (< 2 s)
      - otherwise → parse SDF, atomic-write cache, return

    Atomic write: write to <cache_path>.tmp then os.replace() so a crash
    mid-write never leaves a corrupted cache file.
    """
    global _QM9_3D_DISK_CACHE

    # In-process hit: same process, multiple seeds
    if _QM9_3D_DISK_CACHE is not None:
        return _QM9_3D_DISK_CACHE

    # Disk-cache hit: valid if cache file exists and is newer than SDF
    if os.path.exists(cache_path):
        sdf_mtime   = os.path.getmtime(sdf_path)
        cache_mtime = os.path.getmtime(cache_path)
        if sdf_mtime <= cache_mtime:
            try:
                with open(cache_path, 'rb') as fh:
                    data = pickle.load(fh)
                if isinstance(data, dict) and data:
                    print(f"  [featurizer] QM9 3D cache loaded: {len(data)} entries")
                    _QM9_3D_DISK_CACHE = data
                    return data
                print("  [featurizer] Cache empty or wrong format — rebuilding")
            except Exception as exc:
                print(f"  [featurizer] Cache load failed ({exc}) — rebuilding")

    # Build from SDF
    print(f"  [featurizer] Building QM9 3D cache from {sdf_path} (one-time, ~90 s) ...")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    cache: dict = {}
    for mol in suppl:
        if mol is None:
            continue
        try:
            can = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
            if can in cache:
                continue  # keep first occurrence
            conf = mol.GetConformer()
            heavy_pos = [
                [conf.GetAtomPosition(a.GetIdx()).x,
                 conf.GetAtomPosition(a.GetIdx()).y,
                 conf.GetAtomPosition(a.GetIdx()).z]
                for a in mol.GetAtoms() if a.GetAtomicNum() != 1
            ]
            if heavy_pos:
                cache[can] = np.array(heavy_pos, dtype=np.float32)
        except Exception:
            continue

    # Atomic write
    tmp_path = cache_path + '.tmp'
    with open(tmp_path, 'wb') as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    print(f"  [featurizer] QM9 3D cache saved: {len(cache)} entries → {cache_path}")

    _QM9_3D_DISK_CACHE = cache
    return cache


def load_qm9_3d_from_sdf(smiles_list: list, sdf_path: str = './data/qm9.sdf') -> tuple:
    """
    Load DFT-optimized 3D coordinates for QM9 molecules from SDF file.
    Matches molecules by canonical SMILES (RemoveHs) to smiles_list.
    Returns (pyg_data_list_with_pos, valid_indices).

    Notes:
    - SDF contains all-H coordinates; we strip H after coord extraction
      so that pos rows align with heavy-atom graph features.
    - Molecules not found in SDF or with atom-count mismatch are skipped.
    """
    import torch
    from torch_geometric.data import Data
    import deepchem as dc

    feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    # Load (or build) disk cache: canonical_smiles → heavy-atom pos array
    cache_path = os.path.splitext(sdf_path)[0] + '-3d-cache.pkl'
    sdf_lookup = _load_or_build_qm9_3d_cache(sdf_path, cache_path)

    # Canonicalize query SMILES and extract matching 3D data
    pyg_list, valid = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol_q = Chem.MolFromSmiles(smi)
            if mol_q is None:
                continue
            can_q = Chem.MolToSmiles(Chem.RemoveHs(mol_q), canonical=True)

            if can_q not in sdf_lookup:
                continue

            cache_key = ('qm9_painn', can_q)
            if cache_key in _FEAT_TENSOR_CACHE:
                cached = _FEAT_TENSOR_CACHE[cache_key]
                pyg_list.append(PaiNNData(
                    x=cached['x'],
                    edge_index=cached['edge_index'],
                    edge_attr=cached['edge_attr'],
                    pos=cached['pos'],
                    radius_edge_index=cached['radius_edge_index'],
                    y=torch.tensor([0.0]),
                ))
                valid.append(i)
                continue

            heavy_pos = sdf_lookup[can_q]  # np.ndarray (N_heavy, 3) float32

            # Graph features from canonical SMILES
            graphs = feat.featurize([can_q])
            g = graphs[0]
            if g is None or g.node_features is None or len(g.node_features) == 0:
                continue

            x = torch.tensor(g.node_features, dtype=torch.float)

            # Atom count must match
            if heavy_pos.shape[0] != x.shape[0]:
                continue

            if g.edge_index is not None and len(g.edge_index) > 0:
                ei = np.array(g.edge_index)
                edge_index = torch.tensor(ei if ei.shape[0] == 2 else ei.T, dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = (torch.tensor(g.edge_features, dtype=torch.float)
                         if g.edge_features is not None else None)

            pos_tensor = torch.tensor(heavy_pos)
            radius_edge_index = _precompute_radius_graph(pos_tensor)

            _FEAT_TENSOR_CACHE[cache_key] = {
                'x': x,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'pos': pos_tensor,
                'radius_edge_index': radius_edge_index,
            }
            pyg_list.append(PaiNNData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos_tensor,
                radius_edge_index=radius_edge_index,
                y=torch.tensor([0.0]),
            ))
            valid.append(i)
        except Exception:
            continue

    return pyg_list, valid


def _load_etkdg_cache(dataset: str, data_dir: str = './data') -> dict:
    """
    Load (or return in-process cached) offline ETKDG conformer dict.
    Returns dict[canonical_smiles] = np.ndarray (N_heavy, 3) float32.
    Raises FileNotFoundError if cache file is missing.
    """
    global _ETKDG_DISK_CACHE
    if dataset in _ETKDG_DISK_CACHE:
        return _ETKDG_DISK_CACHE[dataset]

    cache_path = os.path.join(data_dir, f'{dataset}-3d-cache.pkl')
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} 없음. 먼저 실행: "
            f"python scripts/build_conformer_cache.py --dataset {dataset}"
        )

    with open(cache_path, 'rb') as fh:
        data = pickle.load(fh)

    n_mols = sum(1 for k in data if not k.startswith('__'))
    print(f"  [featurizer] {dataset.upper()} 3D cache loaded: {n_mols} entries")
    _ETKDG_DISK_CACHE[dataset] = data
    return data


def featurize_smiles_to_3d(smiles_list: list, seed: int = 0,
                            dataset: str = '',
                            data_dir: str = './data') -> tuple:
    """
    Look up pre-built ETKDG conformers from disk cache for ESOL/Lipo/BACE.
    Returns (pyg_data_list_with_pos, valid_indices).

    Cache must be built before first training run:
        python scripts/build_conformer_cache.py --dataset {dataset}

    Molecules absent from the cache are silently excluded; count is logged.
    The seed parameter is accepted for API compatibility but is unused
    (conformer geometry is fixed at cache-build time).
    """
    import torch
    import deepchem as dc

    if not dataset:
        raise ValueError(
            "dataset must be specified. Pass dataset='esol'|'lipo'|'bace'."
        )

    cache = _load_etkdg_cache(dataset, data_dir)
    feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    pyg_list, valid = [], []
    n_total = len(smiles_list)

    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can_smi = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)

            if can_smi not in cache or can_smi.startswith('__'):
                continue

            cache_key = ('painn', dataset, can_smi)
            if cache_key in _FEAT_TENSOR_CACHE:
                cached = _FEAT_TENSOR_CACHE[cache_key]
                pyg_list.append(PaiNNData(
                    x=cached['x'],
                    edge_index=cached['edge_index'],
                    edge_attr=cached['edge_attr'],
                    pos=cached['pos'],
                    radius_edge_index=cached['radius_edge_index'],
                    y=torch.tensor([0.0]),
                ))
                valid.append(i)
                continue

            heavy_pos = cache[can_smi]  # np.ndarray (N_heavy, 3) float32

            graphs = feat.featurize([can_smi])
            g = graphs[0]
            if g is None or g.node_features is None or len(g.node_features) == 0:
                continue

            x = torch.tensor(g.node_features, dtype=torch.float)
            if heavy_pos.shape[0] != x.shape[0]:
                continue

            if g.edge_index is not None and len(g.edge_index) > 0:
                ei = np.array(g.edge_index)
                edge_index = torch.tensor(
                    ei if ei.shape[0] == 2 else ei.T, dtype=torch.long
                )
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = (
                torch.tensor(g.edge_features, dtype=torch.float)
                if g.edge_features is not None else None
            )

            pos_t = torch.tensor(heavy_pos, dtype=torch.float32)
            radius_edge_index = _precompute_radius_graph(pos_t)

            _FEAT_TENSOR_CACHE[cache_key] = {
                'x': x,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'pos': pos_t,
                'radius_edge_index': radius_edge_index,
            }
            pyg_list.append(PaiNNData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos_t,
                radius_edge_index=radius_edge_index,
                y=torch.tensor([0.0]),
            ))
            valid.append(i)
        except Exception:
            continue

    n_included = len(valid)
    print(
        f"  PaiNN {dataset.upper()}: included {n_included}/{n_total} molecules, "
        f"excluded {n_total - n_included} (no conformer / mismatch)"
    )
    return pyg_list, valid


def featurize_smiles_to_unimol(smiles_list: list, seed: int = 0,
                                dataset: str = '',
                                data_dir: str = './data') -> tuple:
    """
    Build UniMol-format PyG Data objects: .z (int64 atomic numbers) + .pos (3D coords).

    Uses the same ETKDG disk cache as featurize_smiles_to_3d() for ESOL/Lipo/BACE.
    Returns (pyg_data_list, valid_indices).

    Cache must be built before first training run:
        python scripts/build_conformer_cache.py --dataset {dataset}
    """
    import torch
    from torch_geometric.data import Data as PyGData

    if not dataset:
        raise ValueError(
            "dataset must be specified. Pass dataset='esol'|'lipo'|'bace'."
        )

    cache = _load_etkdg_cache(dataset, data_dir)
    pyg_list, valid = [], []
    n_total = len(smiles_list)

    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol_h = Chem.RemoveHs(mol)
            can_smi = Chem.MolToSmiles(mol_h, canonical=True)

            if can_smi not in cache or can_smi.startswith('__'):
                continue

            cache_key = ('unimol', dataset, can_smi)
            if cache_key in _FEAT_TENSOR_CACHE:
                cached = _FEAT_TENSOR_CACHE[cache_key]
                pyg_list.append(PyGData(
                    z=cached['z'],
                    pos=cached['pos'],
                    y=torch.tensor([0.0]),
                ))
                valid.append(i)
                continue

            heavy_pos = cache[can_smi]  # (N_heavy, 3) float32
            z = torch.tensor(
                [a.GetAtomicNum() for a in mol_h.GetAtoms()],
                dtype=torch.long,
            )
            if z.shape[0] != heavy_pos.shape[0]:
                continue

            pos_t = torch.tensor(heavy_pos, dtype=torch.float32)
            _FEAT_TENSOR_CACHE[cache_key] = {'z': z, 'pos': pos_t}
            pyg_list.append(PyGData(
                z=z,
                pos=pos_t,
                y=torch.tensor([0.0]),
            ))
            valid.append(i)
        except Exception:
            continue

    n_included = len(valid)
    print(
        f"  UniMol {dataset.upper()}: included {n_included}/{n_total} molecules, "
        f"excluded {n_total - n_included} (no conformer / mismatch)"
    )
    return pyg_list, valid


def load_qm9_unimol_from_sdf(smiles_list: list,
                               sdf_path: str = './data/qm9.sdf') -> tuple:
    """
    Load DFT-optimized 3D coordinates + atomic numbers for QM9 molecules.

    Reuses the same QM9 3D disk cache as load_qm9_3d_from_sdf().
    Returns (pyg_data_list_with_z_and_pos, valid_indices).
    """
    import torch
    from torch_geometric.data import Data as PyGData

    cache_path = os.path.splitext(sdf_path)[0] + '-3d-cache.pkl'
    sdf_lookup = _load_or_build_qm9_3d_cache(sdf_path, cache_path)

    pyg_list, valid = [], []
    n_total = len(smiles_list)

    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol_h = Chem.RemoveHs(mol)
            can_smi = Chem.MolToSmiles(mol_h, canonical=True)

            if can_smi not in sdf_lookup:
                continue

            cache_key = ('qm9_unimol', can_smi)
            if cache_key in _FEAT_TENSOR_CACHE:
                cached = _FEAT_TENSOR_CACHE[cache_key]
                pyg_list.append(PyGData(
                    z=cached['z'],
                    pos=cached['pos'],
                    y=torch.tensor([0.0]),
                ))
                valid.append(i)
                continue

            heavy_pos = sdf_lookup[can_smi]  # (N_heavy, 3) float32
            z = torch.tensor(
                [a.GetAtomicNum() for a in mol_h.GetAtoms()],
                dtype=torch.long,
            )
            if z.shape[0] != heavy_pos.shape[0]:
                continue

            pos_t = torch.tensor(heavy_pos, dtype=torch.float32)
            _FEAT_TENSOR_CACHE[cache_key] = {'z': z, 'pos': pos_t}
            pyg_list.append(PyGData(
                z=z,
                pos=pos_t,
                y=torch.tensor([0.0]),
            ))
            valid.append(i)
        except Exception:
            continue

    n_included = len(valid)
    print(f"  UniMol QM9: included {n_included}/{n_total} molecules")
    return pyg_list, valid


# ---------------------------------------------------------------------------
# KROVEX atom features (Jang et al. 2026)
# ---------------------------------------------------------------------------

# 8 mendeleev properties matching the original KROVEX repo.
# mendeleev ≥1.1.0 renamed 'atomic_volume' → 'miedema_molar_volume';
# many organic atoms (C, O, N, etc.) have NaN → replaced with 0 by nan_to_num.
_KROVEX_PROPS = [
    'atomic_weight',
    'atomic_radius',
    'miedema_molar_volume',    # proxy for 'atomic_volume' in original paper
    'dipole_polarizability',
    'fusion_heat',
    'thermal_conductivity',
    'vdw_radius',
    'en_pauling',
]

# Process-level cache so the table is fetched only once per process
_KROVEX_ATOM_FEATURES: dict | None = None


def get_krovex_atom_features() -> dict:
    """Return dict[atomic_number (int) → np.ndarray(8,) float32] (z-scored).

    Fetches the mendeleev periodic-table once and caches the result in-process.
    NaN entries (common for organic atoms) are replaced with 0 before z-scoring.
    """
    global _KROVEX_ATOM_FEATURES
    if _KROVEX_ATOM_FEATURES is not None:
        return _KROVEX_ATOM_FEATURES

    from mendeleev.fetch import fetch_table
    tb = fetch_table('elements')

    atomic_nums = np.array(tb['atomic_number'], dtype=int)
    props_raw   = np.nan_to_num(np.array(tb[_KROVEX_PROPS], dtype=float), nan=0.0)

    # Column-wise z-score; guard against zero std
    mu    = props_raw.mean(axis=0)
    sigma = props_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    props_z = ((props_raw - mu) / sigma).astype(np.float32)

    _KROVEX_ATOM_FEATURES = {int(atomic_nums[i]): props_z[i] for i in range(len(atomic_nums))}
    return _KROVEX_ATOM_FEATURES


def featurize_smiles_to_krovex_graph(smiles_list: list) -> tuple:
    """Featurize SMILES into KROVEX-style PyG graphs.

    Each graph has:
      x           : (N_atoms, 8) float32 — mendeleev atom features (z-scored)
      edge_index  : (2, 2*E) long       — bidirectional bonds

    Molecules that fail RDKit parsing are excluded.

    Returns
    -------
    pyg_list     : list[PyG Data]
    valid_indices: list[int]   — positions in smiles_list that succeeded
    """
    import torch
    from torch_geometric.data import Data

    atom_feats = get_krovex_atom_features()
    zero_feat  = np.zeros(len(_KROVEX_PROPS), dtype=np.float32)

    pyg_list, valid = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # Atom features: (N, 8)
            rows = []
            for atom in mol.GetAtoms():
                z = atom.GetAtomicNum()
                rows.append(atom_feats.get(z, zero_feat))
            if not rows:
                continue
            x = torch.tensor(np.stack(rows, axis=0), dtype=torch.float)

            # Bonds → bidirectional edge_index
            src, dst = [], []
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                src += [u, v]
                dst += [v, u]
            if src:
                edge_index = torch.tensor([src, dst], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            pyg_list.append(Data(x=x, edge_index=edge_index, y=torch.tensor([0.0])))
            valid.append(i)
        except Exception:
            continue

    return pyg_list, valid


# ---------------------------------------------------------------------------
# Tensor cache utilities
# ---------------------------------------------------------------------------

def _clear_feat_tensor_cache() -> None:
    """Clear the in-process featurizer tensor cache (for testing or memory relief)."""
    global _FEAT_TENSOR_CACHE
    _FEAT_TENSOR_CACHE.clear()


def _feat_tensor_cache_info() -> dict:
    """Return summary stats about the tensor cache."""
    counts: dict = {}
    for key in _FEAT_TENSOR_CACHE:
        kind = key[0]
        counts[kind] = counts.get(kind, 0) + 1
    return {'total': len(_FEAT_TENSOR_CACHE), 'by_kind': counts}
