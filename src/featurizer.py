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

import warnings
import multiprocessing
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# In-process cache: (canonical_smiles, seed) → PaiNNData
# Avoids re-running ETKDG for the same molecule+seed across different train_sizes/splits
_FEATURIZE_3D_CACHE: dict = {}

# ---------------------------------------------------------------------------
# Subprocess worker for ETKDG (top-level for pickling)
# ---------------------------------------------------------------------------

def _embed_molecule_worker(args):
    """
    Runs EmbedMolecule + MMFF in a subprocess.
    Returns (pos_array, canonical_smi, mmff_status) on success, or (None, None, reason) on failure.
    """
    smi, seed = args
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, 'invalid_smiles'
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = 1000
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        return None, None, 'embed_failed'
    mmff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    mol = Chem.RemoveHs(mol)
    can_smi = Chem.MolToSmiles(mol, canonical=True)
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    mmff_status = 'mmff_not_converged' if mmff_result == 1 else None
    return pos, can_smi, mmff_status


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
    dist2 = torch.cdist(pos, pos)          # (N, N)
    dist2.fill_diagonal_(float('inf'))
    adj = dist2 < cutoff * cutoff           # exclude self-loops
    if max_num_neighbors < n - 1:
        # cap per destination node
        for dst in range(n):
            nbrs = adj[:, dst].nonzero(as_tuple=True)[0]
            if len(nbrs) > max_num_neighbors:
                dists = dist2[nbrs, dst]
                keep = nbrs[dists.argsort()[:max_num_neighbors]]
                mask = torch.ones(n, dtype=torch.bool)
                mask[keep] = False
                mask[dst]  = False          # keep dst itself masked
                adj[mask, dst] = False
    src, dst = adj.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)

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

    # Build lookup: canonical_smiles(heavy) → all-H mol (keep ref to avoid GC)
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    sdf_lookup = {}  # can_smi → all-H mol object
    for mol in suppl:
        if mol is None:
            continue
        try:
            can = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
            if can not in sdf_lookup:  # keep first occurrence
                sdf_lookup[can] = mol  # store all-H mol so GC won't free it
        except Exception:
            continue

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
            all_h_mol = sdf_lookup[can_q]

            # Extract heavy-atom 3D positions from DFT conformer
            conf = all_h_mol.GetConformer()
            heavy_pos = []
            for atom in all_h_mol.GetAtoms():
                if atom.GetAtomicNum() != 1:  # skip hydrogen
                    p = conf.GetAtomPosition(atom.GetIdx())
                    heavy_pos.append([p.x, p.y, p.z])
            heavy_pos = np.array(heavy_pos, dtype=np.float32)

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


def featurize_smiles_to_3d(smiles_list: list, seed: int = 0,
                            fail_log_path: str = None,
                            dataset: str = '') -> tuple:
    """
    Generate 3D conformers via RDKit ETKDGv3.
    Returns (pyg_data_list_with_pos, valid_indices).
    Each PyG Data has .pos attribute (N_atoms, 3).
    seed is passed to ETKDGv3 for reproducible conformer generation.
    Failures are appended to fail_log_path (if given) with reason.
    """
    import torch
    from torch_geometric.data import Data
    import deepchem as dc
    from datetime import datetime

    def _log_fail(idx, smi, reason):
        if fail_log_path is None:
            return
        ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        line = f"{ts} | dataset={dataset} | seed={seed} | idx={idx} | smiles={smi} | reason={reason}\n"
        with open(fail_log_path, 'a') as f:
            f.write(line)

    LARGE_MOL_THRESHOLD = 50  # atoms > this → subprocess for safety
    EMBED_TIMEOUT_SEC = 10

    feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    pool = multiprocessing.Pool(1)  # single worker, reused across molecules

    pyg_list, valid = [], []
    for i, smi in enumerate(smiles_list):
        try:
            # 캐시 확인: 같은 (smiles, seed) 조합은 재계산 생략
            cache_key = (smi, seed)
            if cache_key in _FEATURIZE_3D_CACHE:
                pyg_list.append(_FEATURIZE_3D_CACHE[cache_key])
                valid.append(i)
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                _log_fail(i, smi, 'invalid_smiles')
                continue

            # 1차 방어: 단일 원자 분자 즉시 스킵 (ETKDG 의미 없음)
            if mol.GetNumAtoms() < 2:
                _log_fail(i, smi, 'single_atom')
                continue

            if mol.GetNumAtoms() > LARGE_MOL_THRESHOLD:
                # 2차 방어: 복잡한 분자는 subprocess로 격리 → OS kill로 무한루프 차단
                async_result = pool.apply_async(_embed_molecule_worker, [(smi, seed)])
                try:
                    pos, can_smi, mmff_status = async_result.get(timeout=EMBED_TIMEOUT_SEC)
                except multiprocessing.TimeoutError:
                    _log_fail(i, smi, 'embed_timeout')
                    pool.terminate()
                    pool = multiprocessing.Pool(1)  # 죽은 worker 교체
                    continue
                if pos is None:
                    _log_fail(i, smi, mmff_status)
                    continue
                if mmff_status == 'mmff_not_converged':
                    _log_fail(i, smi, 'mmff_not_converged')
            else:
                # 소/중형 분자: 직접 실행 (빠름)
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                params.maxIterations = 1000
                result = AllChem.EmbedMolecule(mol, params)
                if result != 0:
                    _log_fail(i, smi, 'embed_failed')
                    continue
                mmff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
                if mmff_result == 1:
                    _log_fail(i, smi, 'mmff_not_converged')
                mol = Chem.RemoveHs(mol)
                can_smi = Chem.MolToSmiles(mol, canonical=True)
                pos = mol.GetConformer().GetPositions()

            # Graph features
            graphs = feat.featurize([can_smi])
            g = graphs[0]
            if g is None or g.node_features is None or len(g.node_features) == 0:
                _log_fail(i, smi, 'graph_feat_failed')
                continue

            pos_t = torch.tensor(pos, dtype=torch.float)
            x = torch.tensor(g.node_features, dtype=torch.float)
            if len(pos) != x.shape[0]:
                _log_fail(i, smi, 'atom_count_mismatch')
                continue

            if g.edge_index is not None and len(g.edge_index) > 0:
                ei3d = np.array(g.edge_index)
                edge_index = torch.tensor(ei3d if ei3d.shape[0] == 2 else ei3d.T, dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(g.edge_features, dtype=torch.float) if g.edge_features is not None else None

            radius_edge_index = _precompute_radius_graph(pos_t)
            pyg_data = PaiNNData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                 pos=pos_t, radius_edge_index=radius_edge_index,
                                 y=torch.tensor([0.0]))
            _FEATURIZE_3D_CACHE[cache_key] = pyg_data
            pyg_list.append(pyg_data)
            valid.append(i)
        except Exception as e:
            _log_fail(i, smi, f'exception:{e}')
            continue

    pool.terminate()
    return pyg_list, valid
