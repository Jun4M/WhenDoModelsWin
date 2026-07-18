"""
tests/test_radius_graph.py
===========================
Correctness tests for PaiNNRegressor._radius_graph().

Background:
  A past silent bug applied `dist < cutoff * cutoff` instead of `dist < cutoff`,
  effectively treating cutoff=5Å as 25Å and returning a near-fully-connected graph.
  These tests are designed to catch the same class of bug (unit mismatch,
  squared-distance comparison, wrong axis, wrong edge direction, etc.).

Run:
    pytest tests/test_radius_graph.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import pytest
from scipy.spatial import cKDTree

from src.models import PaiNNRegressor

rg = PaiNNRegressor._radius_graph   # shorthand


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _batch(n, dtype=torch.float32):
    return torch.zeros(n, dtype=torch.long)


def _methane_pos(dtype=torch.float32):
    """CH4: C at origin + 4 H at tetrahedral vertices, C-H = 1.09 Å."""
    d = 1.09 / (3 ** 0.5)
    return torch.tensor([
        [ 0.0,  0.0,  0.0],   # C
        [ d,    d,    d],      # H
        [ d,   -d,   -d],      # H
        [-d,    d,   -d],      # H
        [-d,   -d,    d],      # H
    ], dtype=dtype)


def _edge_set(edge_index):
    """Convert (2, E) edge_index to a sorted frozenset of (src, dst) tuples."""
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    return frozenset(zip(src, dst))


def _ref_radius_graph_scipy(pos_np, cutoff, max_neighbors=32):
    """
    Reference implementation via scipy.spatial.cKDTree.
    Returns directed edges (src, dst) with dist < cutoff, no self-loops,
    capped at max_neighbors per destination node (nearest first).
    """
    tree = cKDTree(pos_np)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')  # (K, 2) undirected
    if len(pairs) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    # Expand to directed edges: (i→j) and (j→i)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    dists_all = np.linalg.norm(pos_np[src] - pos_np[dst], axis=1)

    n = len(pos_np)
    kept_src, kept_dst = [], []
    for node in range(n):
        mask = (dst == node)
        nbr_src = src[mask]
        nbr_dst = dst[mask]
        nbr_d   = dists_all[mask]
        if len(nbr_src) > max_neighbors:
            order   = np.argsort(nbr_d)[:max_neighbors]
            nbr_src = nbr_src[order]
            nbr_dst = nbr_dst[order]
        kept_src.append(nbr_src)
        kept_dst.append(nbr_dst)

    src_out = np.concatenate(kept_src)
    dst_out = np.concatenate(kept_dst)
    return np.stack([src_out, dst_out], axis=0)   # (2, E)


# ─────────────────────────────────────────────────────────────────────────────
# (a) Methane standard geometry — exact edge counts
# ─────────────────────────────────────────────────────────────────────────────

def test_methane_cutoff_5_directed_edges():
    """
    CH4, cutoff=5.0Å: all 5 atoms are within 5Å of each other
    → 5×4 = 20 directed edges (fully-connected minus self-loops).
    Catches: cutoff² bug (would also give 20 here but saturates at large cutoff).
    """
    pos   = _methane_pos()
    batch = _batch(5)
    ei    = rg(pos, r=5.0, batch=batch, max_num_neighbors=32)
    n_edges = ei.shape[1]
    assert n_edges == 20, (
        f"cutoff=5.0Å: expected 20 directed edges for CH4, got {n_edges}. "
        f"If n_edges < 20, cutoff may be too strict (e.g. dist² < cutoff). "
        f"If n_edges > 20, self-loops may not be excluded."
    )


def test_methane_cutoff_05_zero_edges():
    """
    CH4, cutoff=0.5Å: all C-H distances ≈ 1.09Å > 0.5Å → 0 edges.
    Catches: cutoff² bug → 0.5² = 0.25, so dist(1.09) < 0.25 is False → still 0.
    BUT: if dist is stored as dist² ≈ 1.19 > 0.25 → also 0 (can't distinguish).
    This test is most useful when combined with test_methane_cutoff_exact_boundary.
    """
    pos   = _methane_pos()
    batch = _batch(5)
    ei    = rg(pos, r=0.5, batch=batch, max_num_neighbors=32)
    n_edges = ei.shape[1]
    assert n_edges == 0, (
        f"cutoff=0.5Å: expected 0 edges (all C-H ≈ 1.09Å > 0.5Å), got {n_edges}. "
        f"Likely cause: self-loops not removed, or wrong comparison operator."
    )


def test_methane_cutoff_exact_boundary():
    """
    CH4, cutoff just below and just above C-H distance (≈1.09Å).
    cutoff=1.08Å → 0 edges (all H-H ≈ 1.78Å, C-H ≈ 1.09Å > 1.08).
    cutoff=1.10Å → 8 directed edges (4 C-H pairs × 2 directions).

    This is the KEY test for the cutoff² bug:
    - Correct:  1.09 < 1.10  → edge exists          ✓
    - Buggy:    1.09 < 1.10² = 1.21  → also exists  ✓ (false negative — can't distinguish at 1.10)
    - Correct:  1.09 < 1.08  → no edge               ✓
    - Buggy:    1.09 < 1.08² = 1.17  → EDGE EXISTS   ✗ ← this is where bug is caught

    So test with cutoff=1.08Å explicitly catches the squared-cutoff bug.
    """
    pos   = _methane_pos()
    batch = _batch(5)

    # Just below C-H distance: should have 0 C-H edges
    ei_below = rg(pos, r=1.08, batch=batch, max_num_neighbors=32)
    assert ei_below.shape[1] == 0, (
        f"cutoff=1.08Å: expected 0 edges (C-H≈1.09 > 1.08), got {ei_below.shape[1]}. "
        f"CLASSIC CUTOFF² BUG: dist(1.09) < 1.08² (=1.17) is True → false edges appear."
    )

    # Just above C-H distance: 4 C-H pairs × 2 directions = 8 edges
    ei_above = rg(pos, r=1.10, batch=batch, max_num_neighbors=32)
    assert ei_above.shape[1] == 8, (
        f"cutoff=1.10Å: expected 8 C-H directed edges, got {ei_above.shape[1]}."
    )


def test_no_self_loops():
    """Every edge (src, dst) must have src ≠ dst."""
    pos   = _methane_pos()
    batch = _batch(5)
    ei    = rg(pos, r=5.0, batch=batch, max_num_neighbors=32)
    src, dst = ei[0], ei[1]
    self_loops = (src == dst).sum().item()
    assert self_loops == 0, (
        f"Found {self_loops} self-loops in radius graph. "
        f"diagonal should be filled with inf before threshold."
    )


def test_edges_are_directed_symmetric():
    """If (i→j) exists, (j→i) must also exist (undirected graph → symmetric directed)."""
    pos   = _methane_pos()
    batch = _batch(5)
    ei    = rg(pos, r=5.0, batch=batch, max_num_neighbors=32)
    forward = _edge_set(ei)
    reverse = frozenset((d, s) for s, d in forward)
    assert forward == reverse, (
        f"Edge set is not symmetric. Missing reverse edges: {reverse - forward}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (b) Random cube — average degree sanity check
# ─────────────────────────────────────────────────────────────────────────────

def test_random_cube_avg_degree():
    """
    1000 points uniform in [-10, 10]^3, cutoff=1.0Å.
    Expected avg degree ≈ (4/3·π·1³ / 8000) × 999 ≈ 0.523.
    |actual - expected| < 0.20.

    Catches: unit-scale bugs. cutoff² bug would use cutoff=1Å²=1Å (same here, so
    use test_random_cube_avg_degree_squared_cutoff for that distinction).
    """
    rng = np.random.RandomState(0)
    pos_np = rng.uniform(-10, 10, (1000, 3)).astype(np.float32)
    pos    = torch.tensor(pos_np)
    batch  = _batch(1000)
    cutoff = 1.0

    ei = rg(pos, r=cutoff, batch=batch, max_num_neighbors=999)
    n_edges = ei.shape[1]
    avg_degree = n_edges / 1000

    # Expected: volume fraction × (n-1)
    vol_sphere  = (4 / 3) * np.pi * cutoff ** 3
    vol_cube    = 20 ** 3   # [-10,10]^3
    expected_degree = (vol_sphere / vol_cube) * 999  # ≈ 0.523

    assert abs(avg_degree - expected_degree) < 0.20, (
        f"avg_degree={avg_degree:.4f}, expected≈{expected_degree:.4f} "
        f"(|diff|={abs(avg_degree - expected_degree):.4f} > 0.20). "
        f"Likely unit mismatch or wrong distance formula."
    )


def test_random_cube_squared_cutoff_would_fail():
    """
    Adversarial: if cutoff² bug is present (dist < r²), avg_degree with r=2.0
    would be ~4× higher than expected (distance comparison uses r²=4 instead of 2).
    This test directly catches the squared-cutoff class of bug.
    """
    rng = np.random.RandomState(0)
    pos_np = rng.uniform(-10, 10, (500, 3)).astype(np.float32)
    pos    = torch.tensor(pos_np)
    batch  = _batch(500)
    cutoff = 2.0  # distinct from cutoff²=4.0

    ei = rg(pos, r=cutoff, batch=batch, max_num_neighbors=499)
    n_edges = ei.shape[1]
    avg_degree = n_edges / 500

    # Correct expected
    vol_sphere_correct = (4 / 3) * np.pi * 2.0 ** 3  # ≈ 33.5
    vol_sphere_buggy   = (4 / 3) * np.pi * 4.0 ** 3  # ≈ 268
    vol_cube = 20 ** 3
    expected_correct = (vol_sphere_correct / vol_cube) * 499   # ≈ 2.09
    expected_buggy   = (vol_sphere_buggy   / vol_cube) * 499   # ≈ 16.7

    # actual must be close to correct, not buggy
    assert abs(avg_degree - expected_correct) < 1.5, (
        f"avg_degree={avg_degree:.3f}: expected≈{expected_correct:.3f} (correct) "
        f"or ≈{expected_buggy:.3f} (cutoff² bug). "
        f"If avg_degree ≫ {expected_correct:.1f}, cutoff² bug is present."
    )
    assert avg_degree < expected_buggy * 0.5, (
        f"avg_degree={avg_degree:.3f} is suspiciously close to buggy value "
        f"{expected_buggy:.3f} — cutoff² comparison likely active."
    )


# ─────────────────────────────────────────────────────────────────────────────
# (c) Exact match vs scipy cKDTree reference
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_match_scipy_reference():
    """
    100 random points in [-5,5]^3, cutoff=2.0Å, seed=0.
    Edge set must match scipy cKDTree.query_pairs exactly (after sorting).
    Catches: off-by-one (<  vs  <=), wrong axis ordering, wrong src/dst direction.
    """
    rng = np.random.RandomState(0)
    pos_np = rng.uniform(-5, 5, (100, 3)).astype(np.float32)
    pos    = torch.tensor(pos_np)
    batch  = _batch(100)
    cutoff = 2.0

    # Our implementation
    ei_ours = rg(pos, r=cutoff, batch=batch, max_num_neighbors=99)
    edges_ours = _edge_set(ei_ours)

    # Reference: scipy
    ei_ref_np = _ref_radius_graph_scipy(pos_np, cutoff=cutoff, max_neighbors=99)
    if ei_ref_np.shape[1] == 0:
        edges_ref = frozenset()
    else:
        edges_ref = frozenset(zip(ei_ref_np[0].tolist(), ei_ref_np[1].tolist()))

    missing = edges_ref - edges_ours   # in ref but not ours
    extra   = edges_ours - edges_ref   # in ours but not ref

    if missing or extra:
        print(f"\n[exact_match_scipy]  |ours|={len(edges_ours)}  |ref|={len(edges_ref)}")
        if missing:
            sample = list(missing)[:5]
            print(f"  Missing edges (in ref, not ours): {sample} ...")
        if extra:
            sample = list(extra)[:5]
            print(f"  Extra   edges (in ours, not ref): {sample} ...")

    assert not missing, (
        f"{len(missing)} edges present in reference but missing from our impl. "
        f"Sample: {list(missing)[:5]}"
    )
    assert not extra, (
        f"{len(extra)} edges present in our impl but missing from reference. "
        f"Possible cause: strict < should not include boundary (query_pairs uses <)."
    )


@pytest.mark.parametrize("cutoff", [0.5, 1.0, 2.0, 3.5])
def test_exact_match_scipy_multiple_cutoffs(cutoff):
    """
    Same 50-point cloud, four cutoffs. Full edge-set match against scipy at each.
    Catches: fixed-threshold bugs that pass at one cutoff but fail at another.
    """
    rng = np.random.RandomState(42)
    pos_np = rng.uniform(-5, 5, (50, 3)).astype(np.float32)
    pos    = torch.tensor(pos_np)
    batch  = _batch(50)

    ei_ours  = rg(pos, r=cutoff, batch=batch, max_num_neighbors=49)
    ei_ref_np = _ref_radius_graph_scipy(pos_np, cutoff=cutoff, max_neighbors=49)

    edges_ours = _edge_set(ei_ours)
    edges_ref  = (frozenset(zip(ei_ref_np[0].tolist(), ei_ref_np[1].tolist()))
                  if ei_ref_np.shape[1] > 0 else frozenset())

    missing = edges_ref - edges_ours
    extra   = edges_ours - edges_ref

    assert not missing and not extra, (
        f"cutoff={cutoff}: {len(missing)} missing, {len(extra)} extra edges vs scipy. "
        f"Missing sample: {list(missing)[:3]}  Extra sample: {list(extra)[:3]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial: 5 silent-failure modes, each with a dedicated test
# ─────────────────────────────────────────────────────────────────────────────

def test_adversarial_1_squared_distance_comparison():
    """
    BUG CLASS: dist² < cutoff  (or dist < cutoff²)
    Place atom pairs at precisely known distances: 1.5Å and 3.0Å.
    cutoff=2.0Å: only 1.5Å pair should appear.
    cutoff²=4.0:  both would appear (1.5 < 4 and 3.0 < 4) → catches the bug.
    """
    # Pair A: distance = 1.5Å (should be included at cutoff=2.0)
    # Pair B: distance = 3.0Å (should NOT be included at cutoff=2.0, but IS at cutoff²=9)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],   # 1.5Å from node 0
        [5.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],   # 3.0Å from node 2
    ], dtype=torch.float32)
    batch  = _batch(4)
    cutoff = 2.0

    ei = rg(pos, r=cutoff, batch=batch, max_num_neighbors=3)
    edges = _edge_set(ei)

    # Pair A (nodes 0-1): must be present
    assert (0, 1) in edges and (1, 0) in edges, (
        f"[Adversarial #1] Edge 0↔1 (dist=1.5Å) missing at cutoff={cutoff}. "
        f"Edges: {edges}"
    )
    # Pair B (nodes 2-3): must NOT be present
    assert (2, 3) not in edges and (3, 2) not in edges, (
        f"[Adversarial #1] Edge 2↔3 (dist=3.0Å) present at cutoff={cutoff}. "
        f"CUTOFF² BUG: 3.0 < 2.0²=4.0 is True → false edge."
    )


def test_adversarial_2_strict_less_than_boundary():
    """
    BUG CLASS: ≤ instead of < (boundary inclusion).
    Our implementation uses strict < (dist < cutoff).
    Note: scipy.query_pairs uses <= (inclusive), so we do NOT use scipy as
    reference here — we test our own < semantics directly.

    Three atoms at known distances:
      - dist=1.99Å (below cutoff=2.0) → must have edge
      - dist=2.00Å (exactly cutoff)   → must NOT have edge (strict <)
      - dist=2.01Å (above cutoff)     → must NOT have edge
    """
    cutoff = 2.0
    pos = torch.tensor([
        [0.00, 0.0, 0.0],   # node 0
        [1.99, 0.0, 0.0],   # node 1: 1.99Å from 0 → inside
        [2.00, 0.0, 0.0],   # node 2: 2.00Å from 0 → exactly at boundary
        [2.01, 0.0, 0.0],   # node 3: 2.01Å from 0 → outside
    ], dtype=torch.float32)
    batch = _batch(4)

    ei = rg(pos, r=cutoff, batch=batch, max_num_neighbors=3)
    edges = _edge_set(ei)

    # 1.99Å: inside → must be connected
    assert (0, 1) in edges, (
        f"[Adversarial #2] dist=1.99 < cutoff=2.0: edge (0,1) missing. "
        f"Comparison is too strict."
    )
    # 2.00Å: exactly at boundary → strict < means no edge
    assert (0, 2) not in edges, (
        f"[Adversarial #2] dist=2.00 == cutoff=2.0: edge (0,2) present. "
        f"Implementation uses <= instead of <."
    )
    # 2.01Å: outside → must not be connected
    assert (0, 3) not in edges, (
        f"[Adversarial #2] dist=2.01 > cutoff=2.0: edge (0,3) present. "
        f"Threshold is too permissive."
    )


def test_adversarial_3_axis_ordering():
    """
    BUG CLASS: axis swap in cdist or edge_index stacking
    (e.g., returning (dst, src) instead of (src, dst))
    Place asymmetric graph: only node 0 is close to node 1 (but not vice versa,
    which is impossible in Euclidean space — so test src/dst convention instead).
    Verify: edge_index[0] = src (node that sends messages), edge_index[1] = dst.
    Convention: PaiNN uses diff = pos[src] - pos[dst]; src → dst direction.
    """
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ], dtype=torch.float32)
    batch  = _batch(3)
    cutoff = 1.5

    ei = rg(pos, r=cutoff, batch=batch, max_num_neighbors=2)
    src, dst = ei[0].tolist(), ei[1].tolist()
    edges = set(zip(src, dst))

    # Only nodes 0 and 1 are within cutoff
    assert (0, 1) in edges, f"[Adversarial #3] (0→1) missing. src/dst may be swapped."
    assert (1, 0) in edges, f"[Adversarial #3] (1→0) missing. graph is not symmetric."
    assert (0, 2) not in edges, f"[Adversarial #3] (0→2) spurious (dist=5.0 > {cutoff})."
    assert (1, 2) not in edges, f"[Adversarial #3] (1→2) spurious (dist=4.0 > {cutoff})."

    # Verify shape is (2, E)
    assert ei.shape[0] == 2, (
        f"[Adversarial #3] edge_index shape[0]={ei.shape[0]}, expected 2. "
        f"Axes may be transposed (returned (E,2) instead of (2,E))."
    )


def test_adversarial_4_multi_graph_batch_isolation():
    """
    BUG CLASS: batch isolation failure — nodes from different graphs connected.
    Two separate graphs in one batch: graph 0 (nodes 0-1) and graph 1 (nodes 2-3).
    Nodes 1 and 2 are 0.5Å apart, but belong to different graphs → must NOT be connected.
    """
    pos = torch.tensor([
        [0.0,  0.0, 0.0],   # graph 0, node 0
        [1.0,  0.0, 0.0],   # graph 0, node 1
        [1.5,  0.0, 0.0],   # graph 1, node 2  (0.5Å from node 1!)
        [10.0, 0.0, 0.0],   # graph 1, node 3
    ], dtype=torch.float32)
    batch  = torch.tensor([0, 0, 1, 1])
    cutoff = 2.0

    ei = rg(pos, r=cutoff, batch=batch, max_num_neighbors=3)
    edges = _edge_set(ei)

    # Cross-graph edges must not exist
    cross = {(1, 2), (2, 1), (0, 2), (2, 0), (0, 3), (3, 0), (1, 3), (3, 1)}
    found_cross = cross & edges
    assert not found_cross, (
        f"[Adversarial #4] Cross-graph edges found: {found_cross}. "
        f"Batch isolation is broken — nodes from different graphs are being connected."
    )

    # Within-graph edges: 0↔1 (dist=1.0, within cutoff) must exist
    assert (0, 1) in edges and (1, 0) in edges, (
        f"[Adversarial #4] Intra-graph edge 0↔1 missing. edges={edges}"
    )


def test_adversarial_5_distances_are_euclidean_not_squared():
    """
    BUG CLASS: internal distance stored as dist² (L2² instead of L2)
    Then compared as dist² < cutoff → equivalent to dist < sqrt(cutoff),
    which is extremely restrictive and would produce far fewer edges.

    Verify that the distances used internally are true Euclidean (L2), not L2².
    Indirect test: count edges at cutoff=4.0 (true) vs what would happen at
    cutoff=sqrt(4.0)=2.0 if distances were stored as dist².
    """
    rng = np.random.RandomState(7)
    pos_np = rng.uniform(-5, 5, (80, 3)).astype(np.float32)
    pos    = torch.tensor(pos_np)
    batch  = _batch(80)

    ei_c4   = rg(pos, r=4.0, batch=batch, max_num_neighbors=79)
    ei_c2   = rg(pos, r=2.0, batch=batch, max_num_neighbors=79)  # sqrt(4)
    ei_ref  = rg(pos, r=4.0, batch=batch, max_num_neighbors=79)

    # Reference via scipy at cutoff=4.0
    ref_np = _ref_radius_graph_scipy(pos_np, cutoff=4.0, max_neighbors=79)
    ref_edges = (frozenset(zip(ref_np[0].tolist(), ref_np[1].tolist()))
                 if ref_np.shape[1] > 0 else frozenset())

    our_edges_4 = _edge_set(ei_c4)
    our_edges_2 = _edge_set(ei_c2)

    # If distances are stored as dist², calling with cutoff=4.0 would give same
    # edges as correct implementation with cutoff=2.0
    assert our_edges_4 != our_edges_2, (
        f"[Adversarial #5] Edge set at cutoff=4.0 is identical to cutoff=2.0. "
        f"Distances may be stored as L2² and compared to cutoff (not cutoff²), "
        f"so cutoff=4.0 behaves like sqrt(4.0)=2.0."
    )
    assert our_edges_4 == ref_edges, (
        f"[Adversarial #5] cutoff=4.0 does not match scipy reference. "
        f"Missing: {list(ref_edges - our_edges_4)[:3]}  "
        f"Extra: {list(our_edges_4 - ref_edges)[:3]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sanity assertions on _radius_graph output
# ─────────────────────────────────────────────────────────────────────────────

def test_all_distances_within_cutoff():
    """Every returned edge must have actual Euclidean distance strictly < cutoff."""
    pos    = _methane_pos()
    batch  = _batch(5)
    cutoff = 3.0
    ei     = rg(pos, r=cutoff, batch=batch, max_num_neighbors=4)

    src, dst = ei[0], ei[1]
    dists = (pos[src] - pos[dst]).norm(dim=-1)

    max_dist = dists.max().item() if len(dists) > 0 else 0.0
    assert max_dist < cutoff, (
        f"Edge with dist={max_dist:.4f} ≥ cutoff={cutoff}. "
        f"Threshold comparison is wrong (≤ instead of <, or wrong variable)."
    )

    # Sanity thresholds (loose — catch unit/scale bugs, not model quality)
    assert max_dist < 1000, (
        f"Distances suspiciously large: max={max_dist:.2f}. "
        f"Unit mismatch (e.g., Bohr vs Å) or wrong distance formula."
    )


def test_mean_degree_reasonable():
    """
    For methane at cutoff=5.0: every atom sees 4 neighbors → mean_degree=4.
    Sanity assertion: mean_degree < 100 (catches fully-connected / blown-up graphs).
    """
    pos    = _methane_pos()
    batch  = _batch(5)
    cutoff = 5.0
    ei     = rg(pos, r=cutoff, batch=batch, max_num_neighbors=32)

    mean_degree = ei.shape[1] / pos.shape[0]
    assert mean_degree < 100, (
        f"mean_degree={mean_degree:.1f} ≥ 100 for CH4 at cutoff={cutoff}Å. "
        f"Graph is suspiciously dense — cutoff² bug likely (5² = 25 Å)."
    )
    assert mean_degree == 4.0, (
        f"Expected mean_degree=4.0 for CH4 at cutoff=5Å, got {mean_degree:.2f}."
    )
