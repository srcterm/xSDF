import os
import sys
import importlib.util
import time
import numpy as np
import math, torch
from dataclasses import dataclass
from typing import Tuple, Optional

# Load sibling LBVH module (hyphenated filename forces importlib dance here too).
_LBVH_FILE = os.path.join(os.path.dirname(__file__), "lbvh.py")
_lbvh_spec = importlib.util.spec_from_file_location("lbvh", _LBVH_FILE)
lbvh = importlib.util.module_from_spec(_lbvh_spec)
sys.modules["lbvh"] = lbvh
_lbvh_spec.loader.exec_module(lbvh)

# ------------------------------------------------------------------ utilities
def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Choose a torch.device. Honors explicit requests ('cpu', 'mps', 'cuda').
    If prefer is None, auto-pick in order: cuda, mps, cpu.
    """
    if prefer is not None:
        pref = prefer.lower()
        if pref.startswith("cpu"):
            return torch.device("cpu")
        if pref.startswith("mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # Auto-pick when not specified
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@dataclass
class SDFResult:
    phi: torch.Tensor
    origin: torch.Tensor
    dx: float
    grid_shape: Tuple[int, int, int]

# ============================================================================
# LBVH full-grid unsigned distance (|d|) + greedy-leaf warm-start + BFS.
# ============================================================================

def _pairwise_point_triangle_dist(P: torch.Tensor,
                                   A: torch.Tensor,
                                   B: torch.Tensor,
                                   C: torch.Tensor) -> torch.Tensor:
    """Unsigned distance between P_i and triangle (A_i, B_i, C_i). All (N, 3).

    Ericson 6-region test, paired (no outer product). Used inside the leaf
    distance loop where every query has its own (≤ leaf_size) triangles.
    """
    AB = B - A
    AC = C - A
    AP = P - A
    d1 = (AP * AB).sum(-1)
    d2 = (AP * AC).sum(-1)

    BP = P - B
    d3 = (BP * AB).sum(-1)
    d4 = (BP * AC).sum(-1)

    CP = P - C
    d5 = (CP * AB).sum(-1)
    d6 = (CP * AC).sum(-1)

    maskA  = (d1 <= 0.0) & (d2 <= 0.0)
    maskB  = (d3 >= 0.0) & (d4 <= d3)
    maskC  = (d6 >= 0.0) & (d5 <= d6)

    vc = d1 * d4 - d3 * d2
    maskAB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = d1 / (d1 - d3).clamp_min(1e-20) * (d1 >= 0).to(P.dtype)
    v = torch.where((d1 - d3).abs() > 1e-20, d1 / (d1 - d3 + 1e-20), torch.zeros_like(d1))

    vb = d5 * d2 - d1 * d6
    maskAC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w = torch.where((d2 - d6).abs() > 1e-20, d2 / (d2 - d6 + 1e-20), torch.zeros_like(d2))

    va = d3 * d6 - d5 * d4
    denomBC = (d4 - d3) + (d5 - d6)
    t = torch.where(denomBC.abs() > 1e-20,
                    (d4 - d3) / (denomBC + 1e-20),
                    torch.zeros_like(denomBC))
    maskBC = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    maskFace = ~(maskA | maskB | maskC | maskAB | maskAC | maskBC)

    projAB = A + v.unsqueeze(-1) * AB
    projAC = A + w.unsqueeze(-1) * AC
    projBC = B + t.unsqueeze(-1) * (C - B)

    N = torch.linalg.cross(AB, AC, dim=-1)
    N_norm = torch.linalg.norm(N, dim=-1).clamp_min(1e-20)
    N_unit = N / N_norm.unsqueeze(-1)
    dist_plane = (AP * N_unit).sum(-1).abs()

    huge = torch.full_like(d1, 1e20)
    d = huge
    d = torch.minimum(d, torch.where(maskA,    torch.linalg.norm(AP, dim=-1),          huge))
    d = torch.minimum(d, torch.where(maskB,    torch.linalg.norm(BP, dim=-1),          huge))
    d = torch.minimum(d, torch.where(maskC,    torch.linalg.norm(CP, dim=-1),          huge))
    d = torch.minimum(d, torch.where(maskAB,   torch.linalg.norm(P - projAB, dim=-1),  huge))
    d = torch.minimum(d, torch.where(maskAC,   torch.linalg.norm(P - projAC, dim=-1),  huge))
    d = torch.minimum(d, torch.where(maskBC,   torch.linalg.norm(P - projBC, dim=-1),  huge))
    d = torch.minimum(d, torch.where(maskFace, dist_plane,                             huge))
    return d


def _greedy_leaf_warmstart(bvh: "lbvh.LBVH",
                           Q: torch.Tensor,
                           verbose: bool = False,
                           return_closest: bool = False):
    """Tight d_best upper bound via greedy BVH descent to one leaf per query.

    For each query, walk root->leaf picking the child whose AABB is closer.
    At the terminating leaf, evaluate distance to its 1..leaf_size triangles.
    O(Nq · log L) ops on fixed-size tensors — no dynamic reshapes, no host
    syncs in the hot loop — gives BFS a tight enough bound to prune
    aggressively from level 0.

    Bound is ≥ true nearest-surface distance (single-leaf greedy walk may
    miss the truly-closest leaf); the subsequent full BFS tightens it
    exactly.

    If ``return_closest=True``, also returns the original triangle index
    that achieved the warm-start distance per query (long, sentinel = Nt).
    """
    dev = Q.device
    Nq = Q.shape[0]
    if Nq == 0:
        empty = torch.zeros(0, dtype=torch.float32, device=dev)
        if return_closest:
            return empty, torch.zeros(0, dtype=torch.long, device=dev)
        return empty

    L = bvh.num_leaves
    leaf_threshold = L - 1
    leaf_size = bvh.leaf_size
    Nt = bvh.F.shape[0]

    V = bvh.V
    F = bvh.F
    aabb_min = bvh.aabb_min
    aabb_max = bvh.aabb_max
    left = bvh.left
    right = bvh.right
    tri_order = bvh.tri_order.to(torch.long) if bvh.tri_order.dtype != torch.long else bvh.tri_order
    leaf_beg = bvh.leaf_tri_beg.to(torch.long)
    leaf_end = bvh.leaf_tri_end.to(torch.long)

    cur_n = torch.zeros(Nq, dtype=torch.int32, device=dev)
    max_descent = 64  # Karras tree depth ≤ 30 for 30-bit Morton; slack

    for _ in range(max_descent):
        is_leaf = cur_n >= leaf_threshold
        n_long = cur_n.clamp_min(0).to(torch.long)
        lc = left.index_select(0, n_long)
        rc = right.index_select(0, n_long)
        lc_safe = lc.clamp_min(0).to(torch.long)
        rc_safe = rc.clamp_min(0).to(torch.long)

        l_min = aabb_min.index_select(0, lc_safe)
        l_max = aabb_max.index_select(0, lc_safe)
        r_min = aabb_min.index_select(0, rc_safe)
        r_max = aabb_max.index_select(0, rc_safe)

        l_clamped = torch.minimum(torch.maximum(Q, l_min), l_max)
        r_clamped = torch.minimum(torch.maximum(Q, r_min), r_max)
        d_left = torch.linalg.norm(Q - l_clamped, dim=-1)
        d_right = torch.linalg.norm(Q - r_clamped, dim=-1)

        closer_is_left = d_left <= d_right
        next_n = torch.where(closer_is_left, lc, rc)
        cur_n = torch.where(is_leaf, cur_n, next_n)

    # Evaluate triangles at each query's terminating leaf.
    leaf_k = (cur_n - leaf_threshold).clamp_min(0).to(torch.long)
    beg = leaf_beg.index_select(0, leaf_k)
    end = leaf_end.index_select(0, leaf_k)
    d_best = torch.full((Nq,), float("inf"), dtype=torch.float32, device=dev)
    if return_closest:
        closest_tri = torch.full((Nq,), Nt, dtype=torch.long, device=dev)

    for off in range(leaf_size):
        tri_pos = beg + off
        has_tri = tri_pos < end
        tri_pos_safe = tri_pos.clamp_max(Nt - 1)
        tri_orig = tri_order.index_select(0, tri_pos_safe)
        f = F.index_select(0, tri_orig)
        A = V.index_select(0, f[:, 0])
        B = V.index_select(0, f[:, 1])
        C = V.index_select(0, f[:, 2])
        d_tri = _pairwise_point_triangle_dist(Q, A, B, C)
        d_tri = torch.where(has_tri, d_tri, torch.full_like(d_tri, float("inf")))
        if return_closest:
            improving = d_tri < d_best
            closest_tri = torch.where(improving & has_tri, tri_orig, closest_tri)
        d_best = torch.minimum(d_best, d_tri)

    if verbose:
        print(f"[warmstart] greedy-leaf descent  d_best range=["
              f"{float(d_best.min()):.4f}, {float(d_best.max()):.4f}]")
    if return_closest:
        return d_best, closest_tri
    return d_best


# Empirical MPS safe cap for int32 worklist tensors. Above ~20M entries
# (~80 MB) MPS silently corrupts large index_select / cat results, which
# manifests as runaway BFS depth (47 vs the tree's true 30) and ambiguous
# winding numbers. CPU/CUDA pass Nq through unchanged so monolithic BFS
# avoids per-chunk dispatch overhead.
_MPS_CHUNK_THRESH = 16_000_000


def _resolve_chunk_thresh(dev: torch.device, Nq: int, override: Optional[int]) -> int:
    if override is not None:
        return override
    if dev.type == "mps":
        return _MPS_CHUNK_THRESH
    return max(Nq, 1)


def _merge_chunks(chunks, thresh: int):
    """Pack a list of (wl_q, wl_n) chunks back together to reduce dispatch.

    Fast path: when the total fits under ``thresh`` (typical of the BFS's
    shrinking tail), concat everything into a single chunk so subsequent
    iterations launch one set of kernels instead of N. Otherwise pack
    adjacent chunks greedily while their union still fits.
    """
    if len(chunks) <= 1:
        return chunks
    total = sum(c[0].numel() for c in chunks)
    if total <= thresh:
        return [(torch.cat([c[0] for c in chunks]),
                 torch.cat([c[1] for c in chunks]))]
    packed = []
    cur_q, cur_n, cur_size = None, None, 0
    for q, n in chunks:
        sz = q.numel()
        if cur_q is None:
            cur_q, cur_n, cur_size = q, n, sz
        elif cur_size + sz <= thresh:
            cur_q = torch.cat([cur_q, q])
            cur_n = torch.cat([cur_n, n])
            cur_size += sz
        else:
            packed.append((cur_q, cur_n))
            cur_q, cur_n, cur_size = q, n, sz
    if cur_q is not None:
        packed.append((cur_q, cur_n))
    return packed


def _lbvh_unsigned_distance(bvh: "lbvh.LBVH",
                            Q: torch.Tensor,
                            d_best: torch.Tensor,
                            verbose: bool = False,
                            chunk_thresh: Optional[int] = None,
                            closest_tri: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Batched BFS LBVH traversal producing unsigned distance.

    Replaces the per-query state machine with a worklist of (query, node)
    pairs that expands level-by-level. Each BFS iteration is one pass over
    the whole worklist, so total kernel launches scale with tree depth
    (~log₂ L, ≤30 for 30-bit Morton) instead of per-query path length
    (thousands).

    The worklist is held as a list of chunks, each capped at ``chunk_thresh``
    int32 entries. MPS silently corrupts ``index_select``/``cat`` results
    above ~80 MB, which previously inflated BFS depth from 30 to 47 and
    polluted FWN; chunking guarantees individual tensors stay under that
    threshold. After every depth iter, chunks whose union still fits under
    the cap are merged so the dispatch tax only applies during the worklist
    peak, not on the shrinking tail.

    Caller must seed ``d_best`` with a tight upper bound — without it, early
    BFS levels can't prune and the worklist blows up.

    If ``closest_tri`` is provided (long, shape (Nq,), seeded with Nt for
    "no winner"), it is updated in-place alongside d_best — at the end it
    holds the original triangle index achieving min distance per query.
    Atomic (d, tri) tracking uses bit-packed scatter_reduce_(amin) on int64.
    """
    dev = Q.device
    Nq = Q.shape[0]
    if Nq == 0:
        return d_best
    L = bvh.num_leaves
    leaf_threshold = L - 1
    leaf_size = bvh.leaf_size
    Nt = bvh.F.shape[0]

    V = bvh.V
    F = bvh.F
    aabb_min = bvh.aabb_min
    aabb_max = bvh.aabb_max
    left = bvh.left
    right = bvh.right
    tri_order = bvh.tri_order.to(torch.long) if bvh.tri_order.dtype != torch.long else bvh.tri_order
    leaf_beg = bvh.leaf_tri_beg.to(torch.long)
    leaf_end = bvh.leaf_tri_end.to(torch.long)

    thresh = _resolve_chunk_thresh(dev, Nq, chunk_thresh)
    max_depth = 64      # 30-bit Morton caps depth at 30; slack for safety

    # Bit-packed (d, tri) buffer for atomic amin tracking when closest_tri requested.
    # Layout: int64 with float32 d_bits in upper 32 (monotonic for d ≥ 0 in IEEE 754),
    # tri index in lower 32. scatter_reduce_(amin) on this picks (smallest d, smallest
    # tri on ties). Sentinel for "no triangle" is the int Nt (one past last valid).
    track_closest = closest_tri is not None
    if track_closest:
        TRI_MASK = (1 << 32) - 1
        d_bits = d_best.view(torch.int32).to(torch.int64) << 32
        packed_dst = d_bits | (closest_tri & TRI_MASK)

    # Initial worklist: split Q into chunks of ≤ thresh, each rooted at node 0.
    chunks = []
    q_off = 0
    while q_off < Nq:
        qe = min(q_off + thresh, Nq)
        chunks.append((
            torch.arange(q_off, qe, dtype=torch.int32, device=dev),
            torch.zeros(qe - q_off, dtype=torch.int32, device=dev),
        ))
        q_off = qe

    peak_worklist = 0
    peak_chunks = 0
    max_depth_reached = 0

    for depth in range(max_depth):
        new_chunks = []
        P_total = 0
        for wl_q, wl_n in chunks:
            P = wl_q.numel()
            if P == 0:
                continue
            P_total += P

            q_long = wl_q.to(torch.long)
            n_long = wl_n.to(torch.long)

            p = Q.index_select(0, q_long)
            nm = aabb_min.index_select(0, n_long)
            nM = aabb_max.index_select(0, n_long)
            clamped = torch.minimum(torch.maximum(p, nm), nM)
            d_lo = torch.linalg.norm(p - clamped, dim=-1)

            best_pair = d_best.index_select(0, q_long)
            live = d_lo < best_pair
            is_leaf = wl_n >= leaf_threshold

            # ---- Leaf pairs: evaluate triangles, scatter_reduce amin ----
            # No bool().any() guard: empty branches become cheap no-ops,
            # avoiding a host-device sync on MPS each iteration.
            leaf_mask = live & is_leaf
            sub_leaf = leaf_mask.nonzero(as_tuple=True)[0]
            if sub_leaf.numel() > 0:
                leaf_q_long = q_long.index_select(0, sub_leaf)
                leaf_k = (wl_n.index_select(0, sub_leaf) - leaf_threshold).to(torch.long)
                beg = leaf_beg.index_select(0, leaf_k)
                end = leaf_end.index_select(0, leaf_k)
                pts = p.index_select(0, sub_leaf)
                best_upd = best_pair.index_select(0, sub_leaf)
                if track_closest:
                    local_winner_tri = torch.full((sub_leaf.numel(),), Nt,
                                                   dtype=torch.long, device=dev)

                # Always run all leaf_size offsets; has_tri gates per-pair.
                for off in range(leaf_size):
                    tri_pos = beg + off
                    has_tri = tri_pos < end
                    tri_pos_safe = tri_pos.clamp_max(Nt - 1)
                    tri_orig = tri_order.index_select(0, tri_pos_safe)
                    f = F.index_select(0, tri_orig)
                    A = V.index_select(0, f[:, 0])
                    B = V.index_select(0, f[:, 1])
                    C = V.index_select(0, f[:, 2])
                    d_tri = _pairwise_point_triangle_dist(pts, A, B, C)
                    d_tri = torch.where(has_tri, d_tri, torch.full_like(d_tri, float("inf")))
                    if track_closest:
                        improving = (d_tri < best_upd) & has_tri
                        local_winner_tri = torch.where(improving, tri_orig, local_winner_tri)
                    best_upd = torch.minimum(best_upd, d_tri)

                lbvh._scatter_amin(d_best, leaf_q_long, best_upd)
                if track_closest:
                    # Pack (best_upd, local_winner_tri) and amin-scatter into packed_dst.
                    # IEEE 754 non-negative-float bits are monotonic, so amin on the
                    # int64-packed value picks (smaller d, smaller tri on ties).
                    pack_upd = (best_upd.view(torch.int32).to(torch.int64) << 32) \
                               | (local_winner_tri & TRI_MASK)
                    packed_dst.scatter_reduce_(0, leaf_q_long, pack_upd,
                                                'amin', include_self=True)

            # ---- Expand live internal pairs to (left, right) children ----
            expand = live & ~is_leaf
            sub_int = expand.nonzero(as_tuple=True)[0]
            M = sub_int.numel()
            if M == 0:
                continue
            q_int = wl_q.index_select(0, sub_int)
            n_int_long = n_long.index_select(0, sub_int)
            left_c = left.index_select(0, n_int_long).to(torch.int32)
            right_c = right.index_select(0, n_int_long).to(torch.int32)

            if 2 * M <= thresh:
                new_q = torch.empty(2 * M, dtype=torch.int32, device=dev)
                new_q[:M] = q_int; new_q[M:] = q_int
                new_n = torch.empty(2 * M, dtype=torch.int32, device=dev)
                new_n[:M] = left_c; new_n[M:] = right_c
                new_chunks.append((new_q, new_n))
            else:
                # Split each side at thresh-sized boundaries so no single
                # tensor crosses the MPS corruption cap. Adjacent halves get
                # repacked by _merge_chunks if they later fit together.
                for half_q, half_n in ((q_int, left_c), (q_int, right_c)):
                    off = 0
                    while off < M:
                        end = min(off + thresh, M)
                        new_chunks.append((half_q[off:end].clone(),
                                           half_n[off:end].clone()))
                        off = end

        new_chunks = _merge_chunks(new_chunks, thresh)
        if P_total > peak_worklist:
            peak_worklist = P_total
        if len(new_chunks) > peak_chunks:
            peak_chunks = len(new_chunks)
        chunks = new_chunks
        if not chunks:
            break
        max_depth_reached = depth + 1

    if verbose:
        print(f"[lbvh bfs] max depth={max_depth_reached}  peak worklist={peak_worklist}  "
              f"peak chunks={peak_chunks}  thresh={thresh}")

    if track_closest:
        # Decode lower 32 bits of packed_dst as int32 (truncating cast preserves the
        # bit pattern), then to long; the Nt sentinel (set in caller-supplied seed)
        # shouldn't survive if BFS visited every query at least once.
        decoded = (packed_dst & TRI_MASK).to(torch.int32).to(torch.long)
        closest_tri.copy_(decoded)

    return d_best


# =============================================================================
# Barill hierarchical fast winding number (FWN)
# =============================================================================

_INV_4PI = 1.0 / (4.0 * math.pi)


def _lbvh_fwn_winding(bvh: "lbvh.LBVH",
                      Q: torch.Tensor,
                      beta: float = 2.0,
                      verbose: bool = False,
                      chunk_thresh: Optional[int] = None) -> torch.Tensor:
    """Batched BFS LBVH traversal producing the generalized winding number w(p).

    At every BFS level, for each live (query, node) pair:
      - leaf   -> exact Van Oosterom-Strackee solid angle on its 1..leaf_size
                 triangles (scatter_add into w_accum).
      - internal, admissible (||p - c_node|| > beta · r_ball[n])
               -> accept dipole deltaw = (1/4pi) · n_area · (c_node - p) / r^3; done.
      - internal, not admissible -> expand to (left, right) children.

    Admissibility uses c_node = c_area / area_w and r_ball — the L2 norm of
    the per-axis max distance from c_node to the AABB corners (i.e. the
    farthest-corner distance), set by the LBVH builder. Kernel launches
    scale with tree depth, not per-query path length — the key property for
    MPS/CUDA. Worklist chunking + merge mirrors ``_lbvh_unsigned_distance``:
    each chunk stays under the MPS corruption cap, but they're repacked
    once the worklist shrinks below threshold so the dispatch tax is bounded.
    """
    dev = Q.device
    Nq = Q.shape[0]
    if Nq == 0:
        return torch.zeros(0, dtype=torch.float32, device=dev)

    L = bvh.num_leaves
    leaf_threshold = L - 1
    leaf_size = bvh.leaf_size
    Nt = bvh.F.shape[0]
    assert bvh.area_w is not None, "LBVH must carry dipole fields for FWN"

    w_accum = torch.zeros(Nq, dtype=torch.float32, device=dev)

    V = bvh.V
    F = bvh.F
    left = bvh.left
    right = bvh.right
    area_w = bvh.area_w
    c_area = bvh.c_area
    n_area = bvh.n_area
    r_ball = bvh.r_ball
    tri_order = bvh.tri_order.to(torch.long) if bvh.tri_order.dtype != torch.long else bvh.tri_order
    leaf_beg = bvh.leaf_tri_beg.to(torch.long)
    leaf_end = bvh.leaf_tri_end.to(torch.long)

    thresh = _resolve_chunk_thresh(dev, Nq, chunk_thresh)
    max_depth = 64
    eps_denom = 1e-20

    chunks = []
    q_off = 0
    while q_off < Nq:
        qe = min(q_off + thresh, Nq)
        chunks.append((
            torch.arange(q_off, qe, dtype=torch.int32, device=dev),
            torch.zeros(qe - q_off, dtype=torch.int32, device=dev),
        ))
        q_off = qe

    peak_worklist = 0
    peak_chunks = 0
    max_depth_reached = 0

    for depth in range(max_depth):
        new_chunks = []
        P_total = 0
        for wl_q, wl_n in chunks:
            P = wl_q.numel()
            if P == 0:
                continue
            P_total += P

            q_long = wl_q.to(torch.long)
            n_long = wl_n.to(torch.long)
            p = Q.index_select(0, q_long)                     # (P, 3)

            is_leaf = wl_n >= leaf_threshold

            # ---- Admissibility for internal nodes ----
            aw = area_w.index_select(0, n_long).clamp_min(1e-30)
            c_node = c_area.index_select(0, n_long) / aw.unsqueeze(1)
            diff = c_node - p                                 # (P, 3)
            r2 = (diff * diff).sum(dim=-1)
            r = torch.sqrt(r2.clamp_min(eps_denom))
            rball_n = r_ball.index_select(0, n_long)
            admissible = (r > beta * rball_n) & ~is_leaf

            # ---- Accept dipole for admissibles ----
            sub_adm = admissible.nonzero(as_tuple=True)[0]
            if sub_adm.numel() > 0:
                q_sub_long = q_long.index_select(0, sub_adm)
                n_a = n_area.index_select(0, n_long.index_select(0, sub_adm))
                d_a = diff.index_select(0, sub_adm)
                r3 = r.index_select(0, sub_adm).pow(3).clamp_min(eps_denom)
                dw = (n_a * d_a).sum(dim=-1) * (_INV_4PI / r3)
                w_accum.scatter_add_(0, q_sub_long, dw)

            # ---- Leaf: exact VOS per triangle ----
            sub_leaf = is_leaf.nonzero(as_tuple=True)[0]
            if sub_leaf.numel() > 0:
                q_sub_long = q_long.index_select(0, sub_leaf)
                leaf_k = (wl_n.index_select(0, sub_leaf) - leaf_threshold).to(torch.long)
                beg = leaf_beg.index_select(0, leaf_k)
                end = leaf_end.index_select(0, leaf_k)
                pts = p.index_select(0, sub_leaf)
                omega_sum = torch.zeros(pts.shape[0], dtype=torch.float32, device=dev)

                for off in range(leaf_size):
                    tri_pos = beg + off
                    has_tri = tri_pos < end
                    tri_pos_safe = tri_pos.clamp_max(Nt - 1)
                    tri_orig = tri_order.index_select(0, tri_pos_safe)
                    f = F.index_select(0, tri_orig)
                    A = V.index_select(0, f[:, 0]) - pts
                    B = V.index_select(0, f[:, 1]) - pts
                    C = V.index_select(0, f[:, 2]) - pts
                    a_len = torch.linalg.norm(A, dim=-1)
                    b_len = torch.linalg.norm(B, dim=-1)
                    c_len = torch.linalg.norm(C, dim=-1)
                    num = (A * torch.linalg.cross(B, C, dim=-1)).sum(dim=-1)
                    ab = (A * B).sum(dim=-1)
                    bc = (B * C).sum(dim=-1)
                    ca = (C * A).sum(dim=-1)
                    denom = a_len * b_len * c_len + ab * c_len + bc * a_len + ca * b_len + eps_denom
                    omega_tri = 2.0 * torch.atan2(num, denom)
                    omega_tri = torch.where(has_tri, omega_tri, torch.zeros_like(omega_tri))
                    omega_sum = omega_sum + omega_tri

                w_accum.scatter_add_(0, q_sub_long, omega_sum * _INV_4PI)

            # ---- Expand non-admissible internals ----
            expand = ~admissible & ~is_leaf
            sub_int = expand.nonzero(as_tuple=True)[0]
            M = sub_int.numel()
            if M == 0:
                continue
            q_int = wl_q.index_select(0, sub_int)
            n_int_long = n_long.index_select(0, sub_int)
            left_c = left.index_select(0, n_int_long).to(torch.int32)
            right_c = right.index_select(0, n_int_long).to(torch.int32)

            if 2 * M <= thresh:
                new_q = torch.empty(2 * M, dtype=torch.int32, device=dev)
                new_q[:M] = q_int; new_q[M:] = q_int
                new_n = torch.empty(2 * M, dtype=torch.int32, device=dev)
                new_n[:M] = left_c; new_n[M:] = right_c
                new_chunks.append((new_q, new_n))
            else:
                for half_q, half_n in ((q_int, left_c), (q_int, right_c)):
                    off = 0
                    while off < M:
                        end = min(off + thresh, M)
                        new_chunks.append((half_q[off:end].clone(),
                                           half_n[off:end].clone()))
                        off = end

        new_chunks = _merge_chunks(new_chunks, thresh)
        if P_total > peak_worklist:
            peak_worklist = P_total
        if len(new_chunks) > peak_chunks:
            peak_chunks = len(new_chunks)
        chunks = new_chunks
        if not chunks:
            break
        max_depth_reached = depth + 1

    if verbose:
        print(f"[fwn bfs] max depth={max_depth_reached}  peak worklist={peak_worklist}  "
              f"peak chunks={peak_chunks}  thresh={thresh}")

    return w_accum


def _compute_dx_local_max(x_t: torch.Tensor,
                           y_t: torch.Tensor,
                           z_t: torch.Tensor) -> torch.Tensor:
    """Per-voxel max spacing across the three axes.

    For each face-vertex (i,j,k) we take the larger of the forward and
    backward spacing on each axis, then the max across axes. Feeds both the
    narrow-band threshold and the safe-shell seed gate.
    """
    def _axis_max(c: torch.Tensor) -> torch.Tensor:
        dc = c[1:] - c[:-1]                         # (n-1,)
        fwd = torch.cat([dc, dc[-1:]])              # (n,)
        bwd = torch.cat([dc[:1], dc])               # (n,)
        return torch.maximum(fwd, bwd)

    dx_x = _axis_max(x_t)                           # (nx,)
    dx_y = _axis_max(y_t)                           # (ny,)
    dx_z = _axis_max(z_t)                           # (nz,)

    nx, ny, nz = dx_x.numel(), dx_y.numel(), dx_z.numel()
    return torch.maximum(
        torch.maximum(dx_x.view(nx, 1, 1).expand(nx, ny, nz),
                      dx_y.view(1, ny, 1).expand(nx, ny, nz)),
        dx_z.view(1, 1, nz).expand(nx, ny, nz),
    ).contiguous()


def _compute_gradient_nonuniform(U: torch.Tensor,
                                  x_t: torch.Tensor,
                                  y_t: torch.Tensor,
                                  z_t: torch.Tensor) -> torch.Tensor:
    """Central differences on a non-uniform face-vertex grid with one-sided
    boundary stencils, then unit-normalize. Output shape (nx, ny, nz, 3).

    Points where |∇U| is numerically indistinguishable from zero (e.g. a
    perfectly flat region) get a zero gradient so the flood-fill cosine test
    in Stage 6 simply ignores them instead of propagating random directions.
    """
    nx, ny, nz = U.shape

    Gx = torch.empty_like(U)
    dx_center = (x_t[2:] - x_t[:-2]).view(-1, 1, 1)
    Gx[1:-1] = (U[2:] - U[:-2]) / dx_center
    Gx[0]    = (U[1]  - U[0])  / (x_t[1]  - x_t[0])
    Gx[-1]   = (U[-1] - U[-2]) / (x_t[-1] - x_t[-2])

    Gy = torch.empty_like(U)
    dy_center = (y_t[2:] - y_t[:-2]).view(1, -1, 1)
    Gy[:, 1:-1, :] = (U[:, 2:, :] - U[:, :-2, :]) / dy_center
    Gy[:, 0, :]    = (U[:, 1, :]  - U[:, 0, :])  / (y_t[1]  - y_t[0])
    Gy[:, -1, :]   = (U[:, -1, :] - U[:, -2, :]) / (y_t[-1] - y_t[-2])

    Gz = torch.empty_like(U)
    dz_center = (z_t[2:] - z_t[:-2]).view(1, 1, -1)
    Gz[:, :, 1:-1] = (U[:, :, 2:] - U[:, :, :-2]) / dz_center
    Gz[:, :, 0]    = (U[:, :, 1]  - U[:, :, 0])  / (z_t[1]  - z_t[0])
    Gz[:, :, -1]   = (U[:, :, -1] - U[:, :, -2]) / (z_t[-1] - z_t[-2])

    G = torch.stack([Gx, Gy, Gz], dim=-1)           # (nx, ny, nz, 3)

    # Unit-normalize where |G| is non-trivial; zero out flat regions.
    max_dx = max(
        float((x_t[1:] - x_t[:-1]).abs().max()),
        float((y_t[1:] - y_t[:-1]).abs().max()),
        float((z_t[1:] - z_t[:-1]).abs().max()),
    )
    g_norm = torch.linalg.norm(G, dim=-1)
    safe = g_norm > (1e-8 * max_dx)
    G = torch.where(safe.unsqueeze(-1),
                    G / g_norm.clamp_min(1e-20).unsqueeze(-1),
                    torch.zeros_like(G))
    return G


def _narrow_band_and_seed(U: torch.Tensor,
                           dx_local_max: torch.Tensor,
                           fwn_band_width_cells: float,
                           verbose: bool = False):
    """Classify voxels into narrow band vs far field; seed +1 on domain faces
    that sit far enough from the mesh to trust as 'outside'.

    Returns (NB, S, diag) where:
        NB   : (nx, ny, nz) bool — points needing exact FWN signing
        S    : (nx, ny, nz) int8 — 0 unknown, +1 outside-seed 
        diag : dict of per-stage diagnostics (band fraction, seed fractions)
    """
    nx, ny, nz = U.shape
    dev = U.device

    tau_sign = float(fwn_band_width_cells) * dx_local_max
    NB = U < tau_sign

    # Six-face shell mask (one voxel thick).
    is_boundary = torch.zeros_like(U, dtype=torch.bool)
    is_boundary[0, :, :]  = True
    is_boundary[-1, :, :] = True
    is_boundary[:, 0, :]  = True
    is_boundary[:, -1, :] = True
    is_boundary[:, :, 0]  = True
    is_boundary[:, :, -1] = True

    # Seed voxels only where U is clearly far from the mesh (avoids Ahmed's
    # wheel-floor case where a z=0 voxel is inside the tire tread).
    safe = U > 2.0 * dx_local_max
    S = torch.zeros_like(U, dtype=torch.int8)
    S[is_boundary & safe] = 1

    n_total = U.numel()
    nb_frac = float(NB.sum()) / n_total
    n_boundary = int(is_boundary.sum())
    n_seed = int((is_boundary & safe).sum())
    seed_frac = n_seed / max(n_boundary, 1)

    # Per-face safe-seed coverage (Ahmed: z-min face grazes the wheels).
    face_masks = {
        "x-": is_boundary.new_zeros(U.shape, dtype=torch.bool),
        "x+": is_boundary.new_zeros(U.shape, dtype=torch.bool),
        "y-": is_boundary.new_zeros(U.shape, dtype=torch.bool),
        "y+": is_boundary.new_zeros(U.shape, dtype=torch.bool),
        "z-": is_boundary.new_zeros(U.shape, dtype=torch.bool),
        "z+": is_boundary.new_zeros(U.shape, dtype=torch.bool),
    }
    face_masks["x-"][0, :, :]  = True
    face_masks["x+"][-1, :, :] = True
    face_masks["y-"][:, 0, :]  = True
    face_masks["y+"][:, -1, :] = True
    face_masks["z-"][:, :, 0]  = True
    face_masks["z+"][:, :, -1] = True

    face_stats = {}
    for face, mask in face_masks.items():
        n_f = int(mask.sum())
        n_s = int((mask & safe).sum())
        face_stats[face] = n_s / max(n_f, 1)

    diag = {
        "narrow_band_fraction": nb_frac,
        "safe_seed_fraction": seed_frac,
        "face_safe_fraction": face_stats,
    }

    if verbose:
        print(f"[nb/seed] |NB|/N = {nb_frac*100:.2f}%  "
              f"(expect < 10% on well-padded domains)")
        print(f"[nb/seed] safe-shell +1 seeds: {n_seed}/{n_boundary} "
              f"= {seed_frac*100:.2f}%")
        zm = face_stats["z-"]
        if zm < 0.5:
            print(f"[nb/seed] WARNING: z-min face only {zm*100:.1f}% safe — "
                  f"geometry grazes the floor; Stage 8 FWN will cover it")
        if nb_frac > 0.10:
            print(f"[nb/seed] WARNING: narrow-band fraction {nb_frac*100:.2f}% "
                  f"is high; consider widening domain or reducing band cells")

    return NB, S, diag


def _flood_fill_gradient_consistent(S: torch.Tensor,
                                      G: torch.Tensor,
                                      FF: torch.Tensor,
                                      cos_theta_min: float,
                                      max_iters: int,
                                      verbose: bool = False) -> torch.Tensor:
    """Jacobi flood fill gated by gradient alignment.

    For every still-unknown voxel in the far field, a face-neighbor's sign
    only votes when ⟨G_v, G_n⟩ > cos_theta_min and the neighbor itself
    already carries a committed sign (±1, not 0 and not the conflict marker).

    Resolution per iteration for a voxel v with S[v] == 0 and FF[v] True:
        pos > 0, neg == 0  ->  +1
        neg > 0, pos == 0  ->  -1
        pos > 0, neg > 0   ->  -2   
        otherwise          ->   0   (try again next iteration)

    Implementation: slice-based 6-neighbor gather (no fancy indexing) so MPS
    stays on the fast path. Only scalar sync per iter is the ``num_changed``
    reduction used for early termination.
    """
    nx, ny, nz = S.shape
    dev = S.device

    # Dict of 6 (name, center_slice, neighbor_slice) pairs so the 6 shifts
    # share one loop body.
    Sc = slice(None)
    shifts = (
        ("+x", (slice(0, nx - 1), Sc, Sc), (slice(1, nx),     Sc, Sc)),
        ("-x", (slice(1, nx),     Sc, Sc), (slice(0, nx - 1), Sc, Sc)),
        ("+y", (Sc, slice(0, ny - 1), Sc), (Sc, slice(1, ny),     Sc)),
        ("-y", (Sc, slice(1, ny),     Sc), (Sc, slice(0, ny - 1), Sc)),
        ("+z", (Sc, Sc, slice(0, nz - 1)), (Sc, Sc, slice(1, nz))),
        ("-z", (Sc, Sc, slice(1, nz)),     (Sc, Sc, slice(0, nz - 1))),
    )

    int8_pos = torch.tensor( 1, dtype=torch.int8, device=dev)
    int8_neg = torch.tensor(-1, dtype=torch.int8, device=dev)
    int8_cfl = torch.tensor(-2, dtype=torch.int8, device=dev)

    total_changed = 0
    converged_iter = None
    for it in range(max_iters):
        pos_vote = torch.zeros((nx, ny, nz), dtype=torch.int32, device=dev)
        neg_vote = torch.zeros((nx, ny, nz), dtype=torch.int32, device=dev)

        for _name, cs, ns in shifts:
            G_c = G[cs + (Sc,)]
            G_n = G[ns + (Sc,)]
            cos_t = (G_c * G_n).sum(dim=-1)
            S_n = S[ns]
            signed_neighbor = (S_n == 1) | (S_n == -1)
            accept = (cos_t > cos_theta_min) & signed_neighbor
            pos_vote[cs] += (accept & (S_n == 1)).to(torch.int32)
            neg_vote[cs] += (accept & (S_n == -1)).to(torch.int32)

        unknown = (S == 0) & FF
        has_pos = pos_vote > 0
        has_neg = neg_vote > 0
        assign_pos      = unknown & has_pos & ~has_neg
        assign_neg      = unknown & has_neg & ~has_pos
        assign_conflict = unknown & has_pos &  has_neg

        S = S.masked_fill(assign_pos,      int8_pos)
        S = S.masked_fill(assign_neg,      int8_neg)
        S = S.masked_fill(assign_conflict, int8_cfl)

        changed = assign_pos | assign_neg | assign_conflict
        num_changed = int(changed.sum())
        total_changed += num_changed
        if num_changed == 0:
            converged_iter = it
            break

    if verbose:
        n_pos = int((S ==  1).sum())
        n_neg = int((S == -1).sum())
        n_cfl = int((S == -2).sum())
        n_unk = int((S ==  0).sum())
        print(f"[flood] {converged_iter if converged_iter is not None else max_iters} iters, "
              f"{total_changed} voxels updated; "
              f"+1={n_pos} −1={n_neg} conflict={n_cfl} unknown={n_unk}")

    return S


def mesh_to_sdf_tensor_v2(
    V_np,
    F_np,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    *,
    device: Optional[str] = None,
    fwn_beta: float = 2.0,
    fwn_band_width_cells: float = 3.0,
    cos_theta_min: float = 0.8,
    verbose: bool = True,
) -> SDFResult:
    """LBVH + Barill FWN SDF pipeline — torch-tensor result variant.

    Same pipeline as mesh_to_sdf_torch_v2 (which is the numpy-result wrapper);
    skips the .cpu().numpy() conversion so phi remains a torch tensor on
    `device`. The flood-fill + sign-assignment stages still produce a discrete
    int8 sign field (sign is locally constant in any SDF, so its gradient is
    zero a.e. — that's mathematically correct), so this variant is "tensor in,
    tensor out" rather than fully autograd-differentiable. Use ``compute_sdf``
    for the autograd-aware point-query path.

    Pipeline:
      1. Build LBVH (Karras 2012) on device.
      2. Greedy-leaf warm-start: O(Nq . log L) descent giving d_best upper
         bound; lets BFS prune aggressively from level 0.
      3. Batched-BFS LBVH traversal -> exact unsigned distance |d|.
      4. Central-difference gradient on the non-uniform grid.
      5. Narrow-band classification + safe-shell +1 seeding.
      6. Gradient-consistent Jacobi flood fill (cos_theta_min gate).
      7. Gather unresolved voxels (NB ∪ conflict ∪ unknown) for exact FWN.
      8. Batched-BFS Barill FWN on the gathered set -> exact sign per voxel.
      9. Assemble phi = sign . |d|, clamp to 2·domain-diagonal.
    """
    dev = pick_device(device)
    if dev.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if verbose:
        print(f"[v2] LBVH+FWN  device={dev}  "
              f"fwn_beta={fwn_beta} band={fwn_band_width_cells} cos_theta_min={cos_theta_min}")

    # ---- Grid assembly ----
    x_t = torch.as_tensor(x_coords, dtype=torch.float32, device=dev)
    y_t = torch.as_tensor(y_coords, dtype=torch.float32, device=dev)
    z_t = torch.as_tensor(z_coords, dtype=torch.float32, device=dev)
    nx, ny, nz = x_t.numel(), y_t.numel(), z_t.numel()

    X, Y, Z = torch.meshgrid(x_t, y_t, z_t, indexing="ij")
    Q = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)          # (Nq, 3)
    Nq = Q.shape[0]
    origin = torch.tensor([x_t[0].item(), y_t[0].item(), z_t[0].item()],
                          dtype=torch.float32)

    # ---- Stage 1: Build LBVH ----
    t0 = time.time()
    V = torch.as_tensor(V_np, dtype=torch.float32, device=dev)
    F = torch.as_tensor(F_np, dtype=torch.int64, device=dev)
    bvh = lbvh.build_lbvh(V, F, leaf_size=1)
    if verbose:
        print(f"[lbvh] Nt={F.shape[0]} L={bvh.num_leaves} nodes={bvh.num_nodes} "
              f"built in {time.time() - t0:.3f}s")

    # ---- Stage 2: Greedy-leaf warm-start: tight d_best upper bound, O(Nq·log L) ----
    t0 = time.time()
    d_best = _greedy_leaf_warmstart(bvh, Q, verbose=verbose)
    if verbose:
        print(f"[warmstart] {time.time() - t0:.3f}s")

    # ---- Stage 3: Full-grid unsigned distance via batched-BFS LBVH traversal ----
    t0 = time.time()
    d_best = _lbvh_unsigned_distance(bvh, Q, d_best, verbose=verbose)
    if verbose:
        print(f"[traversal] {time.time() - t0:.3f}s  "
              f"|d| range=[{float(d_best.min()):.4f}, {float(d_best.max()):.4f}]")

    U = d_best.reshape(nx, ny, nz)

    # ---- Stage 4: gradient on non-uniform grid (unit-normalized) ----
    t0 = time.time()
    G = _compute_gradient_nonuniform(U, x_t, y_t, z_t)
    if verbose:
        gmag = torch.linalg.norm(G, dim=-1)
        nz_mask = gmag > 0
        mean_mag = float(gmag[nz_mask].mean()) if bool(nz_mask.any()) else 0.0
        print(f"[gradient] {time.time() - t0:.3f}s  "
              f"mean|G|={mean_mag:.4f} (≈1 is eikonal), "
              f"zeroed={int((~nz_mask).sum())}/{U.numel()}")

    # ---- Stages 5: narrow band + safe-shell seed ----
    t0 = time.time()
    dx_local_max = _compute_dx_local_max(x_t, y_t, z_t)
    NB, S, diag = _narrow_band_and_seed(
        U, dx_local_max, fwn_band_width_cells, verbose=verbose)
    if verbose:
        print(f"[nb/seed] {time.time() - t0:.3f}s")

    # ---- Stage 6: gradient-consistent flood fill (Jacobi) ----
    t0 = time.time()
    FF = ~NB
    max_flood_iters = 3 * max(nx, ny, nz)
    S = _flood_fill_gradient_consistent(
        S, G, FF, cos_theta_min=cos_theta_min,
        max_iters=max_flood_iters, verbose=verbose)
    if verbose:
        print(f"[flood] {time.time() - t0:.3f}s")

    # ---- Stage 7: gather unresolved voxels for exact FWN ----
    query_mask = NB | (S == -2) | (S == 0)
    n_query = int(query_mask.sum())
    n_nb = int(NB.sum())
    n_conflict = int((S == -2).sum())
    n_unknown = int(((S == 0) & ~NB).sum())
    if verbose:
        total = U.numel()
        print(f"[fwn query] {n_query}/{total} "
              f"(NB={n_nb}, conflict={n_conflict}, unknown-FF={n_unknown})")
        if n_conflict > 0.01 * total:
            print(f"  WARN: conflict fraction {n_conflict/total*100:.2f}% > 1%")
        if n_unknown > 0.001 * total:
            print(f"  WARN: unknown-FF fraction {n_unknown/total*100:.3f}% > 0.1%")

    # ---- Stage 8: Barill hierarchical FWN on queries ----
    t0 = time.time()
    if n_query > 0:
        q_idx = query_mask.reshape(-1).nonzero(as_tuple=True)[0]
        q_pts = Q.index_select(0, q_idx)
        w = _lbvh_fwn_winding(bvh, q_pts, beta=fwn_beta, verbose=verbose)
        inside_q = w.abs() > 0.5
        if verbose:
            mid_mass = float(((w.abs() > 0.4) & (w.abs() < 0.6)).float().mean())
            print(f"[fwn] {time.time() - t0:.3f}s  "
                  f"|w| range=[{float(w.abs().min()):.3f}, {float(w.abs().max()):.3f}]  "
                  f"inside={int(inside_q.sum())}/{n_query}  "
                  f"ambiguous={mid_mass*100:.2f}%")
            if mid_mass > 0.01:
                print("  WARN: >1% queries have |w|∈[0.4,0.6] — mesh may not be watertight")

        S_flat = S.reshape(-1)
        sign_q = torch.where(inside_q,
                             torch.full_like(w, -1.0),
                             torch.full_like(w,  1.0)).to(torch.int8)
        S_flat.scatter_(0, q_idx, sign_q)
        S = S_flat.reshape(nx, ny, nz)

    # ---- Stage 9: final assembly ----
    # Every voxel's sign is now explicit: S == -1 inside, S == +1 outside.
    sign = torch.where(S == -1,
                       torch.full_like(U, -1.0),
                       torch.full_like(U,  1.0))
    phi = sign * U

    scene_min = torch.stack([x_t.min(), y_t.min(), z_t.min()])
    scene_max = torch.stack([x_t.max(), y_t.max(), z_t.max()])
    diag = float(torch.linalg.norm(scene_max - scene_min))
    max_dist = 2.0 * diag
    phi = torch.nan_to_num(phi, nan=max_dist,
                           posinf=max_dist, neginf=-max_dist)
    phi = phi.clamp(-max_dist, max_dist)

    # dx scalar only if the grid is uniform (same as legacy).
    dx_out = float("nan")
    if nx > 1 and ny > 1 and nz > 1:
        dx_x = float(x_t[1] - x_t[0])
        dx_y = float(y_t[1] - y_t[0])
        dx_z = float(z_t[1] - z_t[0])
        if abs(dx_x - dx_y) < 1e-6 and abs(dx_x - dx_z) < 1e-6:
            dx_out = dx_x

    return SDFResult(phi, origin, dx_out, (nx, ny, nz))


def mesh_to_sdf_torch_v2(
    V_np,
    F_np,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    *,
    device: Optional[str] = None,
    fwn_beta: float = 2.0,
    fwn_band_width_cells: float = 3.0,
    cos_theta_min: float = 0.8,
    verbose: bool = True,
) -> SDFResult:
    """LBVH + Barill FWN SDF pipeline — numpy-result variant (legacy entry).

    Wraps mesh_to_sdf_tensor_v2; converts phi/origin to numpy for back-compat
    with xSDF.py's HDF5 writer and any caller already consuming numpy arrays.
    """
    res = mesh_to_sdf_tensor_v2(
        V_np, F_np, x_coords, y_coords, z_coords,
        device=device,
        fwn_beta=fwn_beta,
        fwn_band_width_cells=fwn_band_width_cells,
        cos_theta_min=cos_theta_min,
        verbose=verbose,
    )
    return SDFResult(res.phi.cpu().numpy(), res.origin.numpy(),
                     res.dx, res.grid_shape)


# ============================================================================
# Public point-query API: compute_sdf + compute_grad.
# Returns torch tensors on-device; autograd graph is preserved end-to-end so
# callers may backprop through points (∇_xyz SDF) or V (∇_V SDF). Sign is
# locally constant so its gradient is zero a.e. — correct for SDFs.
# ============================================================================


def _resolve_dev(points, device) -> torch.device:
    """Pick a target torch.device given the `device` arg and the `points` input."""
    if device is None:
        if isinstance(points, torch.Tensor):
            return points.device
        return pick_device(None)
    if isinstance(device, torch.device):
        return device
    return pick_device(str(device))


def _to_dev(t, dtype, dev):
    """Move t (Tensor or array-like) to `dev` with `dtype`. Avoids unnecessary
    copies when t is already on dev/dtype, preserving any autograd graph."""
    if isinstance(t, torch.Tensor):
        if t.device == dev and t.dtype == dtype:
            return t
        return t.to(device=dev, dtype=dtype)
    return torch.as_tensor(t, dtype=dtype, device=dev)


def compute_sdf(
    V,
    F,
    points,
    *,
    device=None,
    fwn_beta: float = 2.0,
    verbose: bool = False,
) -> torch.Tensor:
    """Signed distance at arbitrary query points.

    Pipeline: build LBVH (Karras 2012), greedy-leaf warm-start + chunked-BFS
    LBVH traversal for unsigned |d|, Barill hierarchical FWN for sign.

    Args:
        V: (Nv, 3) float — mesh vertices. Tensor or numpy.
        F: (Nf, 3) int  — triangle indices. Tensor or numpy.
        points: (Nq, 3) float — query points. Tensor or numpy.
        device: target torch.device (or "cpu"/"mps"/"cuda" string). Defaults to
                points.device if `points` is a tensor; else cuda > mps > cpu.
        fwn_beta: Barill dipole admissibility ratio (default 2.0, ~4 digits).

    Returns:
        (Nq,) float32 torch tensor of signed distances on `device`. Autograd
        graph is preserved through V and points; sign is locally constant so
        ∇sign ≡ 0 a.e. (correct for SDFs).
    """
    dev = _resolve_dev(points, device)
    V_t = _to_dev(V, torch.float32, dev)
    F_t = _to_dev(F, torch.int64, dev)
    Q_t = _to_dev(points, torch.float32, dev)

    # Forward path: BVH topology, BFS unsigned distance, FWN sign — all in
    # no_grad. The BFS uses in-place scatter_reduce_(amin) on shared accumulators
    # which trips autograd's saved-tensor versioning when stacked across
    # iterations. Instead, we extract closest-triangle indices in no_grad and
    # re-evaluate the final point-triangle distance with grad below.
    Nq = Q_t.shape[0]
    Nt = F_t.shape[0]
    with torch.no_grad():
        # build_lbvh internally calls scatter_reduce_ on V-derived AABBs/dipoles;
        # those would also break grad on V via the same iterative-in-place issue.
        # Topology is integer-discrete anyway, so no_grad is correct here.
        bvh = lbvh.build_lbvh(V_t.detach(), F_t, leaf_size=1)

        d_init, closest_tri = _greedy_leaf_warmstart(bvh, Q_t, verbose=verbose,
                                                     return_closest=True)
        _lbvh_unsigned_distance(bvh, Q_t, d_init, verbose=verbose,
                                closest_tri=closest_tri)
        # closest_tri is updated in-place; d_init holds the final unsigned distance,
        # but we discard it and recompute differentiably from closest_tri.

        w = _lbvh_fwn_winding(bvh, Q_t, beta=fwn_beta, verbose=verbose)
        sign = torch.where(w.abs() > 0.5,
                           torch.full((Nq,), -1.0, dtype=torch.float32, device=dev),
                           torch.full((Nq,),  1.0, dtype=torch.float32, device=dev))

    # Re-evaluate point-triangle distance on closest triangles WITH grad. This is
    # where Q_t and V_t feed the autograd graph. The integer index path
    # (closest_tri) is discrete and detached, which is correct.
    closest_tri_safe = closest_tri.clamp_max(Nt - 1)        # guard sentinel
    f = F_t.index_select(0, closest_tri_safe)               # (Nq, 3)
    A = V_t.index_select(0, f[:, 0])
    B = V_t.index_select(0, f[:, 1])
    C = V_t.index_select(0, f[:, 2])
    d_unsigned = _pairwise_point_triangle_dist(Q_t, A, B, C)
    return sign * d_unsigned


def compute_grad(
    V,
    F,
    points,
    *,
    device=None,
    fwn_beta: float = 2.0,
    verbose: bool = False,
) -> torch.Tensor:
    """∇_xyz SDF (surface-normal direction) at query points, via autograd
    backward on `compute_sdf`.

    Single forward + single backward through the same code path as compute_sdf —
    no second implementation of "what's the gradient". Magnitude is ≈ 1
    (eikonal); direction points from the closest surface point outward, with
    sign flipped inside the mesh.

    Returns:
        (Nq, 3) float32 torch tensor of gradient vectors on `device`.
    """
    dev = _resolve_dev(points, device)
    V_t = _to_dev(V, torch.float32, dev)
    F_t = _to_dev(F, torch.int64, dev)
    Q_t = _to_dev(points, torch.float32, dev).detach().clone().requires_grad_(True)

    sdf = compute_sdf(V_t, F_t, Q_t, device=dev, fwn_beta=fwn_beta, verbose=verbose)
    grad, = torch.autograd.grad(sdf.sum(), Q_t, create_graph=False)
    return grad