"""Bounding Volume Hierarchy (BVH) acceleration for xSDF.

Pure-torch implementation (no NumPy) providing:
    - build_bvh_torch: top-down median-split builder. Default runs on CPU because
      the algorithm is inherently sequential (many small per-node ops); GPU
      kernel-launch overhead would dominate. The built tree is small (~5 MB for
      100k tris) and uploaded to the query device with a single transfer.
      Also emits skip-pointer fields (hit_idx, miss_idx) and Barill dipole
      moments (centroid_moment, normal_moment, area_sum, radius) used by the
      GPU FWN / skip-pointer traversal in the fwn accel mode.
    - bvh_to_device: move an entire tree dict to another torch.device.
    - bvh_stats: depth / leaf-size summary for logging.
    - bvh_query_distances: per-point branch-and-bound unsigned distance query
      (CPU hybrid path). Each point keeps a private stack and prunes subtrees
      whose AABBs are farther than its running best distance.
    - point_triangle_distance_pair: per-pair unsigned distance helper used
      inside leaf evaluation during per-point traversal.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


# ------------------------------------------------------------------ build

def build_bvh_torch(
    V: torch.Tensor,
    F: torch.Tensor,
    leaf_size: int = 8,
    build_device: str = "cpu",
) -> Dict:
    """Build a per-triangle BVH using top-down median splits on the longest axis.

    Args:
        V: (N_v, 3) vertex tensor (any float dtype, any device).
        F: (N_t, 3) face tensor (integer dtype, any device).
        leaf_size: maximum triangles per leaf (ragged leaves pad up to this
            size in the traversal kernel).
        build_device: torch device string for the build. Default 'cpu' — see
            module docstring.

    Returns:
        dict with keys (all tensors live on ``build_device``; caller should
        upload via :func:`bvh_to_device`):
            node_min, node_max: (M, 3) float32 AABBs per node.
            left, right: (M,) int32 child indices, -1 when leaf.
            is_leaf: (M,) bool.
            leaf_start, leaf_count: (M,) int32 into ``tri_perm`` (0 when internal).
            tri_perm: (N_t,) int64 permutation — triangles in each leaf are
                contiguous in this order.
            n_nodes: int, == M (for convenience).
            leaf_size: int, stored so the traversal uses the same value.
    """
    if leaf_size < 1:
        raise ValueError("leaf_size must be >= 1")

    dev = torch.device(build_device)
    V_b = V.to(dev, dtype=torch.float32)
    F_b = F.to(dev, dtype=torch.int64)
    T = int(F_b.shape[0])
    if T == 0:
        raise ValueError("build_bvh_torch: empty face tensor")

    A = V_b[F_b[:, 0]]
    B = V_b[F_b[:, 1]]
    C = V_b[F_b[:, 2]]
    tri_min = torch.minimum(torch.minimum(A, B), C)
    tri_max = torch.maximum(torch.maximum(A, B), C)
    centroids = (A + B + C) / 3.0

    # Safe upper bound: a binary tree with up to T leaves has at most 2T-1 nodes.
    # Median splits can land unevenly so leaves may have < leaf_size triangles,
    # making the tighter "2*ceil(T/leaf_size)-1" bound unsafe in practice.
    M_max = max(1, 2 * T - 1)

    node_min = torch.zeros((M_max, 3), device=dev, dtype=torch.float32)
    node_max = torch.zeros((M_max, 3), device=dev, dtype=torch.float32)
    left = torch.full((M_max,), -1, device=dev, dtype=torch.int32)
    right = torch.full((M_max,), -1, device=dev, dtype=torch.int32)
    parent = torch.full((M_max,), -1, device=dev, dtype=torch.int32)
    is_leaf = torch.zeros((M_max,), device=dev, dtype=torch.bool)
    leaf_start = torch.zeros((M_max,), device=dev, dtype=torch.int32)
    leaf_count = torch.zeros((M_max,), device=dev, dtype=torch.int32)
    tri_perm = torch.arange(T, device=dev, dtype=torch.int64)

    work = [(0, T, 0)]
    next_node = 1

    while work:
        s, e, nid = work.pop()
        n = e - s
        perm_slice = tri_perm[s:e]
        slab_min = tri_min.index_select(0, perm_slice).amin(dim=0)
        slab_max = tri_max.index_select(0, perm_slice).amax(dim=0)
        node_min[nid] = slab_min
        node_max[nid] = slab_max

        if n <= leaf_size:
            is_leaf[nid] = True
            leaf_start[nid] = s
            leaf_count[nid] = n
            continue

        axis = int((slab_max - slab_min).argmax().item())
        cvals = centroids.index_select(0, perm_slice)[:, axis]
        order = torch.argsort(cvals)
        tri_perm[s:e] = perm_slice.index_select(0, order)

        mid = s + n // 2
        if mid == s or mid == e:
            is_leaf[nid] = True
            leaf_start[nid] = s
            leaf_count[nid] = n
            continue

        l_id = next_node
        r_id = next_node + 1
        next_node += 2
        left[nid] = l_id
        right[nid] = r_id
        parent[l_id] = nid
        parent[r_id] = nid
        work.append((mid, e, r_id))
        work.append((s, mid, l_id))

    node_min = node_min[:next_node].contiguous()
    node_max = node_max[:next_node].contiguous()
    left = left[:next_node].contiguous()
    right = right[:next_node].contiguous()
    parent = parent[:next_node].contiguous()
    is_leaf = is_leaf[:next_node].contiguous()
    leaf_start = leaf_start[:next_node].contiguous()
    leaf_count = leaf_count[:next_node].contiguous()

    hit_idx, miss_idx = _compute_skip_pointers(left, right, is_leaf, next_node)
    centroid_moment, normal_moment, area_sum, radius, max_leaf_size = _compute_dipoles(
        V_b, F_b, left, right, parent, is_leaf, leaf_start, leaf_count, tri_perm, next_node
    )

    return {
        "node_min": node_min,
        "node_max": node_max,
        "left": left,
        "right": right,
        "is_leaf": is_leaf,
        "leaf_start": leaf_start,
        "leaf_count": leaf_count,
        "tri_perm": tri_perm.contiguous(),
        "n_nodes": next_node,
        "leaf_size": int(leaf_size),
        "max_leaf_size": int(max_leaf_size),
        "hit_idx": hit_idx,
        "miss_idx": miss_idx,
        "centroid_moment": centroid_moment,
        "normal_moment": normal_moment,
        "area_sum": area_sum,
        "radius": radius,
    }


def _compute_skip_pointers(
    left: torch.Tensor,
    right: torch.Tensor,
    is_leaf: torch.Tensor,
    n_nodes: int,
) -> tuple:
    """Stackless-BVH skip pointers.

    hit_idx[n]  = next node to visit when entering n (left child for internals;
                  escape for leaves — i.e. "processed, advance past subtree").
    miss_idx[n] = next node to visit when skipping n's subtree (escape pointer).

    Traversal: current = hit_idx[current] to descend, or miss_idx[current]
    to skip; -1 terminates.
    """
    dev = left.device
    hit = torch.full((n_nodes,), -1, device=dev, dtype=torch.int32)
    miss = torch.full((n_nodes,), -1, device=dev, dtype=torch.int32)
    stack = [(0, -1)]
    while stack:
        nid, esc = stack.pop()
        miss[nid] = esc
        if bool(is_leaf[nid]):
            hit[nid] = esc
        else:
            L = int(left[nid].item())
            R = int(right[nid].item())
            hit[nid] = L
            stack.append((R, esc))
            stack.append((L, R))
    return hit, miss


def _compute_dipoles(
    V_b: torch.Tensor,
    F_b: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    parent: torch.Tensor,
    is_leaf: torch.Tensor,
    leaf_start: torch.Tensor,
    leaf_count: torch.Tensor,
    tri_perm: torch.Tensor,
    n_nodes: int,
) -> tuple:
    """Barill dipole moments + tight cluster radii per BVH node.

    For each node n stores:
        area_sum[n]: sum of triangle areas in the cluster.
        centroid_moment[n] (3,): sum of tri_centroid * tri_area.
        normal_moment[n] (3,): sum of unit_normal * tri_area.
        radius[n]: max distance from cluster centroid (p_bar = centroid_moment/area_sum)
                   to any triangle vertex in the subtree. Tight, not the
                   child-triangle-inequality upper bound.

    Moments aggregate bottom-up by depth; tight radii come from walking each
    triangle up its parent chain and scatter-maxing vertex distances at each
    ancestor node.
    """
    dev = V_b.device
    T = int(F_b.shape[0])

    A = V_b[F_b[:, 0]]
    B = V_b[F_b[:, 1]]
    C = V_b[F_b[:, 2]]
    tri_centroids = (A + B + C) / 3.0
    cross_ab_ac = torch.linalg.cross(B - A, C - A, dim=-1)
    tri_area = 0.5 * torch.linalg.norm(cross_ab_ac, dim=-1)
    tri_area_normal = 0.5 * cross_ab_ac
    tri_area_centroid = tri_centroids * tri_area.unsqueeze(-1)

    # Map each original triangle id to the leaf node that owns it.
    leaf_nids = is_leaf.nonzero().flatten().to(torch.int64)
    leaf_starts_L = leaf_start[leaf_nids].to(torch.int64)
    leaf_counts_L = leaf_count[leaf_nids].to(torch.int64)
    order = torch.argsort(leaf_starts_L)
    leaf_nids_sorted = leaf_nids[order]
    leaf_counts_sorted = leaf_counts_L[order]
    perm_pos_to_nid = torch.repeat_interleave(leaf_nids_sorted, leaf_counts_sorted)
    tri_to_nid = torch.empty((T,), device=dev, dtype=torch.int64)
    tri_to_nid[tri_perm.to(torch.int64)] = perm_pos_to_nid

    area_sum = torch.zeros((n_nodes,), device=dev, dtype=torch.float32)
    centroid_moment = torch.zeros((n_nodes, 3), device=dev, dtype=torch.float32)
    normal_moment = torch.zeros((n_nodes, 3), device=dev, dtype=torch.float32)
    area_sum.index_add_(0, tri_to_nid, tri_area)
    centroid_moment.index_add_(0, tri_to_nid, tri_area_centroid)
    normal_moment.index_add_(0, tri_to_nid, tri_area_normal)

    # Aggregate internals bottom-up by depth for moments.
    depth = torch.zeros((n_nodes,), device=dev, dtype=torch.int32)
    stack = [(0, 0)]
    while stack:
        nid, d = stack.pop()
        depth[nid] = d
        if not bool(is_leaf[nid]):
            L = int(left[nid].item())
            R = int(right[nid].item())
            stack.append((L, d + 1))
            stack.append((R, d + 1))
    max_depth = int(depth.max().item())

    for d in range(max_depth - 1, -1, -1):
        mask = (depth == d) & (~is_leaf)
        nodes_d = mask.nonzero().flatten().to(torch.int64)
        if nodes_d.numel() == 0:
            continue
        L = left[nodes_d].to(torch.int64)
        R = right[nodes_d].to(torch.int64)
        area_sum[nodes_d] = area_sum[L] + area_sum[R]
        centroid_moment[nodes_d] = centroid_moment[L] + centroid_moment[R]
        normal_moment[nodes_d] = normal_moment[L] + normal_moment[R]

    # Tight radius: for each triangle, walk up parent chain; at each ancestor
    # scatter-max the max-vertex-distance to that ancestor's p_bar. O(T*depth).
    p_bar_all = centroid_moment / area_sum.clamp_min(1e-20).unsqueeze(-1)
    radius = torch.zeros((n_nodes,), device=dev, dtype=torch.float32)
    parent64 = parent.to(torch.int64)
    current = tri_to_nid.clone()  # (T,) — each triangle starts at its leaf
    while bool((current != -1).any()):
        mask = current != -1
        m_idx = mask.nonzero().flatten()
        nids = current[m_idx]
        pb = p_bar_all[nids]
        dA = torch.linalg.norm(A[m_idx] - pb, dim=-1)
        dBv = torch.linalg.norm(B[m_idx] - pb, dim=-1)
        dCv = torch.linalg.norm(C[m_idx] - pb, dim=-1)
        tri_max_v = torch.maximum(dA, torch.maximum(dBv, dCv))
        radius.scatter_reduce_(0, nids, tri_max_v, reduce="amax", include_self=True)
        current[m_idx] = parent64[nids]

    max_leaf_size = int(leaf_count.max().item())
    return centroid_moment, normal_moment, area_sum, radius, max_leaf_size


def bvh_to_device(bvh: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in bvh.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def bvh_stats(bvh: Dict) -> Dict:
    left = bvh["left"].cpu()
    right = bvh["right"].cpu()
    is_leaf = bvh["is_leaf"].cpu()
    leaf_count = bvh["leaf_count"].cpu()
    n_nodes = int(bvh["n_nodes"])
    max_depth = 0
    leaf_sizes = []
    stack = [(0, 0)]
    while stack:
        nid, d = stack.pop()
        if d > max_depth:
            max_depth = d
        if bool(is_leaf[nid]):
            leaf_sizes.append(int(leaf_count[nid]))
        else:
            stack.append((int(left[nid]), d + 1))
            stack.append((int(right[nid]), d + 1))
    avg = sum(leaf_sizes) / max(1, len(leaf_sizes))
    return {
        "n_nodes": n_nodes,
        "max_depth": max_depth,
        "n_leaves": len(leaf_sizes),
        "avg_leaf_size": avg,
        "max_leaf_size": max(leaf_sizes) if leaf_sizes else 0,
    }


# ------------------------------------------------------------------ geometry helpers

def point_aabb_min_dist(P: torch.Tensor, nmin: torch.Tensor, nmax: torch.Tensor) -> torch.Tensor:
    """Minimum distance from each point to its paired AABB. All inputs (N, 3)."""
    d_lo = torch.clamp(nmin - P, min=0.0)
    d_hi = torch.clamp(P - nmax, min=0.0)
    d = torch.maximum(d_lo, d_hi)
    return torch.linalg.norm(d, dim=-1)


# ------------------------------------------------------------------ traversal

def point_triangle_distance_pair(
    P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """Per-pair unsigned point-to-triangle distance. All inputs (N, 3) -> (N,).

    Mirrors Ericson's 6-region test but without the outer broadcast — each
    point is paired with exactly one triangle. Used inside per-point BVH leaf
    evaluation where each query point has its own small set of candidates.
    """
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = (AP * AB).sum(-1)
    d2 = (AP * AC).sum(-1)
    maskA = (d1 <= 0.0) & (d2 <= 0.0)
    distA = torch.linalg.norm(AP, dim=-1)

    BP = P - B
    d3 = (BP * AB).sum(-1)
    d4 = (BP * AC).sum(-1)
    maskB = (d3 >= 0.0) & (d4 <= d3)
    distB = torch.linalg.norm(BP, dim=-1)

    CP = P - C
    d5 = (CP * AB).sum(-1)
    d6 = (CP * AC).sum(-1)
    maskC = (d6 >= 0.0) & (d5 <= d6)
    distC = torch.linalg.norm(CP, dim=-1)

    vc = d1 * d4 - d3 * d2
    maskAB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = d1 / (d1 - d3 + 1e-12)
    projAB = A + v.unsqueeze(-1) * AB
    distAB = torch.linalg.norm(P - projAB, dim=-1)

    vb = d5 * d2 - d1 * d6
    maskAC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w = d2 / (d2 - d6 + 1e-12)
    projAC = A + w.unsqueeze(-1) * AC
    distAC = torch.linalg.norm(P - projAC, dim=-1)

    va = d3 * d6 - d5 * d4
    maskBC = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-12)
    projBC = B + t_bc.unsqueeze(-1) * (C - B)
    distBC = torch.linalg.norm(P - projBC, dim=-1)

    maskFace = ~(maskA | maskB | maskC | maskAB | maskAC | maskBC)
    N = torch.linalg.cross(AB, AC, dim=-1)
    N_norm = torch.linalg.norm(N, dim=-1) + 1e-12
    N_unit = N / N_norm.unsqueeze(-1)
    distPlane = torch.abs((AP * N_unit).sum(-1))

    huge = torch.full_like(distA, 1e10)
    dists = torch.where(maskA, distA, huge)
    dists = torch.minimum(dists, torch.where(maskB, distB, huge))
    dists = torch.minimum(dists, torch.where(maskC, distC, huge))
    dists = torch.minimum(dists, torch.where(maskAB, distAB, huge))
    dists = torch.minimum(dists, torch.where(maskAC, distAC, huge))
    dists = torch.minimum(dists, torch.where(maskBC, distBC, huge))
    dists = torch.minimum(dists, torch.where(maskFace, distPlane, huge))
    return dists


def _triangle_solid_angle_vos(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, P: torch.Tensor
) -> torch.Tensor:
    """Van Oosterom–Strackee signed solid angle subtended by triangle (A,B,C) at P.

    All inputs (N, 3) -> (N,). For a closed mesh with CCW-outward triangles
    the sum Σ Ω_i / (4π) = +1 if P is interior, 0 if exterior.
    """
    a = A - P
    b = B - P
    c = C - P
    la = torch.linalg.norm(a, dim=-1)
    lb = torch.linalg.norm(b, dim=-1)
    lc = torch.linalg.norm(c, dim=-1)
    num = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
    den = (la * lb * lc
           + (a * b).sum(-1) * lc
           + (b * c).sum(-1) * la
           + (a * c).sum(-1) * lb)
    return 2.0 * torch.atan2(num, den)


def fwn_query(
    P: torch.Tensor,
    bvh: Dict,
    V: torch.Tensor,
    F: torch.Tensor,
    beta: float = 2.0,
    compact_every: int = 32,
    sync_every: "int | None" = None,
) -> torch.Tensor:
    """Barill Fast Winding Number via skip-pointer BVH traversal (GPU-friendly).

    For each query point walks the tree using single-int ``current`` state (no
    per-point stack, no 2D advanced indexing). At each node:
        - If ``||p - p_bar|| > β · radius`` (cluster is far enough): add the
          aggregated dipole contribution and skip the subtree (``miss_idx``).
        - Else if leaf: sum exact VOS solid angles over the leaf's triangles.
        - Else: descend to left child (``hit_idx``).

    **Fully dense per-iter**, same pattern as :func:`bvh_min_distance_gpu`: the
    leaf work is computed for all N points every iteration and gated via
    ``torch.where``, avoiding ``nonzero()`` (data-dependent shape, forces sync)
    and advanced-indexed assignment (``w_accum[pts_exact] = ...``) which both
    hit MPS slow paths. Termination ``(current >= 0).any()`` is only synced
    every ``sync_every`` iterations.

    Returns:
        w: (N,) float32. For a closed CCW-outward mesh: ~+1 interior, ~0 exterior.
    """
    if sync_every is not None:
        compact_every = int(sync_every)
    dev = P.device
    N = int(P.shape[0])
    pi_4 = 4.0 * math.pi

    node_min = bvh["node_min"]  # unused here but kept for cache warmth of dict
    _ = node_min
    hit_idx = bvh["hit_idx"].to(torch.int64)
    miss_idx = bvh["miss_idx"].to(torch.int64)
    is_leaf = bvh["is_leaf"]
    leaf_start = bvh["leaf_start"].to(torch.int64)
    leaf_count = bvh["leaf_count"].to(torch.int64)
    tri_perm = bvh["tri_perm"]
    centroid_moment = bvh["centroid_moment"]
    normal_moment = bvh["normal_moment"]
    area_sum = bvh["area_sum"]
    radius = bvh["radius"]
    max_leaf_size = int(bvh["max_leaf_size"])
    L = max_leaf_size

    p_bar_all = centroid_moment / area_sum.clamp_min(1e-20).unsqueeze(-1)
    beta_sq = float(beta) * float(beta)

    current = torch.zeros((N,), device=dev, dtype=torch.int64)
    w_accum = torch.zeros((N,), device=dev, dtype=torch.float32)
    offsets = torch.arange(L, device=dev, dtype=torch.int64)
    max_iters = int(bvh["n_nodes"]) + 2

    if dev.type == "cpu":
        # Sparse per-iter (see :func:`bvh_min_distance_gpu`): on CPU, nonzero /
        # scatter are cheap and skipping leaf work for non-need-exact rows wins.
        for _iter in range(max_iters):
            active = current >= 0
            if not bool(active.any()):
                break
            c = current.clamp_min(0)

            pc = p_bar_all[c]
            r = radius[c]
            as_ = area_sum[c]
            diff = pc - P
            dist2 = (diff * diff).sum(-1)
            r2 = r * r
            admissible = (dist2 > beta_sq * r2) & (as_ > 0)

            nm = normal_moment[c]
            dot_nm_diff = (nm * diff).sum(-1)
            inv_r3 = 1.0 / (dist2 * torch.sqrt(dist2.clamp_min(1e-30)) + 1e-30)
            contrib = dot_nm_diff * inv_r3 / pi_4
            add_mask = admissible & active
            w_accum = torch.where(add_mask, w_accum + contrib, w_accum)

            leaf_here = is_leaf[c] & active
            need_exact = leaf_here & (~admissible) & (as_ > 0)
            if bool(need_exact.any()):
                pts_exact = need_exact.nonzero().flatten()
                l_nodes = c[pts_exact]
                ls = leaf_start[l_nodes]
                lc_ = leaf_count[l_nodes]
                valid = offsets.unsqueeze(0) < lc_.unsqueeze(1)
                clamped = torch.where(valid, offsets.unsqueeze(0),
                                      torch.zeros_like(valid, dtype=torch.int64))
                tri_ids_pad = tri_perm[ls.unsqueeze(1) + clamped]
                K = int(pts_exact.numel())
                P_rep = P[pts_exact].unsqueeze(1).expand(-1, L, -1).reshape(-1, 3)
                tri_flat = tri_ids_pad.reshape(-1)
                Av = V[F[tri_flat, 0]]
                Bv = V[F[tri_flat, 1]]
                Cv = V[F[tri_flat, 2]]
                omega = _triangle_solid_angle_vos(Av, Bv, Cv, P_rep)
                w_per = (omega / pi_4).reshape(K, L)
                w_per = torch.where(valid, w_per, torch.zeros_like(w_per))
                w_leaf = w_per.sum(dim=1)
                w_accum[pts_exact] = w_accum[pts_exact] + w_leaf

            skip = admissible | (as_ == 0)
            next_c = torch.where(skip, miss_idx[c], hit_idx[c])
            current = torch.where(active, next_c, current)
        return w_accum

    # Dense per-iter with active-set compaction (MPS/CUDA).
    # Mirrors the structure in :func:`bvh_min_distance_gpu`: every
    # `compact_every` iters we sync once, scatter retired rows' w_accum back
    # to the global output, and shrink the active tensors. Admissibility and
    # leaf work both stop once a point's `current` hits -1.
    alive_idx = torch.arange(N, device=dev, dtype=torch.int64)
    P_a = P
    w_accum_a = w_accum
    current_a = current
    iters_done = 0
    w_out = torch.zeros((N,), device=dev, dtype=torch.float32)

    Na = N
    zeros_NL_i = torch.zeros((Na, L), device=dev, dtype=torch.int64)
    zeros_NL_f = torch.zeros((Na, L), device=dev, dtype=torch.float32)

    while alive_idx.numel() > 0 and iters_done < max_iters:
        Na = alive_idx.numel()
        if zeros_NL_i.shape[0] != Na:
            zeros_NL_i = torch.zeros((Na, L), device=dev, dtype=torch.int64)
            zeros_NL_f = torch.zeros((Na, L), device=dev, dtype=torch.float32)

        inner_budget = min(compact_every, max_iters - iters_done)
        for _ in range(inner_budget):
            active = current_a >= 0
            c = current_a.clamp_min(0)

            pc = p_bar_all[c]
            r = radius[c]
            as_ = area_sum[c]
            diff = pc - P_a
            dist2 = (diff * diff).sum(-1)
            r2 = r * r
            admissible = (dist2 > beta_sq * r2) & (as_ > 0)

            nm = normal_moment[c]
            dot_nm_diff = (nm * diff).sum(-1)
            inv_r3 = 1.0 / (dist2 * torch.sqrt(dist2.clamp_min(1e-30)) + 1e-30)
            contrib = dot_nm_diff * inv_r3 / pi_4
            add_mask = admissible & active
            w_accum_a = torch.where(add_mask, w_accum_a + contrib, w_accum_a)

            leaf_here = is_leaf[c] & active
            need_exact = leaf_here & (~admissible) & (as_ > 0)

            ls = leaf_start[c]
            lc_ = leaf_count[c]
            valid_slot = (offsets.unsqueeze(0) < lc_.unsqueeze(1)) & need_exact.unsqueeze(1)
            clamped = torch.where(valid_slot, offsets.unsqueeze(0), zeros_NL_i)
            tri_ids_pad = tri_perm[ls.unsqueeze(1) + clamped]
            P_rep = P_a.unsqueeze(1).expand(-1, L, -1).reshape(-1, 3)
            tri_flat = tri_ids_pad.reshape(-1)
            Av = V[F[tri_flat, 0]]
            Bv = V[F[tri_flat, 1]]
            Cv = V[F[tri_flat, 2]]
            omega = _triangle_solid_angle_vos(Av, Bv, Cv, P_rep)
            w_per = (omega / pi_4).reshape(Na, L)
            w_per = torch.where(valid_slot, w_per, zeros_NL_f)
            w_accum_a = w_accum_a + w_per.sum(dim=1)

            skip = admissible | (as_ == 0)
            next_c = torch.where(skip, miss_idx[c], hit_idx[c])
            current_a = torch.where(active, next_c, current_a)
            iters_done += 1

        # Compaction: sync once per `compact_every` iters, drop retired rows.
        keep = current_a >= 0
        if bool(keep.all()):
            continue
        retired_mask = ~keep
        w_out[alive_idx[retired_mask]] = w_accum_a[retired_mask]
        alive_idx = alive_idx[keep]
        if alive_idx.numel() == 0:
            break
        P_a = P_a[keep].contiguous()
        w_accum_a = w_accum_a[keep].contiguous()
        current_a = current_a[keep].contiguous()

    if alive_idx.numel() > 0:
        w_out[alive_idx] = w_accum_a

    return w_out


def bvh_min_distance_gpu(
    P: torch.Tensor,
    bvh: Dict,
    V: torch.Tensor,
    F: torch.Tensor,
    max_reasonable_dist: float,
    initial_upper: "torch.Tensor | None" = None,
    early_out_threshold: "float | None" = None,
    compact_every: int = 32,
    sync_every: "int | None" = None,
) -> torch.Tensor:
    """GPU skip-pointer BVH unsigned min-distance query.

    Same traversal skeleton as :func:`fwn_query`: one int64 ``current`` per
    point, no per-point stack. At each node the AABB lower-bound distance is
    compared to ``d_best``; if it can't improve, the subtree is skipped via
    ``miss_idx``. Leaves always compute exact per-triangle distances and then
    advance via ``hit_idx`` (== ``miss_idx`` for leaves).

    **Fully dense per-iter with active-set compaction (MPS/CUDA).** Every
    outer iteration does leaf-distance work for *all* currently-active points,
    masking the update via ``torch.where`` rather than scatter-writing into a
    sparse active set. This avoids the two MPS-hostile patterns that otherwise
    dominate: ``nonzero()`` (data-dependent shape, forces sync) and
    advanced-indexed assignment (``d_best[pts_leaf] = ...``). The net work is
    higher per iteration but throughput on MPS is ~10–30× higher than the
    sparse version.

    Every ``compact_every`` iterations the traversal syncs once, scatters the
    retired rows' ``d_best`` back into the global output, and drops them from
    the active tensors — so later iterations operate on the (usually much
    smaller) surviving working set. This reclaims the straggler tax that the
    dense form otherwise pays: on early-out classify_band most points retire
    within the first few tens of iterations, so after 2–3 compactions the
    working set is ~5–10% of N.

    Args:
        P: (N, 3) query points.
        bvh: dict from :func:`build_bvh_torch`, tensors on ``P.device``.
        V, F: mesh tensors on ``P.device``.
        max_reasonable_dist: initial d_best when ``initial_upper`` is None.
        initial_upper: optional (N,) tensor of pre-computed upper bounds on
            |d|. Seeds ``d_best`` — tighter initial bounds cut iterations.
        early_out_threshold: if set, a point is retired as soon as its
            ``d_best`` drops at or below this value. Useful for band
            classification where an upper bound ≤ threshold is sufficient;
            the returned ``d_best`` for retired points is a valid upper
            bound, not necessarily the exact minimum.
        compact_every: how often (in iterations) the dense MPS/CUDA path
            pauses to sync, scatter retired rows back to the global output,
            and shrink the active tensors. Larger values = fewer syncs but
            more full-N stragglers per sync; 32 is a good default.
        sync_every: deprecated alias for ``compact_every``. If both are
            provided, ``compact_every`` wins.
    """
    if sync_every is not None:
        compact_every = int(sync_every)
    dev = P.device
    N = int(P.shape[0])

    node_min = bvh["node_min"]
    node_max = bvh["node_max"]
    hit_idx = bvh["hit_idx"].to(torch.int64)
    miss_idx = bvh["miss_idx"].to(torch.int64)
    is_leaf = bvh["is_leaf"]
    leaf_start = bvh["leaf_start"].to(torch.int64)
    leaf_count = bvh["leaf_count"].to(torch.int64)
    tri_perm = bvh["tri_perm"]
    max_leaf_size = int(bvh["max_leaf_size"])
    L = max_leaf_size

    if initial_upper is None:
        d_best = torch.full((N,), float(max_reasonable_dist), device=dev, dtype=torch.float32)
    else:
        d_best = initial_upper.to(dev, dtype=torch.float32).clone()

    current = torch.zeros((N,), device=dev, dtype=torch.int64)
    offsets = torch.arange(L, device=dev, dtype=torch.int64)
    retired = torch.full((N,), -1, device=dev, dtype=torch.int64)
    max_iters = int(bvh["n_nodes"]) + 2

    if dev.type == "cpu":
        # Sparse per-iter: nonzero() / advanced-indexed scatter are cheap on
        # CPU (MKL-backed, no kernel launch), and retired points contribute no
        # leaf work. Much faster than the dense form on CPU at large N where
        # most points retire quickly via the early-out.
        for _iter in range(max_iters):
            active = current >= 0
            if not bool(active.any()):
                break
            c = current.clamp_min(0)
            nmin = node_min[c]
            nmax = node_max[c]
            d_box = point_aabb_min_dist(P, nmin, nmax)
            prune = (d_box >= d_best) & active
            leaf_here = is_leaf[c] & active
            do_leaf = leaf_here & (~prune)
            if bool(do_leaf.any()):
                pts_leaf = do_leaf.nonzero().flatten()
                c_leaf = c[pts_leaf]
                ls = leaf_start[c_leaf]
                lc_ = leaf_count[c_leaf]
                valid = offsets.unsqueeze(0) < lc_.unsqueeze(1)
                clamped = torch.where(valid, offsets.unsqueeze(0),
                                      torch.zeros_like(valid, dtype=torch.int64))
                tri_ids_pad = tri_perm[ls.unsqueeze(1) + clamped]
                K = int(pts_leaf.numel())
                P_rep = P[pts_leaf].unsqueeze(1).expand(-1, L, -1).reshape(-1, 3)
                tri_flat = tri_ids_pad.reshape(-1)
                Av = V[F[tri_flat, 0]]
                Bv = V[F[tri_flat, 1]]
                Cv = V[F[tri_flat, 2]]
                d_pair = point_triangle_distance_pair(P_rep, Av, Bv, Cv).reshape(K, L)
                huge = torch.full_like(d_pair, float(max_reasonable_dist))
                d_pair = torch.where(valid, d_pair, huge)
                d_leaf_min = d_pair.amin(dim=1)
                d_best[pts_leaf] = torch.minimum(d_best[pts_leaf], d_leaf_min)
            next_c = torch.where(prune, miss_idx[c], hit_idx[c])
            next_c = torch.where(active, next_c, current)
            if early_out_threshold is not None:
                done = d_best < float(early_out_threshold)
                next_c = torch.where(done, retired, next_c)
            current = next_c
        return d_best

    # Dense per-iter with active-set compaction (MPS/CUDA).
    # Every `compact_every` iters we sync once, scatter retired rows back to
    # the global d_best, and drop them from the active working set. Between
    # compactions the inner loop is identical to the old dense form.
    alive_idx = torch.arange(N, device=dev, dtype=torch.int64)
    P_a = P
    d_best_a = d_best
    current_a = current
    iters_done = 0

    Na = N
    zeros_NL = torch.zeros((Na, L), device=dev, dtype=torch.int64)
    huge_NL = torch.full((Na, L), float(max_reasonable_dist), device=dev, dtype=torch.float32)

    while alive_idx.numel() > 0 and iters_done < max_iters:
        Na = alive_idx.numel()
        if zeros_NL.shape[0] != Na:
            zeros_NL = torch.zeros((Na, L), device=dev, dtype=torch.int64)
            huge_NL = torch.full((Na, L), float(max_reasonable_dist), device=dev, dtype=torch.float32)

        inner_budget = min(compact_every, max_iters - iters_done)
        for _ in range(inner_budget):
            active = current_a >= 0
            c = current_a.clamp_min(0)

            nmin = node_min[c]
            nmax = node_max[c]
            d_box = point_aabb_min_dist(P_a, nmin, nmax)
            prune = (d_box >= d_best_a) & active

            leaf_here = is_leaf[c] & active
            do_leaf = leaf_here & (~prune)

            ls = leaf_start[c]
            lc_ = leaf_count[c]
            valid_slot = (offsets.unsqueeze(0) < lc_.unsqueeze(1)) & do_leaf.unsqueeze(1)
            clamped = torch.where(valid_slot, offsets.unsqueeze(0), zeros_NL)
            tri_ids_pad = tri_perm[ls.unsqueeze(1) + clamped]
            P_rep = P_a.unsqueeze(1).expand(-1, L, -1).reshape(-1, 3)
            tri_flat = tri_ids_pad.reshape(-1)
            Av = V[F[tri_flat, 0]]
            Bv = V[F[tri_flat, 1]]
            Cv = V[F[tri_flat, 2]]
            d_pair = point_triangle_distance_pair(P_rep, Av, Bv, Cv).reshape(Na, L)
            d_pair = torch.where(valid_slot, d_pair, huge_NL)
            d_leaf_min = d_pair.amin(dim=1)
            d_best_a = torch.where(do_leaf, torch.minimum(d_best_a, d_leaf_min), d_best_a)

            next_c = torch.where(prune, miss_idx[c], hit_idx[c])
            next_c = torch.where(active, next_c, current_a)
            if early_out_threshold is not None:
                done = d_best_a < float(early_out_threshold)
                next_c = torch.where(done, torch.full_like(next_c, -1), next_c)
            current_a = next_c
            iters_done += 1

        # Compaction: sync once per `compact_every` iters, drop retired rows.
        keep = current_a >= 0
        if bool(keep.all()):
            continue
        retired_mask = ~keep
        d_best[alive_idx[retired_mask]] = d_best_a[retired_mask]
        alive_idx = alive_idx[keep]
        if alive_idx.numel() == 0:
            break
        P_a = P_a[keep].contiguous()
        d_best_a = d_best_a[keep].contiguous()
        current_a = current_a[keep].contiguous()

    # Scatter whatever remains (e.g. hit max_iters without retiring).
    if alive_idx.numel() > 0:
        d_best[alive_idx] = d_best_a

    return d_best


def bvh_query_distances(
    P: torch.Tensor,
    bvh: Dict,
    V: torch.Tensor,
    F: torch.Tensor,
    max_reasonable_dist: float,
    max_depth: int = 64,
) -> torch.Tensor:
    """Per-point branch-and-bound unsigned distance query (vectorized).

    Each of the N points keeps a private stack of BVH node ids, pops the top,
    tests the node AABB against its running best distance, and pushes the two
    children (farther first so nearer is popped first — near-first order
    tightens ``d_best`` fast and maximizes pruning). Returns once every
    per-point stack is empty.

    Intended to run on CPU: all 2D indexing into ``stack[pts, top]`` and mask
    selects are fast on CPU, while on MPS they hit slow paths that make this
    kernel much slower than the flat AABB prune.

    Args:
        P: (N, 3) query points. Device determines where the traversal runs.
        bvh: dict from :func:`build_bvh_torch`, tensors on ``P.device``.
        V, F: vertices / faces on ``P.device``, used only at leaves.
        max_reasonable_dist: initial upper bound on ``d_best`` (serves as the
            "no candidate found" clamp for points that never hit a leaf).
        max_depth: stack depth per point. 64 covers any realistic tree.

    Returns:
        (N,) float32 tensor of unsigned minimum distances.
    """
    dev = P.device
    N = P.shape[0]
    node_min = bvh["node_min"]
    node_max = bvh["node_max"]
    left = bvh["left"].to(torch.int64)
    right = bvh["right"].to(torch.int64)
    is_leaf = bvh["is_leaf"]
    leaf_start = bvh["leaf_start"].to(torch.int64)
    leaf_count = bvh["leaf_count"].to(torch.int64)
    tri_perm = bvh["tri_perm"]
    leaf_size = int(bvh["leaf_size"])

    stack = torch.zeros((N, max_depth), device=dev, dtype=torch.int64)
    stack_top = torch.zeros((N,), device=dev, dtype=torch.int64)
    stack[:, 0] = 0  # root on every point's stack
    d_best = torch.full((N,), float(max_reasonable_dist), device=dev, dtype=torch.float32)

    point_idx_all = torch.arange(N, device=dev, dtype=torch.int64)
    offsets = torch.arange(leaf_size, device=dev, dtype=torch.int64)

    while True:
        active_mask = stack_top >= 0
        if not bool(active_mask.any()):
            break
        active_idx = point_idx_all[active_mask]
        top_pos = stack_top[active_idx]
        node_ids = stack[active_idx, top_pos]
        stack_top[active_idx] = top_pos - 1

        P_act = P[active_idx]
        d_box = point_aabb_min_dist(P_act, node_min[node_ids], node_max[node_ids])
        keep = d_box < d_best[active_idx]
        if not bool(keep.any()):
            continue

        k_pts = active_idx[keep]
        k_nodes = node_ids[keep]

        leaf_mask = is_leaf[k_nodes]
        if bool(leaf_mask.any()):
            lp = k_pts[leaf_mask]
            ln = k_nodes[leaf_mask]
            ls = leaf_start[ln]
            lc = leaf_count[ln]
            valid = offsets.unsqueeze(0) < lc.unsqueeze(1)
            clamped = torch.where(valid, offsets.unsqueeze(0),
                                  torch.zeros_like(valid, dtype=torch.int64))
            tri_ids_pad = tri_perm[ls.unsqueeze(1) + clamped]

            P_rep = P[lp].unsqueeze(1).expand(-1, leaf_size, -1).reshape(-1, 3)
            tri_flat = tri_ids_pad.reshape(-1)
            A_flat = V[F[tri_flat, 0]]
            B_flat = V[F[tri_flat, 1]]
            C_flat = V[F[tri_flat, 2]]

            d_pair = point_triangle_distance_pair(P_rep, A_flat, B_flat, C_flat)
            d_pair = d_pair.reshape(-1, leaf_size)
            huge = torch.full_like(d_pair, float(max_reasonable_dist))
            d_pair = torch.where(valid, d_pair, huge)
            d_min_per_leaf = d_pair.amin(dim=1)

            d_best[lp] = torch.minimum(d_best[lp], d_min_per_leaf)

        int_mask = ~leaf_mask
        if bool(int_mask.any()):
            ip = k_pts[int_mask]
            in_nodes = k_nodes[int_mask]
            lc_nodes = left[in_nodes]
            rc_nodes = right[in_nodes]
            P_int = P[ip]
            d_l = point_aabb_min_dist(P_int, node_min[lc_nodes], node_max[lc_nodes])
            d_r = point_aabb_min_dist(P_int, node_min[rc_nodes], node_max[rc_nodes])
            near_is_left = d_l <= d_r
            far_child = torch.where(near_is_left, rc_nodes, lc_nodes)
            near_child = torch.where(near_is_left, lc_nodes, rc_nodes)

            cur_top = stack_top[ip]
            can_push_far = cur_top + 1 < max_depth
            can_push_near = cur_top + 2 < max_depth

            push_far_idx = ip[can_push_far]
            if push_far_idx.numel() > 0:
                new_top = stack_top[push_far_idx] + 1
                stack[push_far_idx, new_top] = far_child[can_push_far]
                stack_top[push_far_idx] = new_top

            push_near_idx = ip[can_push_near]
            if push_near_idx.numel() > 0:
                new_top = stack_top[push_near_idx] + 1
                stack[push_near_idx, new_top] = near_child[can_push_near]
                stack_top[push_near_idx] = new_top

    return d_best
