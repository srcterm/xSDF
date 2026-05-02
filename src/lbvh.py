"""Linear Bounding Volume Hierarchy (Karras 2012), pure-PyTorch, device-parallel.

Build pipeline (all work on V.device):
  1. 30-bit Morton codes over per-triangle centroids; ascending torch.sort.
  2. Karras et al.(2012) longest-common-prefix construction over the L sorted
     leaves: per-node range determination + split index, all branch-free.
  3. Bottom-up AABB + Barill et al. (2018) dipole aggregation
     (area_w, c_area, n_area) via the ready-counter pattern.
  4. Barill et al. (2018) r_ball: farthest AABB-corner distance from the area-weighted
     centroid (used as the tight far-field bound in FWN traversal).
  5. Skip pointers via pointer-jumping. the production query kernels in src/torch-meshSDF.py traverse via
     chunked BFS.

Node indexing (unified, size 2L-1 where L = ⌈Nt/leaf_size⌉):
    [0, L-1)    : internal nodes, root = 0
    [L-1, 2L-1) : leaf nodes; leaf k lives at index (L - 1 + k).

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


# =============================================================================
# LBVH dataclass
# =============================================================================

@dataclass
class LBVH:
    # Mesh (not reordered; callers gather via tri_order)
    V: torch.Tensor                # (Nv, 3) float32
    F: torch.Tensor                # (Nt, 3) int64

    # Morton-sort permutation: sorted triangle k is F[tri_order[k]]
    tri_order: torch.Tensor        # (Nt,) int64

    # Tree structure (unified node indexing, size 2L-1)
    aabb_min: torch.Tensor         # (2L-1, 3) float32
    aabb_max: torch.Tensor         # (2L-1, 3) float32
    left:   torch.Tensor           # (2L-1,) int32 — -1 for leaves
    right:  torch.Tensor           # (2L-1,) int32 — -1 for leaves
    parent: torch.Tensor           # (2L-1,) int32 — -1 for root
    skip:   torch.Tensor           # (2L-1,) int32 — next DFS node after subtree, -1 at end

    # Leaf triangle ranges: leaf k holds sorted triangles [leaf_tri_beg[k]:leaf_tri_end[k])
    leaf_tri_beg: torch.Tensor     # (L,) int32
    leaf_tri_end: torch.Tensor     # (L,) int32

    # Barill dipole fields
    area_w: Optional[torch.Tensor] = None   # (2L-1,) float32
    c_area: Optional[torch.Tensor] = None   # (2L-1, 3) float32 — Σ area · centroid
    n_area: Optional[torch.Tensor] = None   # (2L-1, 3) float32 — Σ area · unit_normal
    r_ball: Optional[torch.Tensor] = None   # (2L-1,) float32

    # Meta
    leaf_size: int = 4
    num_leaves: int = 0
    num_nodes: int = 0
    device: Optional[torch.device] = None

    def root(self) -> int:
        return 0

    def is_leaf_node(self, n: int) -> bool:
        return n >= self.num_leaves - 1


# =============================================================================
# Morton codes (30-bit, packed into int32)
# =============================================================================

def _expand_bits_10(v: torch.Tensor) -> torch.Tensor:
    """Spread the low 10 bits of v so each occupies every 3rd bit position.

    Operates on int32; safe because the final pattern fits in 30 bits.
    """
    v = v & 0x000003FF
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8))  & 0x0300F00F
    v = (v | (v << 4))  & 0x030C30C3
    v = (v | (v << 2))  & 0x09249249
    return v


def _morton_codes_30(centroids: torch.Tensor,
                     scene_min: torch.Tensor,
                     scene_max: torch.Tensor) -> torch.Tensor:
    """30-bit interleaved Morton codes for (Nt, 3) centroids.

    Returns int32 codes in [0, 2^30). Extent is clamped to avoid div-by-zero
    on degenerate (coplanar) meshes; ties are broken downstream by index XOR
    in the Karras δ function.
    """
    extent = (scene_max - scene_min).clamp_min(1e-20)
    u = ((centroids - scene_min) / extent).clamp(0.0, 1.0 - 2 ** -20)
    q = (u * 1024.0).to(torch.int32).clamp_(0, 1023)
    mx = _expand_bits_10(q[:, 0])
    my = _expand_bits_10(q[:, 1])
    mz = _expand_bits_10(q[:, 2])
    return (mx << 2) | (my << 1) | mz


# =============================================================================
# Count-leading-zeros (branch-free, vectorized, int32)
# =============================================================================

def _clz32(x: torch.Tensor) -> torch.Tensor:
    """Count leading zeros treating x as unsigned 32-bit.

    Returns 32 where x==0. Works on int32 tensors; uses only shifts/masks so
    it runs on CPU/CUDA/MPS without float conversions.
    """
    x = x.to(torch.int32)
    # Force unsigned view via mask
    x = x & 0x7FFFFFFF | ((x >> 31) & 1) << 31  # no-op but keeps int32 semantics
    n = torch.zeros_like(x)

    m = (x & 0xFFFF0000) == 0
    n = n + m.to(torch.int32) * 16
    x = torch.where(m, x << 16, x)

    m = (x & 0xFF000000) == 0
    n = n + m.to(torch.int32) * 8
    x = torch.where(m, x << 8, x)

    m = (x & 0xF0000000) == 0
    n = n + m.to(torch.int32) * 4
    x = torch.where(m, x << 4, x)

    m = (x & 0xC0000000) == 0
    n = n + m.to(torch.int32) * 2
    x = torch.where(m, x << 2, x)

    m = (x & 0x80000000) == 0
    n = n + m.to(torch.int32)
    # If input was 0 all masks fire ⇒ n = 32. Correct.
    return n


# =============================================================================
# Karras δ (common-prefix length with index tie-break)
# =============================================================================

def _delta(i_arr: torch.Tensor,
           j_arr: torch.Tensor,
           morton: torch.Tensor,
           L: int) -> torch.Tensor:
    """Vectorized common-prefix length. Returns -1 where j is out of [0, L).

    When morton[i] == morton[j], extends with clz of (i XOR j) so equal-code
    ties are still strictly ordered (required for degenerate meshes).
    """
    valid = (j_arr >= 0) & (j_arr < L)
    j_safe = j_arr.clamp(0, max(L - 1, 0))
    m_i = morton[i_arr]
    m_j = morton[j_safe]
    code_xor = (m_i ^ m_j).to(torch.int32)
    # Tie-break with index XOR so equal codes still yield a strict ordering.
    idx_xor = (i_arr ^ j_safe).to(torch.int32)
    clz_code = _clz32(code_xor)
    clz_idx = _clz32(idx_xor)
    prefix = torch.where(code_xor == 0, 32 + clz_idx, clz_code)
    return torch.where(valid, prefix, torch.full_like(prefix, -1))


# =============================================================================
# Karras internal-node construction (ranges, splits, children, parents)
# =============================================================================

def _karras_build_internals(morton: torch.Tensor, L: int):
    """Build internal-node structure for L sorted leaves.

    Returns:
        left_child, right_child, parent : (2L-1,) int32 each
    """
    device = morton.device
    K = L - 1  # number of internal nodes

    # Node arrays (unified indexing).
    left   = torch.full((2 * L - 1,), -1, dtype=torch.int32, device=device)
    right  = torch.full((2 * L - 1,), -1, dtype=torch.int32, device=device)
    parent = torch.full((2 * L - 1,), -1, dtype=torch.int32, device=device)

    if K <= 0:
        # Single leaf: tree is the leaf itself. Empty internal arrays.
        return left, right, parent

    i_arr = torch.arange(K, dtype=torch.int32, device=device)

    # ---- Direction d ∈ {+1, -1}: range grows toward side with larger δ ----
    d_plus  = _delta(i_arr, i_arr + 1, morton, L)   # i+1 always in [1, K], in range for i<K
    d_minus = _delta(i_arr, i_arr - 1, morton, L)   # -1 at i==0
    d = torch.where(d_plus > d_minus,
                    torch.ones_like(i_arr),
                    torch.full_like(i_arr, -1))

    # Min common prefix: δ(i, i - d)
    delta_min = torch.where(d > 0, d_minus, d_plus)

    # ---- Binary search for range length l (unrolled, zero host syncs) ----
    # Schedule: t_k = len_max >> (k+1), k = 0..log2(len_max)-1.
    len_max_exp = max(1, int(math.ceil(math.log2(max(L, 2)))) + 2)
    len_max = 1 << len_max_exp

    l = torch.zeros_like(i_arr)
    for k in range(len_max_exp):
        t = len_max >> (k + 1)
        if t <= 0:
            break
        j_try = i_arr + (l + t) * d
        cond = _delta(i_arr, j_try, morton, L) > delta_min
        l = torch.where(cond, l + t, l)

    # End of range.
    j = i_arr + l * d
    delta_node = _delta(i_arr, j, morton, L)

    # ---- Binary search for split s ∈ [0, l) ----
    s = torch.zeros_like(i_arr)
    for k in range(len_max_exp):
        t = len_max >> (k + 1)
        if t <= 0:
            break
        cond_range = (s + t) <= l
        j_try = i_arr + (s + t) * d
        cond_delta = _delta(i_arr, j_try, morton, L) > delta_node
        cond = cond_range & cond_delta
        s = torch.where(cond, s + t, s)

    # Karras split index γ = i + s*d + min(d, 0).
    gamma = i_arr + s * d + torch.clamp(d, max=0)

    first = torch.minimum(i_arr, j)
    last  = torch.maximum(i_arr, j)

    # ---- Resolve child node indices (leaf vs internal) ----
    # Leaves live at indices [L-1, 2L-1); leaf k ⇒ node (L-1 + k).
    left_is_leaf  = (first == gamma)
    right_is_leaf = ((gamma + 1) == last)
    left_child  = torch.where(left_is_leaf,  (L - 1) + gamma,       gamma)
    right_child = torch.where(right_is_leaf, (L - 1) + (gamma + 1), gamma + 1)

    left[:K]  = left_child
    right[:K] = right_child

    # Parent pointers via scatter. Safe: child indices are unique per parent.
    parent.scatter_(0, left_child.to(torch.long),  i_arr)
    parent.scatter_(0, right_child.to(torch.long), i_arr)

    return left, right, parent


# =============================================================================
# Leaf AABBs + bottom-up aggregation (ready-counter pattern)
# =============================================================================

def _scatter_amin(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    """scatter_reduce 'amin' along dim 0 with a -amax fallback on backends
    that don't support amin (older MPS builds)."""
    try:
        dst.scatter_reduce_(0, index, src, 'amin', include_self=True)
    except (RuntimeError, NotImplementedError):
        tmp = -dst
        tmp.scatter_reduce_(0, index, -src, 'amax', include_self=True)
        dst.copy_(-tmp)


def _scatter_amax(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> None:
    dst.scatter_reduce_(0, index, src, 'amax', include_self=True)


def _aggregate_bottom_up(L: int,
                          parent: torch.Tensor,
                          leaf_aabb_min: torch.Tensor,
                          leaf_aabb_max: torch.Tensor,
                          leaf_area_w: Optional[torch.Tensor] = None,
                          leaf_c_area: Optional[torch.Tensor] = None,
                          leaf_n_area: Optional[torch.Tensor] = None):
    """Bottom-up aggregation via the ready-counter pattern.

    Each internal node becomes ready once both children have propagated; we
    then fold its subtree's AABB and (if provided) dipole fields into it.
    Outer-iteration count is bounded by Karras-tree depth (≤ ~30 for 30-bit
    Morton codes — not log2(L), since equal-Morton ties extend the prefix);
    the loop breaks early once no node is still propagating, with a
    64-iteration safety cap.

    Returns (aabb_min, aabb_max, area_w, c_area, n_area). Dipole outputs are
    None iff the corresponding inputs are None.
    """
    device = parent.device
    N = 2 * L - 1

    aabb_min = torch.full((N, 3), float('inf'),  dtype=torch.float32, device=device)
    aabb_max = torch.full((N, 3), float('-inf'), dtype=torch.float32, device=device)
    aabb_min[L - 1:] = leaf_aabb_min
    aabb_max[L - 1:] = leaf_aabb_max

    have_dipole = leaf_area_w is not None
    if have_dipole:
        area_w = torch.zeros(N, dtype=torch.float32, device=device)
        c_area = torch.zeros((N, 3), dtype=torch.float32, device=device)
        n_area = torch.zeros((N, 3), dtype=torch.float32, device=device)
        area_w[L - 1:] = leaf_area_w
        c_area[L - 1:] = leaf_c_area
        n_area[L - 1:] = leaf_n_area
    else:
        area_w = c_area = n_area = None

    if L <= 1:
        return aabb_min, aabb_max, area_w, c_area, n_area

    ready = torch.zeros(N, dtype=torch.int32, device=device)
    done = torch.zeros(N, dtype=torch.bool, device=device)
    done[L - 1:] = True  # leaves propagate upward first
    propagated = torch.zeros(N, dtype=torch.bool, device=device)

    # Karras-tree depth is bounded by the Morton prefix, which
    # for 30-bit codes can approach 30 — not log2(L). Loop and
    # rely on the early break when no nodes are activating.
    max_iters = 64
    for _ in range(max_iters):
        activating = done & ~propagated
        src_idx = activating.nonzero(as_tuple=True)[0]
        if src_idx.numel() == 0:
            break

        par = parent.index_select(0, src_idx)            # (S,) int32
        valid = par >= 0
        if not bool(valid.any()):
            propagated |= activating
            continue
        par_valid = par[valid].to(torch.long)
        src_valid = src_idx[valid]

        par_idx3 = par_valid.unsqueeze(1).expand(-1, 3)
        _scatter_amin(aabb_min, par_idx3, aabb_min.index_select(0, src_valid))
        _scatter_amax(aabb_max, par_idx3, aabb_max.index_select(0, src_valid))

        if have_dipole:
            area_w.scatter_reduce_(
                0, par_valid, area_w.index_select(0, src_valid),
                'sum', include_self=True)
            c_area.scatter_reduce_(
                0, par_idx3, c_area.index_select(0, src_valid),
                'sum', include_self=True)
            n_area.scatter_reduce_(
                0, par_idx3, n_area.index_select(0, src_valid),
                'sum', include_self=True)

        ones = torch.ones_like(par_valid, dtype=torch.int32)
        ready.scatter_reduce_(0, par_valid, ones, 'sum', include_self=True)

        newly_done = (ready == 2) & ~done
        done |= newly_done
        propagated |= activating

    return aabb_min, aabb_max, area_w, c_area, n_area


# =============================================================================
# Skip pointers via pointer-jumping (no host DFS)
# =============================================================================

def _build_skip_pointers(left: torch.Tensor,
                         right: torch.Tensor,
                         parent: torch.Tensor,
                         L: int) -> torch.Tensor:
    """DFS skip pointer: skip[n] is the next node to visit after n's subtree
    finishes. -1 marks end of traversal.

    Derivation: skip[n] = right_sibling(n) walking up the chain until n is
    a left child; returns -1 if n is always a right child up to the root.
    """
    device = left.device
    N = 2 * L - 1

    if N == 1:
        return torch.full((1,), -1, dtype=torch.int32, device=device)

    # is_left_child[n] == True ⇔ n is the left child of parent[n]
    all_nodes = torch.arange(N, device=device, dtype=torch.int32)
    par_safe = parent.clamp(min=0).to(torch.long)
    left_of_par = left.index_select(0, par_safe)
    is_left_child = (parent >= 0) & (left_of_par == all_nodes)

    # Right sibling of n (if n is left child): right[parent[n]].
    right_of_par = right.index_select(0, par_safe)
    sibling = torch.where(is_left_child,
                          right_of_par,
                          torch.full_like(right_of_par, -1))

    # Pointer-jump. Start cur at the immediate right sibling; if absent (n is
    # a right child), climb through ancestors until a left-child ancestor is
    # found, then its right sibling is the answer.
    cur   = sibling.clone()
    climb = parent.clone()
    max_iters = int(math.ceil(math.log2(max(N, 2)))) + 2
    for _ in range(max_iters):
        need = (cur == -1) & (climb != -1)
        if not bool(need.any()):
            break
        climb_safe = climb.clamp(min=0).to(torch.long)
        new_cur   = sibling.index_select(0, climb_safe)
        new_climb = parent.index_select(0, climb_safe)
        cur   = torch.where(need, new_cur, cur)
        climb = torch.where(need, new_climb, climb)

    return cur


# =============================================================================
# API
# =============================================================================

def build_lbvh(V: torch.Tensor,
               F: torch.Tensor,
               *,
               leaf_size: int = 4) -> LBVH:
    """Build an LBVH for the triangle mesh (V, F).

    All work happens on V.device. The only host↔device sync is the
    early-exit check per aggregation iteration (≤ Karras-tree depth, ~30
    for 30-bit Morton codes).

    Args:
        V: (Nv, 3) float32 vertex positions.
        F: (Nt, 3) int64 triangle indices.
        leaf_size: Max triangles per leaf (default 4). Production callers
                   in src/torch-meshSDF.py pass leaf_size=1 explicitly;
                   larger values reduce tree depth at the cost of more
                   per-leaf brute-force work.

    Returns:
        LBVH populated with: tree topology (left/right/parent/skip),
        per-node AABBs (aabb_min/max), per-leaf triangle ranges
        (leaf_tri_beg/end), and Barill dipole fields
        (area_w, c_area, n_area, r_ball) for FWN.
    """
    assert V.dtype == torch.float32, f"V must be float32, got {V.dtype}"
    assert V.dim() == 2 and V.shape[1] == 3, f"V must be (Nv, 3), got {tuple(V.shape)}"
    assert F.dim() == 2 and F.shape[1] == 3, f"F must be (Nt, 3), got {tuple(F.shape)}"
    assert leaf_size >= 1

    device = V.device
    Nt = int(F.shape[0])
    assert Nt >= 1, "build_lbvh needs at least one triangle"

    F_long = F.to(torch.int64)

    # ---- Per-triangle AABBs (in original triangle order) ----
    A = V.index_select(0, F_long[:, 0])
    B = V.index_select(0, F_long[:, 1])
    C = V.index_select(0, F_long[:, 2])
    tri_min = torch.minimum(torch.minimum(A, B), C)         # (Nt, 3)
    tri_max = torch.maximum(torch.maximum(A, B), C)         # (Nt, 3)
    centroid = (A + B + C) * (1.0 / 3.0)                     # (Nt, 3)

    # ---- Scene bounds over triangle AABBs (tighter than centroid-only) ----
    scene_min = tri_min.amin(dim=0)
    scene_max = tri_max.amax(dim=0)

    # ---- Morton codes → sort ----
    morton = _morton_codes_30(centroid, scene_min, scene_max)
    morton_sorted, tri_order = torch.sort(morton)            # tri_order: (Nt,) int64

    # Reorder per-triangle AABBs to sorted order.
    tri_min_s = tri_min.index_select(0, tri_order)
    tri_max_s = tri_max.index_select(0, tri_order)

    # ---- Leaves: pack leaf_size triangles each, compute per-leaf AABBs ----
    L = (Nt + leaf_size - 1) // leaf_size
    leaf_tri_beg = torch.arange(0, L * leaf_size, leaf_size,
                                dtype=torch.int32, device=device)
    leaf_tri_end = torch.clamp(leaf_tri_beg + leaf_size, max=Nt).to(torch.int32)

    if leaf_size == 1 or L == Nt:
        leaf_aabb_min = tri_min_s
        leaf_aabb_max = tri_max_s
    else:
        leaf_idx_per_tri = (torch.arange(Nt, device=device) // leaf_size).to(torch.long)
        idx3 = leaf_idx_per_tri.unsqueeze(1).expand(-1, 3)
        leaf_aabb_min = torch.full((L, 3), float('inf'),
                                   dtype=torch.float32, device=device)
        leaf_aabb_max = torch.full((L, 3), float('-inf'),
                                   dtype=torch.float32, device=device)
        _scatter_amin(leaf_aabb_min, idx3, tri_min_s)
        _scatter_amax(leaf_aabb_max, idx3, tri_max_s)

    # ---- Leaf Morton codes: min over triangles assigned to each leaf ----
    # Karras needs one sorted code per "leaf" (since tris within a leaf are
    # already consecutive in Morton order, the first is representative).
    leaf_morton = morton_sorted.index_select(0, leaf_tri_beg.to(torch.long))

    # ---- Karras internal nodes ----
    left, right, parent = _karras_build_internals(leaf_morton, L)

    # ---- Per-triangle dipole quantities (Barill 2018) ----
    # cross = (B - A) x (C - A) has magnitude 2*area and direction = 2*area*unit_normal.
    # So area_i = 0.5*||cross_i|| and area_i*unit_normal_i = 0.5*cross_i.
    AB = B - A
    AC = C - A
    cross = torch.linalg.cross(AB, AC, dim=-1)                  # (Nt, 3)
    tri_area = 0.5 * torch.linalg.norm(cross, dim=-1)           # (Nt,)
    tri_an = 0.5 * cross                                        # (Nt, 3) = area * unit_normal
    tri_ac = tri_area.unsqueeze(1) * centroid                   # (Nt, 3) = area * centroid

    # Sort to match Morton-sorted triangle order.
    tri_area_s = tri_area.index_select(0, tri_order)
    tri_an_s   = tri_an.index_select(0, tri_order)
    tri_ac_s   = tri_ac.index_select(0, tri_order)

    # ---- Per-leaf dipole aggregation (triangles in contiguous blocks) ----
    if leaf_size == 1 or L == Nt:
        leaf_area_w = tri_area_s
        leaf_c_area = tri_ac_s
        leaf_n_area = tri_an_s
    else:
        leaf_idx_per_tri = (torch.arange(Nt, device=device) // leaf_size).to(torch.long)
        idx1 = leaf_idx_per_tri
        idx3 = leaf_idx_per_tri.unsqueeze(1).expand(-1, 3)
        leaf_area_w = torch.zeros(L, dtype=torch.float32, device=device)
        leaf_c_area = torch.zeros((L, 3), dtype=torch.float32, device=device)
        leaf_n_area = torch.zeros((L, 3), dtype=torch.float32, device=device)
        leaf_area_w.scatter_reduce_(0, idx1, tri_area_s, 'sum', include_self=True)
        leaf_c_area.scatter_reduce_(0, idx3, tri_ac_s,   'sum', include_self=True)
        leaf_n_area.scatter_reduce_(0, idx3, tri_an_s,   'sum', include_self=True)

    # ---- Bottom-up aggregation (AABB + dipole) ----
    aabb_min, aabb_max, area_w, c_area, n_area = _aggregate_bottom_up(
        L, parent,
        leaf_aabb_min, leaf_aabb_max,
        leaf_area_w, leaf_c_area, leaf_n_area,
    )

    # ---- Barill r_ball: farthest AABB-corner distance from the area-weighted centroid.
    # Per-axis |extent| is max(aabb_max - c, c - aabb_min); true max is sqrt(sum of squared).
    c_node = c_area / area_w.clamp_min(1e-20).unsqueeze(1)
    d_per_axis = torch.maximum(aabb_max - c_node, c_node - aabb_min)
    r_ball = torch.linalg.norm(d_per_axis, dim=-1)              # (N,)

    # ---- Skip pointers ----
    skip = _build_skip_pointers(left, right, parent, L)

    return LBVH(
        V=V,
        F=F_long,
        tri_order=tri_order,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        left=left,
        right=right,
        parent=parent,
        skip=skip,
        leaf_tri_beg=leaf_tri_beg,
        leaf_tri_end=leaf_tri_end,
        area_w=area_w,
        c_area=c_area,
        n_area=n_area,
        r_ball=r_ball,
        leaf_size=int(leaf_size),
        num_leaves=int(L),
        num_nodes=int(2 * L - 1),
        device=device,
    )
