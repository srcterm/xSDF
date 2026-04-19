# Set MPS fallback policy before importing torch
import os
import importlib.util
import time
import numpy as np
import math, torch
from dataclasses import dataclass
from typing import Tuple, Optional

# Load sibling BVH module by file path (matches xSDF.py's loader pattern; works
# whether this file is imported as `torch_meshSDF` via importlib or as a script).
_BVH_FILE = os.path.join(os.path.dirname(__file__), "bvh.py")
_bvh_spec = importlib.util.spec_from_file_location("xsdf_bvh", _BVH_FILE)
_bvh_mod = importlib.util.module_from_spec(_bvh_spec)
_bvh_spec.loader.exec_module(_bvh_mod)

_FF_FILE = os.path.join(os.path.dirname(__file__), "floodfill.py")
_ff_spec = importlib.util.spec_from_file_location("xsdf_floodfill", _FF_FILE)
_ff_mod = importlib.util.module_from_spec(_ff_spec)
_ff_spec.loader.exec_module(_ff_mod)

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


def _device_sync(dev: torch.device) -> None:
    """Drain the queued kernels on ``dev`` so subsequent ``time.time()`` reads
    wall time for completed work, not launch time. No-op on CPU.
    """
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        # torch.mps.synchronize() is parameterless; safe to call on any MPS.
        torch.mps.synchronize()


def _chunk_size_for_bvh(memory_budget_gb: float, max_leaf_size: int) -> int:
    """Upper bound on how many points one dense-per-iter BVH traversal can
    handle under ``memory_budget_gb`` without blowing the (N,L) scratch tensors.

    The MPS/CUDA inner loop materializes several (N,L) tensors per iter (see
    ``bvh_min_distance_gpu``): valid_slot, clamped, tri_ids_pad, d_pair,
    huge_NL, zeros_NL, plus (N·L, 3) P_rep and A/B/C vertex fetches. Empirical
    per-point peak is ~32 B persistent + ~89 B per leaf slot + ~150 B of
    N-wide temps; we round up. Half the budget is reserved for the BVH itself
    and unrelated allocations.
    """
    L = max(1, int(max_leaf_size))
    per_point = 32 + 89 * L + 150  # bytes
    budget_bytes = int(float(memory_budget_gb) * 0.5 * 1e9)
    return max(16_384, budget_bytes // per_point)

# ------------------------------------------------------------------ distance & sign kernels (Torch only)

def point_triangle_distance_torch(P: torch.Tensor,
                                   A: torch.Tensor,
                                   B: torch.Tensor,
                                   C: torch.Tensor):
    """
    Compute unsigned minimum distance from points to triangles.

    Args:
        P: Points (N, 3)
        A, B, C: Triangle vertices (T, 3)

    Returns:
        Minimum unsigned distances (N,)

    Based on Ericson's Real-Time Collision Detection region tests.
    Vectorized; memory-friendly when T is chunked outside.
    """
    # Edges
    AB = B - A  # (T,3)
    AC = C - A  # (T,3)

    AP = P[:, None, :] - A[None, :, :]  # (N,T,3)
    d1 = torch.sum(AP * AB[None, :, :], dim=-1)  # (N,T)
    d2 = torch.sum(AP * AC[None, :, :], dim=-1)

    # Vertex region A
    maskA = (d1 <= 0.0) & (d2 <= 0.0)
    distA = torch.linalg.norm(AP, dim=-1)

    # Vertex region B
    BP = P[:, None, :] - B[None, :, :]
    d3 = torch.sum(BP * AB[None, :, :], dim=-1)
    d4 = torch.sum(BP * AC[None, :, :], dim=-1)
    maskB = (d3 >= 0.0) & (d4 <= d3)
    distB = torch.linalg.norm(BP, dim=-1)

    # Vertex region C
    CP = P[:, None, :] - C[None, :, :]
    d5 = torch.sum(CP * AB[None, :, :], dim=-1)
    d6 = torch.sum(CP * AC[None, :, :], dim=-1)
    maskC = (d6 >= 0.0) & (d5 <= d6)
    distC = torch.linalg.norm(CP, dim=-1)

    # Edge region AB
    vc = d1 * d4 - d3 * d2
    maskAB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = d1 / (d1 - d3 + 1e-12)
    projAB = A[None, :, :] + v[..., None] * AB[None, :, :]
    distAB = torch.linalg.norm(P[:, None, :] - projAB, dim=-1)

    # Edge region AC
    vb = d5 * d2 - d1 * d6
    maskAC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w = d2 / (d2 - d6 + 1e-12)
    projAC = A[None, :, :] + w[..., None] * AC[None, :, :]
    distAC = torch.linalg.norm(P[:, None, :] - projAC, dim=-1)

    # Edge region BC
    va = d3 * d6 - d5 * d4
    maskBC = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    projBC = B[None, :, :] + ((d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-12))[..., None] * (C - B)[None, :, :]
    distBC = torch.linalg.norm(P[:, None, :] - projBC, dim=-1)

    # Face region (inside triangle)
    maskFace = ~(maskA | maskB | maskC | maskAB | maskAC | maskBC)
    N = torch.linalg.cross(AB, AC, dim=-1)  # (T,3)
    N_norm = torch.linalg.norm(N, dim=-1) + 1e-12
    N_unit = N / N_norm[:, None]
    distPlane = torch.abs(torch.sum(AP * N_unit[None, :, :], dim=-1))

    # Use a dtype-safe large value (1e10 overflows fp16)
    if distA.dtype == torch.float16:
        huge_val = torch.finfo(torch.float16).max * 0.25  # ~1.6e4
    else:
        huge_val = 1e10
    huge = torch.full_like(distA, huge_val)
    dists = torch.where(maskA, distA, huge)
    dists = torch.minimum(dists, torch.where(maskB, distB, huge))
    dists = torch.minimum(dists, torch.where(maskC, distC, huge))
    dists = torch.minimum(dists, torch.where(maskAB, distAB, huge))
    dists = torch.minimum(dists, torch.where(maskAC, distAC, huge))
    dists = torch.minimum(dists, torch.where(maskBC, distBC, huge))
    dists = torch.minimum(dists, torch.where(maskFace, distPlane, huge))

    mins = dists.min(dim=1).values
    return mins


def solid_angle_sign_torch(P: torch.Tensor,
                            A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor) -> torch.Tensor:
    """Accumulate solid angle for each point in P with triangles A,B,C batch.
    Returns per-point total solid angle (N,), to be summed over triangle batches.
    Van Oosterom & Strackee (1983) formula.
    """
    Ap = A[None, :, :] - P[:, None, :]  # (N,T,3)
    Bp = B[None, :, :] - P[:, None, :]
    Cp = C[None, :, :] - P[:, None, :]

    a = torch.linalg.norm(Ap, dim=-1)
    b = torch.linalg.norm(Bp, dim=-1)
    c = torch.linalg.norm(Cp, dim=-1)

    num = torch.sum(Ap * torch.linalg.cross(Bp, Cp, dim=-1), dim=-1)  # (N,T)

    ab = torch.sum(Ap * Bp, dim=-1)
    bc = torch.sum(Bp * Cp, dim=-1)
    ca = torch.sum(Cp * Ap, dim=-1)

    eps = 1e-20 if Ap.dtype != torch.float16 else 1e-6
    denom = (a * b * c + a * bc + b * ca + c * ab + eps)
    omega = 2.0 * torch.atan2(num, denom)  # (N,T)
    return torch.sum(omega, dim=1)

# ------------------------------------------------------------------ memory & chunking utilities
def estimate_memory_and_chunks(total_pts: int, total_tris: int, device: torch.device,
                               target_memory_gb: Optional[float] = None):
    """
    Auto-calculate optimal pts_chunk and tri_chunk based on available memory.

    Args:
        total_pts: Total number of points to process
        total_tris: Total number of triangles in mesh
        device: Target torch device
        target_memory_gb: Optional memory budget in GB (auto-detected if None)

    Returns:
        (pts_chunk, tri_chunk, estimated_memory_gb)
    """
    # Detect available memory
    if target_memory_gb is not None:
        available_gb = target_memory_gb
    elif device.type == 'cuda':
        # CUDA: query actual free memory
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        available_gb = (free_mem / (1024**3)) * 0.7  # Use 70% of free memory
    elif device.type == 'mps':
        # MPS: Use heuristic based on system RAM (typically shares with CPU)
        import psutil
        available_gb = (psutil.virtual_memory().available / (1024**3)) * 0.5  # Conservative 50%
    else:  # CPU
        import psutil
        available_gb = (psutil.virtual_memory().available / (1024**3)) * 0.6  # Use 60% of available RAM

    available_bytes = available_gb * (1024**3)

    # Memory calculation for point_triangle_distance_torch:
    # The function creates many intermediate (N,T) and (N,T,3) arrays.
    # Strategy: Keep tri_chunk SMALL for cache efficiency and less memory overhead.
    # The (N,T) arrays are the bottleneck, so prefer large N (pts) and small T (tris).

    tri_chunk = max(2_000, min(10_000, int(0.03 * total_tris)))

    # Memory ≈ pts_chunk × tri_chunk × 48 floats × 4 bytes with 80% of available memory
    bytes_per_float = 4
    floats_per_pair = 48 
    usable_bytes = int(available_bytes * 0.8)  # Use 80% to leave headroom

    pts_chunk = int(usable_bytes / (tri_chunk * floats_per_pair * bytes_per_float))
    pts_chunk = max(2_000, min(pts_chunk, total_pts))  # Lower minimum for memory-constrained scenarios, maybe should be automated..

    # Estimate actual memory usage
    memory_used = pts_chunk * tri_chunk * floats_per_pair * bytes_per_float
    memory_used_gb = memory_used / (1024**3)

    return pts_chunk, tri_chunk, memory_used_gb


def dynamic_chunking(
    batch_num: int,
    current_pts_chunk: int,
    total_pts: int,
    processed: int,
    total_tris: int,
    device: torch.device,
    target_memory_gb: Optional[float] = None,
    reestimate_interval: int = 5
) -> int:
    """
    Dynamically adjust chunk size based on available memory during execution.

    Re-estimates available memory every N batches and reduces chunk size if needed.
    Only reduces (never increases) for stability.

    Args:
        batch_num: Current batch number (0-indexed)
        current_pts_chunk: Current chunk size
        total_pts: Total points to process
        processed: Points already processed
        total_tris: Total triangles in mesh
        device: Torch device
        target_memory_gb: Optional memory budget
        reestimate_interval: How often to re-estimate (default: every 5 batches)

    Returns:
        Updated pts_chunk (potentially reduced if memory pressure detected)
    """
    # Check if re-estimation is needed
    if batch_num == 0 or batch_num % reestimate_interval != 0:
        return current_pts_chunk

    # Re-estimate based on remaining points and current memory
    new_pts_chunk, _, _ = estimate_memory_and_chunks(
        total_pts - processed, total_tris, device, target_memory_gb
    )

    # Only reduce chunk size if memory is tight
    if new_pts_chunk < current_pts_chunk:
        print(f"[Adaptive] Reduced pts_chunk to {new_pts_chunk:,} due to memory pressure")
        return new_pts_chunk

    return current_pts_chunk

# ------------------------------------------------------------------ coordinate preparation
def _prepare_coordinate_arrays(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    device: torch.device,
):
    """
    Convert numpy coordinate arrays to torch tensors and extract domain information.

    Args:
        x_coords, y_coords, z_coords: 1D coordinate arrays (uniform or non-uniform)
        device: Target torch device

    Returns:
        (x_coords_t, y_coords_t, z_coords_t, origin, (Lx, Ly, Lz))
    """
    # Convert to torch tensors
    x_coords_t = torch.as_tensor(x_coords, device=device, dtype=torch.float32)
    y_coords_t = torch.as_tensor(y_coords, device=device, dtype=torch.float32)
    z_coords_t = torch.as_tensor(z_coords, device=device, dtype=torch.float32)

    # Extract origin and domain extents
    origin = torch.tensor([x_coords_t[0].item(), y_coords_t[0].item(), z_coords_t[0].item()],
                         dtype=torch.float32, device=device)

    Lx = float(x_coords_t[-1].item() - x_coords_t[0].item())
    Ly = float(y_coords_t[-1].item() - y_coords_t[0].item())
    Lz = float(z_coords_t[-1].item() - z_coords_t[0].item())

    return x_coords_t, y_coords_t, z_coords_t, origin, (Lx, Ly, Lz)


def _prepare_mesh_data(V: torch.Tensor, F: torch.Tensor):
    """
    Precompute triangle data for SDF computation.

    Args:
        V: Vertices tensor (N, 3)
        F: Faces tensor (T, 3)

    Returns:
        dict with keys: A_tri, B_tri, C_tri, tri_min, tri_max, tri_norm_unit
    """
    A_tri = V[F[:, 0]]
    B_tri = V[F[:, 1]]
    C_tri = V[F[:, 2]]

    # Precompute triangle AABBs for AABB pruning
    tri_min = torch.min(torch.stack((A_tri, B_tri, C_tri)), dim=0).values
    tri_max = torch.max(torch.stack((A_tri, B_tri, C_tri)), dim=0).values

    # Precompute triangle normals for nearest-normal sign method
    tri_edges1 = B_tri - A_tri
    tri_edges2 = C_tri - A_tri
    tri_normals = torch.linalg.cross(tri_edges1, tri_edges2, dim=-1)
    tri_norm_l = torch.linalg.norm(tri_normals, dim=-1).clamp_min(1e-12)
    tri_norm_unit = tri_normals / tri_norm_l[:, None]

    return {
        'A_tri': A_tri,
        'B_tri': B_tri,
        'C_tri': C_tri,
        'tri_min': tri_min,
        'tri_max': tri_max,
        'tri_norm_unit': tri_norm_unit,
    }


def _select_triangles_for_batch(
    P: torch.Tensor,
    mesh_data: dict,
    mode: str,
    domain_extents: Tuple[float, float, float],
    device: torch.device,
    num_triangles: int,
) -> torch.Tensor:
    """
    Select triangles to process for a point batch based on acceleration mode.

    Args:
        P: Point batch (N, 3)
        mesh_data: Dict with tri_min, tri_max from _prepare_mesh_data
        mode: 'aabb' or 'none'
        domain_extents: (Lx, Ly, Lz) for adaptive margin
        device: torch device
        num_triangles: Total number of triangles

    Returns:
        Triangle indices (1D tensor)
    """
    if mode == "aabb":
        # AABB pruning: select triangles whose bounding boxes intersect point batch AABB
        Lx, Ly, Lz = domain_extents
        margin = (Lx + Ly + Lz) / 30.0  # Adaptive margin

        pmin = P.min(dim=0).values - margin
        pmax = P.max(dim=0).values + margin

        tri_min = mesh_data['tri_min']
        tri_max = mesh_data['tri_max']

        mask = ((tri_max[:, 0] >= pmin[0]) & (tri_min[:, 0] <= pmax[0]) &
                (tri_max[:, 1] >= pmin[1]) & (tri_min[:, 1] <= pmax[1]) &
                (tri_max[:, 2] >= pmin[2]) & (tri_min[:, 2] <= pmax[2]))
        tri_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        # Fallback: if AABB pruning produced no triangles, use all
        if tri_ids.numel() == 0:
            tri_ids = torch.arange(num_triangles, device=device, dtype=torch.long)
    else:
        # Brute force: use all triangles
        tri_ids = torch.arange(num_triangles, device=device, dtype=torch.long)

    return tri_ids


def _compute_solid_angle_only(
    P: torch.Tensor,
    tri_ids: torch.Tensor,
    V: torch.Tensor,
    F: torch.Tensor,
    tri_chunk: int,
) -> torch.Tensor:
    """Solid-angle accumulation only — no distance computation."""
    device = P.device
    omega_tot = torch.zeros((P.shape[0],), device=device, dtype=torch.float32)
    for t0 in range(0, tri_ids.shape[0], tri_chunk):
        tid = tri_ids[t0:t0+tri_chunk]
        A = V[F[tid, 0]]
        B = V[F[tid, 1]]
        C = V[F[tid, 2]]
        w_batch = solid_angle_sign_torch(P, A, B, C)
        w_batch = torch.nan_to_num(w_batch, nan=0.0, posinf=0.0, neginf=0.0)
        omega_tot += w_batch
    return omega_tot


def _compute_batch_distances(
    P: torch.Tensor,
    tri_ids: torch.Tensor,
    V: torch.Tensor,
    F: torch.Tensor,
    tri_chunk: int,
    max_reasonable_dist: float,
):
    """
    Compute minimum distances and solid angles for a point batch.

    Uses solid angle method (winding number) for accurate inside/outside determination.

    Args:
        P: Point batch (N, 3)
        tri_ids: Triangle indices to process
        V: Vertices
        F: Faces
        tri_chunk: Triangle chunk size
        max_reasonable_dist: Maximum clamp distance

    Returns:
        (min_d, omega_tot)
        - min_d: minimum unsigned distances (N,)
        - omega_tot: accumulated solid angles (N,)
    """
    device = P.device
    min_d = torch.full((P.shape[0],), max_reasonable_dist, device=device, dtype=torch.float32)
    omega_tot = torch.zeros((P.shape[0],), device=device, dtype=torch.float32)

    # Batch over selected triangles
    for t0 in range(0, tri_ids.shape[0], tri_chunk):
        tid = tri_ids[t0:t0+tri_chunk]
        A = V[F[tid, 0]]
        B = V[F[tid, 1]]
        C = V[F[tid, 2]]

        # Compute distances (unsigned)
        d_batch = point_triangle_distance_torch(P, A, B, C)

        # Sanitize distances (remove NaN/Inf)
        d_batch = torch.nan_to_num(d_batch, nan=max_reasonable_dist,
                                   posinf=max_reasonable_dist, neginf=max_reasonable_dist)

        # Update minimum distance
        upd = d_batch < min_d
        min_d[upd] = d_batch[upd]

        # Accumulate solid angle for sign determination
        w_batch = solid_angle_sign_torch(P, A, B, C)
        w_batch = torch.nan_to_num(w_batch, nan=0.0, posinf=0.0, neginf=0.0)
        omega_tot += w_batch

    return min_d, omega_tot


def _apply_sign_to_distances(
    min_d: torch.Tensor,
    omega_tot: torch.Tensor,
) -> torch.Tensor:
    """
    Apply sign to unsigned distances using solid angle method.

    Uses the accumulated solid angle (winding number) to determine inside/outside.
    Points with |omega| > 2π are inside the mesh (negative distance).

    Args:
        min_d: Unsigned minimum distances (N,)
        omega_tot: Accumulated solid angles (N,)

    Returns:
        Signed distances (N,)
    """
    # Solid angle method: inside if |omega| > 2π
    inside = torch.abs(omega_tot) > math.pi * 2.0
    signed = min_d.clone()
    signed[inside] = -signed[inside]
    return signed


# ------------------------------------------------------------------ FWN pipeline

def _cdist_vertex_warmstart(
    P: torch.Tensor,
    V: torch.Tensor,
    memory_budget_gb: float,
) -> torch.Tensor:
    """Tiled point-to-vertex cdist. Returns (N,) upper bounds on d_surface.

    Valid because min_i ||p - v_i|| >= d_surface for any triangle: the
    closest surface point is never farther than that triangle's closest
    vertex. Used as a tighter seed for bvh_min_distance_gpu than the
    early-outed d_upper from classify_band.
    """
    N = int(P.shape[0])
    nV = int(V.shape[0])
    if N == 0 or nV == 0:
        return torch.empty((N,), dtype=torch.float32, device=P.device)
    bytes_per_pair = 4  # float32
    budget = max(1, int(memory_budget_gb * 1e9) // (nV * bytes_per_pair))
    chunk = min(N, budget)
    out = torch.empty((N,), dtype=torch.float32, device=P.device)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        d = torch.cdist(P[i:j], V)
        out[i:j] = d.amin(dim=1)
    return out


def _run_fwn_pipeline(
    V: torch.Tensor,
    F: torch.Tensor,
    x_coords_t: torch.Tensor,
    y_coords_t: torch.Tensor,
    z_coords_t: torch.Tensor,
    dev: torch.device,
    max_reasonable_dist: float,
    beta: float,
    band_width_cells: float,
    bvh_leaf_size: int,
    bvh_build_device: str,
    memory_budget_gb: float = 2.0,
    use_cdist_warmstart: bool = True,
) -> torch.Tensor:
    """All-device SDF via skip-pointer BVH + Barill FWN + flood-fill + fast sweep.

    Pipeline:
        1. Build BVH (CPU), upload to ``dev``.
        2. Band classify grid points: bvh_min_distance_gpu with d_best pinned
           to band_threshold. Cells under the threshold are "band" cells.
        3. In-band: exact FWN (sign) + bvh_min_distance_gpu (|d|).
        4. Flood-fill sign from a corner seed through non-band cells.
        5. Fast sweep propagates |d| out of the band.
        6. Assemble phi = sign * |d|, pin band cells to their exact values.

    All stages run on ``dev``. The BVH traversal kernels
    (:func:`bvh.bvh_min_distance_gpu`, :func:`bvh.fwn_query`) are written in a
    dense-per-iter form with amortized sync (see their docstrings) so they
    stay fast on MPS — the earlier CPU-routing workaround is no longer needed.
    """
    # --- BVH build + upload to dev ---------------------------------------
    _device_sync(dev)
    t0 = time.time()
    bvh_built = _bvh_mod.build_bvh_torch(V, F, leaf_size=bvh_leaf_size,
                                          build_device=bvh_build_device)
    bvh = _bvh_mod.bvh_to_device(bvh_built, dev)
    stats = _bvh_mod.bvh_stats(bvh)
    _device_sync(dev)
    print(f"[FWN] BVH: {stats['n_nodes']} nodes, depth={stats['max_depth']}, "
          f"leaves={stats['n_leaves']}, avg_leaf={stats['avg_leaf_size']:.1f}, "
          f"max_leaf={stats['max_leaf_size']} | build={(time.time()-t0)*1000:.1f}ms")

    max_leaf = int(bvh["max_leaf_size"])
    max_chunk = _chunk_size_for_bvh(memory_budget_gb, max_leaf)

    Nx = int(x_coords_t.numel())
    Ny = int(y_coords_t.numel())
    Nz = int(z_coords_t.numel())
    Xg, Yg, Zg = torch.meshgrid(
        x_coords_t.to(dev), y_coords_t.to(dev), z_coords_t.to(dev),
        indexing="ij",
    )
    P = torch.stack([Xg, Yg, Zg], dim=-1).reshape(-1, 3)
    total = int(P.shape[0])
    print(f"[FWN] grid: {total:,} points, max_chunk={max_chunk:,} "
          f"(budget={memory_budget_gb:.2f}GB, max_leaf={max_leaf})")

    # Grid spacing (non-uniform safe: use max local spacing).
    def _max_h(coord):
        if coord.numel() < 2:
            return 0.0
        d = (coord[1:] - coord[:-1]).abs()
        return float(d.max())
    dx = _max_h(x_coords_t)
    dy = _max_h(y_coords_t)
    dz = _max_h(z_coords_t)
    h = max(dx, dy, dz)
    band_threshold = float(band_width_cells) * h

    # --- Band classification (chunked) -----------------------------------
    # cdist warm-start is deferred to the band-only slot below: on MPS/CUDA
    # active-set compaction already retires in-band points early via the
    # early_out_threshold path, and on CPU the sparse classify loop retires
    # cheaply on its own — full-grid cdist costs ~10s on Ahmed/MPS for ~0
    # additional speedup.
    _device_sync(dev)
    t1 = time.time()
    band_mask = torch.empty((total,), dtype=torch.bool, device=dev)
    d_upper = torch.empty((total,), dtype=torch.float32, device=dev)
    for i in range(0, total, max_chunk):
        j = min(i + max_chunk, total)
        bm_i, du_i = _ff_mod.classify_band(
            P[i:j], bvh, V, F, band_threshold, _bvh_mod,
        )
        band_mask[i:j] = bm_i
        d_upper[i:j] = du_i
    n_band = int(band_mask.sum())
    _device_sync(dev)
    print(f"[FWN] band: {n_band}/{total}  ({100.0*n_band/max(total,1):.1f}%)  "
          f"threshold={band_threshold:.4g}  ({time.time()-t1:.2f}s)")

    band_idx = band_mask.nonzero().flatten()
    band_mask_3d = band_mask.view(Nx, Ny, Nz)

    # --- Exact sign and |d| in band --------------------------------------
    sign_band = torch.zeros((n_band,), dtype=torch.float32, device=dev)
    d_band = torch.zeros((n_band,), dtype=torch.float32, device=dev)
    if n_band > 0:
        P_band = P[band_idx]

        # Sign via FWN.
        _device_sync(dev)
        t2 = time.time()
        w_band = _bvh_mod.fwn_query(P_band, bvh, V, F, beta=beta)
        # Convention matches _apply_sign_to_distances: inside is negative.
        # CCW outward normals ⇒ w≈+1 inside, ≈0 outside.
        sign_band = torch.where(w_band > 0.5, -1.0, 1.0).to(torch.float32)
        _device_sync(dev)
        print(f"[FWN] fwn_query: {(time.time()-t2)*1000:.1f}ms")

        warm = d_upper[band_idx]
        if use_cdist_warmstart:
            _device_sync(dev)
            tc = time.time()
            d_cdist = _cdist_vertex_warmstart(P_band, V, memory_budget_gb * 0.25)
            warm = torch.minimum(warm, d_cdist)
            _device_sync(dev)
            print(f"[FWN] cdist_warmstart (band-only): {(time.time()-tc)*1000:.1f}ms")

        # Exact |d| (chunked for memory safety on large bands).
        _device_sync(dev)
        t3 = time.time()
        for i in range(0, n_band, max_chunk):
            j = min(i + max_chunk, n_band)
            d_band[i:j] = _bvh_mod.bvh_min_distance_gpu(
                P_band[i:j], bvh, V, F,
                max_reasonable_dist=float(max_reasonable_dist),
                initial_upper=warm[i:j],
            )
        _device_sync(dev)
        print(f"[FWN] bvh_min_distance_gpu: {(time.time()-t3)*1000:.1f}ms")

    # --- Flood-fill sign -------------------------------------------------
    _device_sync(dev)
    t4 = time.time()
    sign_outside_3d = _ff_mod.flood_fill_sign_gpu(band_mask_3d, seed=(0, 0, 0))
    _device_sync(dev)
    print(f"[FWN] flood_fill: {(time.time()-t4)*1000:.1f}ms")

    # --- Fast sweep |d| --------------------------------------------------
    phi_band_abs_3d = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=dev)
    if n_band > 0:
        phi_band_abs_3d.view(-1)[band_idx] = d_band

    _device_sync(dev)
    t5 = time.time()
    u = _ff_mod.fast_sweep_gpu(
        phi_band_abs_3d, band_mask_3d,
        x_coords_t.to(dev), y_coords_t.to(dev), z_coords_t.to(dev),
    )
    _device_sync(dev)
    print(f"[FWN] fast_sweep: {(time.time()-t5)*1000:.1f}ms")

    # --- Assemble phi ----------------------------------------------------
    #   - band cells: sign_band * d_band
    #   - non-band:   sign_outside * u
    phi_flat = (sign_outside_3d * u).view(-1)
    if n_band > 0:
        phi_flat[band_idx] = sign_band * d_band

    _device_sync(dev)
    return phi_flat.view(Nx, Ny, Nz)


# ------------------------------------------------------------------ main driver
def mesh_to_sdf_torch(
    V_np, F_np,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    *,
    device=None,
    compile_kernels=True,
    use_accel=True,
    accel: Optional[str] = None,
    bvh_leaf_size: int = 8,
    bvh_build_device: str = "cpu",
    target_memory_gb: Optional[float] = None,
    fwn_beta: float = 2.0,
    fwn_band_width_cells: float = 2.0,
    fwn_cdist_warmstart: bool = True,
):
    """
    Compute signed distance field (SDF) using PyTorch with automatic chunking and AABB acceleration.

    This function computes the signed distance from grid points to a triangular mesh using the
    solid angle method (winding number) for accurate inside/outside determination. Supports
    both uniform and non-uniform grids with automatic memory management and GPU acceleration.

    Args:
        V_np: Vertex array (N, 3) - mesh vertices
        F_np: Face array (T, 3) - triangle face indices
        x_coords, y_coords, z_coords: 1D coordinate arrays defining the sample points at which
                                     the SDF is evaluated (uniform or non-uniform).
        device: 'cuda'/'mps'/'cpu'/None (auto-select: cuda > mps > cpu)
        compile_kernels: Use torch.compile for 2-3x speedup (default True)
        use_accel: Enable AABB spatial acceleration (default True). False uses brute force.
            Ignored if `accel` is explicitly passed.
        accel: Optional acceleration mode — one of 'none', 'aabb', 'bvh', 'fwn'.
            When None, falls back to the legacy boolean `use_accel` ('aabb' if
            True else 'none'). 'bvh' runs a hybrid: per-point branch-and-bound
            distance query on CPU and solid-angle on the SDF device via the
            flat AABB prune. 'fwn' runs an all-GPU pipeline: skip-pointer BVH
            traversal with Barill fast winding number for sign and exact |d|
            in a narrow band, plus flood-fill sign extension and fast-sweep
            eikonal propagation for the far field.
        bvh_leaf_size: max triangles per BVH leaf (used by 'bvh' and 'fwn').
        bvh_build_device: device for the BVH build ('cpu' recommended — build is
            inherently sequential and per-node kernel launch overhead dominates GPU).
        target_memory_gb: Manual memory budget for auto-chunking (auto-detected if None)
        fwn_beta: Barill β-admissibility threshold (only used when accel='fwn').
            β=2.0 gives ~4-digit sign accuracy. Raise to 3.0 for tighter trees.
        fwn_band_width_cells: Narrow-band half-width in grid cells (accel='fwn').
            Cells within `band_width * max(dx,dy,dz)` of the mesh get exact
            FWN sign + exact |d|; the rest get flood-fill sign + FSM |d|.

    Sign Method:
        Uses solid_angle (winding number) for robust inside/outside determination:
        - Computes winding number via solid angle accumulation (topologically robust)
        - Always correct for watertight meshes, handles complex concave geometry
        - Points with |omega| > 2π are inside the mesh (negative distance)

    Returns:
        SDF Result with:
            - phi: Signed distance field numpy array (nx, ny, nz)
            - origin: Grid origin coordinates (x0, y0, z0) as numpy array
            - dx: Grid spacing if uniform, nan if non-uniform
            - grid_shape: (nx, ny, nz) tuple
    """
    print('Starting SDF computation...')
    dev = pick_device(device)
    if dev.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    print(f"[Device] {dev}")

    # ============ Prepare coordinates ============
    x_coords_t, y_coords_t, z_coords_t, origin, domain_extents = _prepare_coordinate_arrays(
        x_coords, y_coords, z_coords, dev
    )
    nx, ny, nz = len(x_coords_t), len(y_coords_t), len(z_coords_t)
    Lx, Ly, Lz = domain_extents

    # ============ Prepare mesh data ============
    V = torch.from_numpy(V_np).to(device=dev, dtype=torch.float32)
    F = torch.from_numpy(F_np).to(device=dev, dtype=torch.long)
    mesh_data = _prepare_mesh_data(V, F)
    Ntris = F.shape[0]

    # Precompute max reasonable distance (2× diagonal)
    max_reasonable_dist = math.sqrt(Lx*Lx + Ly*Ly + Lz*Lz) * 2.0

    # ============ Build grid indices ============
    grid_idx = torch.stack(torch.meshgrid(
        torch.arange(nx, device=dev),
        torch.arange(ny, device=dev),
        torch.arange(nz, device=dev),
        indexing='ij'), dim=-1).reshape(-1, 3)
    total_pts = grid_idx.shape[0]

    # ============ Auto-chunking: calculate optimal chunk sizes ============
    pts_chunk, tri_chunk, mem_gb = estimate_memory_and_chunks(total_pts, Ntris, dev, target_memory_gb)
    print(f"[Auto-chunking] pts_chunk={pts_chunk:,}, tri_chunk={tri_chunk:,}, est_memory={mem_gb:.2f}GB")

    n_pt_batches = math.ceil(total_pts / pts_chunk)
    phi = torch.empty(total_pts, dtype=torch.float32, device=dev)

    # ============ Acceleration mode ============
    if accel is None:
        mode = "aabb" if use_accel else "none"
    else:
        mode = accel.lower()
        if mode not in ("none", "aabb", "bvh", "fwn"):
            raise ValueError(f"Unknown accel mode '{accel}'. Use 'none', 'aabb', 'bvh', or 'fwn'.")

    bvh_cpu = None
    V_cpu = None
    F_cpu = None
    if mode == "fwn":
        phi_3d = _run_fwn_pipeline(
            V, F, x_coords_t, y_coords_t, z_coords_t, dev,
            max_reasonable_dist=max_reasonable_dist,
            beta=fwn_beta,
            band_width_cells=fwn_band_width_cells,
            bvh_leaf_size=bvh_leaf_size,
            bvh_build_device=bvh_build_device,
            memory_budget_gb=float(target_memory_gb) if target_memory_gb else 2.0,
            use_cdist_warmstart=fwn_cdist_warmstart,
        )
        phi = phi_3d.reshape(-1)
    elif mode == "bvh":
        t_build0 = time.time()
        bvh_built = _bvh_mod.build_bvh_torch(V, F, leaf_size=bvh_leaf_size,
                                              build_device="cpu")
        t_build = time.time() - t_build0
        bvh_cpu = bvh_built  # already on CPU
        V_cpu = V.detach().to("cpu")
        F_cpu = F.detach().to("cpu")
        stats = _bvh_mod.bvh_stats(bvh_cpu)
        print(f"[Accel] BVH (hybrid CPU-distance / GPU-sign): {stats['n_nodes']} nodes, "
              f"depth={stats['max_depth']}, {stats['n_leaves']} leaves, "
              f"avg_leaf={stats['avg_leaf_size']:.1f}, max_leaf={stats['max_leaf_size']} | "
              f"build={t_build*1000:.1f}ms on cpu")
    else:
        print(f"[Accel] {'AABB pruning enabled' if mode == 'aabb' else 'No acceleration (brute force)'}")

    # ============ Optional torch.compile ============
    global point_triangle_distance_torch, solid_angle_sign_torch
    if compile_kernels and hasattr(torch, "compile"):
        point_triangle_distance_torch = torch.compile(point_triangle_distance_torch, mode="max-autotune")
        solid_angle_sign_torch = torch.compile(solid_angle_sign_torch, mode="max-autotune")

    # ============ Main point loop ============
    processed = 0
    batch_num = 0

    with torch.no_grad():
        while mode != "fwn" and processed < total_pts:
            # Dynamic chunking: adapt to memory pressure
            pts_chunk = dynamic_chunking(batch_num, pts_chunk, total_pts, processed, Ntris, dev, target_memory_gb)
            batch_num += 1

            # Get point batch indices
            idx_batch = grid_idx[processed: processed + pts_chunk]

            # Build point coordinates (unified - always use coordinate arrays)
            Px = x_coords_t.index_select(0, idx_batch[:, 0])
            Py = y_coords_t.index_select(0, idx_batch[:, 1])
            Pz = z_coords_t.index_select(0, idx_batch[:, 2])
            P = torch.stack([Px, Py, Pz], dim=1).to(dev)

            # Select triangles and run distance/solid-angle kernels.
            if mode == "bvh":
                # Hybrid: per-point BVH branch-and-bound on CPU for distance,
                # flat AABB prune + solid-angle on GPU for sign. Each point's
                # CPU descent prunes aggressively so far points stop fast;
                # solid-angle stays where the math is big and parallel.
                P_cpu = P.detach().to("cpu")
                min_d_cpu = _bvh_mod.bvh_query_distances(
                    P_cpu, bvh_cpu, V_cpu, F_cpu, max_reasonable_dist
                )
                min_d = min_d_cpu.to(dev)
                min_d = torch.nan_to_num(min_d, nan=max_reasonable_dist,
                                         posinf=max_reasonable_dist, neginf=max_reasonable_dist)
                sign_tri_ids = _select_triangles_for_batch(P, mesh_data, "aabb", domain_extents, dev, Ntris)
                tri_count_for_batch = sign_tri_ids.numel()
                omega_tot = _compute_solid_angle_only(P, sign_tri_ids, V, F, tri_chunk)
            else:
                tri_ids = _select_triangles_for_batch(P, mesh_data, mode, domain_extents, dev, Ntris)
                tri_count_for_batch = tri_ids.numel()
                min_d, omega_tot = _compute_batch_distances(
                    P, tri_ids, V, F, tri_chunk, max_reasonable_dist
                )

            # Apply sign to distances
            signed = _apply_sign_to_distances(min_d, omega_tot)

            # Store results
            phi[processed: processed + P.shape[0]] = signed

            # Progress reporting (per chunk)
            processed += P.shape[0]
            batch_id = processed // pts_chunk
            print(f"[Batch {batch_id}/{n_pt_batches}] {processed}/{total_pts} points | {tri_count_for_batch} triangles used")

    # ============ Finalize and return ============
    phi = phi.reshape(nx, ny, nz)

    # Final safety: replace any NaNs/Infs, then clamp to physical range
    phi = torch.nan_to_num(phi,
                           nan=max_reasonable_dist,
                           posinf=max_reasonable_dist,
                           neginf=-max_reasonable_dist)
    phi = torch.clamp(phi, -max_reasonable_dist, max_reasonable_dist)

    # Detect if grid is uniform by checking coordinate spacing
    is_uniform = False
    if nx > 1 and ny > 1 and nz > 1:
        dx_x = (x_coords_t[1] - x_coords_t[0]).item()
        dx_y = (y_coords_t[1] - y_coords_t[0]).item()
        dx_z = (z_coords_t[1] - z_coords_t[0]).item()
        if abs(dx_x - dx_y) < 1e-6 and abs(dx_x - dx_z) < 1e-6:
            is_uniform = True
            dx_out = dx_x
    if not is_uniform:
        dx_out = float('nan')

    # Always return numpy arrays
    return SDFResult(phi.cpu().float().numpy(), origin.cpu().float().numpy(), dx_out, (nx, ny, nz))