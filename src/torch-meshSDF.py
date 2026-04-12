# Set MPS fallback policy before importing torch
import os
import numpy as np
import math, torch
from dataclasses import dataclass
from typing import Tuple, Optional

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
    target_memory_gb: Optional[float] = None,
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
        target_memory_gb: Manual memory budget for auto-chunking (auto-detected if None)

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
    mode = "aabb" if use_accel else "none"
    print(f"[Accel] {'AABB pruning enabled' if use_accel else 'No acceleration (brute force)'}")

    # ============ Optional torch.compile ============
    global point_triangle_distance_torch, solid_angle_sign_torch
    if compile_kernels and hasattr(torch, "compile"):
        point_triangle_distance_torch = torch.compile(point_triangle_distance_torch, mode="max-autotune")
        solid_angle_sign_torch = torch.compile(solid_angle_sign_torch, mode="max-autotune")

    # ============ Main point loop ============
    processed = 0
    batch_num = 0

    with torch.no_grad():
        while processed < total_pts:
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

            # Select triangles for this batch
            tri_ids = _select_triangles_for_batch(P, mesh_data, mode, domain_extents, dev, Ntris)
            tri_count_for_batch = tri_ids.numel()

            # Compute distances and solid angles
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