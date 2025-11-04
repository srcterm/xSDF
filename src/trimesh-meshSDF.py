"""
Trimesh-based SDF computation backend.

This module provides CPU-based signed distance field computation using the trimesh library.
Implements chunked processing with double-batching to work around trimesh memory leaks.

Date: 2025-01-01
"""

import numpy as np
import trimesh
import gc
from typing import Dict, Optional


def mesh_to_sdf_trimesh(mesh: trimesh.Trimesh,
                        x_coords: np.ndarray,
                        y_coords: np.ndarray,
                        z_coords: np.ndarray,
                        memory_budget_gb: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute SDF using trimesh on a grid defined by 1D coordinate arrays (uniform or non-uniform).

    Uses chunked processing for memory efficiency with progress reporting.

    Args:
        mesh: The watertight mesh to compute SDF from
        x_coords, y_coords, z_coords: 1D arrays of cell-center coordinates
        memory_budget_gb: Optional memory budget in GB for chunk size calculation.
                         If None, uses default chunk_size of 64.

    Returns:
        sdf: 3D array of signed distance values
        origin: Grid origin point
        coord_dict: Dictionary with coordinate arrays
    """
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    total_voxels = nx * ny * nz

    # Calculate chunk size from memory budget
    if memory_budget_gb is not None:
        usable_bytes = int(memory_budget_gb * (1024**3) * 0.05)  # 5% of budget
        chunk_size = max(4, int((usable_bytes / 4) ** (1/3)))
        chunk_size = min(chunk_size, 28)
    else:
        chunk_size = 28

    # Clamp to grid dimensions
    chunk_size = max(4, min(chunk_size, nx, ny, nz))

    # Calculate chunking
    chunks_x = int(np.ceil(nx / chunk_size))
    chunks_y = int(np.ceil(ny / chunk_size))
    chunks_z = int(np.ceil(nz / chunk_size))
    total_chunks = chunks_x * chunks_y * chunks_z

    print(f"Trimesh SDF computation:")
    print(f"  Grid shape: {nx}×{ny}×{nz} = {total_voxels:,} voxels")
    print(f"  Chunk size: {chunk_size}³ = {chunk_size**3:,} voxels per chunk")
    print(f"  Processing in {chunks_x}×{chunks_y}×{chunks_z} = {total_chunks} chunks")

    # Initialize SDF array
    sdf = np.zeros((nx, ny, nz), dtype=np.float32)
    chunk_count = 0

    # Process in chunks
    for i0 in range(0, nx, chunk_size):
        i1 = min(i0 + chunk_size, nx)
        for j0 in range(0, ny, chunk_size):
            j1 = min(j0 + chunk_size, ny)
            for k0 in range(0, nz, chunk_size):
                k1 = min(k0 + chunk_size, nz)
                chunk_count += 1

                # Create meshgrid for this chunk
                Xc, Yc, Zc = np.meshgrid(x_coords[i0:i1], y_coords[j0:j1], z_coords[k0:k1], indexing='ij')
                pts = np.column_stack([Xc.ravel(), Yc.ravel(), Zc.ravel()])

                # Process points in small sub-batches 
                point_batch_size = 2000  # Process 2000 points at a time
                n_points = pts.shape[0]
                d = np.zeros(n_points, dtype=np.float32)

                for p0 in range(0, n_points, point_batch_size):
                    p1 = min(p0 + point_batch_size, n_points)
                    d[p0:p1] = trimesh.proximity.signed_distance(mesh, pts[p0:p1])
                    mesh._cache.clear()
                sdf_chunk = (-d).reshape(Xc.shape).astype(np.float32)
                sdf[i0:i1, j0:j1, k0:k1] = sdf_chunk

                # Clean up
                del Xc, Yc, Zc, pts, d, sdf_chunk
                if chunk_count % 2 == 0:
                    gc.collect()

                # Progress reporting
                if chunk_count % max(1, total_chunks // 20) == 0:
                    print(f"  Progress: {chunk_count/total_chunks*100:.1f}% ({chunk_count}/{total_chunks} chunks)")

    print(f"  SDF stats: min={sdf.min():.6f}, max={sdf.max():.6f}")
    print(f"  Interior voxels (φ<0): {np.sum(sdf < 0):,}")
    print(f"  Exterior voxels (φ>0): {np.sum(sdf > 0):,}")

    origin = np.array([x_coords[0], y_coords[0], z_coords[0]], dtype=np.float32)
    coord_dict = {
        "x_coords": x_coords.astype(np.float32),
        "y_coords": y_coords.astype(np.float32),
        "z_coords": z_coords.astype(np.float32)
    }

    return sdf, origin, coord_dict
