"""Narrow-band + grid propagation helpers for the FWN accel mode.

Pure-torch, GPU-friendly:
    - classify_band: run bvh_min_distance_gpu with max_reasonable_dist set to
      the band threshold. Points whose distance is below the threshold are in
      the band (and their returned value is a useful tight upper bound).
    - flood_fill_sign_gpu: BFS from an "outside" seed through the grid,
      blocked by band cells. Implemented as iterative mask dilation with
      torch slicing — no Python per-cell loops.
    - fast_sweep_gpu: 3D Fast Sweeping Method (Zhao 2005) for the eikonal
      equation, seeded by unsigned distances in the band. 8 Gauss-Seidel
      sweeps per pass, configurable number of passes.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


# ------------------------------------------------------------------ band classification

def classify_band(
    P: torch.Tensor,
    bvh: Dict,
    V: torch.Tensor,
    F: torch.Tensor,
    band_threshold: float,
    bvh_mod,
    initial_upper: "torch.Tensor | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mark which query points lie within ``band_threshold`` of the mesh.

    Uses :func:`bvh.bvh_min_distance_gpu` with ``max_reasonable_dist`` pinned
    to ``band_threshold`` so the traversal aggressively prunes cells whose
    true distance exceeds the threshold. Returns a boolean mask plus the
    tight upper-bound tensor (useful for seeding downstream kernels).

    Args:
        initial_upper: optional pre-computed upper bound on ``|d|`` per point
            (e.g. a point-to-vertex cdist seed clamped to ``band_threshold``).
            Forwarded to ``bvh_min_distance_gpu``; cells that already prove
            sub-threshold via the seed retire at iter 0.
    """
    d_upper = bvh_mod.bvh_min_distance_gpu(
        P, bvh, V, F,
        max_reasonable_dist=band_threshold,
        initial_upper=initial_upper,
        early_out_threshold=band_threshold,
    )
    band_mask = d_upper < band_threshold
    return band_mask, d_upper


# ------------------------------------------------------------------ flood-fill sign

def flood_fill_sign_gpu(
    band_mask_3d: torch.Tensor,
    seed: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """Mark "outside" grid cells reachable from ``seed`` without crossing the band.

    Starts from the seed cell, expands into non-band 6-connected neighbors
    until a fixed point is reached. Cells that end up unreached are declared
    inside.

    Args:
        band_mask_3d: (Nx, Ny, Nz) bool. Cells inside the band act as walls.
        seed: (i, j, k) tuple. Must be a non-band cell. Typically a domain corner.

    Returns:
        sign: (Nx, Ny, Nz) float32 tensor of +1 (reached from seed) or -1
        (unreached). Band cells are marked +1 here as a placeholder — the
        caller will overwrite them with sign(phi_band).
    """
    dev = band_mask_3d.device
    Nx, Ny, Nz = band_mask_3d.shape
    if bool(band_mask_3d[seed]):
        # Seed fell inside the band. The caller should pick a different seed
        # or pad the domain. Fall back to "all outside" to avoid silent bugs
        # — sign will be wrong but not catastrophic at large distances.
        return torch.ones_like(band_mask_3d, dtype=torch.float32)

    outside = torch.zeros_like(band_mask_3d, dtype=torch.bool)
    outside[seed] = True
    not_band = ~band_mask_3d

    for _ in range(Nx + Ny + Nz + 2):
        prev_sum = int(outside.sum().item())
        nxt = outside.clone()
        # x neighbors
        nxt[1:, :, :] |= outside[:-1, :, :]
        nxt[:-1, :, :] |= outside[1:, :, :]
        # y neighbors
        nxt[:, 1:, :] |= outside[:, :-1, :]
        nxt[:, :-1, :] |= outside[:, 1:, :]
        # z neighbors
        nxt[:, :, 1:] |= outside[:, :, :-1]
        nxt[:, :, :-1] |= outside[:, :, 1:]
        # Mask off band cells
        nxt &= not_band
        if int(nxt.sum().item()) == prev_sum:
            break
        outside = nxt

    sign = torch.where(outside, 1.0, -1.0).to(torch.float32)
    return sign


# ------------------------------------------------------------------ fast sweeping

def fast_sweep_gpu(
    phi_band_abs: torch.Tensor,
    band_mask_3d: torch.Tensor,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    z_coords: torch.Tensor,
    n_passes: int | None = None,
    inf_sentinel: float = 1e20,
    tol: float = 1e-4,
) -> torch.Tensor:
    """3D Fast Sweeping Method for the eikonal equation |∇u| = 1.

    Seeds ``u`` with the band's unsigned distances and runs parallel Godunov
    updates until the field converges. Band cells stay pinned. Non-uniform
    grids are handled correctly: the eikonal update uses each cell's local
    upwind spacing rather than a single global h.

    Each sub-sweep is a full-field Jacobi-style update (slice-assigns can't
    see their own updates mid-sweep in vectorized torch). Information
    advances one cell per update in the worst case, so we iterate until the
    max change per pass falls below ``tol``, capped at
    ``max(Nx, Ny, Nz)`` passes as a safety bound.

    Args:
        phi_band_abs: (Nx, Ny, Nz) unsigned distance on band cells, ignored
            (can be any value) elsewhere.
        band_mask_3d: (Nx, Ny, Nz) bool. Band cells are the pinned boundary.
        x_coords, y_coords, z_coords: 1D tensors of cell coordinates per axis.
            Spacings are derived per-cell from consecutive differences.
        n_passes: if given, fixed number of full 8-sweep passes (no convergence
            check). If None (default), iterate until converged.
        inf_sentinel: initial value for non-band cells.
        tol: convergence tolerance on max |u_new - u_old| across a full pass.

    Returns:
        u: (Nx, Ny, Nz) float32 unsigned distance tensor.
    """
    u = torch.where(band_mask_3d, phi_band_abs, torch.full_like(phi_band_abs, inf_sentinel))
    pinned = band_mask_3d

    hx_neg, hx_pos = _per_cell_spacing(x_coords, inf_sentinel)
    hy_neg, hy_pos = _per_cell_spacing(y_coords, inf_sentinel)
    hz_neg, hz_pos = _per_cell_spacing(z_coords, inf_sentinel)

    Nx, Ny, Nz = u.shape
    max_passes = n_passes if n_passes is not None else max(Nx, Ny, Nz) + 2

    for _pass in range(max_passes):
        u_prev = u
        for sx, sy, sz in [
            (+1, +1, +1), (-1, +1, +1), (+1, -1, +1), (+1, +1, -1),
            (-1, -1, +1), (-1, +1, -1), (+1, -1, -1), (-1, -1, -1),
        ]:
            u = _godunov_update(
                u, pinned,
                hx_neg, hx_pos, hy_neg, hy_pos, hz_neg, hz_pos,
                sx, sy, sz, inf_sentinel,
            )
        if n_passes is None:
            # Only sample convergence on finite cells to avoid inf-inf = nan.
            finite = u < inf_sentinel * 0.5
            if finite.any():
                delta = (u[finite] - u_prev[finite]).abs().max().item()
                if delta < tol:
                    break

    return u


def _per_cell_spacing(
    coord: torch.Tensor, inf_sentinel: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-cell upwind spacings along one axis.

    ``h_neg[i] = coord[i] - coord[i-1]`` (distance to the i-1 neighbor);
    ``h_pos[i] = coord[i+1] - coord[i]`` (distance to the i+1 neighbor).
    Out-of-range ends are filled with ``inf_sentinel`` so the associated
    neighbor direction (also ``inf_sentinel``) is never selected.
    """
    if coord.numel() < 2:
        return (
            torch.full_like(coord, inf_sentinel),
            torch.full_like(coord, inf_sentinel),
        )
    d = (coord[1:] - coord[:-1]).abs()
    pad = torch.full((1,), inf_sentinel, dtype=coord.dtype, device=coord.device)
    h_neg = torch.cat([pad, d])
    h_pos = torch.cat([d, pad])
    return h_neg, h_pos


def _godunov_update(
    u: torch.Tensor,
    pinned: torch.Tensor,
    hx_neg: torch.Tensor, hx_pos: torch.Tensor,
    hy_neg: torch.Tensor, hy_pos: torch.Tensor,
    hz_neg: torch.Tensor, hz_pos: torch.Tensor,
    sx: int, sy: int, sz: int,
    inf_sentinel: float,
) -> torch.Tensor:
    """Apply one Godunov eikonal update over the whole grid.

    For each axis, pick the smaller of the two neighbors (strict Godunov
    upwind) and the spacing that goes with it — this is the non-uniform-grid
    generalization of "min of both neighbors with scalar h".
    """
    _ = (sx, sy, sz)  # signature parity; strict-Godunov picks the min neighbor.

    big = torch.full_like(u, inf_sentinel)

    # x axis: neighbor selection + matching per-cell spacing
    nx_neg = torch.cat([big[:1, :, :], u[:-1, :, :]], dim=0)
    nx_pos = torch.cat([u[1:, :, :], big[:1, :, :]], dim=0)
    use_neg_x = nx_neg < nx_pos
    a = torch.where(use_neg_x, nx_neg, nx_pos)
    ha = torch.where(use_neg_x, hx_neg.view(-1, 1, 1), hx_pos.view(-1, 1, 1))

    ny_neg = torch.cat([big[:, :1, :], u[:, :-1, :]], dim=1)
    ny_pos = torch.cat([u[:, 1:, :], big[:, :1, :]], dim=1)
    use_neg_y = ny_neg < ny_pos
    b = torch.where(use_neg_y, ny_neg, ny_pos)
    hb = torch.where(use_neg_y, hy_neg.view(1, -1, 1), hy_pos.view(1, -1, 1))

    nz_neg = torch.cat([big[:, :, :1], u[:, :, :-1]], dim=2)
    nz_pos = torch.cat([u[:, :, 1:], big[:, :, :1]], dim=2)
    use_neg_z = nz_neg < nz_pos
    c = torch.where(use_neg_z, nz_neg, nz_pos)
    hc = torch.where(use_neg_z, hz_neg.view(1, 1, -1), hz_pos.view(1, 1, -1))

    new_u = _solve_eikonal_3d(a, b, c, ha * ha, hb * hb, hc * hc)
    new_u = torch.minimum(u, new_u)
    new_u = torch.where(pinned, u, new_u)
    return new_u


def _solve_eikonal_3d(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    ha2: torch.Tensor, hb2: torch.Tensor, hc2: torch.Tensor,
) -> torch.Tensor:
    """Solve (u-a)^2/ha2 + (u-b)^2/hb2 + (u-c)^2/hc2 = 1 for u>=max(a,b,c).

    Includes degenerate 1D and 2D cases when some neighbors are +inf. All
    ops are elementwise. ``ha2/hb2/hc2`` broadcast over the cell grid.
    """
    # Broadcast per-axis spacings up to full 3D so swaps below work uniformly.
    ha2 = torch.broadcast_to(ha2, a.shape).contiguous()
    hb2 = torch.broadcast_to(hb2, b.shape).contiguous()
    hc2 = torch.broadcast_to(hc2, c.shape).contiguous()

    # Sort so a ≤ b ≤ c.
    # Swap a/b if a > b.
    swap_ab = a > b
    a, b = torch.where(swap_ab, b, a), torch.where(swap_ab, a, b)
    ha2, hb2 = torch.where(swap_ab, hb2, ha2), torch.where(swap_ab, ha2, hb2)
    # Swap b/c if b > c.
    swap_bc = b > c
    b, c = torch.where(swap_bc, c, b), torch.where(swap_bc, b, c)
    hb2, hc2 = torch.where(swap_bc, hc2, hb2), torch.where(swap_bc, hb2, hc2)
    # Swap a/b again (after bc swap b may now be < a).
    swap_ab2 = a > b
    a, b = torch.where(swap_ab2, b, a), torch.where(swap_ab2, a, b)
    ha2, hb2 = torch.where(swap_ab2, hb2, ha2), torch.where(swap_ab2, ha2, hb2)

    # 1D trial: u = a + h_a
    u1 = a + torch.sqrt(ha2)

    # 2D trial: use a and b with spacings ha2, hb2.
    # Solve (u-a)^2/ha2 + (u-b)^2/hb2 = 1  =>  (hb2 + ha2) u^2 - 2(hb2*a + ha2*b) u + (hb2*a^2 + ha2*b^2 - ha2*hb2) = 0
    A2 = ha2 + hb2
    B2 = ha2 * b + hb2 * a
    C2 = hb2 * a * a + ha2 * b * b - ha2 * hb2
    disc2 = B2 * B2 - A2 * C2
    disc2_ok = disc2 >= 0
    u2 = (B2 + torch.sqrt(torch.clamp(disc2, min=0.0))) / A2
    use_2d = (u1 > b) & disc2_ok

    # 3D trial: all three.
    # (u-a)^2/ha2 + (u-b)^2/hb2 + (u-c)^2/hc2 = 1
    A3 = hb2 * hc2 + ha2 * hc2 + ha2 * hb2
    B3 = hb2 * hc2 * a + ha2 * hc2 * b + ha2 * hb2 * c
    C3 = hb2 * hc2 * a * a + ha2 * hc2 * b * b + ha2 * hb2 * c * c - ha2 * hb2 * hc2
    disc3 = B3 * B3 - A3 * C3
    disc3_ok = disc3 >= 0
    u3 = (B3 + torch.sqrt(torch.clamp(disc3, min=0.0))) / A3

    # If 2D trial's u2 > c, try 3D.
    use_3d = use_2d & (u2 > c) & disc3_ok

    u = torch.where(use_3d, u3, torch.where(use_2d, u2, u1))
    return u
