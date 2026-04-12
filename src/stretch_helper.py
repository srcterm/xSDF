#!/usr/bin/env python3
"""
stretch_helper.py  —  Unified geometric/uniform grid coordinate builder.

Features
--------
- Generate 1D cell-face coordinates with geometric stretching or uniform spacing
- Geometric stretching: clustering around a focus point with growth ratio r_max > 1.0
- Uniform spacing: simply set r_max = 1.0
- Specify domain [a,b], center point, minimum spacing (dx_min), and growth ratio (r_max)
- Optional matplotlib preview of spacing distribution

Output convention
-----------------
All builders return **cell-face** coordinates (length n+1 for n cells). The SDF is 
subsequently evaluated on the face-vertex grid formed by these 1D arrays.

Usage
-----
# Geometric stretching (r_max > 1.0)
faces = geom_coords(a=-5.0, b=7.0, center=0.0, dx_min=0.05, r_max=1.075)

# Uniform grid (r_max = 1.0)
faces = geom_coords(a=-5.0, b=7.0, center=0.0, dx_min=0.05, r_max=1.0)
"""
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Unified geometric/uniform grid builder
def geom_coords(a: float, b: float, center: float, dx_min: float, r_max: float) -> np.ndarray:
    """Return cell-face coordinates with geometric growth from center, or uniform if r_max=1.0.

    Args:
        a, b: Domain bounds (returned as first/last face positions)
        center: Focus point for clustering (ignored if r_max ≈ 1.0)
        dx_min: Minimum cell spacing (at center for stretched, everywhere for uniform)
        r_max: Maximum growth ratio. r_max=1.0 produces uniform spacing.

    Returns:
        1D array of cell-face coordinates (length n+1 for n cells), spanning [a, b].
    """
    assert r_max >= 1.0, "r_max must be >= 1.0"

    # Special case: uniform grid when r_max ≈ 1.0
    if abs(r_max - 1.0) < 1e-6:
        L = b - a
        n = max(1, int(math.ceil(L / dx_min)))
        # Uniform face positions exactly spanning [a, b]
        faces = np.linspace(a, b, n + 1)
        return faces.astype(np.float64)

    # Geometric stretching (r_max > 1.0) — build face positions outward from center.
    L_left = center - a
    L_right = b - center

    def _side(L, sign):
        """Return face offsets extending outward from centre by cumulative geometric spacing."""
        if L <= 1e-12:
            return np.array([], dtype=np.float64)
        n = math.ceil(math.log(1 + (r_max - 1) * L / dx_min) / math.log(r_max))
        spacing = dx_min * r_max ** np.arange(n)  # smallest first, grow outward
        offset = np.cumsum(spacing)
        return center + sign * offset

    left = _side(L_left, -1)   # faces on the negative side, farthest first after flip
    right = _side(L_right, +1)  # faces on the positive side
    # Assemble faces: [outermost left, ..., innermost left, center, innermost right, ..., outermost right]
    faces = np.concatenate((left[::-1], [center], right)).astype(np.float64)
    # Snap the outermost faces to the requested domain bounds to guarantee [a, b] is exact.
    if left.size > 0:
        faces[0] = a
    if right.size > 0:
        faces[-1] = b
    return faces

def metrics(x):
    dx = np.diff(x)
    # protect against zero/negative spacings to avoid divide-by-zero
    dx_safe = np.where(dx <= 1e-15, 1e-15, dx)
    if dx_safe.size > 1:
        r_neighbor = np.maximum(dx_safe[1:] / dx_safe[:-1], dx_safe[:-1] / dx_safe[1:])
        r_nb_max = r_neighbor.max()
    else:
        r_nb_max = 1.0
    return dict(dx_min=dx_safe.min(),
                dx_max=dx_safe.max(),
                ratio_minmax=dx_safe.min()/dx_safe.max(),
                r_neighbor_max=r_nb_max)

# --- Piecewise multi-segment grid builder ---

def _solve_ratio(dx_fine, L, N, r0=1.1, tol=1e-9):
    """Newton-solve geometric ratio: dx_fine * r*(r^N-1)/(r-1) = L."""
    C = L / dx_fine
    assert C > N, f"Segment too short: L={L:.4g} < {N}*dx={N*dx_fine:.4g}"
    r = r0
    for _ in range(100):
        f = C - r * (r**N - 1) / (r - 1)
        if abs(f) < tol:
            assert r > 1.0, f"Ratio {r:.4f} <= 1.0"
            return r
        df = -1.0 / (r - 1)**2 * (N * r**(N+1) - (N+1) * r**N + 1)
        r -= f / df
    raise RuntimeError(f"Ratio did not converge (res={abs(f):.2e})")


def piecewise_coords(segments, dx_target, r_target, snap_bounds=True, snap_tol=1e-6):
    """Build 1D cell-face positions from ordered segments.

    Each segment: {"type": "CONSTANT"|"INCREASING"|"DECREASING",
                   "lower_bound": float, "upper_bound": float}

    Cell counts are auto-computed:
      CONSTANT  — cells = round(L / dx_target); requires L to be (approximately)
                  an integer multiple of dx_target so dx_fine == dx_target exactly.
                  If ``snap_bounds`` is True (default), segment bounds are adjusted
                  silently so that ``dx = dx_target`` holds; neighboring segments
                  are updated to match.
      INCREASING/DECREASING — N from geometric series inversion using r_target,
                               then Newton-solve exact ratio for N cells to span L.

    Args:
        segments: list of segment dicts (ordered lower→upper along the axis)
        dx_target: the fine cell size that CONSTANT regions must produce exactly
        r_target: geometric growth ratio for stretched regions
        snap_bounds: if True (default), auto-adjust CONSTANT segment bounds so that
                     L is an exact integer multiple of dx_target; if False, raise
                     ValueError when a mismatch exceeds ``snap_tol``.
        snap_tol: tolerance (in cells) for deciding whether snapping is required.

    Returns:
        1D array of cell-face coordinates (length total_cells+1) for this axis.
    """
    # --- Make a working copy so we can safely mutate bounds (for auto-snap)
    segs = [dict(s) for s in segments]

    # --- Validate neighbor constraints (unchanged)
    for i, s in enumerate(segs):
        if s["type"] == "INCREASING":
            assert i > 0 and segs[i-1]["type"] == "CONSTANT", \
                f"Segment {i}: INCREASING requires CONSTANT to its left"
        if s["type"] == "DECREASING":
            assert i < len(segs)-1 and segs[i+1]["type"] == "CONSTANT", \
                f"Segment {i}: DECREASING requires CONSTANT to its right"

    # --- Enforce exact dx_target in CONSTANT segments.
    # CONSTANT segments are the "anchors"; their bounds determine the fine dx.
    # If the user-specified length L isn't an integer multiple of dx_target,
    # either raise (snap_bounds=False) or snap the upper_bound to the nearest
    # multiple and propagate to the adjacent INCREASING segment's lower_bound.
    for i, s in enumerate(segs):
        if s["type"] != "CONSTANT":
            continue
        a = s["lower_bound"]
        b = s["upper_bound"]
        L = b - a
        n_float = L / dx_target
        n = max(1, int(round(n_float)))
        if abs(n_float - n) > snap_tol:
            if not snap_bounds:
                raise ValueError(
                    f"Segment {i} CONSTANT region has length {L:.6g} which is not an "
                    f"integer multiple of dx_target={dx_target:.6g} (nearest {n} cells "
                    f"would require L={n*dx_target:.6g}). Adjust the segment bounds or "
                    f"dx_target, or pass snap_bounds=True to auto-adjust."
                )
            new_b = a + n * dx_target
            print(f"[piecewise_coords] snapping segment {i} CONSTANT upper_bound "
                  f"{b:.6g} -> {new_b:.6g} (delta={new_b-b:+.3e}) to enforce "
                  f"dx={dx_target:.6g} exactly over {n} cells.")
            s["upper_bound"] = new_b
            # Propagate to the neighbor on the right (if any) so faces still match.
            if i + 1 < len(segs):
                segs[i+1]["lower_bound"] = new_b

    # --- Pass 1: CONSTANT segments (now exact)
    faces, dx = [None]*len(segs), {}
    for i, s in enumerate(segs):
        if s["type"] == "CONSTANT":
            L = s["upper_bound"] - s["lower_bound"]
            n = max(1, int(round(L / dx_target)))
            faces[i] = np.linspace(s["lower_bound"], s["upper_bound"], n + 1)
            dx[i] = dx_target  

    # --- Pass 2: stretched segments
    for i, s in enumerate(segs):
        if s["type"] == "CONSTANT":
            continue
        d = dx[i-1] if s["type"] == "INCREASING" else dx[i+1]
        a, b = s["lower_bound"], s["upper_bound"]
        L = b - a
        n = math.ceil(math.log(1 + L * (r_target - 1) / (d * r_target)) / math.log(r_target))
        r = _solve_ratio(d, L, n)
        sz = d * r ** np.arange(1, n+1)
        if s["type"] == "DECREASING":
            f = np.flip(b - np.cumsum(sz)); f[0] = a; f = np.concatenate([f, [b]])
        else:
            f = a + np.cumsum(sz); f[-1] = b; f = np.concatenate([[a], f])
        faces[i] = f

    # --- Concat (get rid of duplicate boundary faces between segments)
    all_f = np.concatenate([f[:-1] if i < len(segs)-1 else f
                            for i, f in enumerate(faces)])
    assert np.all(np.diff(all_f) > 0), "piecewise_coords produced non-monotonic faces"
    return all_f.astype(np.float64)


__all__ = ["geom_coords", "piecewise_coords", "metrics"]
