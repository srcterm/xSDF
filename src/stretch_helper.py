#!/usr/bin/env python3
"""
stretch_helper.py  —  Unified geometric/uniform grid coordinate builder.

Features
--------
- Generate 1D cell-center coordinates with geometric stretching or uniform spacing
- Geometric stretching: clustering around a focus point with growth ratio r_max > 1.0
- Uniform spacing: simply set r_max = 1.0
- Specify domain [a,b], center point, minimum spacing (dx_min), and growth ratio (r_max)
- Optional matplotlib preview of spacing distribution

Usage
-----
# Geometric stretching (r_max > 1.0)
coords = geom_coords(a=-5.0, b=7.0, center=0.0, dx_min=0.05, r_max=1.075)

# Uniform grid (r_max = 1.0)
coords = geom_coords(a=-5.0, b=7.0, center=0.0, dx_min=0.05, r_max=1.0)
"""
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Unified geometric/uniform grid builder
def geom_coords(a: float, b: float, center: float, dx_min: float, r_max: float) -> np.ndarray:
    """Return coordinates with geometric growth from center, or uniform if r_max=1.0.

    Args:
        a, b: Domain bounds
        center: Focus point for clustering (ignored if r_max ≈ 1.0)
        dx_min: Minimum cell spacing (at center for stretched, everywhere for uniform)
        r_max: Maximum growth ratio. r_max=1.0 produces uniform spacing.

    Returns:
        1D array of cell-center coordinates
    """
    assert r_max >= 1.0, "r_max must be >= 1.0"

    # Special case: uniform grid when r_max ≈ 1.0
    if abs(r_max - 1.0) < 1e-6:
        L = b - a
        n = int(math.ceil(L / dx_min))
        # Generate uniform cell centers
        coords = a + (np.arange(n) + 0.5) * dx_min
        return coords.astype(np.float64)

    # Geometric stretching (r_max > 1.0)
    L_left  = center - a
    L_right = b - center

    def _side(L, sign):
        """Return coords extending *outward* from centre by cumulative geometric spacing."""
        if L <= 1e-12:
            return np.array([], dtype=np.float64)
        n = math.ceil(math.log(1 + (r_max - 1) * L / dx_min) / math.log(r_max))
        spacing = dx_min * r_max ** np.arange(n)  # smallest first, grow outward
        offset = np.cumsum(spacing)
        return center + sign * offset

    left  = _side(L_left,  -1)  # negative direction
    right = _side(L_right, +1)  # positive direction
    coords = np.concatenate((left[::-1], [center], right)).astype(np.float64)
    return coords

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


def piecewise_coords(segments, dx_target, r_target):
    """Build 1D cell centers from ordered segments.

    Each segment: {"type": "CONSTANT"|"INCREASING"|"DECREASING",
                   "lower_bound": float, "upper_bound": float}

    Cell counts are auto-computed:
      CONSTANT  — cells = ceil(L / dx_target)
      INCREASING/DECREASING — N from geometric series inversion using r_target,
                               then Newton-solve exact ratio for N cells to span L.
    """
    # Validate neighbor constraints
    for i, s in enumerate(segments):
        if s["type"] == "INCREASING":
            assert i > 0 and segments[i-1]["type"] == "CONSTANT", \
                f"Segment {i}: INCREASING requires CONSTANT to its left"
        if s["type"] == "DECREASING":
            assert i < len(segments)-1 and segments[i+1]["type"] == "CONSTANT", \
                f"Segment {i}: DECREASING requires CONSTANT to its right"

    # Pass 1: CONSTANT segments
    faces, dx = [None]*len(segments), {}
    for i, s in enumerate(segments):
        if s["type"] == "CONSTANT":
            L = s["upper_bound"] - s["lower_bound"]
            n = math.ceil(L / dx_target)
            faces[i] = np.linspace(s["lower_bound"], s["upper_bound"], n + 1)
            dx[i] = L / n

    # Pass 2: stretched segments
    for i, s in enumerate(segments):
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

    # Concatenate, strip duplicate boundary faces between segments
    all_f = np.concatenate([f[:-1] if i < len(segments)-1 else f
                            for i, f in enumerate(faces)])
    centers = 0.5 * (all_f[1:] + all_f[:-1])
    assert np.all(np.diff(centers) > 0), "piecewise_coords produced non-monotonic centers"
    return centers.astype(np.float64)


__all__ = ["geom_coords", "piecewise_coords", "metrics"]