#!/usr/bin/env python3
"""Benchmark xSDF (LBVH + Barill FWN) vs trimesh on the Stanford bunny + dragon.

Auto-downloads the meshes from the Stanford 3D Scanning Repository on first run
into ``meshes/`` (gitignored). Runs both backends head-to-head on the bunny and
xSDF only on the full ~870k-tri dragon (trimesh would take ~30 min there).
Prints a markdown speed table, a sign-agreement / max|Δφ| accuracy summary for
the bunny, and saves a side-by-side mid-z slice render to
``docs/stanford_bench.png``.

Usage
-----
    python tests/bench_stanford.py                        # full showcase run
    python tests/bench_stanford.py --bunny-n 64 --dragon-n 96   # smoke test
    python tests/bench_stanford.py --device cpu           # force CPU
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import trimesh.repair  # noqa: F401  # ensure repair submodule is loaded


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
MESH_DIR = REPO / "meshes"
DOCS_DIR = REPO / "docs"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(SRC))
import stretch_helper  # noqa: E402

torch_sdf = _load_module("torch_meshSDF", SRC / "torch-meshSDF.py")
trimesh_sdf = _load_module("trimesh_meshSDF", SRC / "trimesh-meshSDF.py")


ASSETS = {
    "bunny": dict(
        url="http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
        member="bunny/reconstruction/bun_zipper.ply",
        local="bun_zipper.ply",
    ),
    "dragon": dict(
        url="http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
        member="dragon_recon/dragon_vrip.ply",
        local="dragon_vrip.ply",
    ),
}


def fetch(name: str) -> Path:
    cfg = ASSETS[name]
    target = MESH_DIR / cfg["local"]
    if target.exists():
        return target
    MESH_DIR.mkdir(exist_ok=True)
    tarball = MESH_DIR / f"{name}.tar.gz"
    if not tarball.exists():
        print(f"[fetch] downloading {cfg['url']}")
        urllib.request.urlretrieve(cfg["url"], tarball)
    print(f"[fetch] extracting {cfg['member']}")
    with tarfile.open(tarball) as t:
        t.extract(cfg["member"], path=MESH_DIR)
    extracted = MESH_DIR / cfg["member"]
    extracted.rename(target)
    return target


def load_and_repair(path: Path):
    mesh = trimesh.load(str(path), process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight and len(mesh.faces) < 250_000:
        # Trimesh's fill_holes only closes triangle/quad-sized holes. For
        # large openings (e.g. the bunny's open bottom scan hole), fall back
        # to pymeshfix (Sven Behnke's MeshFix). pymeshfix segfaults on very
        # large meshes (~ >250k tris), so we gate it; meshes that big tend
        # to be non-manifold artifacts rather than open holes anyway and
        # don't trigger the flood-fill leak.
        import pymeshfix
        fixer = pymeshfix.MeshFix(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
        )
        fixer.repair()
        mesh = trimesh.Trimesh(vertices=fixer.v, faces=fixer.f, process=True)
        trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        print(f"[load_and_repair] WARN: mesh non-watertight after repair "
              f"(F={len(mesh.faces)}); flood-fill may misbehave near holes.")
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int64)
    return mesh, V, F


def uniform_grid(mesh: trimesh.Trimesh, n: int, pad_frac: float = 0.10):
    bb_min, bb_max = mesh.bounds[0], mesh.bounds[1]
    pad = pad_frac * float((bb_max - bb_min).max())
    lo = bb_min - pad
    hi = bb_max + pad
    extent = hi - lo
    max_extent = float(extent.max())
    coords = []
    for axis in range(3):
        n_axis = max(2, int(round(n * extent[axis] / max_extent)))
        dx_min = (hi[axis] - lo[axis]) / n_axis
        c = stretch_helper.geom_coords(
            float(lo[axis]), float(hi[axis]),
            center=float(0.5 * (lo[axis] + hi[axis])),
            dx_min=float(dx_min),
            r_max=1.0,
        )
        coords.append(np.asarray(c, dtype=np.float32))
    return tuple(coords)


def run_xsdf(V, F, coords, device):
    t0 = time.perf_counter()
    res = torch_sdf.mesh_to_sdf_torch_v2(
        V, F, coords[0], coords[1], coords[2],
        device=device, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    phi = np.asarray(res.phi)
    return elapsed, phi


def run_trimesh(mesh, coords):
    t0 = time.perf_counter()
    sdf, _, _ = trimesh_sdf.mesh_to_sdf_trimesh(
        mesh, coords[0], coords[1], coords[2]
    )
    return time.perf_counter() - t0, sdf


def accuracy(phi_x, phi_t):
    diff = np.abs(phi_x - phi_t)
    sign_agree = float((np.sign(phi_x) == np.sign(phi_t)).mean())
    return float(diff.max()), float(diff.mean()), sign_agree


def slice_plot(out_path: Path, phi_x, phi_t, coords, title: str):
    out_path.parent.mkdir(exist_ok=True)
    z_idx = phi_x.shape[2] // 2
    extent = [float(coords[0][0]), float(coords[0][-1]),
              float(coords[1][0]), float(coords[1][-1])]
    sx = phi_x[:, :, z_idx].T
    st = phi_t[:, :, z_idx].T
    vmax = float(max(np.abs(sx).max(), np.abs(st).max()))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, s, lab in zip(axs, [sx, st], ["xSDF (LBVH+FWN)", "trimesh"]):
        ax.imshow(s, origin="lower", extent=extent, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax)
        ax.contour(s, levels=[0], colors="k", linewidths=0.8, extent=extent)
        ax.set_title(lab)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axs[0].set_ylabel("y")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bunny-n", type=int, default=128,
                   help="voxels along longest axis for bunny grid (default 128)")
    p.add_argument("--dragon-n", type=int, default=256,
                   help="voxels along longest axis for dragon grid (default 256)")
    p.add_argument("--device", default=None,
                   help="torch device override: 'mps' / 'cuda' / 'cpu' (default auto)")
    args = p.parse_args()

    dev = torch_sdf.pick_device(args.device)
    dev_str = str(dev)
    print(f"[device] {dev_str}")

    rows = []

    print("\n=== Bunny ===")
    bunny_path = fetch("bunny")
    mesh_b, V_b, F_b = load_and_repair(bunny_path)
    coords_b = uniform_grid(mesh_b, args.bunny_n)
    voxels_b = coords_b[0].size * coords_b[1].size * coords_b[2].size
    print(f"[bunny] tris={F_b.shape[0]:,}  voxels={voxels_b:,} "
          f"({coords_b[0].size}×{coords_b[1].size}×{coords_b[2].size})")

    print(f"[bunny] running xSDF on {dev_str}...")
    t_x, phi_x = run_xsdf(V_b, F_b, coords_b, dev_str)
    print(f"[bunny] xSDF: {t_x:.2f}s  phi=[{phi_x.min():.4f}, {phi_x.max():.4f}]")

    print("[bunny] running trimesh (CPU)...")
    t_t, phi_t = run_trimesh(mesh_b, coords_b)
    print(f"[bunny] trimesh: {t_t:.2f}s")

    max_d, mean_d, sign_pct = accuracy(phi_x, phi_t)
    print(f"[bunny] accuracy: max|Δφ|={max_d:.3e} "
          f"mean|Δφ|={mean_d:.3e} "
          f"sign agreement={100*sign_pct:.2f}%")

    slice_path = DOCS_DIR / "stanford_bench.png"
    slice_plot(slice_path, phi_x, phi_t, coords_b, "Stanford bunny  —  mid-z slice")
    print(f"[bunny] saved slice -> {slice_path}")

    rows.append(("bunny", F_b.shape[0], voxels_b, dev_str,
                 f"{t_x:.2f}s", f"{t_t:.2f}s", f"{t_t/t_x:.1f}×"))

    print("\n=== Dragon ===")
    dragon_path = fetch("dragon")
    mesh_d, V_d, F_d = load_and_repair(dragon_path)
    coords_d = uniform_grid(mesh_d, args.dragon_n)
    voxels_d = coords_d[0].size * coords_d[1].size * coords_d[2].size
    print(f"[dragon] tris={F_d.shape[0]:,}  voxels={voxels_d:,} "
          f"({coords_d[0].size}×{coords_d[1].size}×{coords_d[2].size})")

    print(f"[dragon] running xSDF on {dev_str}...")
    t_x_d, _ = run_xsdf(V_d, F_d, coords_d, dev_str)
    print(f"[dragon] xSDF: {t_x_d:.2f}s")

    rows.append(("dragon", F_d.shape[0], voxels_d, dev_str,
                 f"{t_x_d:.2f}s", "n/a", "—"))

    print()
    print("| mesh   | triangles | voxels     | device | xSDF   | trimesh | speedup |")
    print("|--------|----------:|-----------:|--------|-------:|--------:|--------:|")
    for r in rows:
        print(f"| {r[0]:<6} | {r[1]:>9,} | {r[2]:>10,} | {r[3]:<6} | "
              f"{r[4]:>6} | {r[5]:>7} | {r[6]:>7} |")
    print()
    print(f"[accuracy bunny] max|Δφ|={max_d:.3e}  "
          f"mean|Δφ|={mean_d:.3e}  "
          f"sign agreement={100*sign_pct:.2f}%")
    print(f"[saved] {slice_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
