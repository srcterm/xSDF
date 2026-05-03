'''
Main script for generation of a signed distance field (SDF) of a geometry
(eg .stl,.ply, etc.) and surrounding domain. The main feature is GPU acceleration
using torch-meshSDF backend, which utilizes a reformulation of the SDF in PyTorch
tensors for fast computation on large grids (see description in file/readme.md). 

A trimesh backend is also available as a fallback option but is significantly slower for large grids or complex geometries
requiring high resolution meshes. This was why the gpu accelerated torch-meshSDF.py
was created.

The option to use geometric stretching for finer resolution of the geometry and coarsening of the background domain is 
possible in the config. file, sdf_config.json.
'''

import numpy as np
import h5py
import time
import json
import sys
from typing import Dict, Optional
import importlib.util, os

from src import stretch_helper, plot_utils

# ---- Load torch SDF backend ----
_TORCH_SDF_FILE = os.path.join(os.path.dirname(__file__), "src", "torch-meshSDF.py")
_spec = importlib.util.spec_from_file_location("torch_meshSDF", _TORCH_SDF_FILE)
torch_meshSDF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(torch_meshSDF)

# ---- Load Trimesh SDF backend (fallback) ----
import trimesh
_TRIMESH_SDF_FILE = os.path.join(os.path.dirname(__file__), "src", "trimesh-meshSDF.py")
_spec2 = importlib.util.spec_from_file_location("trimesh_meshSDF", _TRIMESH_SDF_FILE)
trimesh_meshSDF = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(trimesh_meshSDF)


def create_cube_mesh(side_length=1.0, center=(0.0, 0.0, 0.0)):
    '''
    Create a cube mesh with specified side length and center.
    '''
    cube = trimesh.creation.box(extents=[side_length, side_length/1.5, side_length*2.0])
    cube.apply_translation(center)
    return cube


def create_cylinder_mesh(radius=1.0, height=2.0, scale=1.0,
                          translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0)):
    '''
    Create a cylinder mesh with specified radius, height, and center.
    '''
    cylinder = trimesh.creation.cylinder(radius=radius, height=height)
    cylinder.apply_scale(scale)
    rotation_matrix = trimesh.transformations.euler_matrix(
        np.radians(rotate[0]), np.radians(rotate[1]), np.radians(rotate[2]))
    cylinder.apply_transform(rotation_matrix)
    cylinder.apply_translation(translate)
    return cylinder


def load_stl_mesh(stl_path, scale=1.0, translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0)):
    """Load and transform STL mesh with positioning control."""
    mesh = trimesh.load(stl_path, process=True)

    # If trimesh returns a Scene (multiple bodies), concatenate into a single mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Apply transformations
    mesh.apply_scale(scale)
    rotation_matrix = trimesh.transformations.euler_matrix(
        np.radians(rotate[0]), np.radians(rotate[1]), np.radians(rotate[2])
    )
    mesh.apply_transform(rotation_matrix)
    mesh.apply_translation(translate)
    
    if not mesh.is_watertight:
        print("Warning: STL mesh is not watertight. Attempting to repair...")
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_inversion(mesh)
    
    print(f"Loaded STL: {len(mesh.faces)} faces, watertight: {mesh.is_watertight}")
    print(f"Bounds: {mesh.bounds}")

    if not mesh.is_watertight:
        print("WARNING: Mesh is still not watertight after repair. SDF sign may be unreliable.")
    
    # Ensure consistent winding for correct sign computation
    if not mesh.is_winding_consistent:
        try:
            trimesh.repair.fix_winding(mesh)
            print("Applied winding fix (trimesh.repair.fix_winding)")
        except Exception as e:
            print("Winding fix failed:", e)
    return mesh


def load_config(config_path: str) -> dict:
    """
    Load and process SDF configuration from JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Process grid stretching: fill in null values
    target_min_size = config['grid']['target_min_size']
    stretch_factor = config['grid']['stretch_factor']

    for axis in ['x', 'y', 'z']:
        if axis in config['grid']['stretch_axes']:
            ax_cfg = config['grid']['stretch_axes'][axis]
            if ax_cfg.get('type') == 'piecewise':
                ax_cfg.setdefault('dx_target', target_min_size)
                ax_cfg.setdefault('r_target', stretch_factor)
                continue
            # Inherit dx_min from target_min_size if null
            if ax_cfg.get('dx_min') is None:
                ax_cfg['dx_min'] = target_min_size
            # Inherit r_max from stretch_factor if null
            if ax_cfg.get('r_max') is None:
                ax_cfg['r_max'] = stretch_factor

    return config


def main(domain_bounds, save_name, target_voxel_size=0.5, geom='cube', geom_path=None,
          scale=1.0, translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0),
          method='trimesh',
          memory_budget_gb=None,
          # Torch SDF options (LBVH + Barill FWN pipeline)
          torch_device='mps',
          fwn_beta: float = 2.0,
          fwn_band_width_cells: float = 3.0,
          cos_theta_min: float = 0.8,
          # Grid coordinate options (non-uniform by default, set r_max=1.0 for uniform)
          stretch_axes: Optional[Dict[str, Dict]] = None,
          preview_stretch: bool = False
         ):
    """
    Generate signed distance field (SDF) for a watertight mesh.

    This function always builds coordinate arrays using geometric stretching.
    For uniform grids, simply set r_max=1.0 in stretch_axes (or leave as None for defaults).

    Args:
        domain_bounds: Dict with 'x', 'y', 'z' keys, each a [min, max] list
        save_name: Output H5 filename
        target_voxel_size: Default cell size (used if stretch_axes not provided)
        geom: 'cube' or 'stl'
        geom_path: Path to STL/PLY file if geom='stl'
        scale, translate, rotate: Geometry transformations
        method: 'torch' (GPU) or 'trimesh' (CPU) backend
        memory_budget_gb: Optional memory budget in GB (both backends).
                         If None, backends use their default memory management.
        torch_*: PyTorch backend options (only used if method='torch')
        stretch_axes: Dict specifying grid stretching per axis:
                     {'x': {'center': 0.0, 'dx_min': 0.05, 'r_max': 1.075}, ...}
                     Set r_max=1.0 for uniform spacing on any axis.
                     If None, uses uniform grid with target_voxel_size.
        preview_stretch: Show grid preview plot before computation

    Returns:
        sdf: 3D numpy array of signed distances
    """
    
    if geom=='cube':
        side_length = 1.0
        center = (0.0, 0.0, 0.0)
        mesh_ = create_cube_mesh(side_length=side_length, center=center)
    elif geom=='cylinder':
        radius = 10.0
        height = 100
        mesh_ = create_cylinder_mesh(radius=radius, height=height, 
                                     scale=scale, translate=translate, rotate=rotate)
    elif geom=='stl':
        mesh_ = load_stl_mesh(geom_path, scale, translate, rotate)

    print(f"Geom. mesh: {len(mesh_.faces)} faces, watertight: {mesh_.is_watertight}")
    print(f"Geom. bounds: {mesh_.bounds}")

    # -------- Grid generation: always build coordinate arrays --------
    # If stretch_axes is None, use default uniform config (r_max=1.0)
    if stretch_axes is None:
        stretch_axes = {}

    def _build_axis(ax_name: str):
        """Build coordinates for one axis using geometric stretching.
        For uniform grids, simply set r_max=1.0 in the configuration (default).
        """
        cfg = stretch_axes.get(ax_name, {})
        a, b = domain_bounds[ax_name]

        if cfg.get("type") == "piecewise":
            return stretch_helper.piecewise_coords(
                cfg["segments"], cfg["dx_target"], cfg["r_target"])

        # Extract parameters with sensible defaults
        center = cfg.get("center", 0.5 * (a + b))  # default to domain center (need to expose this later for control)
        dx_min = cfg.get("dx_min", target_voxel_size)  # default to target size
        r_max = cfg.get("r_max", 1.0)  # default to uniform (no stretching)

        # Always use geom_coords (handles both stretched and uniform via r_max)
        return stretch_helper.geom_coords(a, b, center, dx_min, r_max)

    # Always build coordinate arrays (uniform when r_max=1.0, stretched otherwise)
    x_coords = _build_axis('x')
    y_coords = _build_axis('y')
    z_coords = _build_axis('z')

    # ---------------- Calculate grid size and spacing ----------------
    # Face-centric convention: coord arrays are cell-face positions (length n_cells+1).
    gx, gy, gz = len(x_coords), len(y_coords), len(z_coords)
    total_pts = gx * gy * gz
    nx_cells, ny_cells, nz_cells = gx - 1, gy - 1, gz - 1
    total_cells = nx_cells * ny_cells * nz_cells

    # Calculate spacing info for display
    dx_min = min(np.diff(x_coords).min(), np.diff(y_coords).min(), np.diff(z_coords).min())
    dx_max = max(np.diff(x_coords).max(), np.diff(y_coords).max(), np.diff(z_coords).max())

    print(f"\n[Grid Size] {nx_cells}x{ny_cells}x{nz_cells} = {total_cells:,} cells "
          f"({gx}x{gy}x{gz} = {total_pts:,} face-vertex sample points)")
    print(f"[Spacing] Δmin={dx_min:.6g}, Δmax={dx_max:.6g}, ratio={dx_max/dx_min:.2f}x")

    # ------ Preview grid if requested ------
    if preview_stretch:
        try:
            should_proceed = plot_utils.preview_grid_coords(x_coords, y_coords, z_coords, mesh=mesh_)
            if not should_proceed:
                print("Script terminated by user.")
                return None
        except Exception as e:
            print(f"[preview] Failed to plot stretch preview: {e}")

    # ============ Unified backend selection: torch or trimesh ============
    print(f"\nComputing SDF using method='{method}'...")

    if method == 'torch':
        # PyTorch GPU-accelerated SDF computation
        if torch_meshSDF is None:
            raise ImportError("torch-meshSDF.py not found. Cannot use method='torch'.")
        print("Using PyTorch SDF backend (torch-meshSDF.py)...")
        V_np = np.asarray(mesh_.vertices, dtype=np.float32)
        F_np = np.asarray(mesh_.faces, dtype=np.int64)
        
        t_sdf = time.perf_counter() # Time the SDF evaluation only
        res = torch_meshSDF.mesh_to_sdf_torch_v2(
            V_np, F_np,
            x_coords, y_coords, z_coords,
            device=torch_device,
            fwn_beta=fwn_beta,
            fwn_band_width_cells=fwn_band_width_cells,
            cos_theta_min=cos_theta_min,
        )
        t_sdf = time.perf_counter() - t_sdf
        sdf = res.phi
        origin = res.origin
        print(f"\n >> [Total SDF compute] {t_sdf:.2f}s  (torch / {torch_device})")
    elif method == 'trimesh':
        # CPU trimesh SDF computation
        print("Using trimesh CPU backend...")
        t_sdf = time.perf_counter()
        sdf, origin, coord_dict = trimesh_meshSDF.mesh_to_sdf_trimesh(mesh_, x_coords, y_coords, z_coords, memory_budget_gb)
        t_sdf = time.perf_counter() - t_sdf
        print(f"\n >> [Total SDF compute] {t_sdf:.2f}s  (trimesh / cpu)")
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'torch' or 'trimesh'.")

    # Always create coord_dict for saving
    coord_dict = {
        "x_coords": x_coords.astype(np.float32),
        "y_coords": y_coords.astype(np.float32),
        "z_coords": z_coords.astype(np.float32)
    }

    # Print final stats
    print(f"\nFinal SDF stats: min={sdf.min():.6f}, max={sdf.max():.6f}, shape={sdf.shape}")
    print(f"Grid spacing: Δmin={dx_min:.6g}, Δmax={dx_max:.6g}, voxels={sdf.size:,}")
    print(f"Grid origin: {origin}")

    # Detect if grid is uniform (all spacings equal within tolerance)
    dx_x = np.diff(x_coords)
    dx_y = np.diff(y_coords)
    dx_z = np.diff(z_coords)
    is_uniform_x = np.allclose(dx_x, dx_x[0], rtol=1e-6)
    is_uniform_y = np.allclose(dx_y, dx_y[0], rtol=1e-6)
    is_uniform_z = np.allclose(dx_z, dx_z[0], rtol=1e-6)
    is_uniform = is_uniform_x and is_uniform_y and is_uniform_z

    # Save the SDF to .h5 file
    print(f"\nSaving SDF to {save_name}...")
    with h5py.File(save_name, "w") as f:
        f.create_dataset("levelset", data=sdf)
        f.create_dataset("origin", data=origin)
        # Always save coordinate arrays
        f.create_dataset("x_coords", data=coord_dict["x_coords"])
        f.create_dataset("y_coords", data=coord_dict["y_coords"])
        f.create_dataset("z_coords", data=coord_dict["z_coords"])
        # Always use nan for dx (downstream code computes from coordinates)
        f.create_dataset("dx", data=np.array([np.nan], dtype=np.float32))
        f.create_dataset("grid_size", data=sdf.shape)
        # Set grid type attribute based on actual spacing
        f.attrs["non_uniform_grid"] = not is_uniform
        f.attrs["description"] = f"SDF using {method} method"

    print(f"Done saving SDF ({'uniform' if is_uniform else 'non-uniform'} grid)")
    return sdf


if __name__ == "__main__":
    # Load configuration from JSON file <python xSDF.py [config.json]>
    # Default sdf_config.json if no args
    config_file = sys.argv[1] if len(sys.argv) > 1 else "sdf_config.json"

    print(f"Loading configuration from: {config_file}")
    cfg = load_config(config_file)

    # Run SDF computation.
    sdf = main(
        domain_bounds=cfg['domain']['bounds'],
        save_name=cfg['output']['save_name'],
        geom=cfg['geometry']['type'],
        geom_path=cfg['geometry'].get('path'),
        scale=cfg['geometry']['transformations']['scale'],
        translate=tuple(cfg['geometry']['transformations']['translate']),
        rotate=tuple(cfg['geometry']['transformations']['rotate']),
        stretch_axes=cfg['grid']['stretch_axes'],
        preview_stretch=cfg['grid']['preview_stretch'],
        method=cfg['backend']['method'],
        memory_budget_gb=cfg['backend']['memory_budget_gb'],
        torch_device=cfg['backend']['torch'].get('device', 'mps'),
        fwn_beta=cfg['backend']['torch'].get('fwn_beta', 2.0),
        fwn_band_width_cells=cfg['backend']['torch'].get('fwn_band_width_cells', 3.0),
        cos_theta_min=cfg['backend']['torch'].get('cos_theta_min', 0.8),
    )

    # Visualize the results
    if cfg['output']['visualize'] and sdf is not None:
        plot_utils.visualize(cfg['output']['save_name'])
    elif sdf is None:
        print("No SDF generated - skipping visualization.")