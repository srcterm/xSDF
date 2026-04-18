"""
plot_utils.py  —  Visualization utilities for SDF mesher.

Functions for visualizing:
- Stretched grid layouts (matplotlib scatter plots)
- Geometry and domain boundaries (PyVista 3D)
- Computed SDF fields (PyVista 3D with contours)
- SDF value distributions (matplotlib histograms)
- Grid spacing analysis (matplotlib)
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pyvista as pv


def preview_grid_coords(x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray, mesh=None) -> bool:
    """Line-plots of cell faces for a stretched grid.

    Each face is drawn as a single thin line (axvline/axhline) so that local cell
    spacing is visually unambiguous, regardless of marker size or panel aspect
    ratio. If *mesh* (trimesh.Trimesh) is provided, overlay its vertex projection
    for context. Returns True if user wants to proceed, False to stop.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    line_kw = dict(color='C0', linewidth=0.4, alpha=0.8)

    def _draw_grid(ax, h_coords, v_coords, h_range, v_range):
        for xv in h_coords:
            ax.plot([xv, xv], v_range, **line_kw)
        for yv in v_coords:
            ax.plot(h_range, [yv, yv], **line_kw)
        ax.set_xlim(h_range)
        ax.set_ylim(v_range)

    x_range = (float(x_coords[0]), float(x_coords[-1]))
    y_range = (float(y_coords[0]), float(y_coords[-1]))
    z_range = (float(z_coords[0]), float(z_coords[-1]))

    # ------ XY plane ------
    _draw_grid(axes[0], x_coords, y_coords, x_range, y_range)
    if mesh is not None:
        verts = mesh.vertices
        axes[0].scatter(verts[:, 0], verts[:, 1], s=1, alpha=0.3, color='gray', label='Geometry verts')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'XY grid faces ({len(x_coords)}×{len(y_coords)})')
    axes[0].set_aspect('equal', adjustable='box')

    # ------ YZ plane ------
    _draw_grid(axes[1], y_coords, z_coords, y_range, z_range)
    if mesh is not None:
        verts = mesh.vertices
        axes[1].scatter(verts[:, 1], verts[:, 2], s=1, alpha=0.3, color='gray', label='Geometry verts')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('z')
    axes[1].set_title(f'YZ grid faces ({len(y_coords)}×{len(z_coords)})')
    axes[1].set_aspect('equal', adjustable='box')

    # ------ XZ plane ------
    _draw_grid(axes[2], x_coords, z_coords, x_range, z_range)
    if mesh is not None:
        verts = mesh.vertices
        axes[2].scatter(verts[:, 0], verts[:, 2], s=1, alpha=0.3, color='gray', label='Geometry verts')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('z')
    axes[2].set_title(f'XZ grid faces ({len(x_coords)}×{len(z_coords)})')
    axes[2].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

    plt.close('all')

    # User prompt
    while True:
        response = input("\nGrid preview shown. Does the mesh and domain look correct? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("Proceeding with SDF calculation...")
            return True
        elif response in ['n', 'no']:
            print("Stopping script...")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def visualize_domain_and_geometry(mesh, domain_bounds):
    """Visualize the domain bounds and loaded geometry before SDF computation."""
    print("Visualizing domain and geometry...")

    # Create plotter
    plotter = pv.Plotter()

    # Convert trimesh to pyvista mesh
    vertices = mesh.vertices
    faces = mesh.faces
    # PyVista expects faces with count prefix
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()
    pv_mesh = pv.PolyData(vertices, pv_faces)

    # Add the geometry mesh
    plotter.add_mesh(pv_mesh, color='red', opacity=0.8, label='Geometry')

    # Create domain boundary box
    domain_box = pv.Box(bounds=[
        domain_bounds['x'][0], domain_bounds['x'][1],  # x bounds
        domain_bounds['y'][0], domain_bounds['y'][1],  # y bounds
        domain_bounds['z'][0], domain_bounds['z'][1]   # z bounds
    ])
    plotter.add_mesh(domain_box, style='wireframe', color='blue', line_width=2, label='Domain Boundary')

    # Add coordinate axes
    plotter.add_axes(label_size=(0.1, 0.1))

    # Add legend
    plotter.add_legend()

    # Set camera position for good view
    plotter.camera_position = 'iso'

    # Add title
    plotter.add_title("Domain and Geometry Preview", font_size=16)

    # Show geometry info in the plot
    geom_bounds = mesh.bounds
    info_text = f"Geometry bounds:\nX: [{geom_bounds[0][0]:.2f}, {geom_bounds[1][0]:.2f}]\n"
    info_text += f"Y: [{geom_bounds[0][1]:.2f}, {geom_bounds[1][1]:.2f}]\n"
    info_text += f"Z: [{geom_bounds[0][2]:.2f}, {geom_bounds[1][2]:.2f}]\n\n"
    info_text += f"Domain bounds:\nX: {domain_bounds['x']}\n"
    info_text += f"Y: {domain_bounds['y']}\n"
    info_text += f"Z: {domain_bounds['z']}"

    plotter.add_text(info_text, position='upper_left', font_size=10)

    print(f"Geometry bounds: {geom_bounds}")
    print(f"Domain bounds: X{domain_bounds['x']}, Y{domain_bounds['y']}, Z{domain_bounds['z']}")

    plotter.show()


def visualize(h5file):
    """Visualize the generated SDF from H5 file with 3D rendering."""
    with h5py.File(h5file) as f:
        phi = f["levelset"][()]
        grid_shape = f["grid_size"][()]
        non_uniform = bool(f.attrs.get("non_uniform_grid", False))

        if non_uniform or np.isnan(f["dx"][()][0]):
            # --- Non‑uniform grid (Rectilinear) ---
            x = f["x_coords"][()]
            y = f["y_coords"][()]
            z = f["z_coords"][()]
            grid = pv.RectilinearGrid(x, y, z)
            origin = np.array([x[0], y[0], z[0]], dtype=np.float32)
            dx_min = float(min(np.diff(x).min(), np.diff(y).min(), np.diff(z).min()))
            dx_info = f"non‑uniform (Δmin≈{dx_min:.4g})"
        else:
            # --- Uniform grid (ImageData) ---
            origin = f["origin"][()]
            dx = float(f["dx"][()])
            grid = pv.ImageData(dimensions=grid_shape, origin=origin, spacing=(dx, dx, dx))
            dx_info = f"uniform (dx={dx:.4g})"

    # Attach data
    grid["phi"] = phi.flatten(order="F")

    # Print stats
    print("SDF visualization stats:")
    print(f"  SDF range: [{phi.min():.6f}, {phi.max():.6f}]")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Origin: {origin}")
    print(f"  Spacing: {dx_info}")
    print(f"  Values near zero: {np.sum(np.abs(phi) < 0.01):,}")

    # Create plotter
    plotter = pv.Plotter()

    # Try to add the zero‑level surface (geometry boundary)
    try:
        contour = grid.contour([0])
        if contour.n_points > 0:
            plotter.add_mesh(contour, color='red', opacity=0.8, label='Geometry Surface (SDF=0)')
            print(f"Successfully added contour with {contour.n_points} points")
        else:
            print("Warning: Zero-level contour is empty!")
    except Exception as e:
        print(f"Error creating contour: {e}")

    # Mid‑z slice as fallback visualization
    if grid.n_cells > 0:
        if isinstance(grid, pv.RectilinearGrid):
            # RectilinearGrid: z coords are defined, use them
            z_mid = 0.5 * (z[0] + z[-1])
        else:
            # ImageData: dx is defined, use it
            z_mid = origin[2] + grid_shape[2]*dx/2
        slice_z = grid.slice(normal='z', origin=(origin[0], origin[1], z_mid))
        clim = (0.0, phi.max()) if phi.min() < 0 else (phi.min(), phi.max())
        plotter.add_mesh(slice_z, scalars="phi", cmap="RdBu", clim=clim, label='SDF Mid‑Slice')

    # Add domain boundaries as wireframe box
    if isinstance(grid, pv.RectilinearGrid):
        # RectilinearGrid: x, y, z coords are defined
        bounds = [x[0], x[-1], y[0], y[-1], z[0], z[-1]]
    else:
        # ImageData: origin and dx are defined
        bounds = [origin[0], origin[0] + grid_shape[0]*dx,
                  origin[1], origin[1] + grid_shape[1]*dx,
                  origin[2], origin[2] + grid_shape[2]*dx]
    domain_box = pv.Box(bounds=bounds)
    plotter.add_mesh(domain_box, style='wireframe', color='blue', line_width=2, label='Domain Boundary')

    # Add axes, legend, title
    plotter.add_axes(label_size=(0.1, 0.1))
    plotter.add_legend()
    plotter.camera_position = 'iso'
    plotter.add_title("SDF Visualization", font_size=14)
    plotter.show()


def visualize_sdf_histogram(h5file):
    """Show histogram of SDF values to understand distribution."""
    with h5py.File(h5file) as f:
        phi = f["levelset"][()]

    plt.figure(figsize=(12, 4))

    # Histogram 1: Full range
    plt.subplot(1, 3, 1)
    plt.hist(phi.flatten(), bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('SDF Value')
    plt.ylabel('Count')
    plt.title('SDF Distribution (Full Range)')
    plt.grid(True, alpha=0.3)

    # Histogram 2: Focus on near-zero values
    plt.subplot(1, 3, 2)
    near_zero = phi.flatten()
    near_zero = near_zero[np.abs(near_zero) < 2.0]  # Focus on values near the surface
    plt.hist(near_zero, bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('SDF Value')
    plt.ylabel('Count')
    plt.title('SDF Distribution (Near Surface)')
    plt.axvline(x=0, color='red', linestyle='--', label='Surface (SDF=0)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Statistics
    plt.subplot(1, 3, 3)
    stats_text = f"""SDF Statistics:
Min: {phi.min():.6f}
Max: {phi.max():.6f}
Mean: {phi.mean():.6f}
Std: {phi.std():.6f}

Voxel Counts:
Interior (< 0): {np.sum(phi < 0):,}
Exterior (> 0): {np.sum(phi > 0):,}
Near-zero (|φ| < 0.1): {np.sum(np.abs(phi) < 0.1):,}

Grid Shape: {phi.shape}
Total Voxels: {phi.size:,}"""

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    plt.axis('off')
    plt.title('SDF Statistics')

    plt.tight_layout()
    plt.show()


def plot_spacing(x: np.ndarray):
    """Plot cell spacing distribution for 1D coordinate array.

    Shows both the cell size distribution and the mapping function.
    """
    dx = np.diff(x)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(range(len(dx)), dx, marker='o')
    axes[0].set_xlabel('cell index')
    axes[0].set_ylabel('Δx')
    axes[0].set_title('Cell size distribution')
    axes[1].plot(x, np.linspace(0, 1, len(x)), marker='.')
    axes[1].set_xlabel('x coordinate')
    axes[1].set_ylabel('normalized index')
    axes[1].set_title('Mapping function')
    fig.tight_layout()
    plt.show()


__all__ = [
    "preview_grid_coords",
    "visualize_domain_and_geometry",
    "visualize",
    "visualize_sdf_histogram",
    "plot_spacing"
]
