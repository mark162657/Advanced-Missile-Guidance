"""
High-performance DEM visualization using PyVista (VTK-based).
Supports interactive 3D terrain viewing with better memory management than VisPy.

Controls:
    - Mouse drag: Rotate view
    - Mouse wheel: Zoom in/out
    - Middle mouse drag: Pan
    - R: Reset view
    - Q/ESC: Exit
"""

import sys
import numpy as np
from pathlib import Path
import pyvista as pv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.guidance.pathfinding import Pathfinding
from src.terrain.dem_loader import DEMLoader


class PyVistaDEMViewer:
    """PyVista-based 3D terrain viewer with pathfinding overlay."""
    
    def __init__(self, dem_path: Path, downsample: int = 4, max_vertices: int = 2_000_000):
        """
        Initialize the DEM viewer.
        
        Args:
            dem_path: Path to DEM file
            downsample: Initial downsample factor (1 = full res, 2 = half, 4 = quarter, etc.)
            max_vertices: Maximum number of vertices to prevent OOM (default: 2 million)
        """
        print("\n" + "="*60)
        print("INITIALIZING PYVISTA DEM VIEWER")
        print("="*60)
        
        self.dem_loader = DEMLoader(dem_path)
        
        # Load DEM data
        print(f"\nLoading DEM: {dem_path.name}")
        print(f"Original shape: {self.dem_loader.shape} ({self.dem_loader.shape[0] * self.dem_loader.shape[1]:,} pixels)")
        
        # Auto-adjust downsample to stay within vertex limit
        original_vertices = self.dem_loader.shape[0] * self.dem_loader.shape[1]
        self.downsample = downsample
        
        while True:
            downsampled_vertices = (self.dem_loader.shape[0] // self.downsample) * (self.dem_loader.shape[1] // self.downsample)
            if downsampled_vertices <= max_vertices:
                break
            self.downsample += 1
        
        if self.downsample != downsample:
            print(f"⚠️  Auto-adjusted downsample from {downsample} to {self.downsample} to stay within {max_vertices:,} vertex limit")
        
        # Downsample for performance
        self.dem_data = self.dem_loader.data[::self.downsample, ::self.downsample].astype(np.float32)
        self.dem_data[self.dem_data <= -100] = np.nan  # Remove nodata
        
        print(f"Viewing shape: {self.dem_data.shape} ({self.dem_data.size:,} pixels)")
        print(f"Reduction: {original_vertices / self.dem_data.size:.1f}x")
        
        # Normalize elevation
        valid_data = self.dem_data[~np.isnan(self.dem_data)]
        self.elev_min = np.percentile(valid_data, 2)
        self.elev_max = np.percentile(valid_data, 98)
        self.elev_range = self.elev_max - self.elev_min
        
        print(f"Elevation range: {self.elev_min:.0f}m - {self.elev_max:.0f}m")
        
        # Vertical exaggeration
        self.v_exag = 3.0
        
        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.set_background('lightblue')
        
        # Create terrain mesh
        self._create_terrain_mesh()
        
        # Path data
        self.path = None
        self.path_actor = None
        
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  Mouse Drag: Rotate | Wheel: Zoom | Middle Drag: Pan")
        print("  R: Reset view | Q/ESC: Exit")
        print("="*60 + "\n")
    
    def _create_terrain_mesh(self):
        """Create the 3D terrain mesh using PyVista."""
        print("Generating 3D terrain mesh...")
        
        rows, cols = self.dem_data.shape
        
        # Create coordinate arrays
        x = np.arange(0, cols, dtype=np.float32)
        y = np.arange(0, rows, dtype=np.float32)
        x, y = np.meshgrid(x, y)
        
        # Z is elevation with vertical exaggeration
        z = (self.dem_data - self.elev_min) / self.elev_range * self.v_exag * 1000
        z = np.nan_to_num(z, nan=0.0)
        
        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)
        
        # Add elevation as scalar for coloring
        grid['elevation'] = self.dem_data.ravel(order='F')
        
        # Add mesh to plotter with terrain colormap
        self.plotter.add_mesh(
            grid,
            scalars='elevation',
            cmap='terrain',
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Elevation (m)',
                'vertical': True,
                'position_x': 0.85,
                'position_y': 0.1,
            },
            lighting=True,
            smooth_shading=True,
        )
        
        # Add axes
        self.plotter.show_axes()
        
        print("✅ Terrain mesh created")
    
    def load_path(self, start_gps: tuple[float, float], end_gps: tuple[float, float], 
                  heuristic_weight: float = 1.5):
        """
        Compute pathfinding and add overlay.
        
        Args:
            start_gps: (lat, lon) start coordinate
            end_gps: (lat, lon) end coordinate
            heuristic_weight: A* heuristic weight
        """
        print("\n" + "="*60)
        print("COMPUTING PATH...")
        print("="*60)
        
        pf = Pathfinding()
        
        # Convert GPS to pixel
        start_row, start_col = pf.dem_loader.lat_lon_to_pixel(start_gps[0], start_gps[1])
        end_row, end_col = pf.dem_loader.lat_lon_to_pixel(end_gps[0], end_gps[1])
        
        start_pixel = (start_row, start_col)
        end_pixel = (end_row, end_col)
        
        print(f"Start: {start_gps} → Pixel {start_pixel}")
        print(f"End:   {end_gps} → Pixel {end_pixel}")
        
        # Compute path
        path = pf.pathfinding(start_pixel, end_pixel, heuristic_weight=heuristic_weight)
        
        if path:
            self.path = path
            print(f"✅ Path computed: {len(path)} waypoints")
            self._add_path_overlay()
        else:
            print("❌ No path found")
    
    def _add_path_overlay(self):
        """Add 3D path overlay to the visualization."""
        if not self.path:
            return
        
        # Convert path to 3D coordinates
        path_3d = []
        for row, col in self.path:
            # Downsample coordinates
            ds_row = row // self.downsample
            ds_col = col // self.downsample
            
            # Get elevation (with vertical exaggeration)
            if 0 <= ds_row < self.dem_data.shape[0] and 0 <= ds_col < self.dem_data.shape[1]:
                z_val = (self.dem_data[ds_row, ds_col] - self.elev_min) / self.elev_range * self.v_exag * 1000
                if np.isnan(z_val):
                    z_val = 0.0
                path_3d.append([ds_col, ds_row, z_val + 50])  # +50m above ground
        
        path_3d = np.array(path_3d, dtype=np.float32)
        
        # Create polyline
        path_line = pv.Spline(path_3d, n_points=len(path_3d) * 2)
        
        # Add to plotter
        self.path_actor = self.plotter.add_mesh(
            path_line,
            color='red',
            line_width=5,
            label='Flight Path'
        )
        
        print("✅ Path overlay added")
    
    def run(self):
        """Start the PyVista viewer."""
        print("\n▶️  Starting viewer...")
        self.plotter.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PyVista 3D DEM Viewer with Path Overlay')
    parser.add_argument('--downsample', type=int, default=4, 
                        help='Downsample factor (1=full res, 2=half, 4=quarter, etc.)')
    parser.add_argument('--max-vertices', type=int, default=2_000_000,
                        help='Maximum vertices to prevent OOM (default: 2 million)')
    parser.add_argument('--path', action='store_true', 
                        help='Enable pathfinding overlay')
    parser.add_argument('--start-lat', type=float, default=56.0)
    parser.add_argument('--start-lon', type=float, default=95.0)
    parser.add_argument('--end-lat', type=float, default=56.0)
    parser.add_argument('--end-lon', type=float, default=95.5)
    parser.add_argument('--weight', type=float, default=1.5, 
                        help='A* heuristic weight')
    
    args = parser.parse_args()
    
    # DEM path
    dem_path = Path(__file__).parent / 'data' / 'dem' / 'merged_dem_sib_N54_N59_E090_E100.tif'
    
    if not dem_path.exists():
        print(f"❌ ERROR: DEM file not found at {dem_path}")
        sys.exit(1)
    
    # Create viewer
    viewer = PyVistaDEMViewer(dem_path, downsample=args.downsample, max_vertices=args.max_vertices)
    
    # Load path if requested
    if args.path:
        start_gps = (args.start_lat, args.start_lon)
        end_gps = (args.end_lat, args.end_lon)
        viewer.load_path(start_gps, end_gps, heuristic_weight=args.weight)
    
    # Run
    viewer.run()
