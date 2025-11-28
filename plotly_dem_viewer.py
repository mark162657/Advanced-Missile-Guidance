"""
High-performance DEM visualization using Plotly (web-based 3D).
Supports interactive 3D terrain viewing with excellent performance.

Controls (in browser):
    - Mouse drag: Rotate view
    - Mouse wheel: Zoom in/out
    - Right click drag: Pan
    - Double click: Reset view
"""

import sys
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.guidance.pathfinding import Pathfinding
from src.terrain.dem_loader import DEMLoader


class PlotlyDEMViewer:
    """Plotly-based 3D terrain viewer with pathfinding overlay."""
    
    def __init__(self, dem_path: Path, downsample: int = 4, max_vertices: int = 500_000):
        """
        Initialize the DEM viewer.
        
        Args:
            dem_path: Path to DEM file
            downsample: Initial downsample factor (1 = full res, 2 = half, 4 = quarter, etc.)
            max_vertices: Maximum number of vertices to prevent slowdown (default: 500k for Plotly)
        """
        print("\n" + "="*60)
        print("INITIALIZING PLOTLY DEM VIEWER")
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
            print(f"WARNING: Auto-adjusted downsample from {downsample} to {self.downsample} to stay within {max_vertices:,} vertex limit")
        
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
        
        # Create figure
        self.fig = go.Figure()
        
        # Create terrain surface
        self._create_terrain_surface()
        
        # Path data
        self.path = None
        
        print("\n" + "="*60)
        print("CONTROLS (in browser):")
        print("  Mouse Drag: Rotate | Wheel: Zoom | Right Drag: Pan")
        print("  Double Click: Reset view")
        print("="*60 + "\n")
    
    def _create_terrain_surface(self):
        """Create the 3D terrain surface using Plotly."""
        print("Generating 3D terrain surface...")
        
        rows, cols = self.dem_data.shape
        
        # Create coordinate arrays
        x = np.arange(0, cols)
        y = np.arange(0, rows)
        
        # Z is elevation with vertical exaggeration
        z = (self.dem_data - self.elev_min) / self.elev_range * self.v_exag * 1000
        z = np.nan_to_num(z, nan=0.0)
        
        # Create surface
        surface = go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale='Earth',  # Beautiful terrain colormap
            colorbar=dict(
                title='Elevation (m)',
                x=1.02,
            ),
            surfacecolor=self.dem_data,  # Color by actual elevation
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.2,
                roughness=0.8,
            ),
            lightposition=dict(
                x=1000,
                y=1000,
                z=2000,
            ),
        )
        
        self.fig.add_trace(surface)
        
        # Update layout
        self.fig.update_layout(
            title='3D DEM Terrain Viewer',
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Elevation (m)',
                aspectmode='manual',
                aspectratio=dict(x=2, y=2, z=0.5),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0),
                ),
            ),
            width=1400,
            height=900,
        )
        
        print("[OK] Terrain surface created")
    
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
            print(f"[OK] Path computed: {len(path)} waypoints")
            self._add_path_overlay()
        else:
            print("[ERROR] No path found")
    
    def _add_path_overlay(self):
        """Add 3D path overlay to the visualization."""
        if not self.path:
            return
        
        # Convert path to 3D coordinates
        path_x, path_y, path_z = [], [], []
        
        for row, col in self.path:
            # Downsample coordinates
            ds_row = row // self.downsample
            ds_col = col // self.downsample
            
            # Get elevation (with vertical exaggeration)
            if 0 <= ds_row < self.dem_data.shape[0] and 0 <= ds_col < self.dem_data.shape[1]:
                z_val = (self.dem_data[ds_row, ds_col] - self.elev_min) / self.elev_range * self.v_exag * 1000
                if np.isnan(z_val):
                    z_val = 0.0
                
                path_x.append(ds_col)
                path_y.append(ds_row)
                path_z.append(z_val + 50)  # +50m above ground
        
        # Create 3D line
        path_line = go.Scatter3d(
            x=path_x,
            y=path_y,
            z=path_z,
            mode='lines',
            line=dict(
                color='red',
                width=8,
            ),
            name='Flight Path',
        )
        
        self.fig.add_trace(path_line)
        
        # Add start and end markers
        start_marker = go.Scatter3d(
            x=[path_x[0]],
            y=[path_y[0]],
            z=[path_z[0]],
            mode='markers',
            marker=dict(
                size=10,
                color='green',
                symbol='diamond',
            ),
            name='Start',
        )
        
        end_marker = go.Scatter3d(
            x=[path_x[-1]],
            y=[path_y[-1]],
            z=[path_z[-1]],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond',
            ),
            name='End',
        )
        
        self.fig.add_trace(start_marker)
        self.fig.add_trace(end_marker)
        
        print("[OK] Path overlay added")
    
    def show(self):
        """Display the visualization in browser."""
        print("\n>> Opening viewer in browser...")
        self.fig.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plotly 3D DEM Viewer with Path Overlay')
    parser.add_argument('--downsample', type=int, default=10, 
                        help='Downsample factor (1=full res, 2=half, 4=quarter, etc.)')
    parser.add_argument('--max-vertices', type=int, default=500_000,
                        help='Maximum vertices to prevent slowdown (default: 500k)')
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
        print(f"[ERROR] DEM file not found at {dem_path}")
        sys.exit(1)
    
    # Create viewer
    viewer = PlotlyDEMViewer(dem_path, downsample=args.downsample, max_vertices=args.max_vertices)
    
    # Load path if requested
    if args.path:
        start_gps = (args.start_lat, args.start_lon)
        end_gps = (args.end_lat, args.end_lon)
        viewer.load_path(start_gps, end_gps, heuristic_weight=args.weight)
    
    # Show
    viewer.show()
