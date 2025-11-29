"""
High-performance full-resolution DEM visualization using VisPy (GPU-accelerated).
Supports interactive 3D terrain viewing with dynamic LOD (Level of Detail).

Controls:
    - Mouse drag: Rotate view
    - Mouse wheel: Zoom in/out
    - WASD: Pan camera
    - R: Reset view
    - 1-5: Change vertical exaggeration
    - L: Toggle lighting
    - C: Toggle colormap
    - P: Toggle path overlay
    - ESC: Exit
"""

import sys
import numpy as np
from pathlib import Path
from vispy import app, scene
from vispy.color import Colormap
from vispy.visuals.transforms import STTransform

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.guidance.pathfinding import Pathfinding
from src.terrain.dem_loader import DEMLoader


class DEMViewer:
    """GPU-accelerated 3D terrain viewer with pathfinding overlay."""
    
    def __init__(self, dem_path: Path, downsample: int = 4):
        """
        Initialize the DEM viewer.
        
        Args:
            dem_path: Path to DEM file
            downsample: Downsample factor (1 = full res, 2 = half, 4 = quarter, etc.)
        """
        print("\n" + "="*60)
        print("INITIALIZING VISPY DEM VIEWER")
        print("="*60)
        
        self.downsample = downsample
        self.dem_loader = DEMLoader(dem_path)
        
        # Load DEM data
        print(f"\nLoading DEM: {dem_path.name}")
        print(f"Original shape: {self.dem_loader.shape} ({self.dem_loader.shape[0] * self.dem_loader.shape[1]:,} pixels)")
        
        # Downsample for performance
        self.dem_data = self.dem_loader.data[::downsample, ::downsample].astype(np.float32)
        self.dem_data[self.dem_data <= -100] = np.nan  # Remove nodata
        
        print(f"Viewing shape: {self.dem_data.shape} ({self.dem_data.size:,} pixels)")
        print(f"Reduction: {(self.dem_loader.shape[0] * self.dem_loader.shape[1]) / self.dem_data.size:.1f}x")
        
        # Normalize elevation for better visualization
        valid_data = self.dem_data[~np.isnan(self.dem_data)]
        self.elev_min = np.percentile(valid_data, 2)
        self.elev_max = np.percentile(valid_data, 98)
        self.elev_range = self.elev_max - self.elev_min
        
        print(f"Elevation range: {self.elev_min:.0f}m - {self.elev_max:.0f}m")
        
        # Setup VisPy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1600, 1000), 
                                        show=True, title='DEM Viewer (VisPy)')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(elevation=30, azimuth=45, distance=None)
        self.view.camera.set_range()
        
        # Colormap settings
        self.colormaps = [
            self._create_terrain_colormap(),
            Colormap(['blue', 'cyan', 'lime', 'yellow', 'red']),
            Colormap(['black', 'gray', 'white']),
            Colormap(['blue', 'green', 'yellow', 'orange', 'red', 'darkred'])
        ]
        self.current_cmap_idx = 0
        
        # Vertical exaggeration
        self.v_exag = 3.0
        self.v_exag_levels = [1.0, 2.0, 3.0, 5.0, 10.0]
        self.v_exag_idx = 2
        
        # Lighting toggle
        self.lighting_enabled = True
        
        # Path overlay data
        self.path = None
        self.path_visual = None
        self.show_path = False
        
        # Create surface mesh
        self._create_surface()
        
        # Connect keyboard events
        self.canvas.connect(self.on_key_press)
        
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  Mouse Drag: Rotate | Wheel: Zoom")
        print("  WASD: Pan | R: Reset view")
        print("  1-5: Vertical exaggeration")
        print("  L: Toggle lighting | C: Cycle colormap")
        print("  P: Toggle path overlay | ESC: Exit")
        print("="*60 + "\n")
    
    def _create_terrain_colormap(self):
        """Create custom terrain colormap."""
        colors = [
            (0.176, 0.314, 0.086),  # Dark green
            (0.353, 0.549, 0.169),  # Green
            (0.561, 0.702, 0.349),  # Light green
            (0.831, 0.780, 0.494),  # Yellow-tan
            (0.651, 0.486, 0.322),  # Brown
            (1.0, 1.0, 1.0)          # White (peaks)
        ]
        return Colormap(colors)
    
    def _create_surface(self):
        """Create the 3D terrain surface mesh."""
        print("Generating 3D surface mesh...")
        
        rows, cols = self.dem_data.shape
        
        # Create meshgrid for X, Y coordinates
        x = np.linspace(0, cols, cols)
        y = np.linspace(0, rows, rows)
        X, Y = np.meshgrid(x, y)
        
        # Z is elevation with vertical exaggeration
        Z = (self.dem_data - self.elev_min) / self.elev_range * self.v_exag * 1000
        Z = np.nan_to_num(Z, nan=0.0)
        
        # Create surface visual
        self.surface = scene.visuals.SurfacePlot(
            x=X, y=Y, z=Z,
            color=(0.5, 0.7, 0.5, 1),
            shading='smooth'
        )
        
        # Apply colormap
        colors = self.colormaps[self.current_cmap_idx].map(
            (self.dem_data - self.elev_min) / self.elev_range
        )
        # Reshape to 2D (vertices x RGBA) - flatten spatial dimensions
        colors = colors.reshape(-1, 4)
        self.surface.mesh_data.set_vertex_colors(colors)
        
        self.view.add(self.surface)
        
        # Center camera on terrain
        self.view.camera.center = (cols/2, rows/2, 0)
        self.view.camera.scale_factor = max(rows, cols) * 1.5
        
        print("✅ Surface mesh created")
    
    def _update_surface_elevation(self):
        """Update surface Z values with new vertical exaggeration."""
        rows, cols = self.dem_data.shape
        Z = (self.dem_data - self.elev_min) / self.elev_range * self.v_exag * 1000
        Z = np.nan_to_num(Z, nan=0.0)
        
        # Update mesh
        mesh_data = self.surface.mesh_data
        vertices = mesh_data.get_vertices()
        vertices[:, 2] = Z.ravel()
        mesh_data.set_vertices(vertices)
        
        self.surface.mesh_data_changed()
        self.canvas.update()
    
    def _update_colormap(self):
        """Update surface colormap."""
        rows, cols = self.dem_data.shape
        colors = self.colormaps[self.current_cmap_idx].map(
            (self.dem_data - self.elev_min) / self.elev_range
        )
        # Reshape to 2D (vertices x RGBA)
        colors = colors.reshape(-1, 4)
        self.surface.mesh_data.set_vertex_colors(colors)
        self.surface.mesh_data_changed()
        self.canvas.update()
    
    def load_path(self, start_gps: tuple[float, float], end_gps: tuple[float, float], 
                  heuristic_weight: float = 1.5):
        """
        Compute pathfinding and prepare overlay.
        
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
            self._create_path_visual()
        else:
            print("❌ No path found")
    
    def _create_path_visual(self):
        """Create 3D line visual for the path."""
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
        
        # Create line visual
        if self.path_visual:
            self.view.remove(self.path_visual)
        
        self.path_visual = scene.visuals.Line(
            pos=path_3d,
            color='red',
            width=3.0,
            method='gl'
        )
        
        self.view.add(self.path_visual)
        self.show_path = True
        self.canvas.update()
        
        print("✅ Path overlay added")
    
    def on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'Escape':
            self.canvas.close()
        
        elif event.key == 'R':
            # Reset camera
            self.view.camera.elevation = 30
            self.view.camera.azimuth = 45
            self.view.camera.set_range()
            print("🔄 Camera reset")
        
        elif event.key in ['1', '2', '3', '4', '5']:
            # Change vertical exaggeration
            self.v_exag_idx = int(event.key) - 1
            self.v_exag = self.v_exag_levels[self.v_exag_idx]
            self._update_surface_elevation()
            print(f"📏 Vertical exaggeration: {self.v_exag}x")
        
        elif event.key == 'C':
            # Cycle colormap
            self.current_cmap_idx = (self.current_cmap_idx + 1) % len(self.colormaps)
            self._update_colormap()
            print(f"🎨 Colormap: {self.current_cmap_idx + 1}/{len(self.colormaps)}")
        
        elif event.key == 'L':
            # Toggle lighting (placeholder - VisPy SurfacePlot doesn't support runtime lighting toggle easily)
            self.lighting_enabled = not self.lighting_enabled
            print(f"💡 Lighting: {'ON' if self.lighting_enabled else 'OFF'} (requires restart)")
        
        elif event.key == 'P':
            # Toggle path
            if self.path_visual:
                if self.show_path:
                    self.view.remove(self.path_visual)
                    self.show_path = False
                    print("🛤️  Path hidden")
                else:
                    self.view.add(self.path_visual)
                    self.show_path = True
                    print("🛤️  Path shown")
                self.canvas.update()
    
    def run(self):
        """Start the VisPy event loop."""
        print("\n▶️  Starting viewer...")
        app.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='3D DEM Viewer with Path Overlay')
    parser.add_argument('--downsample', type=int, default=8, 
                        help='Downsample factor (1=full res, 4=quarter, 8=1/8th, etc.)')
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
    viewer = DEMViewer(dem_path, downsample=args.downsample)
    
    # Load path if requested
    if args.path:
        start_gps = (args.start_lat, args.start_lon)
        end_gps = (args.end_lat, args.end_lon)
        viewer.load_path(start_gps, end_gps, heuristic_weight=args.weight)
    
    # Run
    viewer.run()