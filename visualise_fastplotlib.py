"""
High-performance DEM visualization using fastplotlib (GPU-accelerated).
Supports interactive 3D terrain viewing with real-time path overlay.

Controls:
    - Mouse drag: Rotate view
    - Mouse wheel: Zoom in/out
    - Arrow keys: Pan camera
    - R: Reset view
    - ESC: Exit
"""

import sys
import numpy as np
from pathlib import Path
import fastplotlib as fpl
from fastplotlib import ImageGraphic

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.guidance.pathfinding import Pathfinding
from src.terrain.dem_loader import DEMLoader


class FastDEMViewer:
    """GPU-accelerated DEM viewer using fastplotlib."""
    
    def __init__(self, dem_path: Path, downsample: int = 8):
        """
        Initialize the DEM viewer.
        
        Args:
            dem_path: Path to DEM file
            downsample: Downsample factor (1 = full res, 8 = 1/8th, etc.)
        """
        print("\n" + "="*60)
        print("FASTPLOTLIB DEM VIEWER")
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
        print(f"Downsample: {downsample}x")
        
        # Normalize elevation for colormap
        valid_data = self.dem_data[~np.isnan(self.dem_data)]
        self.elev_min = np.percentile(valid_data, 2)
        self.elev_max = np.percentile(valid_data, 98)
        self.elev_range = self.elev_max - self.elev_min
        
        print(f"Elevation range: {self.elev_min:.0f}m - {self.elev_max:.0f}m")
        
        # Normalize data for display (0-1 range)
        self.dem_normalized = (self.dem_data - self.elev_min) / self.elev_range
        self.dem_normalized = np.nan_to_num(self.dem_normalized, nan=0.0)
        
        # Create figure
        self.figure = None
        self.image_graphic = None
        self.path_graphic = None
        
        self._create_visualization()
        
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  Mouse: Drag to pan, Wheel to zoom")
        print("  R: Reset view | ESC: Exit")
        print("="*60 + "\n")
    
    def _create_visualization(self):
        """Create the fastplotlib visualization."""
        print("Generating visualization...")
        
        try:
            # Create figure with single subplot
            self.figure = fpl.Figure(size=(1600, 1000), canvas="glfw")
            
            # Add DEM as image with terrain colormap
            self.image_graphic = self.figure[0, 0].add_image(
                data=self.dem_normalized,
                cmap="viridis",
                name="DEM"
            )
            
            # Set camera to fit data
            self.figure[0, 0].auto_scale()
            
            print("✅ Visualization created")
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            print("Trying fallback method...")
            
            # Fallback: simpler initialization
            self.figure = fpl.Figure(size=(1200, 800))
            self.image_graphic = self.figure[0, 0].add_image(
                data=self.dem_normalized,
                cmap="gray"
            )
            print("✅ Fallback visualization created")
    
    def add_path(self, start_gps: tuple[float, float], end_gps: tuple[float, float], 
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
            print(f"✅ Path computed: {len(path)} waypoints")
            
            # Convert path to downsampled coordinates
            path_coords = np.array([[col // self.downsample, row // self.downsample] 
                                   for row, col in path], dtype=np.float32)
            
            # Add path as line graphic
            self.path_graphic = self.figure[0, 0].add_line(
                data=path_coords,
                thickness=3,
                colors="red",
                name="Path"
            )
            
            # Add start/end markers
            start_marker = np.array([[start_col // self.downsample, start_row // self.downsample]], dtype=np.float32)
            end_marker = np.array([[end_col // self.downsample, end_row // self.downsample]], dtype=np.float32)
            
            self.figure[0, 0].add_scatter(
                data=start_marker,
                sizes=15,
                colors="lime",
                name="Start"
            )
            
            self.figure[0, 0].add_scatter(
                data=end_marker,
                sizes=15,
                colors="yellow",
                name="End"
            )
            
            print("✅ Path overlay added")
        else:
            print("❌ No path found")
    
    def show(self):
        """Display the visualization."""
        print("\n▶️  Starting viewer...")
        try:
            self.figure.show(maintain_aspect=False)
        except Exception as e:
            print(f"❌ Viewer crashed: {e}")
            print("This may be due to GPU/driver compatibility issues.")
            print("Try: pip install --upgrade fastplotlib wgpu glfw")
            print("Or use the matplotlib version (visualise.py) instead.")


def compare_paths(start_gps: tuple[float, float], end_gps: tuple[float, float],
                  weights: list[float] = [1.0, 1.5, 2.0], downsample: int = 8):
    """
    Create multi-panel comparison of different heuristic weights.
    
    Args:
        start_gps: (lat, lon) start coordinate
        end_gps: (lat, lon) end coordinate
        weights: List of heuristic weights to compare
        downsample: Downsample factor
    """
    print("\n" + "="*60)
    print("MULTI-PATH COMPARISON")
    print("="*60)
    
    # Load DEM once
    dem_path = Path(__file__).parent / 'data' / 'dem' / 'merged_dem_sib_N54_N59_E090_E100.tif'
    dem_loader = DEMLoader(dem_path)
    
    print(f"\nLoading DEM: {dem_path.name}")
    dem_data = dem_loader.data[::downsample, ::downsample].astype(np.float32)
    dem_data[dem_data <= -100] = np.nan
    
    # Normalize
    valid_data = dem_data[~np.isnan(dem_data)]
    elev_min = np.percentile(valid_data, 2)
    elev_max = np.percentile(valid_data, 98)
    dem_normalized = (dem_data - elev_min) / (elev_max - elev_min)
    dem_normalized = np.nan_to_num(dem_normalized, nan=0.0)
    
    # Initialize pathfinding
    pf = Pathfinding()
    start_row, start_col = pf.dem_loader.lat_lon_to_pixel(start_gps[0], start_gps[1])
    end_row, end_col = pf.dem_loader.lat_lon_to_pixel(end_gps[0], end_gps[1])
    
    start_pixel = (start_row, start_col)
    end_pixel = (end_row, end_col)
    
    print(f"\nStart: {start_gps} → Pixel {start_pixel}")
    print(f"End:   {end_gps} → Pixel {end_pixel}")
    
    # Compute paths for each weight
    paths = {}
    for weight in weights:
        print(f"\nComputing path with weight {weight}...")
        path = pf.pathfinding(start_pixel, end_pixel, heuristic_weight=weight)
        if path:
            paths[weight] = path
            print(f"✅ Path found: {len(path)} waypoints")
        else:
            print(f"❌ No path found for weight {weight}")
    
    if not paths:
        print("\n❌ No valid paths computed. Exiting.")
        return
    
    # Create figure with subplots
    n_cols = min(len(paths), 3)
    n_rows = (len(paths) + n_cols - 1) // n_cols
    
    print(f"\nCreating {n_rows}x{n_cols} comparison grid...")
    figure = fpl.Figure(shape=(n_rows, n_cols), size=(600 * n_cols, 500 * n_rows))
    
    # Add each path to its subplot
    colors = ['red', 'cyan', 'yellow', 'lime', 'magenta']
    
    for idx, (weight, path) in enumerate(paths.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        # Add DEM
        figure[row, col].add_image(
            data=dem_normalized,
            cmap="viridis",
            name=f"DEM_w{weight}"
        )
        
        # Add path
        path_coords = np.array([[c // downsample, r // downsample] 
                               for r, c in path], dtype=np.float32)
        
        figure[row, col].add_line(
            data=path_coords,
            thickness=3,
            colors=colors[idx % len(colors)],
            name=f"Path_w{weight}"
        )
        
        # Add markers
        start_marker = np.array([[start_col // downsample, start_row // downsample]], dtype=np.float32)
        end_marker = np.array([[end_col // downsample, end_row // downsample]], dtype=np.float32)
        
        figure[row, col].add_scatter(data=start_marker, sizes=12, colors="lime")
        figure[row, col].add_scatter(data=end_marker, sizes=12, colors="white")
        
        # Auto-scale view
        figure[row, col].auto_scale()
    
    print("✅ Comparison created")
    print("\n▶️  Starting viewer...")
    figure.show(maintain_aspect=False)


if __name__ == "__main__":
    import argparse
    import random
    import math
    
    parser = argparse.ArgumentParser(description='Fastplotlib DEM Viewer')
    parser.add_argument('--downsample', type=int, default=8,
                        help='Downsample factor (default: 8)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple heuristic weights')
    parser.add_argument('--weights', nargs='+', type=float, default=[1.0, 1.5, 2.0],
                        help='Heuristic weights to test (default: 1.0 1.5 2.0)')
    parser.add_argument('--start-lat', type=float, default=None)
    parser.add_argument('--start-lon', type=float, default=None)
    parser.add_argument('--end-lat', type=float, default=None)
    parser.add_argument('--end-lon', type=float, default=None)
    parser.add_argument('--distance', type=float, default=10.0,
                        help='Random path distance in km (default: 10)')
    
    args = parser.parse_args()
    
    # DEM path
    dem_path = Path(__file__).parent / 'data' / 'dem' / 'merged_dem_sib_N54_N59_E090_E100.tif'
    
    if not dem_path.exists():
        print(f"❌ ERROR: DEM file not found at {dem_path}")
        sys.exit(1)
    
    # Generate random coordinates if not provided
    if args.start_lat is None or args.end_lat is None:
        safe_lat_min, safe_lat_max = 54.5, 57.5
        safe_lon_min, safe_lon_max = 92.5, 97.5
        
        start_lat = random.uniform(safe_lat_min, safe_lat_max)
        start_lon = random.uniform(safe_lon_min, safe_lon_max)
        
        direction = random.uniform(0, 2 * math.pi)
        deg_per_km_lon = 1 / (111.32 * 0.559193)
        deg_per_km_lat = 1 / 111.32
        
        end_lat = start_lat + (args.distance * math.cos(direction) * deg_per_km_lat)
        end_lon = start_lon + (args.distance * math.sin(direction) * deg_per_km_lon)
        
        print(f"\n🎲 Random {args.distance}km path: Direction {math.degrees(direction):.1f}°")
    else:
        start_lat, start_lon = args.start_lat, args.start_lon
        end_lat, end_lon = args.end_lat, args.end_lon
    
    start_gps = (start_lat, start_lon)
    end_gps = (end_lat, end_lon)
    
    if args.compare:
        # Multi-path comparison mode
        compare_paths(start_gps, end_gps, weights=args.weights, downsample=args.downsample)
    else:
        # Single view mode
        viewer = FastDEMViewer(dem_path, downsample=args.downsample)
        viewer.add_path(start_gps, end_gps, heuristic_weight=args.weights[0])
        viewer.show()
