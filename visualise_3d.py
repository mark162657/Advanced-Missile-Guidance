"""
3D Plotly visualization of pathfinding results on DEM terrain.
Shows paths in 3D with elevation, allowing interactive exploration.
"""

import sys
import time
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.guidance.pathfinding import Pathfinding


def visualize_path_3d(start_gps: tuple[float, float], end_gps: tuple[float, float], 
                      weights: list[float] = [1.5, 2.0],
                      downsample: int = 10,
                      v_exag: float = 3.0):
    """
    Run pathfinding with multiple heuristic weights and visualize all paths in 3D.
    
    Args:
        start_gps: (lat, lon) start coordinate
        end_gps: (lat, lon) end coordinate
        weights: List of heuristic weights to test
        downsample: Terrain downsample factor for visualization
        v_exag: Vertical exaggeration for terrain
    """
    
    print("\n" + "="*60)
    print("3D PATH VISUALIZATION")
    print("="*60)
    
    # Initialize pathfinding
    pf = Pathfinding()
    
    # Convert GPS to pixel coordinates
    start_row, start_col = pf.dem_loader.lat_lon_to_pixel(start_gps[0], start_gps[1])
    end_row, end_col = pf.dem_loader.lat_lon_to_pixel(end_gps[0], end_gps[1])
    
    start_pixel = (start_row, start_col)
    end_pixel = (end_row, end_col)
    
    print(f"\nStart GPS: {start_gps} -> Pixel: {start_pixel}")
    print(f"End GPS:   {end_gps} -> Pixel: {end_pixel}")
    
    # Compute paths for each weight
    paths_data = []
    
    for weight in weights:
        print(f"\n--- Testing Weight: {weight} ---")
        start_time = time.time()
        
        path = pf.pathfinding(start_pixel, end_pixel, heuristic_weight=weight)
        
        elapsed = time.time() - start_time
        
        if path:
            path_length = len(path)
            
            # Calculate total distance and climb
            total_dist = 0.0
            total_climb = 0.0
            for i in range(len(path) - 1):
                curr_idx = pf.pixel_to_idx(path[i][0], path[i][1])
                next_idx = pf.pixel_to_idx(path[i+1][0], path[i+1][1])
                total_dist += pf._heuristic(curr_idx, next_idx)
                
                # Calculate elevation change
                curr_elev = pf.dem[path[i]]
                next_elev = pf.dem[path[i+1]]
                climb = max(0, next_elev - curr_elev)
                total_climb += climb
            
            paths_data.append({
                'weight': weight,
                'path': path,
                'length': path_length,
                'distance': total_dist,
                'climb': total_climb,
                'time': elapsed
            })
            
            print(f"[OK] Path found: {path_length} nodes, {total_dist/1000:.2f}km, {total_climb:.0f}m climb")
            print(f"   Time: {elapsed:.3f}s")
        else:
            print(f"[ERROR] No path found")
    
    if not paths_data:
        print("\n[ERROR] No paths found. Exiting.")
        return
    
    # Calculate bounding box for all paths with padding
    print("\n" + "="*60)
    print("GENERATING 3D VISUALIZATION...")
    print("="*60)
    
    all_rows = []
    all_cols = []
    for data in paths_data:
        for point in data['path']:
            all_rows.append(point[0])
            all_cols.append(point[1])
    
    min_row, max_row = min(all_rows), max(all_rows)
    min_col, max_col = min(all_cols), max(all_cols)
    
    # Add padding (20% of path dimensions)
    row_padding = int((max_row - min_row) * 0.2) + 100
    col_padding = int((max_col - min_col) * 0.2) + 100
    
    min_row = max(0, min_row - row_padding)
    max_row = min(pf.row - 1, max_row + row_padding)
    min_col = max(0, min_col - col_padding)
    max_col = min(pf.col - 1, max_col + col_padding)
    
    print(f"Zoom area: rows [{min_row}:{max_row}], cols [{min_col}:{max_col}]")
    
    # Auto-adjust downsample to maximize detail while staying within vertex limit
    # Mesh3d is more memory intensive than Surface, so use lower limit
    max_vertices = 500_000  # 500k vertices max for Mesh3d
    zoom_rows = max_row - min_row
    zoom_cols = max_col - min_col
    original_vertices = zoom_rows * zoom_cols
    
    # Start with downsample=1 (full res) and increase if needed
    optimal_downsample = 1
    while True:
        downsampled_vertices = (zoom_rows // optimal_downsample) * (zoom_cols // optimal_downsample)
        if downsampled_vertices <= max_vertices:
            break
        optimal_downsample += 1
    
    # Use the optimal downsample (ignore the input parameter for zoomed areas)
    downsample = optimal_downsample
    
    if downsample == 1:
        print(f"Using FULL RESOLUTION ({original_vertices:,} vertices)")
    else:
        print(f"Auto-adjusted downsample to {downsample} ({original_vertices:,} -> {downsampled_vertices:,} vertices)")
    
    # Extract zoomed DEM region with downsampling
    dem_region = pf.dem[min_row:max_row:downsample, min_col:max_col:downsample].astype(float)
    dem_region[dem_region <= -100] = np.nan  # Remove nodata
    
    # Get elevation stats
    valid_elev = dem_region[~np.isnan(dem_region)]
    elev_min = np.percentile(valid_elev, 2)
    elev_max = np.percentile(valid_elev, 98)
    elev_range = elev_max - elev_min
    
    print(f"Elevation range: {elev_min:.0f}m - {elev_max:.0f}m")
    
    # Create coordinate arrays for terrain
    rows_ds, cols_ds = dem_region.shape
    
    # Replace NaN with elev_min
    dem_clean = np.nan_to_num(dem_region, nan=elev_min)
    
    # Create 1D arrays for vertices
    x_1d = []
    y_1d = []
    z_1d = []
    
    for i in range(rows_ds):
        for j in range(cols_ds):
            x_1d.append(j)
            y_1d.append(i)
            elev = dem_clean[i, j]
            z_val = ((elev - elev_min) / elev_range) * v_exag * 1000
            z_1d.append(z_val)
    
    # Create triangular faces (indices into vertex arrays)
    faces_i = []
    faces_j = []
    faces_k = []
    
    for i in range(rows_ds - 1):
        for j in range(cols_ds - 1):
            # Two triangles per grid cell
            v0 = i * cols_ds + j
            v1 = i * cols_ds + (j + 1)
            v2 = (i + 1) * cols_ds + j
            v3 = (i + 1) * cols_ds + (j + 1)
            
            # Triangle 1
            faces_i.append(v0)
            faces_j.append(v1)
            faces_k.append(v2)
            
            # Triangle 2
            faces_i.append(v1)
            faces_j.append(v3)
            faces_k.append(v2)
    
    # Create intensity for coloring based on elevation
    intensity = [(dem_clean[i // cols_ds, i % cols_ds] - elev_min) / elev_range for i in range(len(x_1d))]
    
    print(f"Terrain mesh: {len(x_1d):,} vertices, {len(faces_i):,} triangles")
    print(f"Z range: {min(z_1d):.1f}m to {max(z_1d):.1f}m")
    
    # Create figure
    fig = go.Figure()
    
    # Add terrain mesh
    mesh = go.Mesh3d(
        x=x_1d,
        y=y_1d,
        z=z_1d,
        i=faces_i,
        j=faces_j,
        k=faces_k,
        intensity=intensity,
        colorscale='Earth',
        colorbar=dict(
            title='Elevation (m)',
            x=1.02,
        ),
        lighting=dict(
            ambient=0.5,
            diffuse=0.8,
            specular=0.2,
            roughness=0.5,
        ),
        lightposition=dict(
            x=1000,
            y=1000,
            z=2000,
        ),
        showscale=True,
        name='Terrain',
    )
    
    fig.add_trace(mesh)
    
    # Add paths with different colors
    path_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for i, data in enumerate(paths_data):
        path = data['path']
        weight = data['weight']
        
        # Convert path to 3D coordinates (local coordinate system)
        path_x, path_y, path_z = [], [], []
        
        for row, col in path:
            # Convert to local downsampled coordinates
            ds_row = (row - min_row) // downsample
            ds_col = (col - min_col) // downsample
            
            # Get elevation
            if 0 <= ds_row < dem_region.shape[0] and 0 <= ds_col < dem_region.shape[1]:
                elev = dem_region[ds_row, ds_col]
                if np.isnan(elev):
                    elev = elev_min
                
                z_val = (elev - elev_min) / elev_range * v_exag * 1000
                
                path_x.append(ds_col)  # Use local coordinates
                path_y.append(ds_row)
                path_z.append(z_val + 50)  # +50m above ground
        
        # Add path line
        path_line = go.Scatter3d(
            x=path_x,
            y=path_y,
            z=path_z,
            mode='lines',
            line=dict(
                color=path_colors[i % len(path_colors)],
                width=8,
            ),
            name=f'w={weight} ({data["length"]} nodes, {data["distance"]/1000:.1f}km)',
        )
        
        fig.add_trace(path_line)
    
    # Add start marker
    start_ds_row = (start_row - min_row) // downsample
    start_ds_col = (start_col - min_col) // downsample
    if 0 <= start_ds_row < dem_region.shape[0] and 0 <= start_ds_col < dem_region.shape[1]:
        start_elev = dem_region[start_ds_row, start_ds_col]
        if np.isnan(start_elev):
            start_elev = elev_min
        start_z = (start_elev - elev_min) / elev_range * v_exag * 1000 + 100
        
        start_marker = go.Scatter3d(
            x=[start_ds_col],  # Use local coordinates
            y=[start_ds_row],
            z=[start_z],
            mode='markers',
            marker=dict(
                size=15,
                color='lime',
                symbol='diamond',
                line=dict(color='black', width=2),
            ),
            name='Start',
        )
        fig.add_trace(start_marker)
    
    # Add end marker
    end_ds_row = (end_row - min_row) // downsample
    end_ds_col = (end_col - min_col) // downsample
    if 0 <= end_ds_row < dem_region.shape[0] and 0 <= end_ds_col < dem_region.shape[1]:
        end_elev = dem_region[end_ds_row, end_ds_col]
        if np.isnan(end_elev):
            end_elev = elev_min
        end_z = (end_elev - elev_min) / elev_range * v_exag * 1000 + 100
        
        end_marker = go.Scatter3d(
            x=[end_ds_col],  # Use local coordinates
            y=[end_ds_row],
            z=[end_z],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=2),
            ),
            name='End',
        )
        fig.add_trace(end_marker)
    
    # Update layout
    fig.update_layout(
        title='3D Path Comparison on DEM Terrain',
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
        width=1600,
        height=1000,
        showlegend=True,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for data in paths_data:
        print(f"\nWeight {data['weight']}:")
        print(f"  Nodes: {data['length']}")
        print(f"  Distance: {data['distance']/1000:.2f} km")
        print(f"  Climb: {data['climb']:.0f} m")
        print(f"  Time: {data['time']:.3f} s")
    
    best_time = min(paths_data, key=lambda x: x['time'])
    best_distance = min(paths_data, key=lambda x: x['distance'])
    best_climb = min(paths_data, key=lambda x: x['climb'])
    print("2. Medium path (~50km)")
    print("3. Long path (~100km)")
    print("4. Custom coordinates")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    # Base coordinate (Siberia safe land)
    base_lat, base_lon = 56.0, 95.0
    
    # Degrees per km at latitude 56°
    deg_per_km_lon = 1 / (111.32 * 0.559193)
    deg_per_km_lat = 1 / 111.32
    
    if choice == "1":
        # 10km east
        start_gps = (base_lat, base_lon)
        end_gps = (base_lat, base_lon + 10 * deg_per_km_lon)
        
    elif choice == "2":
        # 50km northeast
        start_gps = (base_lat, base_lon)
        end_gps = (base_lat + 25 * deg_per_km_lat, base_lon + 43.3 * deg_per_km_lon)
        
    elif choice == "3":
        # 100km east
        start_gps = (base_lat, base_lon)
        end_gps = (base_lat, base_lon + 100 * deg_per_km_lon)
        
    elif choice == "4":
        print("\nEnter custom coordinates:")
        start_lat = float(input("Start Latitude: "))
        start_lon = float(input("Start Longitude: "))
        end_lat = float(input("End Latitude: "))
        end_lon = float(input("End Longitude: "))
        
        start_gps = (start_lat, start_lon)
        end_gps = (end_lat, end_lon)
    else:
        print("Invalid choice. Using default 10km test.")
        start_gps = (base_lat, base_lon)
        end_gps = (base_lat, base_lon + 10 * deg_per_km_lon)
    
    # Run 3D visualization with multiple weights
    visualize_path_3d(start_gps, end_gps, weights=[1.5, 2.0], downsample=10, v_exag=3.0)
