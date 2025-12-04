import sys
import numpy as np
import time
import math
from pathlib import Path
from ..terrain.dem_loader import DEMLoader

# ---- Import C++ Backend ----
try:
    from . import missile_backend
    CPP_AVAILABLE = True
except ImportError as e:
    print("---------- ERROR! ----------")
    print(f"  WARNING: C++ Engine not found ({e}).")
    print("   Please compile the C++ code in src/guidance/cpp/ first.")
    print("   Ensure the .so/.pyd file is in src/guidance/")
    print("----------------------------")
    CPP_AVAILABLE = False

class Pathfinding:
    def __init__(self):
        # Load DEM
        tif_path = Path(__file__).parent.parent.parent / 'data' / 'dem' / 'merged_dem_sib_N54_N59_E090_E100.tif'
        dem = DEMLoader(tif_path)
        self.dem_loader = dem

        # Load data into ascontiguousarray for C/C++, float32 for dem
        self.dem = np.ascontiguousarray(dem.data, dtype=np.float32)

        # Initiating the lookup table and column value for future use
        self.rows = self.dem_loader.shape[0] # shape: (rows, columns)
        self.cols = self.dem_loader.shape[1]

        # Vertical Resolution (Meters per Pixel Z / Latitude)
        self.meter_per_z = abs(self.dem_loader.transform[4] * 111320)
        
        start_lat = self.dem_loader.transform[5]
        pixel_height = self.dem_loader.transform[4]

        # Setup lookup table (an array of latitude for every row)
        # Fix: Typo in numpy function name (arrange -> arange)
        row_indices = np.arange(self.rows, dtype=np.float64)
        self.latitudes = start_lat + (row_indices * pixel_height)

        # Horizontal (east / west) setup - x: width in meter for each latitude
        base_width_meters = abs(self.dem_loader.transform[0] * 111320)

        # Setting up lookup table for C++ (float64)
        meters_per_x_raw = base_width_meters * np.cos(np.radians(self.latitudes))
        self.meters_per_x_lookup = np.ascontiguousarray(meters_per_x_raw, dtype=np.float64)

        # Initiate C++ engine
        if CPP_AVAILABLE:
            print(f"Initializing C++ Engine with {self.rows * self.cols} pixels...")
            
            # start timer for knowing the execution speed
            start_time = time.time()

            # Instantiate the C++ Class
            self.engine = missile_backend.PathfinderCPP(
                self.dem,
                self.meters_per_x_lookup,
                self.meter_per_z,
                self.rows,
                self.cols
            )

            print(f"C++ Engine Ready ({time.time() - start_time:.4f}s)")
        
        else:
            self.engine = None

    def find_path(self, start: tuple[int, int], end: tuple[int, int], heuristic_weight: float=1.3) -> list:
        """
        Main Interface to run the A* algorithm.
        Args:
            start: (row, col) tuple
            end: (row, col) tuple
            heuristic_weight: 
                1.1 = Safe, Standard A* behavior
                1.3 = Sweet Spot (Fast)
        """
        if not self.engine:
            raise RuntimeError("C++ Engine is not loaded. Cannot run pathfinding.")

        # 1. Bounds Check
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            print(f"Start coordinates out of bounds: {start}")
            return None
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            print(f"End coordinates out of bounds: {end}")
            return None

        # 2. Convert Tuples to Packed Indices (C++ expects Ints)
        start_idx = start[0] * self.cols + start[1]
        end_idx = end[0] * self.cols + end[1]

        # 3. Call C++ Engine
        start_t = time.time()
        path = self.engine.find_path(start_idx, end_idx, heuristic_weight)
        duration = time.time() - start_t

        if not path:
            print(f"No path found (C++ returned empty list). Time: {duration:.4f}s")
            return None
            
        print(f"Path found! Length: {len(path)}. Time: {duration:.4f}s")
        return path

    def convert_path_to_gps(self, pixel_path: list) -> list:
        """
        Batch converts the pixel path to GPS coordinates (Lat, Lon).
        Useful for map plotting (2D).
        """
        if not pixel_path: return []
        
        # Unzip into two lists
        rows, cols = zip(*pixel_path)
        
        # Use Rasterio transform for conversion
        lons, lats = self.dem_loader.pixel_to_lat_lon(rows, cols)
        
        # Zip back into (lat, lon) tuples
        return list(zip(lats, lons))

    def get_3d_path_points(self, pixel_path: list) -> list:
        """
        Converts pixel path to 3D World Coordinates (Y-Up).
        Format: [(x, y, z), ...]
        
        X = Longitude (Meters relative to top-left)
        Y = Altitude (Meters above sea level - from DEM)
        Z = Latitude (Meters relative to top-left)
        """
        if not pixel_path: return []
        
        path_3d = []
        
        for r, c in pixel_path:
            # Y = Altitude
            altitude = float(self.dem[r, c])
            
            # Z = Latitude Displacement (Row * Meters_Per_Pixel_Z)
            z_pos = r * self.meter_per_z
            
            # X = Longitude Displacement (Col * Meters_Per_Pixel_X at that row)
            x_pos = c * self.meters_per_x_lookup[r]
            
            path_3d.append((x_pos, altitude, z_pos))
            
        return path_3d
    def get_surfcae_distance(self, loc1: tuple[float, float], loc2: tuple[float, float]) -> float:
        """
        Get the distance (ground distance, ignoring height) of two GPS points. Take into consideration of shrink of latitude shrink.
        Mainly using the Haversine distance formula to acheive the purpose.

        Args:
            - loc1 / loc2: tuple that stores the lat/lon coordinate
        """

        lat1, lon1 = loc1
        lat2, lon2 = loc2

        # Convert to radians
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

        dlon = lon2_r - lon1_r
        dlat = lat2_r - lat1_r

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000 # Earth radius in meter
        
        return c * r