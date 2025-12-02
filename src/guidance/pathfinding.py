from multiprocessing import heap
from ..terrain.dem_loader import DEMLoader
import math
from pathlib import Path
import sys
import time
import numpy as np


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

        # Load ata into ascontiguousarray for C/C++, float32 for dem, float64 for lookup
        self.dem = np.ascontiguousarray(dem.data, dtype=np.float32)

        # Initiating the lookup table and column value for future use
        self.row = self.dem_loader.shape[0] # shape: (rows, columns)
        self.col = self.dem_loader.shape[1]

        self.meter_per_z = abs(self.dem_loader.transform[4] * 111320)
        start_lat = self.dem_loader.transform[5]
        pixel_height = self.dem_loader.transform[4]


        # Setup lookup table (an array of latitude for every row)
        row_indices = np.arrange(self.row, dtype=np.float64)
        self.latitudes = start_lat + (row_indices * pixel_height)

        # Horizontal (east / west) setup - x: width in meter for each latitude
        base_width_meters = abs(self.dem_loader.transform[0] * 111320)

        # Setting up lookup table for C++
        meter_per_x_row = base_width_meters * np.cos(np.radians(self.latitudes))
        self.meter_per_x_lookup = np.ascontiguousarray(meter_per_x_row, dtype=np.float64)


        # Initiate C++ engine
        if CPP_AVAILABLE:
            print(f"Initializing C++ Engine with {self.rows * self.cols} pixels...")
            
            # start timer for knowing the execution speed
            start_time = time.time()



            print(f"C++ Engine Ready ({time.time() - start_time:.4f}s)")
        
        else:
            self.engine = None



    def find_path(self):
        pass