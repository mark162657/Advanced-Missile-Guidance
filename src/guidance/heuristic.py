import math
import numpy as np
from ..terrain.dem_loader import DEMLoader
from pathlib import Path

class Heuristic:
    def __init__(self, ):
        tif_path = Path(__file__).parent.parent.parent / 'data' / 'dem' / 'merged_dem_sib_N54_N59_E090_E100.tif'
        dem = DEMLoader(tif_path)

        self.dem_loader = dem
        self.dem = dem.data


        # Vertical (north / south) setup - y
        self.meter_per_y = abs(self.dem_loader.transform[4] * 111,320) # 111320 is meter per degree (lat)

        # Horizontal (east / west) setup - x

        # Initiating the lookup table

        self.row = self.dem_loader.shape[0]

        





        

