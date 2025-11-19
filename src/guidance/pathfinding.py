from ..terrain.dem_loader import DEMLoader
import math
from pathlib import Path
import numpy as np

class Pathfinding:
    """
    Using A*pathfinding algorithm to find the most ideal path that considers the horizontal and vertical movement.
    """
    def __init__(self):
        tif_path = Path(__file__).parent.parent.parent / 'data' / 'dem' / 'merged_dem_sib_N54_N59_E090_E100.tif'
        dem = DEMLoader(tif_path)
        self.dem_loader = dem
        self.dem = dem.data

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

    def heuristic(self, node1, node2) -> float:
        """

        """
        

        


        
        





















