import numpy as np
import math
from ..terrain.dem_loader import DEMLoader

class GPS:
    def __init__(self, horizontal_accuracy: float, vertical_accuracy: float, update_freq: int=1):
        """
        Assume (feature might be added in future, depends):
            - signal travel time does not exists + processing delay
            - PPS + WAGE enhancement signal (acheiving around 1 meter accuracy)
            - ignore signal processing layer, solved triangulation and output Cartesian coordinates (x, y, z) directly
        """
        
        self.update_freq = update_freq
        self.horizontal_accuracy = horizontal_accuracy
        self.vertical_accuracy = vertical_accuracy

        dem = DEMLoader()
        self.dem_laoder = dem

    def get_measurement(self, current_position: tuple[int, int]):
        """
        We accept the "perfect" true position, add it to the error/drift and return the simulated non-perfect
        GPS location.

        Args:
            - current_position: (row, col) tuple
        """
        curr_lat, curr_lon = self.dem_loader.pixel_to_lat_lon(current_position[0], current_position[1])
        position = [curr_lat, curr_lon]






    def is_ready(self, current_time):
        pass

    def is_jammed(self) -> bool:
        pass

        


    
    