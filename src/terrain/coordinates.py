import math
from typing import Tuple

class CoordinateSystem:
    def __init__(self, origin_lat: float, origin_lon: float) -> None:
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon

        # Precompute Earth scale factor (meter per degree)
        # Earth lon / lat cosine
        self.meter_per_deg_lat = 111320
        self.meter_per_deg_lon = 111320 * math.cos(math.radians(origin_lat))

    def latlong_to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        """

        :param lat:
        :param lon:
        :return:
        """

        delta_lat = lat - self.origin_lat
        delta_lon = lon - self.origin_lon

        y = delta_lat * 111320

        x = delta_lon * 111320 * math.cos(math.radians(self.origin_lat))

        return x, y

    def get_distance(self, orig_lon: float, orig_lat: float, dest_lon: float, dest_lat: float) -> float:
        """
            Get the distance between two points using Haversine Distance.
            Which takes consider of the Earth radius and scale.

            Args:
                - orig_lon: longitude for start (origin)
                - orig_lat: latitude for start (origin)
                - dest_lon: longitude for destination
                - dest_lat: latitude for destination

            Return:
                the distance in meters
        """

        # distance between latitudes and longitudes
        dLat = (dest_lat - orig_lat) * math.pi / 180.0
        dLon = (dest_lon - orig_lon) * math.pi / 180.0

        # convert to radians
        orig_lat = (orig_lat) * math.pi / 180.0
        dest_lat = (dest_lat) * math.pi / 180.0

        # apply formulae
        a = (pow(math.sin(dLat / 2), 2) +
             pow(math.sin(dLon / 2), 2) *
             math.cos(orig_lat) * math.cos(dest_lat));
        rad = 6371000 # Earth's radius (in meters)
        c = 2 * math.asin(math.sqrt(a))
        return rad * c
