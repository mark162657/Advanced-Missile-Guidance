from dem_loader import DEMLoader

class TerrainDatabase:
    def __init__(self, dem_path) -> None:
        self.dem = DEMLoader(dem_path) # load the dem file (file path)

    def get_elevation(self, lat: float, lon: float) -> float | None:
        if lat < 0 or lon < 0:
            return None
        return self.dem.get_elevation(lat, lon)


    def get_elevation_patch(self, lat: float, lon: float):
        if lat < 0 or lon < 0:
            return None
        return self.dem.get_elevation_patch(lat, lon)

    
    

    
        