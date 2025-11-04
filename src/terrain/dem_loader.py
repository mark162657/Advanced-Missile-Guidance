# src/terrain/dem_loader.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import rowcol, xy


class DEMLoader:
    """Loads and queries elevation data from a single SRTM file."""

    def __init__(self, dem_path: Path) -> None:
        """
        Initialise DEM loader.

        For the data:
            read first band as NumPy array (rows, cols) = (height, width)
            so the index will be array[row, col]

        Args:
            dem_path: the path to the dem file
        """

        # Check existance
        self.path = Path(dem_path)
        if not self.path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.path}. Check again.")

        # Load DEM data
        with rasterio.open(self.path) as src: # path will be provided by user
            self.data = src.read(1)
            self.transform = src.transform # row, col <-> gps
            self.crs = src.crs
            self.bounds = src.bounds # see if coordinate is inside the bound
            self.shape = self.data.shape
            self.nodata = src.nodata # invalid data (pixel)


    def get_elevation(self, lat, lon):
        """
        Get elevation at GPS coordinates.

        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)

        Returns:
            float: Elevation in meters, or None if out of bounds
        """
        try: 
            row, col = rowcol(self.transform, lon, lat)
        
            if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
                elev = float(self.data[row, col])
                return elev if elev != self.nodata else None # return elevation if its not nodata
            return None
    
        except Exception:
            return None
    
            
    

    # def get_elevation_patch(self, lat, lon, patch_size=7):
    #     """
    #     Extract terrain patch around coordinates (for TERCOM).

    #     Args:
    #         lat, lon: Center coordinates
    #         patch_size: Patch will be (patch_size × patch_size)

    #     Returns:
    #         np.ndarray: Normalized elevation patch, or None
    #     """
    #     try:
    #         row, col = rowcol(self.transform, lon, lat)
    #         half = patch_size // 2

    #         # Extract patch
    #         r_start = max(0, row - half)
    #         r_end = min(self.shape[0], row + half + 1)
    #         c_start = max(0, col - half)
    #         c_end = min(self.shape[1], col + half + 1)

    #         patch = self.data[r_start:r_end, c_start:c_end]

    #         # Normalize (zero mean, unit variance)
    #         patch = patch.astype(float)
    #         patch = (patch - patch.mean()) / (patch.std() + 1e-6)

    #         return patch
    #     except Exception:
    #         return None

    def lat_lon_to_pixel(self, lat, lon):
        """Convert GPS to pixel coordinates."""
        return rowcol(self.transform, lon, lat)

    def pixel_to_lat_lon(self, row, col):
        """Convert pixel to GPS coordinates."""
        lon, lat = xy(self.transform, row, col)
        return lat, lon

    def close(self):
        """Cleanup (data already loaded, but keeps interface consistent)."""
        pass


# Quick test
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    # Set the root for the project
    project_root = script_dir.parents[1]
    # This guides where the tif file is located, for testing
    dem_path = project_root / "data" / "dem" / "merged_dem_sib_N54_N59_E090_E100.tif"

    dem = DEMLoader(dem_path)
    print(f"✓ DEM loaded: {dem.path.name}")
    print(f"  Shape: {dem.shape}")
    print(f"  Bounds: {dem.bounds}")

    # Test the lat/lon to elevation query, should be in range
    lat, lon = 47.0, 33.0
    elev = dem.get_elevation(lat, lon)
    if elev:
        print(f"  Elevation at ({lat}, {lon}): {elev:.2f}m")
    else:
        print(f"  ⚠️  Coordinate ({lat}, {lon}) outside tile bounds")

    # Load DEM data for plotting and handle NoData values
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(float)  # Convert to float for NaN handling
        nodata = src.nodata

        # Replace NoData values with NaN to prevent matplotlib errors
        if nodata is not None:
            dem_data[dem_data == nodata] = np.nan

    # Plot DEM
    plt.figure(figsize=(10, 8))
    plt.imshow(dem_data, cmap='terrain')
    plt.colorbar(label='Elevation (meters)')
    plt.title('DEM Elevation (SRTM)')
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    plt.gca().invert_yaxis()  # having north on top (traditional map layout)
    plt.show()
