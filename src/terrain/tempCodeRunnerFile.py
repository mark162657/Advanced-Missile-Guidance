# src/terrain/dem_loader.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
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
        
            if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
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
    lat, lon = 56.0, 95.0
    elev = dem.get_elevation(lat, lon)
    if elev:
        print(f"  Elevation at ({lat}, {lon}): {elev:.2f}m")
    else:
        print(f"  ⚠️  Coordinate ({lat}, {lon}) outside tile bounds")

    # Load DEM data for plotting and handle NoData values
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(float)  # Convert to float for NaN handling
        nodata = src.nodata
        bounds = src.bounds

        # Replace NoData values with NaN to prevent matplotlib errors
        if nodata is not None:
            dem_data[dem_data == nodata] = np.nan

    # Handle downsampling
    downsample_size = 10
    dem_downsampled = dem_data[::downsample_size, ::downsample_size]

    # Print console
    print(f"\n📊 Visualization Info:\n")
    print(f"  Original shape: {dem_data.shape} | Total pixel: {dem_data.size:,})") # show shape and total pixels
    print(f"  Downsampled shae: {dem_downsampled.shape} | Total pixel: {dem_downsampled.size:,}")
    print(f"  Reduction by: {dem_data.size / dem_downsampled.size:.0f}x")

    # AI generated sections about custom colour sgements: (im so bad at colour and art...) 
    # green -> yellow -> brown -> white
    colors = ['#2d5016', '#5a8c2b', '#8fb359', '#d4c77e', '#a67c52', '#ffffff']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('terrain_land', colors, N=n_bins)

    # AI assisted: Calculate elevation percentiles to exclude extreme outliers
    vmin = np.nanpercentile(dem_downsampled, 2)   # 2nd percentile
    vmax = np.nanpercentile(dem_downsampled, 98)  # 98th percentile

    # AI: Create hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem_downsampled, vert_exag=2, dx=1, dy=1)


    # Plot DEM
    plt.figure(figsize=(14, 10))
    plt.imshow(hillshade, cmap='gray', extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), alpha=0.3)
    plt.imshow(dem_downsampled, cmap=cmap_custom, aspect='auto', extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), interpolation='bilinear', alpha=0.7, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Elevation (meters)', fraction=0.046, pad=0.04)
    plt.title(f'Downsampled 1:{downsample_size} DEM of Siberia', fontsize=14, fontweight='bold')
    plt.xlabel('Longitude (degrees)', fontsize=12)
    plt.ylabel('Latitude (degrees)', fontsize=12)
    plt.gca().invert_yaxis()  # having north on top (traditional map layout)
    plt.tight_layout()
    plt.show()

    print('\nPlot generated')

