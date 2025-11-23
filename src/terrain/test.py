import os
import numpy as np
import rasterio as rio
from rasterio.plot import show, show_hist
from pathlib import Path

# Use absolute path relative to this script
dem_path = Path(__file__).parent.parent.parent / "data" / "dem" / "merged_dem_sib_N54_N59_E090_E100.tif"

with rio.open(dem_path) as src:
    map = src.read(1)

    print(map)
    rows = src.height   # Number of pixel rows
    cols = src.width    # Number of pixel columns

print(f'Rows (height): {rows}, Columns (width): {cols}')