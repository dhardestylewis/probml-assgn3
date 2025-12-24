# -*- coding: utf-8 -*-
from probml.core.utils import get_logger

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union

# Attempt to import geopandas, but make its usage conditional
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
# from .utils import get_logger

logger = get_logger(__name__)

class SpatialAnalyzer:
    """
    Handles spatial analysis tasks, such as grid assignment and
    spatial pattern analysis.
    Requires geopandas for most functionalities.
    """
    def __init__(self, random_state: Optional[int] = None, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        self.grid_gdf: Optional['gpd.GeoDataFrame'] = None
        self.grid_params: Optional[Dict[str, Any]] = None
        self.cell_to_rows_lookup: Optional[Dict[str, list]] = None

        if not GEOPANDAS_AVAILABLE:
            logger.warning("Geopandas not available. Spatial analysis features will be limited.")

    def _log(self, message: str, level: str = "info"):
        if self.verbose:
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)

    def assign_grid_indices(self,
                            df: pd.DataFrame,
                            x_coord_col: str,
                            y_coord_col: str,
                            grid_size: Union[float, int], # e.g., 250 meters or degrees
                            crs: Optional[str] = None # Optional CRS for GeoDataFrame
                           ) -> pd.DataFrame:
        """
        Assigns spatial grid indices to properties based on their coordinates.

        Args:
            df: DataFrame with property data including coordinate columns.
            x_coord_col: Name of the column with x-coordinates (e.g., 'longitude', 'xcoord').
            y_coord_col: Name of the column with y-coordinates (e.g., 'latitude', 'ycoord').
            grid_size: The size of each grid cell in the units of the coordinates.
            crs: Coordinate Reference System string (e.g., 'EPSG:4326' for WGS84).
                 Required if creating a GeoDataFrame for self.grid_gdf.

        Returns:
            DataFrame with added 'grid_i', 'grid_j', and 'grid_cell_id' columns.
        """
        self._log(f"\n--- Assigning Spatial Grid Indices (Size: {grid_size}) ---")
        if not GEOPANDAS_AVAILABLE:
            self._log("Geopandas not available, cannot perform advanced grid assignment or create GeoDataFrame.", level="warning")

        if x_coord_col not in df.columns or y_coord_col not in df.columns:
            logger.error(f"Coordinate columns '{x_coord_col}' or '{y_coord_col}' not found.")
            return df.copy()

        _df = df.copy()
        _df[x_coord_col] = pd.to_numeric(_df[x_coord_col], errors='coerce')
        _df[y_coord_col] = pd.to_numeric(_df[y_coord_col], errors='coerce')

        n_initial_coords = len(_df)
        _df.dropna(subset=[x_coord_col, y_coord_col], inplace=True)
        n_valid_coords = len(_df)
        if n_initial_coords > n_valid_coords:
            self._log(f"Dropped {n_initial_coords - n_valid_coords} rows with NaN coordinates.")

        if n_valid_coords == 0:
            self._log("No valid coordinates remaining for grid assignment.", level="warning")
            _df['grid_i'] = pd.NA
            _df['grid_j'] = pd.NA
            _df['grid_cell_id'] = pd.NA
            return _df

        x_coords = _df[x_coord_col].values
        y_coords = _df[y_coord_col].values

        q_low, q_high = 0.001, 0.999
        finite_x_coords = x_coords[np.isfinite(x_coords)]
        finite_y_coords = y_coords[np.isfinite(y_coords)]

        if len(finite_x_coords) < 2 or len(finite_y_coords) < 2:
            self._log("Too few finite coordinate points to define robust grid boundaries using quantiles. Using min/max.", level="warning")
            x_min, x_max = np.min(finite_x_coords), np.max(finite_x_coords)
            y_min, y_max = np.min(finite_y_coords), np.max(finite_y_coords)
        else:
            x_min = np.quantile(finite_x_coords, q_low)
            x_max = np.quantile(finite_x_coords, q_high)
            y_min = np.quantile(finite_y_coords, q_low)
            y_max = np.quantile(finite_y_coords, q_high)

        x_min -= grid_size / 2
        x_max += grid_size / 2
        y_min -= grid_size / 2
        y_max += grid_size / 2

        grid_i = np.floor((x_coords - x_min) / grid_size).astype(int)
        grid_j = np.floor((y_coords - y_min) / grid_size).astype(int)

        n_cols = int(np.ceil((x_max - x_min) / grid_size))
        n_rows = int(np.ceil((y_max - y_min) / grid_size))

        grid_i = np.clip(grid_i, 0, n_cols -1 if n_cols > 0 else 0)
        grid_j = np.clip(grid_j, 0, n_rows -1 if n_rows > 0 else 0)

        _df['grid_i'] = grid_i
        _df['grid_j'] = grid_j
        _df['grid_cell_id'] = _df['grid_i'].astype(str) + "_" + _df['grid_j'].astype(str)

        self._log(f"Assigned grid indices. Grid dimensions: {n_cols} cols x {n_rows} rows.")
        self._log(f"Unique grid cells populated: {_df['grid_cell_id'].nunique()}")

        self.grid_params = {
            'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max,
            'n_cols': n_cols, 'n_rows': n_rows, 'grid_size': grid_size, 'crs': crs
        }

        self.cell_to_rows_lookup = _df.groupby('grid_cell_id').groups

        if GEOPANDAS_AVAILABLE and crs:
            polygons = []
            cell_ids_for_gdf = []
            for i_idx in range(n_cols):
                for j_idx in range(n_rows):
                    cell_id = f"{i_idx}_{j_idx}"
                    if cell_id in self.cell_to_rows_lookup:
                        cell_x_min = x_min + i_idx * grid_size
                        cell_y_min = y_min + j_idx * grid_size
                        cell_x_max = cell_x_min + grid_size
                        cell_y_max = cell_y_min + grid_size
                        polygons.append(Polygon([
                            (cell_x_min, cell_y_min), (cell_x_max, cell_y_min),
                            (cell_x_max, cell_y_max), (cell_x_min, cell_y_max)
                        ]))
                        cell_ids_for_gdf.append(cell_id)

            if polygons:
                self.grid_gdf = gpd.GeoDataFrame({'cell_id': cell_ids_for_gdf, 'geometry': polygons}, crs=crs)  # type: ignore
                self._log(f"Created GeoDataFrame for {len(self.grid_gdf)} populated grid cells.")
            else:
                self._log("No populated grid cells to create a GeoDataFrame.", level="warning")

        return _df

    # Future methods:
    # - analyze_spatial_autocorrelation (e.g., Moran's I on prices/residuals)
    # - create_spatial_lag_features
    # - visualize_spatial_patterns (choropleth maps etc.)

