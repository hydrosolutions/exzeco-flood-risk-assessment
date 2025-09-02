#!/usr/bin/env python
"""
EXZECO (Extraction des Zones d'Ecoulement) Implementation
==========================================================

This module implements the EXZECO method for preliminary flood risk assessment
based on the methodology described by CEREMA.

The method uses Monte Carlo simulation with DEM perturbation to identify
potentially flooded areas by calculating flow accumulation paths multiple times
with random terrain modifications.

Windows Compatibility Notes:
- Uses pathlib.Path for all file operations
- Specifies encoding for text file operations
- Avoids Unix-specific subprocess calls
- Uses cross-platform parallel processing

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from shapely.geometry import Point, Polygon, box
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import xarray as xr
import rioxarray as rxr
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
import warnings
from tqdm import tqdm
# Import joblib with Windows-compatible backend
try:
    # Import Parallel and delayed for parallel execution and parallel_backend for context management
    from joblib import Parallel, delayed, parallel_backend
    import os
except ImportError as e:
    print(f"Warning: joblib import failed ({e}). Parallel processing will be disabled.")
    Parallel = None
    delayed = None
import numba as nb
from pathlib import Path
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExzecoConfig:
    """Configuration parameters for EXZECO analysis."""
    noise_levels: List[float] = None  # Noise levels in meters
    iterations: int = 100  # Number of Monte Carlo iterations
    min_drainage_area: float = 0.01  # Minimum drainage area in km²
    drainage_classes: List[float] = None  # Drainage area classes in km²
    n_jobs: int = -1  # Number of parallel jobs
    chunk_size: int = 1000  # Chunk size for processing
    seed: Optional[int] = 42  # Random seed for reproducibility
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        if self.drainage_classes is None:
            self.drainage_classes = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]


class ExzecoAnalysis:
    """
    Main class for EXZECO flood risk assessment.
    
    This class implements the complete EXZECO workflow including:
    - DEM preprocessing and pit filling
    - Monte Carlo simulation with random noise
    - D8 flow direction and accumulation
    - Multi-level analysis (20cm to 100cm)
    - Endorheic basin detection
    - Result aggregation and export
    """
    
    def __init__(self, config: Optional[ExzecoConfig] = None):
        """
        Initialize EXZECO analysis.
        
        Parameters
        ----------
        config : ExzecoConfig, optional
            Configuration parameters. If None, uses defaults.
        """
        self.config = config or ExzecoConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Flow direction encoding (D8 algorithm)
        self.flow_directions = {
            1: (0, 1),    # East
            2: (1, 1),    # Southeast
            4: (1, 0),    # South
            8: (1, -1),   # Southwest
            16: (0, -1),  # West
            32: (-1, -1), # Northwest
            64: (-1, 0),  # North
            128: (-1, 1)  # Northeast
        }
        
        # Store results
        self.results = {}
        self.dem_data = None
        self.transform = None
        self.crs = None
        
    def load_dem(self, dem_path: Union[str, Path], bounds: Optional[Tuple] = None) -> np.ndarray:
        """
        Load and preprocess DEM data.
        
        Parameters
        ----------
        dem_path : str or Path
            Path to DEM file (GeoTIFF format)
        bounds : tuple, optional
            Bounding box (minx, miny, maxx, maxy) to clip DEM
            
        Returns
        -------
        np.ndarray
            Preprocessed DEM array
        """
        logger.info(f"Loading DEM from {dem_path}")
        
        with rasterio.open(dem_path) as src:
            if bounds:
                # Clip to bounds
                window = rasterio.windows.from_bounds(*bounds, src.transform)
                dem = src.read(1, window=window)
                self.transform = rasterio.windows.transform(window, src.transform)
                dem_bounds = rasterio.windows.bounds(window, src.transform)
                # Convert tuple to BoundingBox-like access
                minx, miny, maxx, maxy = dem_bounds
            else:
                dem = src.read(1)
                self.transform = src.transform
                dem_bounds = src.bounds
                # dem_bounds is already a BoundingBox object
                minx, miny, maxx, maxy = dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top
            
            self.crs = src.crs
            
            # FIX: Calculate actual ground resolution in meters
            if src.crs.is_geographic:
                # Geographic coordinates (lat/lon) - convert to meters
                lon_extent = maxx - minx
                lat_extent = maxy - miny
                
                # Approximate conversion to meters
                lat_center = (maxy + miny) / 2
                lon_extent_m = lon_extent * 111320 * np.cos(np.radians(lat_center))
                lat_extent_m = lat_extent * 111320
                
                self.resolution_x = lon_extent_m / src.width if bounds is None else lon_extent_m / dem.shape[1]
                self.resolution_y = lat_extent_m / src.height if bounds is None else lat_extent_m / dem.shape[0]
                self.resolution = (self.resolution_x + self.resolution_y) / 2  # Average
                
                logger.info(f"Geographic CRS detected. Calculated resolution: X={self.resolution_x:.1f}m, Y={self.resolution_y:.1f}m")
            else:
                # Projected coordinates - use directly
                self.resolution = src.res[0]
                self.resolution_x = src.res[0]
                self.resolution_y = src.res[1]
            
        # Handle nodata values
        dem = np.where(dem < -9999, np.nan, dem)
        
        # Fill pits (essential for flow routing)
        dem_filled = self._fill_pits(dem)
        
        self.dem_data = dem_filled
        self.shape = dem_filled.shape
        
        logger.info(f"DEM loaded: shape={self.shape}, resolution={self.resolution:.1f}m")
        
        return dem_filled
    
    @staticmethod
    @nb.jit(nopython=True, parallel=False)
    def _fill_pits_numba(dem: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Fast pit filling using Numba JIT compilation.
        
        Uses the Planchon-Darboux algorithm for pit filling.
        """
        rows, cols = dem.shape
        filled = np.copy(dem)
        filled[np.isnan(filled)] = -9999
        
        # Initialize with very high values except at edges
        w = np.full_like(dem, 1e10)
        
        # Set edges to DEM values
        w[0, :] = filled[0, :]
        w[-1, :] = filled[-1, :]
        w[:, 0] = filled[:, 0]
        w[:, -1] = filled[:, -1]
        
        # Iterative filling
        changed = True
        while changed:
            changed = False
            
            # Forward pass
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if filled[i, j] == -9999:
                        continue
                        
                    neighbors = [
                        w[i-1, j], w[i+1, j],
                        w[i, j-1], w[i, j+1],
                        w[i-1, j-1], w[i-1, j+1],
                        w[i+1, j-1], w[i+1, j+1]
                    ]
                    
                    min_neighbor = min(neighbors)
                    new_val = max(filled[i, j], min_neighbor + epsilon)
                    
                    if abs(w[i, j] - new_val) > epsilon:
                        w[i, j] = new_val
                        changed = True
            
            # Backward pass
            for i in range(rows - 2, 0, -1):
                for j in range(cols - 2, 0, -1):
                    if filled[i, j] == -9999:
                        continue
                        
                    neighbors = [
                        w[i-1, j], w[i+1, j],
                        w[i, j-1], w[i, j+1],
                        w[i-1, j-1], w[i-1, j+1],
                        w[i+1, j-1], w[i+1, j+1]
                    ]
                    
                    min_neighbor = min(neighbors)
                    new_val = max(filled[i, j], min_neighbor + epsilon)
                    
                    if abs(w[i, j] - new_val) > epsilon:
                        w[i, j] = new_val
                        changed = True
        
        # Replace nodata
        w[filled == -9999] = np.nan
        
        return w
    
    def _fill_pits(self, dem: np.ndarray) -> np.ndarray:
        """
        Fill pits in DEM for hydrological correctness.
        """
        logger.info("Filling pits in DEM...")
        
        # Use Numba-accelerated version if possible
        try:
            filled = self._fill_pits_numba(dem)
        except:
            # Fallback to scipy
            logger.warning("Numba pit filling failed, using scipy fallback")
            filled = ndimage.generic_filter(dem, np.nanmean, size=3)
            
        return filled
    
    def _add_noise(self, dem: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add random noise to DEM.
        
        Parameters
        ----------
        dem : np.ndarray
            Original DEM
        noise_level : float
            Maximum noise level in meters
            
        Returns
        -------
        np.ndarray
            DEM with added noise
        """
        # Create random mask (20% of pixels get noise as per EXZECO spec)
        mask = np.random.random(dem.shape) < 0.2
        
        # Add noise only to masked pixels
        noise = np.zeros_like(dem)
        noise[mask] = noise_level
        
        return dem + noise
    
    @staticmethod
    @nb.jit(nopython=True)
    def _compute_flow_direction_d8(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute D8 flow direction and slopes using Numba.
        
        Returns flow direction grid and slope grid.
        """
        rows, cols = dem.shape
        flow_dir = np.zeros((rows, cols), dtype=np.int32)
        slopes = np.zeros((rows, cols), dtype=np.float32)
        
        # D8 neighbor offsets and powers of 2
        offsets = [
            (0, 1, 1),    # E
            (1, 1, 2),    # SE
            (1, 0, 4),    # S
            (1, -1, 8),   # SW
            (0, -1, 16),  # W
            (-1, -1, 32), # NW
            (-1, 0, 64),  # N
            (-1, 1, 128)  # NE
        ]
        
        for i in range(rows):
            for j in range(cols):
                if np.isnan(dem[i, j]):
                    continue
                
                max_slope = -np.inf
                max_dir = 0
                
                for di, dj, direction in offsets:
                    ni, nj = i + di, j + dj
                    
                    if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(dem[ni, nj]):
                        # Calculate slope
                        distance = np.sqrt(di*di + dj*dj)
                        slope = (dem[i, j] - dem[ni, nj]) / distance
                        
                        if slope > max_slope:
                            max_slope = slope
                            max_dir = direction
                
                flow_dir[i, j] = max_dir
                slopes[i, j] = max_slope if max_slope > 0 else 0
        
        return flow_dir, slopes
    
    def _compute_flow_accumulation(self, flow_dir: np.ndarray) -> np.ndarray:
        """
        Compute flow accumulation from flow direction.
        
        Parameters
        ----------
        flow_dir : np.ndarray
            D8 flow direction grid
            
        Returns
        -------
        np.ndarray
            Flow accumulation grid (number of upstream cells)
        """
        rows, cols = flow_dir.shape
        flow_acc = np.ones((rows, cols), dtype=np.float32)
        flow_acc[np.isnan(self.dem_data)] = 0
        
        # Build dependency graph
        dependencies = {}
        for i in range(rows):
            for j in range(cols):
                if flow_dir[i, j] == 0:
                    continue
                    
                # Find downstream cell
                direction = flow_dir[i, j]
                di, dj = self.flow_directions.get(direction, (0, 0))
                ni, nj = i + di, j + dj
                
                if 0 <= ni < rows and 0 <= nj < cols:
                    key = (ni, nj)
                    if key not in dependencies:
                        dependencies[key] = []
                    dependencies[key].append((i, j))
        
        # Iterative topological sort and accumulation to avoid recursion limit
        visited = np.zeros((rows, cols), dtype=bool)
        
        def accumulate_iterative(start_i, start_j):
            """Iterative implementation to avoid recursion depth issues"""
            stack = [(start_i, start_j)]
            processing_stack = []
            
            # Build dependency chain using DFS
            while stack:
                i, j = stack.pop()
                
                if visited[i, j]:
                    continue
                    
                processing_stack.append((i, j))
                visited[i, j] = True
                
                # Add upstream cells to stack for processing
                if (i, j) in dependencies:
                    for ui, uj in dependencies[(i, j)]:
                        if not visited[ui, uj]:
                            stack.append((ui, uj))
            
            # Process in reverse order to ensure upstream cells are processed first
            while processing_stack:
                i, j = processing_stack.pop()
                
                if (i, j) in dependencies:
                    for ui, uj in dependencies[(i, j)]:
                        flow_acc[i, j] += flow_acc[ui, uj]
        
        # Process all cells
        for i in range(rows):
            for j in range(cols):
                if not visited[i, j] and flow_dir[i, j] != 0:
                    accumulate_iterative(i, j)
        
        return flow_acc
    
    def _single_iteration(self, noise_level: float) -> np.ndarray:
        """
        Single Monte Carlo iteration with noise addition.
        
        Parameters
        ----------
        noise_level : float
            Noise level to add to DEM
            
        Returns
        -------
        np.ndarray
            Binary flood zone mask for this iteration
        """
        # Add noise to DEM
        dem_noisy = self._add_noise(self.dem_data, noise_level)
        
        # Compute flow direction
        flow_dir, _ = self._compute_flow_direction_d8(dem_noisy)
        
        # Compute flow accumulation
        flow_acc = self._compute_flow_accumulation(flow_dir)
        
        # FIX: Convert to drainage area using correct pixel area calculation
        pixel_area_m2 = self.resolution_x * self.resolution_y  # Area in m²
        pixel_area_km2 = pixel_area_m2 / 1e6  # Convert to km²
        drainage_area = flow_acc * pixel_area_km2
        
        # Create binary mask for areas above threshold
        mask = drainage_area >= self.config.min_drainage_area
        
        return mask.astype(np.float32)
    
    def run_monte_carlo(self, noise_level: float, progress_bar: bool = True) -> np.ndarray:
        """
        Run Monte Carlo simulation for a specific noise level.
        
        Parameters
        ----------
        noise_level : float
            Noise level in meters
        progress_bar : bool
            Show progress bar
            
        Returns
        -------
        np.ndarray
            Probability map (0-1) of flood zones
        """
        logger.info(f"Running Monte Carlo for noise level {noise_level}m with {self.config.iterations} iterations")
        
        # Parallel execution with Windows compatibility
        if self.config.n_jobs != 1 and Parallel is not None:
            try:
                iterator = tqdm(range(self.config.iterations), desc=f"MC {noise_level}m") if progress_bar else range(self.config.iterations)
                
                # Use threading backend on Windows to avoid subprocess issues
                if os.name == 'nt':  # Windows
                    from joblib import parallel_backend
                    with parallel_backend('threading', n_jobs=self.config.n_jobs):
                        results = Parallel(n_jobs=self.config.n_jobs)(
                            delayed(self._single_iteration)(noise_level) for _ in iterator
                        )
                else:
                    results = Parallel(n_jobs=self.config.n_jobs)(
                        delayed(self._single_iteration)(noise_level) for _ in iterator
                    )
                
                # Aggregate results
                probability_map = np.mean(results, axis=0)
            except Exception as e:
                logger.warning(f"Parallel execution failed ({e}), falling back to sequential")
                # Fall back to sequential execution
                probability_map = np.zeros(self.shape, dtype=np.float32)
                iterator = tqdm(range(self.config.iterations), desc=f"MC {noise_level}m") if progress_bar else range(self.config.iterations)
                for _ in iterator:
                    mask = self._single_iteration(noise_level)
                    probability_map += mask
                probability_map /= self.config.iterations
        else:
            # Sequential execution
            probability_map = np.zeros(self.shape, dtype=np.float32)
            
            iterator = tqdm(range(self.config.iterations), desc=f"MC {noise_level}m") if progress_bar else range(self.config.iterations)
            
            for _ in iterator:
                mask = self._single_iteration(noise_level)
                probability_map += mask
            
            probability_map /= self.config.iterations
        
        return probability_map
    
    def run_full_analysis(self, dem_path: Union[str, Path], bounds: Optional[Tuple] = None) -> Dict:
        """
        Run complete EXZECO analysis for all noise levels.
        
        Parameters
        ----------
        dem_path : str or Path
            Path to DEM file
        bounds : tuple, optional
            Bounding box for analysis area
            
        Returns
        -------
        dict
            Results dictionary with probability maps for each noise level
        """
        # Load DEM
        self.load_dem(dem_path, bounds)
        
        # Run analysis for each noise level
        results = {}
        
        for noise_level in self.config.noise_levels:
            logger.info(f"Processing noise level: {noise_level}m")
            
            # Run Monte Carlo
            prob_map = self.run_monte_carlo(noise_level)
            
            # Apply incremental DEM modification for next level
            if noise_level < self.config.noise_levels[-1]:
                # Modify DEM for areas with high flow accumulation
                flow_dir, _ = self._compute_flow_direction_d8(self.dem_data)
                flow_acc = self._compute_flow_accumulation(flow_dir)
                
                # FIX: Use correct pixel area calculation
                pixel_area_m2 = self.resolution_x * self.resolution_y
                pixel_area_km2 = pixel_area_m2 / 1e6
                drainage_area = flow_acc * pixel_area_km2
                
                # Increase elevation where drainage > 0.1 km²
                mask = drainage_area > 0.1
                self.dem_data[mask] += noise_level
            
            # Store results
            results[f"exzeco_{int(noise_level*100)}cm"] = {
                'probability_map': prob_map,
                'noise_level': noise_level,
                'threshold': 0.5  # Default threshold for binary classification
            }
        
        self.results = results
        return results
    
    def classify_drainage_areas(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Classify flood zones by drainage area.
        
        Parameters
        ----------
        prob_map : np.ndarray
            Probability map from Monte Carlo
            
        Returns
        -------
        np.ndarray
            Classified drainage areas
        """
        # Compute flow accumulation for original DEM
        flow_dir, _ = self._compute_flow_direction_d8(self.dem_data)
        flow_acc = self._compute_flow_accumulation(flow_dir)
        
        # FIX: Convert to drainage area using correct pixel area
        pixel_area_m2 = self.resolution_x * self.resolution_y
        pixel_area_km2 = pixel_area_m2 / 1e6
        drainage_area = flow_acc * pixel_area_km2
        
        # Create classified array
        classified = np.zeros_like(drainage_area, dtype=np.int8)
        
        # Apply probability threshold
        flood_mask = prob_map > 0.5
        
        # Classify by drainage area
        for i, threshold in enumerate(self.config.drainage_classes):
            if i == 0:
                mask = (drainage_area >= threshold) & flood_mask
            else:
                mask = (drainage_area >= threshold) & (drainage_area < self.config.drainage_classes[i-1]) & flood_mask
            
            classified[mask] = i + 1
        
        return classified
    
    def detect_endorheic_basins(self) -> np.ndarray:
        """
        Detect endorheic (closed) basins.
        
        Returns
        -------
        np.ndarray
            Binary mask of endorheic areas
        """
        logger.info("Detecting endorheic basins...")
        
        # Find local minima (pits)
        dem_smooth = ndimage.gaussian_filter(self.dem_data, sigma=1)
        local_min = ndimage.minimum_filter(dem_smooth, size=3)
        pits = (dem_smooth == local_min) & ~np.isnan(dem_smooth)
        
        # Label connected components
        labeled, num_features = ndimage.label(pits)
        
        # Calculate basin properties
        endorheic_mask = np.zeros_like(self.dem_data, dtype=bool)
        
        for i in range(1, num_features + 1):
            basin_mask = labeled == i
            
            # Check if basin drains to edge
            if np.any(basin_mask[0, :]) or np.any(basin_mask[-1, :]) or \
               np.any(basin_mask[:, 0]) or np.any(basin_mask[:, -1]):
                continue  # Drains to edge, not endorheic
            
            endorheic_mask |= basin_mask
        
        return endorheic_mask
    
    def export_results(self, output_dir: Union[str, Path], format: str = 'geotiff') -> None:
        """
        Export analysis results with Windows compatibility.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        format : str
            Export format ('geotiff', 'shapefile', 'geojson')
        """
        # Use pathlib for cross-platform compatibility
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting results to {output_dir}")
        
        for name, data in self.results.items():
            prob_map = data['probability_map']
            
            if format == 'geotiff':
                # Export as GeoTIFF with cross-platform path handling
                output_path = output_dir / f"{name}.tif"
                
                with rasterio.open(
                    str(output_path),  # Convert Path to string for rasterio compatibility
                    'w',
                    driver='GTiff',
                    height=prob_map.shape[0],
                    width=prob_map.shape[1],
                    count=1,
                    dtype=prob_map.dtype,
                    crs=self.crs,
                    transform=self.transform,
                    compress='lzw'
                ) as dst:
                    dst.write(prob_map, 1)
                    
            elif format in ['shapefile', 'geojson']:
                # Vectorize and export with proper path handling
                shapes = features.shapes(
                    (prob_map > 0.5).astype(np.uint8),
                    transform=self.transform
                )
                
                geometries = []
                values = []
                
                for geom, value in shapes:
                    if value == 1:  # Only flood zones
                        geometries.append(Polygon(geom['coordinates'][0]))
                        values.append(data['noise_level'])
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    {'noise_level': values},
                    geometry=geometries,
                    crs=self.crs
                )
                
                # Export with cross-platform path handling
                if format == 'shapefile':
                    output_path = output_dir / f"{name}.shp"
                    gdf.to_file(str(output_path))  # Convert Path to string
                else:  # geojson
                    output_path = output_dir / f"{name}.geojson"
                    gdf.to_file(str(output_path), driver='GeoJSON')  # Convert Path to string
            
            logger.info(f"Exported {name} to {output_path}")
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate summary report of analysis.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        report_data = []
        
        for name, data in self.results.items():
            prob_map = data['probability_map']
            flood_mask = prob_map > 0.5
            
            # FIX: Use correct pixel area calculation
            pixel_area_m2 = self.resolution_x * self.resolution_y
            pixel_area_km2 = pixel_area_m2 / 1e6
            
            stats = {
                'Analysis': name,
                'Noise Level (m)': data['noise_level'],
                'Total Area (km²)': np.sum(~np.isnan(self.dem_data)) * pixel_area_km2,
                'Flood Area (km²)': np.sum(flood_mask) * pixel_area_km2,
                'Flood Area (%)': 100 * np.sum(flood_mask) / np.sum(~np.isnan(self.dem_data)),
                'Mean Probability': np.nanmean(prob_map),
                'Max Probability': np.nanmax(prob_map),
                'Pixels > 0.8 Prob': np.sum(prob_map > 0.8)
            }
            
            report_data.append(stats)
        
        return pd.DataFrame(report_data)


def load_config(config_path: Union[str, Path]) -> ExzecoConfig:
    """
    Load configuration from YAML file with Windows compatibility.
    
    Parameters
    ----------
    config_path : str or Path
        Path to config file
        
    Returns
    -------
    ExzecoConfig
        Configuration object
    """
    # Use pathlib for cross-platform compatibility
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        # Specify encoding for Windows compatibility
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")
    except Exception as e:
        raise IOError(f"Error reading configuration file: {e}")
    
    if not isinstance(config_dict, dict):
        raise ValueError("Configuration file must contain a YAML dictionary")
        
    exzeco_params = config_dict.get('exzeco', {})
    processing_params = config_dict.get('processing', {})
    
    return ExzecoConfig(
        noise_levels=exzeco_params.get('noise_levels'),
        iterations=exzeco_params.get('iterations', 100),
        min_drainage_area=exzeco_params.get('min_drainage_area', 0.01),
        drainage_classes=exzeco_params.get('drainage_classes'),
        n_jobs=processing_params.get('n_jobs', -1),
        chunk_size=processing_params.get('chunk_size', 1000),
        seed=processing_params.get('seed', 42)
    )


if __name__ == "__main__":
    # Example usage
    config = ExzecoConfig(iterations=50)  # Reduced for testing
    analyzer = ExzecoAnalysis(config)
    
    # Run analysis (requires DEM file)
    # results = analyzer.run_full_analysis("path/to/dem.tif")
    # analyzer.export_results("output/")
    # report = analyzer.generate_report()
    # print(report)
    
    print("EXZECO module loaded successfully")