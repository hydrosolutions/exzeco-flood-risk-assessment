#!/usr/bin/env python
"""
EXZECO (Extraction des Zones d'Ecoulement) Implementation
==========================================================

This module implements the EXZECO method for preliminary flood risk assessment
based on the methodology described by CEREMA.

The method uses Monte Carlo simulation with DEM perturbation to identify
potentially flooded areas by calculating flow accumulation paths multiple times
with random terrain modifications.

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from shapely.geometry import Point, Polygon, MultiPolygon, box
import rasterio
from rasterio import features
from rasterio.transform import from_bounds, Affine
from rasterio.warp import reproject, Resampling
import xarray as xr
import rioxarray as rxr
from typing import Tuple, Dict, List, Optional, Union, Any, Callable
from numpy.typing import NDArray
from dataclasses import dataclass
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import numba as nb
from pathlib import Path
import yaml
import logging

# Import new core modules
from core import (
    FlowAnalyzer,
    MonteCarloSimulator,
    GeometryProcessor,
    DrainageClassifier,
    ClassificationThresholds,
    ResultExporter
)

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExzecoConfig:
    """Configuration parameters for EXZECO analysis."""
    noise_levels: Optional[List[float]] = None  # Noise levels in meters
    iterations: int = 100  # Number of Monte Carlo iterations
    min_drainage_area: float = 0.01  # Minimum drainage area in km²
    drainage_classes: Optional[List[float]] = None  # Drainage area classes in km²
    n_jobs: int = -1  # Number of parallel jobs
    chunk_size: int = 1000  # Chunk size for processing
    seed: Optional[int] = 42  # Random seed for reproducibility
    shapefile_path: Optional[str] = None  # Path to shapefile for study area definition
    bounds: Optional[Tuple[float, float, float, float]] = None  # Fallback bounding box
    
    def __post_init__(self) -> None:
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
    
    def __init__(self, config: Optional[ExzecoConfig] = None) -> None:
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
        
        # Initialize core modules using composition pattern
        self.flow_analyzer = FlowAnalyzer()
        self.geometry_processor = GeometryProcessor()
        
        # Classification thresholds based on config
        classification_thresholds = ClassificationThresholds(
            very_low=0.001,
            low=0.01,
            medium=0.1,
            high=1.0,
            very_high=10.0
        )
        self.drainage_classifier = DrainageClassifier(classification_thresholds)
        
        # Monte Carlo simulator will be initialized with flow analyzer
        self.monte_carlo_simulator = MonteCarloSimulator(self.flow_analyzer)
        
        # Result exporter will be initialized when output directory is known
        self.result_exporter: Optional[ResultExporter] = None
        
        # Flow direction encoding (D8 algorithm) - kept for backward compatibility
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
        self.results: Dict[str, Dict[str, Any]] = {}
        self.dem_data: Optional[NDArray[np.floating]] = None
        self.transform: Optional[Affine] = None
        self.crs: Optional[Any] = None
        self.study_areas: Optional[gpd.GeoDataFrame] = None  # For storing individual subcatchments
        self.total_study_area: Optional[gpd.GeoDataFrame] = None  # For storing entire domain
        self.resolution: Optional[float] = None
        self.resolution_x: Optional[float] = None
        self.resolution_y: Optional[float] = None
        
    def load_study_areas(self, shapefile_path: Optional[str] = None, bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[gpd.GeoDataFrame, Tuple[float, float, float, float]]:
        """
        Load study areas from shapefile or bounds.
        
        Parameters
        ----------
        shapefile_path : str, optional
            Path to shapefile/geopackage containing study area polygons
        bounds : tuple, optional
            Bounding box (minx, miny, maxx, maxy) as fallback
            
        Returns
        -------
        Tuple[gpd.GeoDataFrame, Tuple[float, float, float, float]]
            (GeoDataFrame of study areas, total bounds)
            
        Raises
        ------
        ValueError
            If neither shapefile nor bounds are provided or valid
        """
        # Try shapefile first using GeometryProcessor
        if shapefile_path is not None:
            try:
                gdf = self.geometry_processor.load_study_area(shapefile_path)
                
                # Store individual subcatchments
                self.study_areas = gdf
                
                # Create total study area (dissolved geometry)
                total_geom = gdf.geometry.unary_union
                if hasattr(total_geom, 'geoms'):
                    # If it's a MultiPolygon, keep as is
                    from shapely.geometry import MultiPolygon
                    if not isinstance(total_geom, MultiPolygon):
                        total_geom = MultiPolygon([total_geom])
                
                total_gdf = gpd.GeoDataFrame([{'name': 'total_domain'}], 
                                           geometry=[total_geom], 
                                           crs=gdf.crs)
                self.total_study_area = total_gdf
                
                total_bounds = self.geometry_processor.get_bbox(gdf)
                
                logger.info(f"Loaded {len(gdf)} subcatchments from shapefile")
                logger.info(f"Total study area bounds: {total_bounds}")
                
                return gdf, total_bounds
                
            except Exception as e:
                logger.warning(f"Failed to load shapefile {shapefile_path}: {e}")
        
        # Fall back to bounding box
        if bounds is not None and len(bounds) == 4:
            logger.info(f"Using bounding box: {bounds}")
            
            # Create a rectangular polygon from bounds
            from shapely.geometry import box
            geom = box(*bounds)
            
            # Create GeoDataFrame with single feature
            gdf = gpd.GeoDataFrame([{'name': 'bounding_box'}], 
                                 geometry=[geom], 
                                 crs='EPSG:4326')  # Assume WGS84 for bounds
            
            self.study_areas = gdf
            self.total_study_area = gdf.copy()
            
            return gdf, bounds
        
        # If nothing works, raise error
        raise ValueError("Neither valid shapefile nor bounds provided. Please specify either a valid shapefile path in config.yml or bounding box coordinates.")
    
    def mask_raster_by_geometry(self, raster: np.ndarray, geometry: Union[Polygon, MultiPolygon], transform: Affine) -> np.ndarray:
        """
        Mask raster data by geometry.
        
        Parameters
        ----------
        raster : np.ndarray
            Input raster array
        geometry : shapely geometry
            Geometry to use as mask
        transform : Affine
            Raster transform
            
        Returns
        -------
        np.ndarray
            Masked raster array
        """
        # Create GeoDataFrame from geometry
        gdf = gpd.GeoDataFrame([{'id': 1}], geometry=[geometry], crs=self.crs)
        
        # Use GeometryProcessor to clip raster
        clipped_raster, _ = self.geometry_processor.clip_raster_to_geometry(
            raster, transform, gdf, nodata_value=np.nan
        )
        
        return clipped_raster
    
    
        
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
                # Get dimensions from clipped DEM
                height, width = dem.shape
            else:
                dem = src.read(1)
                self.transform = src.transform
                dem_bounds = src.bounds
                # dem_bounds is already a BoundingBox object
                minx, miny, maxx, maxy = dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top
                # Get dimensions from full DEM
                height, width = src.height, src.width
            
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
                
                self.resolution_x = lon_extent_m / width if width > 0 else src.res[0]
                self.resolution_y = lat_extent_m / height if height > 0 else src.res[1]
                self.resolution = (self.resolution_x + self.resolution_y) / 2  # Average
                
                logger.info(f"Geographic CRS detected. Calculated resolution: X={self.resolution_x:.1f}m, Y={self.resolution_y:.1f}m")
            else:
                # Projected coordinates - use directly
                self.resolution = src.res[0]
                self.resolution_x = src.res[0]
                self.resolution_y = src.res[1]
            
        # Handle nodata values
        dem = np.where(dem < -9999, np.nan, dem)
        
        # Fill pits (essential for flow routing) - use FlowAnalyzer
        dem_filled = self.flow_analyzer.fill_pits(dem)
        
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
        
        def accumulate_iterative(start_i: int, start_j: int) -> None:
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
        # Use MonteCarloSimulator for single iteration
        return self.monte_carlo_simulator.single_iteration(
            self.dem_data, 
            noise_level,
            self.config.min_drainage_area,
            self.resolution_x,
            self.resolution_y
        )
    
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
        # Use MonteCarloSimulator for full simulation
        return self.monte_carlo_simulator.run_simulation(
            self.dem_data,
            noise_level,
            self.config.iterations,
            self.config.min_drainage_area,
            self.resolution_x,
            self.resolution_y,
            self.config.n_jobs,
            progress_bar
        )
    
    def compute_flow_direction(self, dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute D8 flow direction and slopes.
        
        Public wrapper for flow analysis functionality needed by notebooks.
        
        Parameters
        ----------
        dem : np.ndarray
            Digital elevation model
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Flow direction grid and slope grid
        """
        return self.flow_analyzer.compute_flow_direction(dem)
    
    def compute_flow_accumulation(self, flow_dir: np.ndarray, dem_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute flow accumulation from flow direction.
        
        Public wrapper for flow analysis functionality needed by notebooks.
        
        Parameters
        ----------
        flow_dir : np.ndarray
            Flow direction grid
        dem_data : np.ndarray, optional
            DEM data for masking. If None, uses self.dem_data
            
        Returns
        -------
        np.ndarray
            Flow accumulation grid
        """
        if dem_data is None:
            dem_data = self.dem_data
        return self.flow_analyzer.compute_flow_accumulation(flow_dir, dem_data)
    
    def run_full_analysis(self, 
                         dem_path: Union[str, Path], 
                         bounds: Optional[Tuple] = None,
                         shapefile_path: Optional[str] = None) -> Dict:
        """
        Run complete EXZECO analysis for all noise levels.
        
        Parameters
        ----------
        dem_path : str or Path
            Path to DEM file
        bounds : tuple, optional
            Bounding box for analysis area (fallback if no shapefile)
        shapefile_path : str, optional
            Path to shapefile/geopackage for study area definition
            
        Returns
        -------
        dict
            Results dictionary with probability maps for each noise level
        """
        # Load study areas first
        try:
            study_areas_gdf, total_bounds = self.load_study_areas(shapefile_path, bounds)
        except ValueError as e:
            logger.error(f"Failed to load study areas: {e}")
            raise
        
        # Load DEM using total bounds
        self.load_dem(dem_path, total_bounds)
        
        # Run analysis for each noise level
        results = {}
        
        for noise_level in self.config.noise_levels:
            logger.info(f"Processing noise level: {noise_level}m")
            
            # Run Monte Carlo for entire domain
            prob_map = self.run_monte_carlo(noise_level)
            
            # Apply incremental DEM modification for next level
            if noise_level < self.config.noise_levels[-1]:
                # Modify DEM for areas with high flow accumulation using FlowAnalyzer
                flow_dir, _ = self.flow_analyzer.compute_flow_direction(self.dem_data)
                flow_acc = self.flow_analyzer.compute_flow_accumulation(flow_dir, self.dem_data)
                
                # FIX: Use correct pixel area calculation
                pixel_area_m2 = self.resolution_x * self.resolution_y
                pixel_area_km2 = pixel_area_m2 / 1e6
                drainage_area = flow_acc * pixel_area_km2
                
                # Increase elevation where drainage > 0.1 km²
                mask = drainage_area > 0.1
                self.dem_data[mask] += noise_level
            
            # Store results for entire domain
            results[f"exzeco_{int(noise_level*100)}cm"] = {
                'probability_map': prob_map,
                'noise_level': noise_level,
                'threshold': 0.5,  # Default threshold for binary classification
                'total_domain': True
            }
            
            # If we have subcatchments, calculate statistics for each
            if self.study_areas is not None and len(self.study_areas) > 1:
                subcatchment_results = {}
                
                for idx, row in self.study_areas.iterrows():
                    subcatch_name = row.get('NAME_EN', row.get('name', f'subcatchment_{idx}'))
                    
                    # Transform geometry to raster CRS if needed
                    geom = row.geometry
                    if self.study_areas.crs != self.crs:
                        geom_gdf = gpd.GeoDataFrame([row], crs=self.study_areas.crs)
                        geom_gdf = geom_gdf.to_crs(self.crs)
                        geom = geom_gdf.geometry.iloc[0]
                    
                    # Mask probability map by subcatchment geometry
                    masked_prob = self.mask_raster_by_geometry(prob_map, geom, self.transform)
                    
                    subcatchment_results[subcatch_name] = {
                        'probability_map': masked_prob,
                        'geometry': geom,
                        'original_data': row
                    }
                
                results[f"exzeco_{int(noise_level*100)}cm"]['subcatchments'] = subcatchment_results
        
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
        # Compute flow accumulation for original DEM using FlowAnalyzer
        flow_dir, _ = self.flow_analyzer.compute_flow_direction(self.dem_data)
        flow_acc = self.flow_analyzer.compute_flow_accumulation(flow_dir, self.dem_data)
        
        # Convert to drainage area using correct pixel area
        pixel_area_m2 = self.resolution_x * self.resolution_y
        pixel_area_km2 = pixel_area_m2 / 1e6
        drainage_area = flow_acc * pixel_area_km2
        
        # Use DrainageClassifier for classification
        classified = self.drainage_classifier.classify_drainage_areas(drainage_area, prob_map)
        
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
        Export analysis results with descriptive naming.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        format : str
            Export format ('geotiff', 'shapefile', 'geojson')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting results to {output_dir}")
        
        for name, data in self.results.items():
            prob_map = data['probability_map']
            
            # Extract noise level from name (e.g., 'exzeco_200cm' -> '200cm')
            noise_level_cm = name.split('_')[-1] if '_' in name else '0cm'
            
            # Create descriptive filename: exzeco_{noise_level}_{iterations}_{drainage_threshold}
            drainage_threshold_str = str(self.config.min_drainage_area).replace('.', 'p')
            descriptive_name = f"exzeco_{noise_level_cm}_{self.config.iterations}_{drainage_threshold_str}km2"
            
            # Export total domain results
            if format == 'geotiff':
                # Export as GeoTIFF
                output_path = output_dir / f"{descriptive_name}.tif"
                
                with rasterio.open(
                    output_path,
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
                # Vectorize and export
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
                
                # Export
                if format == 'shapefile':
                    output_path = output_dir / f"{descriptive_name}.shp"
                    gdf.to_file(output_path)
                else:  # geojson
                    output_path = output_dir / f"{descriptive_name}.geojson"
                    gdf.to_file(output_path, driver='GeoJSON')
            
            logger.info(f"Exported {name} to {output_path}")
            
            # Export subcatchment results if available
            if 'subcatchments' in data:
                subcatch_dir = output_dir / 'subcatchments'
                subcatch_dir.mkdir(exist_ok=True)
                
                for subcatch_name, subcatch_data in data['subcatchments'].items():
                    subcatch_prob = subcatch_data['probability_map']
                    
                    # Clean subcatchment name for filename
                    clean_name = "".join(c for c in subcatch_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    clean_name = clean_name.replace(' ', '_')
                    
                    if format == 'geotiff':
                        # Export subcatchment raster with descriptive naming
                        subcatch_output_path = subcatch_dir / f"{descriptive_name}_{clean_name}.tif"
                        
                        with rasterio.open(
                            subcatch_output_path,
                            'w',
                            driver='GTiff',
                            height=subcatch_prob.shape[0],
                            width=subcatch_prob.shape[1],
                            count=1,
                            dtype=subcatch_prob.dtype,
                            crs=self.crs,
                            transform=self.transform,
                            compress='lzw'
                        ) as dst:
                            dst.write(subcatch_prob, 1)
                    
                    elif format in ['shapefile', 'geojson']:
                        # Vectorize subcatchment results
                        shapes = features.shapes(
                            (subcatch_prob > 0.5).astype(np.uint8),
                            transform=self.transform
                        )
                        
                        geometries = []
                        values = []
                        
                        for geom, value in shapes:
                            if value == 1:  # Only flood zones
                                geometries.append(Polygon(geom['coordinates'][0]))
                                values.append(data['noise_level'])
                        
                        if geometries:  # Only create file if there are flood zones
                            # Create GeoDataFrame
                            subcatch_gdf = gpd.GeoDataFrame(
                                {
                                    'noise_level': values,
                                    'subcatchment': subcatch_name
                                },
                                geometry=geometries,
                                crs=self.crs
                            )
                            
                            # Export
                            if format == 'shapefile':
                                subcatch_output_path = subcatch_dir / f"{descriptive_name}_{clean_name}.shp"
                                subcatch_gdf.to_file(subcatch_output_path)
                            else:  # geojson
                                subcatch_output_path = subcatch_dir / f"{descriptive_name}_{clean_name}.geojson"
                                subcatch_gdf.to_file(subcatch_output_path, driver='GeoJSON')
                    
                    logger.info(f"Exported subcatchment {subcatch_name} to {subcatch_output_path}")
    
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate summary report of analysis.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics for total domain and individual subcatchments
        """
        report_data = []
        
        for name, data in self.results.items():
            prob_map = data['probability_map']
            flood_mask = prob_map > 0.5
            
            # FIX: Use correct pixel area calculation
            pixel_area_m2 = self.resolution_x * self.resolution_y
            pixel_area_km2 = pixel_area_m2 / 1e6
            
            # Total domain statistics
            total_valid_pixels = np.sum(~np.isnan(prob_map))
            total_flood_pixels = np.sum(flood_mask & ~np.isnan(prob_map))
            
            stats = {
                'Analysis': name,
                'Area_Type': 'Total Domain',
                'Area_Name': 'Total Domain',
                'Noise Level (m)': data['noise_level'],
                'Total Area (km²)': total_valid_pixels * pixel_area_km2,
                'Flood Area (km²)': total_flood_pixels * pixel_area_km2,
                'Flood Area (%)': 100 * total_flood_pixels / total_valid_pixels if total_valid_pixels > 0 else 0,
                'Mean Probability': np.nanmean(prob_map),
                'Max Probability': np.nanmax(prob_map),
                'Pixels > 0.8 Prob': np.sum((prob_map > 0.8) & ~np.isnan(prob_map))
            }
            
            report_data.append(stats)
            
            # If we have subcatchment results, add them
            if 'subcatchments' in data:
                for subcatch_name, subcatch_data in data['subcatchments'].items():
                    subcatch_prob = subcatch_data['probability_map']
                    subcatch_flood_mask = subcatch_prob > 0.5
                    
                    # Calculate statistics for this subcatchment
                    subcatch_valid_pixels = np.sum(~np.isnan(subcatch_prob))
                    subcatch_flood_pixels = np.sum(subcatch_flood_mask & ~np.isnan(subcatch_prob))
                    
                    if subcatch_valid_pixels > 0:
                        subcatch_stats = {
                            'Analysis': name,
                            'Area_Type': 'Subcatchment',
                            'Area_Name': subcatch_name,
                            'Noise Level (m)': data['noise_level'],
                            'Total Area (km²)': subcatch_valid_pixels * pixel_area_km2,
                            'Flood Area (km²)': subcatch_flood_pixels * pixel_area_km2,
                            'Flood Area (%)': 100 * subcatch_flood_pixels / subcatch_valid_pixels,
                            'Mean Probability': np.nanmean(subcatch_prob),
                            'Max Probability': np.nanmax(subcatch_prob),
                            'Pixels > 0.8 Prob': np.sum((subcatch_prob > 0.8) & ~np.isnan(subcatch_prob))
                        }
                        
                        report_data.append(subcatch_stats)
        
        return pd.DataFrame(report_data)


def load_config(config_path: Union[str, Path]) -> ExzecoConfig:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to config file
        
    Returns
    -------
    ExzecoConfig
        Configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    exzeco_params = config_dict.get('exzeco', {})
    processing_params = config_dict.get('processing', {})
    study_area_params = config_dict.get('study_area', {})
    
    return ExzecoConfig(
        noise_levels=exzeco_params.get('noise_levels'),
        iterations=exzeco_params.get('iterations', 100),
        min_drainage_area=exzeco_params.get('min_drainage_area', 0.01),
        drainage_classes=exzeco_params.get('drainage_classes'),
        n_jobs=processing_params.get('n_jobs', -1),
        chunk_size=processing_params.get('chunk_size', 1000),
        seed=processing_params.get('seed', 42),
        shapefile_path=study_area_params.get('shapefile_path'),
        bounds=study_area_params.get('bounds')
    )


def run_exzeco_with_config(config_path: Union[str, Path], dem_path: Union[str, Path], output_dir: Union[str, Path]) -> Tuple[ExzecoAnalysis, Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Run EXZECO analysis using configuration file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to configuration YAML file
    dem_path : str or Path
        Path to DEM file
    output_dir : str or Path
        Output directory for results
        
    Returns
    -------
    tuple
        (ExzecoAnalysis instance, results dictionary, report DataFrame)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize analyzer
    analyzer = ExzecoAnalysis(config)
    
    # Run analysis - the method will automatically handle shapefile vs bounds
    results = analyzer.run_full_analysis(
        dem_path=dem_path,
        bounds=config.bounds,
        shapefile_path=config.shapefile_path
    )
    
    # Export results
    analyzer.export_results(output_dir, format='geotiff')
    analyzer.export_results(output_dir, format='geojson')
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report
    output_dir = Path(output_dir)
    report.to_csv(output_dir / 'exzeco_report.csv', index=False)
    report.to_excel(output_dir / 'exzeco_report.xlsx', index=False)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    return analyzer, results, report


if __name__ == "__main__":
    # Example usage with configuration file
    config_path = "config/config.yml"
    dem_path = "data/dem/cache/study_area_dem.tif"  
    output_dir = "data/outputs"
    
    try:
        analyzer, results, report = run_exzeco_with_config(config_path, dem_path, output_dir)
        print("EXZECO analysis completed successfully!")
        print(f"\nSummary Report:")
        print(report)
    except Exception as e:
        print(f"Analysis failed: {e}")
        
    print("EXZECO module loaded successfully")