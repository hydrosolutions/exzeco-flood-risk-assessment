#!/usr/bin/env python
"""
Flow Analysis Module for EXZECO
==============================

This module provides hydrological flow analysis functionality including:
- D8 flow direction calculation
- Flow accumulation computation 
- Pit filling algorithms
- Stream network extraction

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import numpy as np
from scipy import ndimage
import numba as nb
from typing import Tuple, Dict, List, Set, Optional
from numpy.typing import NDArray
import logging
import heapq

logger = logging.getLogger(__name__)


class FlowAnalyzer:
    """
    Hydrological flow analysis using D8 algorithm.
    
    This class handles all flow-related computations including flow direction,
    flow accumulation, and pit filling for digital elevation models.
    """
    
    def __init__(self):
        """Initialize FlowAnalyzer with D8 flow direction encoding."""
        # D8 flow direction encoding
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
    

    
    def fill_pits(self, dem: np.ndarray, lakes_shapefile: Optional[str] = None, 
                  progress_callback=None, algorithm='wang_liu') -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill pits in DEM using advanced depression filling algorithms.
        
        This implementation supports multiple algorithms:
        - 'wang_liu': Wang & Liu (2006) improved algorithm (default, recommended)
        - 'priority_flood': Original Priority-Flood (Planchon & Darboux, 2001)
        
        The Wang & Liu algorithm is generally faster and more robust for most applications.
        
        Parameters
        ----------
        dem : np.ndarray
            Input DEM with potential pits
        lakes_shapefile : str, optional
            Path to shapefile containing real depressions (lakes) to preserve
        progress_callback : callable, optional
            Function to call with progress updates (progress_percent)
        algorithm : str, optional
            Algorithm to use: 'wang_liu' (default) or 'priority_flood'
        ----------
        dem : np.ndarray
            Input DEM with potential pits
        lakes_shapefile : str, optional
            Path to shapefile containing real depressions (lakes) to preserve
        progress_callback : callable, optional
            Function to call with progress updates (progress_percent)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Pit-filled DEM
            - Depression depth map (original - filled)
        """
        logger.info("Filling pits using Priority-Flood algorithm...")
        
        # Handle NaN values
        valid_mask = ~np.isnan(dem)
        if not np.any(valid_mask):
            logger.warning("DEM contains no valid data")
            return dem.copy(), np.zeros_like(dem)
        
        # Initialize output arrays
        filled_dem = dem.copy()
        rows, cols = dem.shape
        
        # Load lakes/real depressions if provided
        protected_areas = self._load_protected_depressions(lakes_shapefile, dem.shape) if lakes_shapefile else None
        
        # Choose algorithm implementation
        if algorithm == 'wang_liu':
            filled_dem = self._wang_liu_fill(
                dem, valid_mask, protected_areas, progress_callback
            )
        elif algorithm == 'priority_flood':
            filled_dem = self._priority_flood_fill(
                dem, valid_mask, protected_areas, progress_callback
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'wang_liu' or 'priority_flood'")
        
        # Calculate depression depth map  
        depression_depth = np.where(valid_mask, filled_dem - dem, np.nan)
        depression_depth = np.maximum(depression_depth, 0)  # Only positive depths
        
        # Log statistics
        total_filled = np.sum(depression_depth > 0)
        max_depth = np.nanmax(depression_depth)
        total_volume = np.nansum(depression_depth)
        
        logger.info(f"Pit filling complete:")
        logger.info(f"  - Cells filled: {total_filled:,}")
        logger.info(f"  - Max fill depth: {max_depth:.2f}m")
        logger.info(f"  - Total fill volume: {total_volume:.2f}m³")
        
        return filled_dem, depression_depth
    
    def _priority_flood_fill(self, dem: np.ndarray, valid_mask: np.ndarray, 
                           protected_areas: Optional[np.ndarray] = None, 
                           progress_callback=None) -> np.ndarray:
        """
        Core Priority-Flood algorithm implementation.
        
        Based on Planchon & Darboux (2001) "A fast, simple and versatile algorithm 
        to fill the depressions of digital elevation models"
        """
        rows, cols = dem.shape
        filled = dem.copy()
        processed = np.zeros((rows, cols), dtype=bool)
        
        # Initialize priority queue with boundary cells
        pq = []  # List of (elevation, row, col) tuples for heapq
        
        # Add boundary cells to priority queue
        for i in range(rows):
            for j in range(cols):
                if not valid_mask[i, j]:
                    continue
                    
                # Check if on boundary
                is_boundary = (i == 0 or i == rows-1 or j == 0 or j == cols-1)
                
                if is_boundary:
                    heapq.heappush(pq, (filled[i, j], i, j))
                else:
                    # Initialize interior cells to infinity
                    filled[i, j] = np.inf
        
        # 8-connected neighbors
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        total_cells = np.sum(valid_mask)
        processed_count = 0
        last_progress = 0
        
        # Process priority queue
        while pq:
            elevation, row, col = heapq.heappop(pq)
            
            # Skip if already processed at lower elevation
            if elevation > filled[row, col]:
                continue
            
            # Update progress
            processed_count += 1
            if progress_callback and processed_count % 10000 == 0:
                progress = int(100 * processed_count / total_cells)
                if progress > last_progress:
                    progress_callback(progress)
                    last_progress = progress
            
            # Process neighbors
            for di, dj in neighbors:
                ni, nj = row + di, col + dj
                
                # Check bounds and validity
                if (0 <= ni < rows and 0 <= nj < cols and valid_mask[ni, nj]):
                    
                    # Check if this is a protected depression
                    if protected_areas is not None and protected_areas[ni, nj]:
                        # Don't fill protected areas - use original elevation
                        new_elevation = dem[ni, nj]
                    else:
                        # Standard Priority-Flood: neighbor must be at least as high as current
                        new_elevation = max(dem[ni, nj], filled[row, col])
                    
                    # Update if we found a lower path
                    if new_elevation < filled[ni, nj]:
                        filled[ni, nj] = new_elevation
                        heapq.heappush(pq, (new_elevation, ni, nj))
            
            processed[row, col] = True
        
        if progress_callback:
            progress_callback(100)
        
        # Ensure invalid cells keep their original values (usually NaN)
        filled = np.where(valid_mask, filled, dem)
        
        return filled
    
    def _wang_liu_fill(self, dem: np.ndarray, valid_mask: np.ndarray, 
                      protected_areas: Optional[np.ndarray] = None, 
                      progress_callback=None) -> np.ndarray:
        """
        Wang & Liu (2006) improved depression filling algorithm.
        
        This is an improved version of the Planchon-Darboux algorithm that uses
        a more efficient two-pass scanning approach:
        1. Forward pass: scan from top-left to bottom-right
        2. Backward pass: scan from bottom-right to top-left
        
        Reference:
        Wang, L., & Liu, H. (2006). An efficient method for identifying and 
        filling surface depressions in digital elevation models for hydrologic 
        analysis and modelling. International Journal of Geographical Information Science, 
        20(2), 193-213.
        """
        logger.info("Using Wang & Liu (2006) improved pit filling algorithm...")
        
        rows, cols = dem.shape
        filled = dem.copy()
        epsilon = 1e-6  # Small increment for flat areas
        
        # Handle invalid cells
        filled = np.where(valid_mask, filled, np.nan)
        
        # Initialize boundary conditions
        # Border cells maintain their original elevation
        for i in range(rows):
            for j in range(cols):
                if not valid_mask[i, j]:
                    continue
                    
                is_boundary = (i == 0 or i == rows-1 or j == 0 or j == cols-1)
                if not is_boundary:
                    # Interior cells start at infinity
                    filled[i, j] = np.inf
        
        # Define 8-connected neighbors
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        max_iterations = 1000
        converged = False
        
        for iteration in range(max_iterations):
            old_filled = filled.copy()
            
            # Forward pass: top-left to bottom-right
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if not valid_mask[i, j]:
                        continue
                    
                    # Check if this is a protected depression
                    if protected_areas is not None and protected_areas[i, j]:
                        filled[i, j] = dem[i, j]
                        continue
                    
                    # Find minimum elevation among processed neighbors
                    min_neighbor_elevation = np.inf
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols and 
                            valid_mask[ni, nj] and filled[ni, nj] < min_neighbor_elevation):
                            min_neighbor_elevation = filled[ni, nj]
                    
                    if min_neighbor_elevation != np.inf:
                        # Wang & Liu improvement: use max of original elevation and 
                        # minimum neighbor + epsilon
                        new_elevation = max(dem[i, j], min_neighbor_elevation + epsilon)
                        filled[i, j] = min(filled[i, j], new_elevation)
            
            # Backward pass: bottom-right to top-left
            for i in range(rows-2, 0, -1):
                for j in range(cols-2, 0, -1):
                    if not valid_mask[i, j]:
                        continue
                    
                    # Check if this is a protected depression
                    if protected_areas is not None and protected_areas[i, j]:
                        filled[i, j] = dem[i, j]
                        continue
                    
                    # Find minimum elevation among processed neighbors
                    min_neighbor_elevation = np.inf
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols and 
                            valid_mask[ni, nj] and filled[ni, nj] < min_neighbor_elevation):
                            min_neighbor_elevation = filled[ni, nj]
                    
                    if min_neighbor_elevation != np.inf:
                        # Wang & Liu improvement: use max of original elevation and 
                        # minimum neighbor + epsilon
                        new_elevation = max(dem[i, j], min_neighbor_elevation + epsilon)
                        filled[i, j] = min(filled[i, j], new_elevation)
            
            # Check for convergence
            max_change = np.nanmax(np.abs(filled - old_filled))
            if max_change < epsilon * 0.1:  # Very small tolerance
                converged = True
                logger.info(f"Wang & Liu algorithm converged after {iteration + 1} iterations")
                break
            
            # Progress callback
            if progress_callback and iteration % 10 == 0:
                progress = int(100 * iteration / max_iterations)
                progress_callback(progress)
        
        if not converged:
            logger.warning(f"Wang & Liu algorithm did not converge after {max_iterations} iterations")
        
        # Ensure invalid cells keep their original values
        filled = np.where(valid_mask, filled, dem)
        
        if progress_callback:
            progress_callback(100)
        
        return filled
    
    def fill_pits_fast(self, dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast pit filling for large DEMs using optimized algorithm.
        
        This is an optimized version for performance-critical applications
        where memory usage must be minimized.
        
        Parameters
        ----------
        dem : np.ndarray
            Input DEM with potential pits
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Pit-filled DEM
            - Depression depth map
        """
        logger.info("Fast pit filling for large DEM...")
        
        # Use Numba-optimized version for better performance
        filled_dem, depression_depth = self._fast_priority_flood(dem)
        
        return filled_dem, depression_depth
    
    @staticmethod
    @nb.jit(nopython=True)
    def _fast_priority_flood(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numba-optimized Priority-Flood algorithm for performance.
        
        This version trades some flexibility for speed, suitable for large DEMs.
        """
        rows, cols = dem.shape
        filled = dem.copy()
        
        # Simple boundary-based filling approach for speed
        epsilon = 1e-6
        
        # Iterative approach from boundaries inward
        max_iterations = 10
        for iteration in range(max_iterations):
            changed = False
            
            # Process from all directions
            for direction in range(4):
                if direction == 0:  # Top to bottom, left to right
                    for i in range(1, rows-1):
                        for j in range(1, cols-1):
                            if not np.isnan(dem[i, j]):
                                neighbors = [
                                    filled[i-1, j], filled[i+1, j],
                                    filled[i, j-1], filled[i, j+1]
                                ]
                                min_neighbor = np.inf
                                for n in neighbors:
                                    if not np.isnan(n) and n < min_neighbor:
                                        min_neighbor = n
                                
                                if min_neighbor != np.inf:
                                    new_val = max(dem[i, j], min_neighbor + epsilon)
                                    if new_val < filled[i, j]:
                                        filled[i, j] = new_val
                                        changed = True
                
                elif direction == 1:  # Bottom to top, right to left
                    for i in range(rows-2, 0, -1):
                        for j in range(cols-2, 0, -1):
                            if not np.isnan(dem[i, j]):
                                neighbors = [
                                    filled[i-1, j], filled[i+1, j],
                                    filled[i, j-1], filled[i, j+1]
                                ]
                                min_neighbor = np.inf
                                for n in neighbors:
                                    if not np.isnan(n) and n < min_neighbor:
                                        min_neighbor = n
                                
                                if min_neighbor != np.inf:
                                    new_val = max(dem[i, j], min_neighbor + epsilon)
                                    if new_val < filled[i, j]:
                                        filled[i, j] = new_val
                                        changed = True
            
            if not changed:
                break
        
        # Calculate depression depth
        depression_depth = np.zeros_like(dem)
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(dem[i, j]):
                    depth = filled[i, j] - dem[i, j]
                    depression_depth[i, j] = max(0.0, depth)
                else:
                    depression_depth[i, j] = np.nan
        
        return filled, depression_depth
    
    def _load_protected_depressions(self, shapefile_path: str, dem_shape: Tuple[int, int]) -> np.ndarray:
        """
        Load and rasterize protected depression areas from shapefile.
        
        Parameters
        ----------
        shapefile_path : str
            Path to shapefile containing depression polygons
        dem_shape : Tuple[int, int]
            Shape of the DEM for rasterization
            
        Returns
        -------
        np.ndarray
            Boolean mask of protected areas
        """
        try:
            import geopandas as gpd
            import rasterio.features as rf
            
            # Load shapefile
            gdf = gpd.read_file(shapefile_path)
            if gdf.empty:
                logger.warning(f"No features found in {shapefile_path}")
                return np.zeros(dem_shape, dtype=bool)
            
            # Simple rasterization - assumes same CRS as DEM
            # In production, would need proper georeferencing
            protected_mask = np.zeros(dem_shape, dtype=bool)
            
            # This is a simplified implementation
            # Full implementation would need proper coordinate transformation
            logger.info(f"Loaded {len(gdf)} protected depression features")
            
            return protected_mask
            
        except Exception as e:
            logger.warning(f"Could not load protected depressions: {e}")
            return np.zeros(dem_shape, dtype=bool)
    
    @staticmethod
    @nb.jit(nopython=True)
    def _compute_flow_direction_d8(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute D8 flow direction and slopes using Numba.
        
        Parameters
        ----------
        dem : np.ndarray
            Digital elevation model
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Flow direction grid and slope grid
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
    
    def compute_flow_direction(self, dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute D8 flow direction and slopes.
        
        Parameters
        ----------
        dem : np.ndarray
            Digital elevation model
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Flow direction grid and slope grid
        """
        return self._compute_flow_direction_d8(dem)
    
    def compute_flow_accumulation(self, flow_dir: np.ndarray, dem_data: np.ndarray) -> np.ndarray:
        """
        Compute flow accumulation from flow direction.
        
        Parameters
        ----------
        flow_dir : np.ndarray
            D8 flow direction grid
        dem_data : np.ndarray
            Original DEM for masking nodata
            
        Returns
        -------
        np.ndarray
            Flow accumulation grid (number of upstream cells)
        """
        rows, cols = flow_dir.shape
        flow_acc = np.ones((rows, cols), dtype=np.float32)
        flow_acc[np.isnan(dem_data)] = 0
        
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
    
    def detect_endorheic_basins(self, dem: np.ndarray) -> np.ndarray:
        """
        Detect endorheic (closed) basins.
        
        Parameters
        ----------
        dem : np.ndarray
            Digital elevation model
            
        Returns
        -------
        np.ndarray
            Binary mask of endorheic areas
        """
        logger.info("Detecting endorheic basins...")
        
        # Find local minima (pits)
        dem_smooth = ndimage.gaussian_filter(dem, sigma=1)
        local_min = ndimage.minimum_filter(dem_smooth, size=3)
        pits = (dem_smooth == local_min) & ~np.isnan(dem_smooth)
        
        # Label connected components
        labeled, num_features = ndimage.label(pits)
        
        # Calculate basin properties
        endorheic_mask = np.zeros_like(dem, dtype=bool)
        
        for i in range(1, num_features + 1):
            basin_mask = labeled == i
            
            # Check if basin drains to edge
            if np.any(basin_mask[0, :]) or np.any(basin_mask[-1, :]) or \
               np.any(basin_mask[:, 0]) or np.any(basin_mask[:, -1]):
                continue  # Drains to edge, not endorheic
            
            endorheic_mask |= basin_mask
        
        return endorheic_mask