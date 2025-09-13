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
from typing import Tuple, Dict
from numpy.typing import NDArray
import logging

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
    

    
    def fill_pits(self, dem: np.ndarray) -> np.ndarray:
        """
        Fill pits in DEM for hydrological correctness.
        
        Uses scipy.ndimage filtering to fill pits by smoothing.
        
        Parameters
        ----------
        dem : np.ndarray
            Input DEM with potential pits
            
        Returns
        -------
        np.ndarray
            Pit-filled DEM
        """
        logger.info("Filling pits in DEM using scipy...")
        
        # Use scipy ndimage filtering for pit filling
        filled = ndimage.generic_filter(dem, np.nanmean, size=3)
        
        return filled
    
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