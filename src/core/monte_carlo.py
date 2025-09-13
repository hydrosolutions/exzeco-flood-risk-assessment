#!/usr/bin/env python
"""
Monte Carlo Simulation Module for EXZECO
========================================

This module provides Monte Carlo simulation functionality including:
- Random noise addition to DEMs
- Parallel and sequential simulation execution
- Single iteration processing
- Probability map generation

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import numpy as np
from typing import Optional, List
from numpy.typing import NDArray
from tqdm import tqdm
from joblib import Parallel, delayed
import logging

from .flow_analysis import FlowAnalyzer

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for flood risk assessment.
    
    This class handles the Monte Carlo simulation process including
    noise addition, parallel execution, and result aggregation.
    """
    
    def __init__(self, flow_analyzer: FlowAnalyzer):
        """
        Initialize Monte Carlo simulator.
        
        Parameters
        ----------
        flow_analyzer : FlowAnalyzer
            Flow analysis instance for hydrological computations
        """
        self.flow_analyzer = flow_analyzer
    
    def add_noise(self, dem: np.ndarray, noise_level: float) -> np.ndarray:
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
    
    def single_iteration(self, 
                        dem_data: np.ndarray, 
                        noise_level: float,
                        min_drainage_area: float,
                        resolution_x: float,
                        resolution_y: float) -> np.ndarray:
        """
        Single Monte Carlo iteration with noise addition.
        
        Parameters
        ----------
        dem_data : np.ndarray
            Original DEM data
        noise_level : float
            Noise level to add to DEM
        min_drainage_area : float
            Minimum drainage area threshold in km²
        resolution_x : float
            X resolution in meters
        resolution_y : float
            Y resolution in meters
            
        Returns
        -------
        np.ndarray
            Binary flood zone mask for this iteration
        """
        # Add noise to DEM
        dem_noisy = self.add_noise(dem_data, noise_level)
        
        # Compute flow direction
        flow_dir, _ = self.flow_analyzer.compute_flow_direction(dem_noisy)
        
        # Compute flow accumulation
        flow_acc = self.flow_analyzer.compute_flow_accumulation(flow_dir, dem_data)
        
        # Convert to drainage area using correct pixel area calculation
        pixel_area_m2 = resolution_x * resolution_y  # Area in m²
        pixel_area_km2 = pixel_area_m2 / 1e6  # Convert to km²
        drainage_area = flow_acc * pixel_area_km2
        
        # Create binary mask for areas above threshold
        mask = drainage_area >= min_drainage_area
        
        return mask.astype(np.float32)
    
    def run_simulation(self, 
                      dem_data: np.ndarray,
                      noise_level: float,
                      iterations: int,
                      min_drainage_area: float,
                      resolution_x: float,
                      resolution_y: float,
                      n_jobs: int = -1,
                      progress_bar: bool = True) -> np.ndarray:
        """
        Run Monte Carlo simulation for a specific noise level.
        
        Parameters
        ----------
        dem_data : np.ndarray
            DEM data array
        noise_level : float
            Noise level in meters
        iterations : int
            Number of Monte Carlo iterations
        min_drainage_area : float
            Minimum drainage area threshold in km²
        resolution_x : float
            X resolution in meters
        resolution_y : float
            Y resolution in meters
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        progress_bar : bool
            Show progress bar
            
        Returns
        -------
        np.ndarray
            Probability map (0-1) of flood zones
        """
        logger.info(f"Running Monte Carlo for noise level {noise_level}m with {iterations} iterations")
        
        # Parallel execution
        if n_jobs != 1:
            iterator = tqdm(range(iterations), desc=f"MC {noise_level}m") if progress_bar else range(iterations)
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.single_iteration)(
                    dem_data, noise_level, min_drainage_area, resolution_x, resolution_y
                ) for _ in iterator
            )
            
            # Aggregate results
            probability_map = np.mean(results, axis=0)
        else:
            # Sequential execution
            probability_map = np.zeros(dem_data.shape, dtype=np.float32)
            
            iterator = tqdm(range(iterations), desc=f"MC {noise_level}m") if progress_bar else range(iterations)
            
            for _ in iterator:
                mask = self.single_iteration(
                    dem_data, noise_level, min_drainage_area, resolution_x, resolution_y
                )
                probability_map += mask
            
            probability_map /= iterations
        
        return probability_map