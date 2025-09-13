#!/usr/bin/env python
"""
Geometry Processing Module for EXZECO
====================================

This module provides geometry processing functionality including:
- Study area loading and validation
- Coordinate transformation and projection
- Bounding box calculations
- Spatial data validation

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from pyproj import CRS
import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class GeometryProcessor:
    """
    Handles geometry processing for study areas and coordinate transformations.
    
    This class provides functionality for loading study areas, coordinate
    transformations, and spatial data validation.
    """
    
    def __init__(self):
        """Initialize geometry processor."""
        pass
    
    def load_study_area(self, study_area_path: str) -> gpd.GeoDataFrame:
        """
        Load study area from file.
        
        Parameters
        ----------
        study_area_path : str
            Path to study area shapefile or geopackage
            
        Returns
        -------
        gpd.GeoDataFrame
            Study area geometry
            
        Raises
        ------
        FileNotFoundError
            If study area file doesn't exist
        ValueError
            If study area is invalid
        """
        if not os.path.exists(study_area_path):
            raise FileNotFoundError(f"Study area file not found: {study_area_path}")
        
        logger.info(f"Loading study area from: {study_area_path}")
        study_area = gpd.read_file(study_area_path)
        
        if study_area.empty:
            raise ValueError("Study area is empty")
        
        if not study_area.is_valid.all():
            logger.warning("Invalid geometries found, attempting to fix")
            study_area['geometry'] = study_area.geometry.buffer(0)
        
        # Ensure single polygon (union if multiple)
        if len(study_area) > 1:
            logger.info("Multiple polygons found, creating union")
            unified = study_area.unary_union
            study_area = gpd.GeoDataFrame([1], geometry=[unified], crs=study_area.crs)
        
        return study_area
    
    def get_bbox(self, geometry: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
        """
        Get bounding box of geometry.
        
        Parameters
        ----------
        geometry : gpd.GeoDataFrame
            Input geometry
            
        Returns
        -------
        Tuple[float, float, float, float]
            Bounding box as (minx, miny, maxx, maxy)
        """
        bounds = geometry.total_bounds
        return tuple(bounds)
    
    def reproject_geometry(self, 
                          geometry: gpd.GeoDataFrame, 
                          target_crs: Union[str, int, CRS]) -> gpd.GeoDataFrame:
        """
        Reproject geometry to target CRS.
        
        Parameters
        ----------
        geometry : gpd.GeoDataFrame
            Input geometry
        target_crs : Union[str, int, CRS]
            Target coordinate reference system
            
        Returns
        -------
        gpd.GeoDataFrame
            Reprojected geometry
        """
        if geometry.crs is None:
            raise ValueError("Input geometry has no CRS defined")
        
        if geometry.crs != target_crs:
            logger.info(f"Reprojecting from {geometry.crs} to {target_crs}")
            return geometry.to_crs(target_crs)
        
        return geometry
    
    def create_mask_from_geometry(self, 
                                 geometry: gpd.GeoDataFrame,
                                 transform: rasterio.transform.Affine,
                                 shape: Tuple[int, int]) -> np.ndarray:
        """
        Create raster mask from geometry.
        
        Parameters
        ----------
        geometry : gpd.GeoDataFrame
            Input geometry
        transform : rasterio.transform.Affine
            Raster transformation matrix
        shape : Tuple[int, int]
            Output raster shape (height, width)
            
        Returns
        -------
        np.ndarray
            Boolean mask array
        """
        logger.info("Creating raster mask from geometry")
        
        # Rasterize geometry
        mask = rasterize(
            geometry.geometry,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        
        return mask.astype(bool)
    
    def calculate_utm_crs(self, longitude: float, latitude: float) -> CRS:
        """
        Calculate appropriate UTM CRS for given coordinates.
        
        Parameters
        ----------
        longitude : float
            Longitude coordinate
        latitude : float
            Latitude coordinate
            
        Returns
        -------
        CRS
            UTM coordinate reference system
        """
        # Calculate UTM zone
        utm_zone = int((longitude + 180) / 6) + 1
        
        # Determine hemisphere
        hemisphere = 'north' if latitude >= 0 else 'south'
        
        # Create UTM CRS
        utm_crs = CRS(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs")
        
        logger.info(f"Calculated UTM CRS: {utm_crs}")
        
        return utm_crs
    
    def validate_spatial_overlap(self, 
                                geometry: gpd.GeoDataFrame,
                                raster_bounds: Tuple[float, float, float, float],
                                raster_crs: CRS) -> bool:
        """
        Validate that geometry overlaps with raster bounds.
        
        Parameters
        ----------
        geometry : gpd.GeoDataFrame
            Study area geometry
        raster_bounds : Tuple[float, float, float, float]
            Raster bounds (left, bottom, right, top)
        raster_crs : CRS
            Raster coordinate reference system
            
        Returns
        -------
        bool
            True if geometry overlaps with raster
        """
        # Reproject geometry to raster CRS if needed
        if geometry.crs != raster_crs:
            geometry_proj = geometry.to_crs(raster_crs)
        else:
            geometry_proj = geometry
        
        # Create bounding box from raster bounds
        raster_bbox = box(*raster_bounds)
        
        # Check intersection
        geometry_bounds = geometry_proj.unary_union
        overlap = geometry_bounds.intersects(raster_bbox)
        
        if not overlap:
            logger.error("No spatial overlap between study area and raster data")
        
        return overlap
    
    def clip_raster_to_geometry(self,
                               raster_data: np.ndarray,
                               raster_transform: rasterio.transform.Affine,
                               geometry: gpd.GeoDataFrame,
                               nodata_value: Optional[float] = None) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """
        Clip raster data to geometry bounds.
        
        Parameters
        ----------
        raster_data : np.ndarray
            Input raster data
        raster_transform : rasterio.transform.Affine
            Raster transformation matrix
        geometry : gpd.GeoDataFrame
            Clipping geometry
        nodata_value : Optional[float]
            Value to use for areas outside geometry
            
        Returns
        -------
        Tuple[np.ndarray, rasterio.transform.Affine]
            Clipped raster data and updated transform
        """
        # Create mask
        mask = self.create_mask_from_geometry(
            geometry, raster_transform, raster_data.shape
        )
        
        # Apply mask
        clipped_data = raster_data.copy()
        if nodata_value is not None:
            clipped_data[~mask] = nodata_value
        
        return clipped_data, raster_transform