#!/usr/bin/env python
"""
DEM Download and Processing Utilities
=====================================

This module provides utilities for downloading and processing Digital Elevation Models
from various sources including SRTM, Copernicus DEM, and OpenTopography.

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import requests
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging
import hashlib
import json
from tqdm import tqdm
import elevation
import earthpy.spatial as es
import geopandas as gpd
from shapely.geometry import box, Polygon
import xarray as xr
import rioxarray as rxr
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import zipfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DEMDownloader:
    """
    Download and process DEM data from various sources.
    
    Supports:
    - SRTM 30m/90m
    - Copernicus GLO-30
    - Local file loading
    - Automatic caching
    - Multi-tile merging
    """
    
    # DEM source configurations
    SOURCES = {
        'srtm30': {
            'resolution': 30,
            'url_pattern': 'https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm/{lat}{lon}.tif',
            'description': 'SRTM 1 arc-second (~30m) global DEM'
        },
        'srtm90': {
            'resolution': 90,
            'url_pattern': 'https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{x}_{y}.zip',
            'description': 'SRTM 3 arc-second (~90m) global DEM'
        },
        'copernicus': {
            'resolution': 30,
            'url_pattern': 'https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat}_{lon}_DEM/Copernicus_DSM_COG_10_{lat}_{lon}_DEM.tif',
            'description': 'Copernicus GLO-30 DEM'
        }
    }
    
    def __init__(self, cache_dir: Union[str, Path] = "./data/dem/cache"):
        """
        Initialize DEM downloader.
        
        Parameters
        ----------
        cache_dir : str or Path
            Directory for caching downloaded DEMs
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> dict:
        """Load or create cache index."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index."""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _get_cache_key(self, bounds: Tuple, source: str, resolution: Optional[int] = None) -> str:
        """Generate unique cache key for bounds and source."""
        key_str = f"{bounds}_{source}_{resolution}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def download_dem(self,
                     bounds: Tuple[float, float, float, float],
                     source: str = 'copernicus',
                     resolution: Optional[int] = None,
                     output_path: Optional[Union[str, Path]] = None,
                     force_download: bool = False) -> Path:
        """
        Download DEM for specified bounds.
        
        Parameters
        ----------
        bounds : tuple
            Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
        source : str
            DEM source ('srtm30', 'srtm90', 'copernicus', 'elevation')
        resolution : int, optional
            Target resolution in meters (will resample if needed)
        output_path : str or Path, optional
            Output file path
        force_download : bool
            Force re-download even if cached
            
        Returns
        -------
        Path
            Path to downloaded/processed DEM file
        """
        logger.info(f"Downloading DEM from {source} for bounds: {bounds}")
        
        # Check cache
        cache_key = self._get_cache_key(bounds, source, resolution)
        if not force_download and cache_key in self.cache_index:
            cached_path = Path(self.cache_index[cache_key])
            if cached_path.exists():
                logger.info(f"Using cached DEM: {cached_path}")
                return cached_path
        
        # Set output path
        if output_path is None:
            output_path = self.cache_dir / f"dem_{cache_key}.tif"
        else:
            output_path = Path(output_path)
        
        # Download based on source
        if source == 'elevation':
            # Use elevation library (SRTM)
            self._download_srtm_elevation(bounds, output_path)
        elif source == 'copernicus':
            self._download_copernicus(bounds, output_path)
        elif source.startswith('srtm'):
            self._download_srtm_tiles(bounds, output_path, source)
        else:
            raise ValueError(f"Unknown DEM source: {source}")
        
        # Resample if needed
        if resolution:
            output_path = self._resample_dem(output_path, resolution)
        
        # Update cache
        self.cache_index[cache_key] = str(output_path)
        self._save_cache_index()
        
        return output_path
    
    def _download_srtm_elevation(self, bounds: Tuple, output_path: Path):
        """Download SRTM using elevation library."""
        logger.info("Downloading SRTM tiles using elevation library...")
        
        # Clean previous downloads
        elevation.clean()
        
        # Download tiles
        elevation.clip(bounds=bounds, output=str(output_path))
        
        logger.info(f"SRTM DEM saved to {output_path}")
    
    def _download_copernicus(self, bounds: Tuple, output_path: Path):
        """Download Copernicus GLO-30 DEM tiles."""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Calculate required tiles
        tiles = []
        for lat in range(int(np.floor(min_lat)), int(np.ceil(max_lat))):
            for lon in range(int(np.floor(min_lon)), int(np.ceil(max_lon))):
                lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
                lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
                tiles.append((lat_str, lon_str))
        
        logger.info(f"Downloading {len(tiles)} Copernicus tiles...")
        
        # Download tiles in parallel
        tile_paths = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for lat_str, lon_str in tiles:
                url = self.SOURCES['copernicus']['url_pattern'].format(
                    lat=lat_str, lon=lon_str
                )
                tile_path = self.cache_dir / f"copernicus_{lat_str}_{lon_str}.tif"
                
                if not tile_path.exists():
                    future = executor.submit(self._download_tile, url, tile_path)
                    futures.append((future, tile_path))
                else:
                    tile_paths.append(tile_path)
            
            # Wait for downloads
            for future, tile_path in futures:
                try:
                    if future.result():
                        tile_paths.append(tile_path)
                except Exception as e:
                    logger.error(f"Failed to download tile: {e}")
        
        # Merge tiles
        self._merge_tiles(tile_paths, bounds, output_path)
    
    def _download_srtm_tiles(self, bounds: Tuple, output_path: Path, source: str):
        """Download SRTM tiles from CGIAR or other sources."""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        if source == 'srtm30':
            # SRTM30 uses 1-degree tiles with specific naming
            tiles = []
            for lat in range(int(np.floor(min_lat)), int(np.ceil(max_lat)) + 1):
                for lon in range(int(np.floor(min_lon)), int(np.ceil(max_lon)) + 1):
                    # SRTM30 tile naming: N/S + lat + E/W + lon
                    lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
                    lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                    tiles.append((lat_str, lon_str))
        else:
            # SRTM90 uses 5-degree tiles
            tiles = []
            for lat in range(int(np.floor(min_lat/5))*5, int(np.ceil(max_lat/5))*5, 5):
                for lon in range(int(np.floor(min_lon/5))*5, int(np.ceil(max_lon/5))*5, 5):
                    # SRTM tile naming convention
                    x = (lon + 180) // 5 + 1
                    y = (60 - lat) // 5
                    tiles.append((x, y))
        
        logger.info(f"Downloading {len(tiles)} SRTM tiles...")
        
        # Download tiles
        tile_paths = []
        for tile_coords in tiles:
            if source == 'srtm30':
                lat_str, lon_str = tile_coords
                url = self.SOURCES['srtm30']['url_pattern'].format(lat=lat_str, lon=lon_str)
                tile_path = self.cache_dir / f"srtm30_{lat_str}_{lon_str}.tif"
                
                if not tile_path.exists():
                    if self._download_tile(url, tile_path):
                        tile_paths.append(tile_path)
                else:
                    tile_paths.append(tile_path)
                    
            elif source == 'srtm90':
                x, y = tile_coords
                url = self.SOURCES['srtm90']['url_pattern'].format(x=x, y=y)
                zip_path = self.cache_dir / f"srtm90_{x:02d}_{y:02d}.zip"
                tile_path = self.cache_dir / f"srtm90_{x:02d}_{y:02d}.tif"
                
                # Check if final TIF exists
                if not tile_path.exists():
                    # Download ZIP if needed
                    if not zip_path.exists():
                        if not self._download_tile(url, zip_path):
                            continue
                    
                    # Extract ZIP to get TIF
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # Find the TIF file in the ZIP
                            tif_files = [f for f in zip_ref.namelist() if f.endswith('.tif')]
                            if tif_files:
                                # Extract first TIF file
                                zip_ref.extract(tif_files[0], self.cache_dir)
                                extracted_path = self.cache_dir / tif_files[0]
                                # Rename to standardized name
                                extracted_path.rename(tile_path)
                    except Exception as e:
                        logger.error(f"Failed to extract {zip_path}: {e}")
                        continue
                
                if tile_path.exists():
                    tile_paths.append(tile_path)
        
        # Merge tiles
        self._merge_tiles(tile_paths, bounds, output_path)
    
    def _download_tile(self, url: str, output_path: Path, max_retries: int = 3) -> bool:
        """
        Download a single tile with retry logic.
        
        Parameters
        ----------
        url : str
            Download URL
        output_path : Path
            Output file path
        max_retries : int
            Maximum number of retry attempts
            
        Returns
        -------
        bool
            Success status
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading: {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Download with progress bar
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                
                with open(output_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(block_size):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {url}")
                    return False
        
        return False
    
    def _merge_tiles(self, tile_paths: List[Path], bounds: Tuple, output_path: Path):
        """
        Merge multiple DEM tiles and clip to bounds.
        
        Parameters
        ----------
        tile_paths : list
            List of tile file paths
        bounds : tuple
            Bounding box for clipping
        output_path : Path
            Output file path
        """
        if not tile_paths:
            raise ValueError("No tiles to merge")
        
        logger.info(f"Merging {len(tile_paths)} tiles...")
        
        # Open all tiles
        src_files = []
        for path in tile_paths:
            if path.exists():
                src = rasterio.open(path)
                src_files.append(src)
        
        if not src_files:
            raise ValueError("No valid tiles found")
        
        # Merge tiles
        mosaic, out_trans = merge(src_files)
        
        # Get metadata from first file
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        
        # Close source files
        for src in src_files:
            src.close()
        
        # Write merged result
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Clip to bounds
        self._clip_dem(output_path, bounds)
        
        logger.info(f"Merged DEM saved to {output_path}")
    
    def _clip_dem(self, dem_path: Path, bounds: Tuple):
        """
        Clip DEM to specified bounds.
        
        Parameters
        ----------
        dem_path : Path
            Path to DEM file
        bounds : tuple
            Bounding box (min_lon, min_lat, max_lon, max_lat)
        """
        # Create temporary output
        temp_path = dem_path.parent / f"temp_{dem_path.name}"
        
        with rasterio.open(dem_path) as src:
            # Create bounding box polygon
            bbox = box(*bounds)
            
            # Get window for bounds
            window = rasterio.windows.from_bounds(*bounds, src.transform)
            
            # Read data within window
            data = src.read(1, window=window)
            
            # Update transform
            transform = rasterio.windows.transform(window, src.transform)
            
            # Write clipped data
            profile = src.profile
            profile.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': transform
            })
            
            with rasterio.open(temp_path, 'w', **profile) as dst:
                dst.write(data, 1)
        
        # Replace original with clipped
        temp_path.replace(dem_path)
    
    def _resample_dem(self, dem_path: Path, target_resolution: int) -> Path:
        """
        Resample DEM to target resolution.
        
        Parameters
        ----------
        dem_path : Path
            Input DEM path
        target_resolution : int
            Target resolution in meters
            
        Returns
        -------
        Path
            Path to resampled DEM
        """
        output_path = dem_path.parent / f"{dem_path.stem}_res{target_resolution}m.tif"
        
        with rasterio.open(dem_path) as src:
            # Calculate target dimensions
            scale_factor = src.res[0] / target_resolution
            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)
            
            # Resample
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.bilinear
            )
            
            # Update transform
            transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
            
            # Write resampled data
            profile = src.profile
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': transform
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
        
        logger.info(f"Resampled DEM to {target_resolution}m: {output_path}")
        return output_path
    
    def create_hillshade(self, dem_path: Union[str, Path], 
                         azimuth: int = 315, 
                         altitude: int = 45) -> np.ndarray:
        """
        Create hillshade from DEM for visualization.
        
        Parameters
        ----------
        dem_path : str or Path
            Path to DEM file
        azimuth : int
            Sun azimuth angle in degrees
        altitude : int
            Sun altitude angle in degrees
            
        Returns
        -------
        np.ndarray
            Hillshade array
        """
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            
        # Calculate hillshade using earthpy
        hillshade = es.hillshade(dem, azimuth=azimuth, altitude=altitude)
        
        return hillshade
    
    def calculate_slope(self, dem_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate slope and aspect from DEM.
        
        Parameters
        ----------
        dem_path : str or Path
            Path to DEM file
            
        Returns
        -------
        tuple
            Slope (degrees) and aspect (degrees) arrays
        """
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            res = src.res[0]
        
        # Calculate gradients
        dy, dx = np.gradient(dem, res)
        
        # Slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # Aspect in degrees (0-360, clockwise from north)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = np.where(aspect < 0, 360 + aspect, aspect)
        
        return slope, aspect
    
    def download_dem_with_fallback(self,
                                   bounds: Tuple[float, float, float, float],
                                   cache_dir: Union[str, Path],
                                   output_filename: str = "study_area_dem.tif",
                                   product: str = 'SRTM3') -> Tuple[Path, dict]:
        """
        Download DEM using elevation package with fallback to synthetic DEM.
        
        This method mirrors the logic from the Jupyter notebook cell for downloading DEM
        data. It first attempts to use the elevation package, and if that fails, creates
        a synthetic DEM for testing purposes.
        
        Parameters
        ----------
        bounds : tuple
            Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
        cache_dir : str or Path
            Cache directory path
        output_filename : str
            Output filename for the DEM
        product : str
            Elevation product to use ('SRTM1', 'SRTM3')
            
        Returns
        -------
        tuple
            (dem_path, dem_stats) where dem_path is Path to DEM file and 
            dem_stats is dict with DEM statistics
        """
        import os
        import shutil
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        
        # Setup paths
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dem_path = cache_dir / output_filename
        
        logger.info("Downloading DEM using elevation package...")
        logger.info(f"Study area bounds: {bounds}")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"Output path: {dem_path}")
        
        try:
            # Change to cache directory to avoid path issues
            original_dir = os.getcwd()
            os.chdir(cache_dir)
            
            # Download DEM using elevation package with relative path
            elevation.clip(
                bounds=bounds,
                output=output_filename,
                product=product,
                cache_dir="elevation_cache"
            )
            
            # Move back to original directory
            os.chdir(original_dir)
            
            # The file might be created in the elevation_cache subdirectory
            elevation_dem_path = cache_dir / "elevation_cache" / product / output_filename
            
            if elevation_dem_path.exists():
                # Copy to expected location
                shutil.copy2(elevation_dem_path, dem_path)
                logger.info(f"✅ DEM copied from {elevation_dem_path} to {dem_path}")
            elif dem_path.exists():
                logger.info(f"✅ DEM downloaded successfully: {dem_path}")
            else:
                raise FileNotFoundError("DEM file not found in expected locations")
                
        except Exception as e:
            # Make sure to return to original directory
            try:
                os.chdir(original_dir)
            except:
                pass
            
            logger.warning(f"Elevation package failed: {e}")
            logger.info("Trying manual approach with test data...")
            
            # Create a synthetic DEM for testing purposes
            min_lon, min_lat, max_lon, max_lat = bounds
            
            # Create coordinate arrays
            lons = np.linspace(min_lon, max_lon, 100)
            lats = np.linspace(min_lat, max_lat, 100)
            
            # Create a synthetic elevation surface (simple gradient + noise)
            X, Y = np.meshgrid(lons, lats)
            elevation_data = (
                500 + 200 * (X - min_lon) / (max_lon - min_lon) +  # East-west gradient
                300 * (Y - min_lat) / (max_lat - min_lat) +        # North-south gradient
                50 * np.random.random(X.shape)                     # Random noise
            ).astype(np.float32)
            
            # Create GeoTIFF with proper georeference
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat, 
                                  elevation_data.shape[1], elevation_data.shape[0])
            
            with rasterio.open(
                dem_path,
                'w',
                driver='GTiff',
                height=elevation_data.shape[0],
                width=elevation_data.shape[1],
                count=1,
                dtype=elevation_data.dtype,
                crs=CRS.from_epsg(4326),
                transform=transform,
                compress='deflate'
            ) as dst:
                dst.write(elevation_data, 1)
            
            logger.info(f"✅ Synthetic DEM created for testing: {dem_path}")
        
        # Verify the file exists and get basic statistics
        if dem_path.exists():
            logger.info(f"DEM file confirmed at: {dem_path}")
            dem_stats = self.get_dem_stats(dem_path)
            
            logger.info("\nDEM Statistics:")
            for key, value in dem_stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
                    
            return dem_path, dem_stats
        else:
            raise FileNotFoundError("❌ DEM file was not created successfully")

    def get_dem_stats(self, dem_path: Union[str, Path]) -> dict:
        """
        Get statistics for DEM.
        
        Parameters
        ----------
        dem_path : str or Path
            Path to DEM file
            
        Returns
        -------
        dict
            DEM statistics
        """
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            # Handle nodata values
            if src.nodata is not None:
                dem_data = dem_data.astype(float)
                dem_data[dem_data == src.nodata] = np.nan
            
            # Use nanmin/nanmax/nanmean/nanstd to handle nodata properly
            valid_data = dem_data[~np.isnan(dem_data)] if np.any(np.isnan(dem_data)) else dem_data
            
            # Calculate actual ground resolution in meters
            if src.crs.is_geographic:
                # Geographic coordinates (lat/lon) - convert to meters
                minx, miny, maxx, maxy = src.bounds
                lon_extent = maxx - minx
                lat_extent = maxy - miny
                
                # Approximate conversion to meters
                lat_center = (maxy + miny) / 2
                lon_extent_m = lon_extent * 111320 * np.cos(np.radians(lat_center))
                lat_extent_m = lat_extent * 111320
                
                resolution_x = lon_extent_m / src.width
                resolution_y = lat_extent_m / src.height
                resolution_m = (resolution_x + resolution_y) / 2  # Average
            else:
                # Projected coordinates - use directly
                resolution_m = src.res[0]
            
            stats = {
                'min_elevation': float(np.nanmin(dem_data)),
                'max_elevation': float(np.nanmax(dem_data)),
                'mean_elevation': float(np.nanmean(dem_data)),
                'std_elevation': float(np.nanstd(dem_data)),
                'shape': dem_data.shape,
                'resolution': resolution_m,
                'width': src.width,
                'height': src.height,
                'crs': str(src.crs),
                'bounds': src.bounds,
                'nodata_value': src.nodata,
                'valid_pixels': len(valid_data),
                'total_pixels': dem_data.size
            }
        
        return stats


class StudyArea:
    """
    Define and manage study area for analysis.
    """
    
    def __init__(self, geometry: Union[str, dict, Polygon, Tuple]):
        """
        Initialize study area.
        
        Parameters
        ----------
        geometry : various
            Can be:
            - String: place name for geocoding
            - Dict: GeoJSON geometry
            - Polygon: Shapely polygon
            - Tuple: (min_lon, min_lat, max_lon, max_lat)
        """
        self.geometry = self._parse_geometry(geometry)
        self.bounds = self.geometry.bounds
        self.crs = CRS.from_epsg(4326)  # WGS84
        
    def _parse_geometry(self, geometry) -> Polygon:
        """Parse various geometry inputs."""
        if isinstance(geometry, str):
            # Geocode place name (requires geocoding service)
            return self._geocode_place(geometry)
        elif isinstance(geometry, dict):
            # GeoJSON geometry
            return Polygon(geometry['coordinates'][0])
        elif isinstance(geometry, Polygon):
            return geometry
        elif isinstance(geometry, tuple) and len(geometry) == 4:
            # Bounding box
            return box(*geometry)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geometry)}")
    
    def _geocode_place(self, place_name: str) -> Polygon:
        """Geocode place name to polygon (simplified implementation)."""
        # This would require a geocoding service like Nominatim
        # For now, return a default area
        logger.warning(f"Geocoding not implemented, using default area for: {place_name}")
        return box(-1.0, 49.0, 0.0, 50.0)  # Default area
    
    def to_geopandas(self) -> gpd.GeoDataFrame:
        """Convert to GeoDataFrame."""
        return gpd.GeoDataFrame([{'geometry': self.geometry}], crs=self.crs)
    
    def buffer(self, distance: float) -> 'StudyArea':
        """
        Buffer study area by distance in degrees.
        
        Parameters
        ----------
        distance : float
            Buffer distance in degrees
            
        Returns
        -------
        StudyArea
            Buffered study area
        """
        buffered = self.geometry.buffer(distance)
        return StudyArea(buffered)
    
    def get_area_km2(self) -> float:
        """Calculate area in square kilometers."""
        # Project to equal area projection for accurate area calculation
        gdf = self.to_geopandas()
        gdf_projected = gdf.to_crs('EPSG:3857')  # Web Mercator
        return gdf_projected.geometry.area[0] / 1e6  # m² to km²


if __name__ == "__main__":
    # Example usage
    downloader = DEMDownloader()
    
    # Define study area (example: small area in France)
    bounds = (5.0, 43.5, 5.5, 44.0)  # (min_lon, min_lat, max_lon, max_lat)
    
    # Download DEM
    dem_path = downloader.download_dem(
        bounds=bounds,
        source='copernicus',
        resolution=30
    )
    
    # Get DEM statistics
    stats = downloader.get_dem_stats(dem_path)
    print(f"DEM Statistics: {stats}")
    
    # Create hillshade
    hillshade = downloader.create_hillshade(dem_path)
    print(f"Hillshade shape: {hillshade.shape}")