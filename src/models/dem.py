"""
DEM Domain Models
=================

Domain models for Digital Elevation Model data structures used in EXZECO analysis.
These models provide type-safe, well-structured representations of DEM data,
metadata, and configuration parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


@dataclass
class DEMBounds:
    """
    Represents geographical bounds for DEM data.
    
    Attributes
    ----------
    min_lon : float
        Minimum longitude (west bound) in WGS84
    min_lat : float  
        Minimum latitude (south bound) in WGS84
    max_lon : float
        Maximum longitude (east bound) in WGS84
    max_lat : float
        Maximum latitude (north bound) in WGS84
    crs : str
        Coordinate reference system (default: EPSG:4326)
    """
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    crs: str = "EPSG:4326"
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert bounds to tuple format (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)
    
    def to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert bounds to bbox format (min_x, min_y, max_x, max_y)."""
        return self.to_tuple()
    
    @classmethod
    def from_tuple(cls, bounds: Tuple[float, float, float, float], crs: str = "EPSG:4326") -> "DEMBounds":
        """Create DEMBounds from tuple (min_lon, min_lat, max_lon, max_lat)."""
        return cls(
            min_lon=bounds[0],
            min_lat=bounds[1], 
            max_lon=bounds[2],
            max_lat=bounds[3],
            crs=crs
        )


@dataclass
class DEMInfo:
    """
    Metadata information about a DEM dataset.
    
    Attributes
    ----------
    width : int
        Number of columns in the DEM
    height : int
        Number of rows in the DEM
    resolution : float
        Spatial resolution in meters
    crs : str or CRS
        Coordinate reference system
    transform : Affine
        Affine transformation matrix
    nodata : float, optional
        No data value
    bounds : DEMBounds
        Geographical bounds of the DEM
    dtype : str
        Data type of elevation values
    source : str, optional
        Source of the DEM data (e.g., 'srtm30', 'copernicus')
    """
    width: int
    height: int
    resolution: float
    crs: Union[str, CRS]
    transform: Affine
    bounds: DEMBounds
    dtype: str = "float32"
    nodata: Optional[float] = None
    source: Optional[str] = None
    
    @classmethod
    def from_rasterio(cls, dataset: rasterio.DatasetReader, source: Optional[str] = None) -> "DEMInfo":
        """Create DEMInfo from rasterio dataset."""
        bounds = DEMBounds(
            min_lon=dataset.bounds.left,
            min_lat=dataset.bounds.bottom,
            max_lon=dataset.bounds.right,
            max_lat=dataset.bounds.top,
            crs=str(dataset.crs)
        )
        
        return cls(
            width=dataset.width,
            height=dataset.height,
            resolution=dataset.res[0],  # Assuming square pixels
            crs=dataset.crs,
            transform=dataset.transform,
            bounds=bounds,
            dtype=str(dataset.dtypes[0]),
            nodata=dataset.nodata,
            source=source
        )


@dataclass
class DEMData:
    """
    Container for DEM elevation data and associated metadata.
    
    Attributes
    ----------
    elevation : np.ndarray
        2D array of elevation values
    info : DEMInfo
        Metadata about the DEM
    file_path : Path, optional
        Path to the source file
    processing_history : List[str]
        History of processing operations applied
    """
    elevation: np.ndarray
    info: DEMInfo
    file_path: Optional[Path] = None
    processing_history: List[str] = field(default_factory=list)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the elevation array (height, width)."""
        return self.elevation.shape
    
    @property
    def bounds(self) -> DEMBounds:
        """Geographical bounds of the DEM."""
        return self.info.bounds
    
    @property
    def resolution(self) -> float:
        """Spatial resolution in meters."""
        return self.info.resolution
    
    def add_processing_step(self, step: str) -> None:
        """Add a processing step to the history."""
        self.processing_history.append(step)
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate basic statistics of elevation data."""
        valid_data = self.elevation[~np.isnan(self.elevation)]
        if len(valid_data) == 0:
            return {}
        
        return {
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "count": len(valid_data),
            "nodata_count": int(np.sum(np.isnan(self.elevation)))
        }


@dataclass
class DEMSourceConfig:
    """
    Configuration for a specific DEM source.
    
    Attributes
    ----------
    name : str
        Name identifier for the source
    resolution : int
        Native resolution in meters
    url_pattern : str
        URL pattern for downloading
    description : str
        Human-readable description
    max_tiles : int
        Maximum number of tiles to download
    requires_auth : bool
        Whether authentication is required
    """
    name: str
    resolution: int
    url_pattern: str
    description: str
    max_tiles: int = 100
    requires_auth: bool = False


@dataclass  
class DEMDownloadConfig:
    """
    Configuration parameters for DEM downloading and processing.
    
    Attributes
    ----------
    source : str
        DEM source identifier
    resolution : int, optional
        Target resolution in meters (will resample if different from source)
    cache_dir : Path
        Directory for caching downloaded DEMs
    force_download : bool
        Force re-download even if cached
    merge_tiles : bool
        Whether to merge multiple tiles into single file
    fill_pits : bool
        Whether to apply pit filling preprocessing
    resampling_method : str
        Resampling method when changing resolution
    output_format : str
        Output file format
    compress : bool
        Whether to compress output files
    sources : Dict[str, DEMSourceConfig]
        Available DEM source configurations
    """
    source: str = "copernicus"
    resolution: Optional[int] = None
    cache_dir: Path = Path("./data/dem/cache")
    force_download: bool = False
    merge_tiles: bool = True
    fill_pits: bool = False
    resampling_method: str = "bilinear"
    output_format: str = "GTiff"
    compress: bool = True
    sources: Dict[str, DEMSourceConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default DEM sources if none provided."""
        if not self.sources:
            self.sources = {
                'srtm30': DEMSourceConfig(
                    name='srtm30',
                    resolution=30,
                    url_pattern='https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm/{lat}{lon}.tif',
                    description='SRTM 1 arc-second (~30m) global DEM'
                ),
                'srtm90': DEMSourceConfig(
                    name='srtm90', 
                    resolution=90,
                    url_pattern='https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{x}_{y}.zip',
                    description='SRTM 3 arc-second (~90m) global DEM'
                ),
                'copernicus': DEMSourceConfig(
                    name='copernicus',
                    resolution=30,
                    url_pattern='https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat}_{lon}_DEM/Copernicus_DSM_COG_10_{lat}_{lon}_DEM.tif',
                    description='Copernicus GLO-30 DEM'
                )
            }
    
    def get_source_config(self, source: str) -> Optional[DEMSourceConfig]:
        """Get configuration for a specific DEM source."""
        return self.sources.get(source)
    
    def get_cache_key(self, bounds: DEMBounds, source: str) -> str:
        """Generate unique cache key for bounds and source."""
        import hashlib
        key_str = f"{bounds.to_tuple()}_{source}_{self.resolution}"
        return hashlib.md5(key_str.encode()).hexdigest()