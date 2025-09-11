"""
Study Area Domain Models
========================

Domain models for study area definitions and management in EXZECO analysis.
These models provide structured representations for study area geometry,
configuration, and metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, box
from rasterio.crs import CRS
from rasterio.transform import Affine


@dataclass
class StudyAreaBounds:
    """
    Represents geographical bounds for a study area.
    
    Attributes
    ----------
    min_x : float
        Minimum X coordinate (west bound)
    min_y : float  
        Minimum Y coordinate (south bound)
    max_x : float
        Maximum X coordinate (east bound)
    max_y : float
        Maximum Y coordinate (north bound)
    crs : str
        Coordinate reference system
    """
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: str = "EPSG:4326"
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert bounds to tuple format (min_x, min_y, max_x, max_y)."""
        return (self.min_x, self.min_y, self.max_x, self.max_y)
    
    def to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert bounds to bbox format."""
        return self.to_tuple()
    
    def width(self) -> float:
        """Calculate width of the bounds."""
        return self.max_x - self.min_x
    
    def height(self) -> float:
        """Calculate height of the bounds."""
        return self.max_y - self.min_y
    
    def center(self) -> Tuple[float, float]:
        """Calculate center point of the bounds."""
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)
    
    @classmethod
    def from_tuple(cls, bounds: Tuple[float, float, float, float], crs: str = "EPSG:4326") -> "StudyAreaBounds":
        """Create StudyAreaBounds from tuple (min_x, min_y, max_x, max_y)."""
        return cls(
            min_x=bounds[0],
            min_y=bounds[1], 
            max_x=bounds[2],
            max_y=bounds[3],
            crs=crs
        )
    
    @classmethod
    def from_geometry(cls, geometry: Union[Polygon, MultiPolygon], crs: str = "EPSG:4326") -> "StudyAreaBounds":
        """Create StudyAreaBounds from shapely geometry."""
        bounds = geometry.bounds
        return cls.from_tuple(bounds, crs)


@dataclass
class StudyAreaGeometry:
    """
    Container for study area geometry with metadata.
    
    Attributes
    ----------
    geometry : Union[Polygon, MultiPolygon]
        The shapely geometry object
    name : str, optional
        Name identifier for the study area
    properties : Dict[str, Any]
        Additional properties/attributes
    crs : str
        Coordinate reference system
    area_m2 : float, optional
        Area in square meters (calculated when needed)
    area_km2 : float, optional
        Area in square kilometers (calculated when needed)
    """
    geometry: Union[Polygon, MultiPolygon]
    name: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    crs: str = "EPSG:4326"
    area_m2: Optional[float] = None
    area_km2: Optional[float] = None
    
    @property
    def bounds(self) -> StudyAreaBounds:
        """Get bounds of the geometry."""
        return StudyAreaBounds.from_geometry(self.geometry, self.crs)
    
    @property
    def centroid(self) -> Point:
        """Get centroid of the geometry."""
        return self.geometry.centroid
    
    def calculate_area(self, target_crs: str = "EPSG:3857") -> Tuple[float, float]:
        """
        Calculate area in square meters and square kilometers.
        
        Parameters
        ----------
        target_crs : str
            Equal area projection for accurate calculation
            
        Returns
        -------
        Tuple[float, float]
            Area in (square meters, square kilometers)
        """
        # Convert to GeoDataFrame for reprojection
        gdf = gpd.GeoDataFrame([{'name': self.name or 'study_area'}], 
                             geometry=[self.geometry], 
                             crs=self.crs)
        
        # Reproject to equal area projection
        gdf_projected = gdf.to_crs(target_crs)
        
        # Calculate area
        area_m2 = gdf_projected.geometry.area.iloc[0]
        area_km2 = area_m2 / 1e6
        
        # Update stored values
        self.area_m2 = area_m2
        self.area_km2 = area_km2
        
        return area_m2, area_km2
    
    def to_geopandas(self) -> gpd.GeoDataFrame:
        """Convert to GeoDataFrame."""
        return gpd.GeoDataFrame(
            [self.properties] if self.properties else [{}],
            geometry=[self.geometry],
            crs=self.crs
        )


@dataclass
class StudyArea:
    """
    Complete study area definition with geometry and metadata.
    
    Attributes
    ----------
    geometries : List[StudyAreaGeometry]
        List of study area geometries (subcatchments)
    total_geometry : StudyAreaGeometry, optional
        Dissolved total geometry for the entire study area
    source_path : Path, optional
        Path to source shapefile/geopackage
    source_type : str
        Source type ('shapefile', 'bounds', 'geometry')
    processing_history : List[str]
        History of processing operations
    metadata : Dict[str, Any]
        Additional metadata about the study area
    """
    geometries: List[StudyAreaGeometry]
    total_geometry: Optional[StudyAreaGeometry] = None
    source_path: Optional[Path] = None
    source_type: str = "geometry"
    processing_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate total geometry if not provided."""
        if self.total_geometry is None and self.geometries:
            self._calculate_total_geometry()
    
    @property
    def bounds(self) -> StudyAreaBounds:
        """Get bounds of the entire study area."""
        if self.total_geometry:
            return self.total_geometry.bounds
        elif self.geometries:
            # Calculate combined bounds
            all_bounds = [geom.bounds for geom in self.geometries]
            min_x = min(b.min_x for b in all_bounds)
            min_y = min(b.min_y for b in all_bounds)
            max_x = max(b.max_x for b in all_bounds)
            max_y = max(b.max_y for b in all_bounds)
            return StudyAreaBounds(min_x, min_y, max_x, max_y, self.geometries[0].crs)
        else:
            raise ValueError("No geometries defined")
    
    @property
    def crs(self) -> str:
        """Get CRS of the study area."""
        if self.total_geometry:
            return self.total_geometry.crs
        elif self.geometries:
            return self.geometries[0].crs
        else:
            return "EPSG:4326"
    
    @property
    def total_area_km2(self) -> float:
        """Get total area in square kilometers."""
        if self.total_geometry and self.total_geometry.area_km2:
            return self.total_geometry.area_km2
        elif self.total_geometry:
            _, area_km2 = self.total_geometry.calculate_area()
            return area_km2
        else:
            # Sum individual areas
            total = 0.0
            for geom in self.geometries:
                if geom.area_km2:
                    total += geom.area_km2
                else:
                    _, area_km2 = geom.calculate_area()
                    total += area_km2
            return total
    
    def _calculate_total_geometry(self):
        """Calculate total (dissolved) geometry."""
        if not self.geometries:
            return
            
        # Combine all geometries
        all_geoms = [g.geometry for g in self.geometries]
        
        # Use shapely unary_union for dissolving
        from shapely.ops import unary_union
        dissolved = unary_union(all_geoms)
        
        # Ensure it's a proper MultiPolygon if needed
        if hasattr(dissolved, 'geoms') and len(dissolved.geoms) > 1:
            if not isinstance(dissolved, MultiPolygon):
                dissolved = MultiPolygon(dissolved.geoms)
        
        # Create total geometry
        self.total_geometry = StudyAreaGeometry(
            geometry=dissolved,
            name="total_study_area",
            crs=self.geometries[0].crs,
            properties={"source": "dissolved", "subcatchment_count": len(self.geometries)}
        )
    
    def add_geometry(self, geometry: StudyAreaGeometry):
        """Add a new geometry to the study area."""
        self.geometries.append(geometry)
        self._calculate_total_geometry()  # Recalculate total
        self.processing_history.append(f"Added geometry: {geometry.name}")
    
    def to_geopandas(self, include_total: bool = False) -> gpd.GeoDataFrame:
        """
        Convert to GeoDataFrame.
        
        Parameters
        ----------
        include_total : bool
            Whether to include the total dissolved geometry
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with all geometries
        """
        geoms_to_include = self.geometries.copy()
        if include_total and self.total_geometry:
            geoms_to_include.append(self.total_geometry)
        
        if not geoms_to_include:
            return gpd.GeoDataFrame()
        
        # Create list of geometries and properties
        geometries = [g.geometry for g in geoms_to_include]
        properties_list = []
        
        for g in geoms_to_include:
            props = g.properties.copy()
            props['name'] = g.name
            if g.area_km2:
                props['area_km2'] = g.area_km2
            properties_list.append(props)
        
        return gpd.GeoDataFrame(properties_list, geometry=geometries, crs=self.crs)
    
    @classmethod
    def from_shapefile(cls, shapefile_path: Union[str, Path], name_column: Optional[str] = None) -> "StudyArea":
        """
        Create StudyArea from shapefile/geopackage.
        
        Parameters
        ----------
        shapefile_path : str or Path
            Path to shapefile or geopackage
        name_column : str, optional
            Column to use for geometry names
            
        Returns
        -------
        StudyArea
            Study area loaded from file
        """
        path = Path(shapefile_path)
        if not path.exists():
            raise FileNotFoundError(f"Shapefile not found: {path}")
        
        # Load with geopandas
        gdf = gpd.read_file(path)
        
        if len(gdf) == 0:
            raise ValueError("Shapefile contains no features")
        
        # Ensure valid geometries
        gdf = gdf[gdf.geometry.is_valid]
        
        if len(gdf) == 0:
            raise ValueError("Shapefile contains no valid geometries")
        
        # Create geometry objects
        geometries = []
        for idx, row in gdf.iterrows():
            name = None
            if name_column and name_column in row:
                name = str(row[name_column])
            else:
                name = f"subcatchment_{idx + 1}"
            
            # Get properties (exclude geometry column)
            properties = {k: v for k, v in row.items() if k != gdf.geometry.name}
            
            geom = StudyAreaGeometry(
                geometry=row.geometry,
                name=name,
                properties=properties,
                crs=str(gdf.crs)
            )
            geometries.append(geom)
        
        return cls(
            geometries=geometries,
            source_path=path,
            source_type="shapefile",
            metadata={
                "original_crs": str(gdf.crs),
                "feature_count": len(gdf),
                "columns": list(gdf.columns)
            }
        )
    
    @classmethod
    def from_bounds(cls, bounds: Tuple[float, float, float, float], crs: str = "EPSG:4326", name: str = "bounding_box") -> "StudyArea":
        """
        Create StudyArea from bounding box.
        
        Parameters
        ----------
        bounds : Tuple[float, float, float, float]
            Bounds as (min_x, min_y, max_x, max_y)
        crs : str
            Coordinate reference system
        name : str
            Name for the study area
            
        Returns
        -------
        StudyArea
            Study area from bounding box
        """
        # Create box geometry
        geom = box(*bounds)
        
        study_geom = StudyAreaGeometry(
            geometry=geom,
            name=name,
            crs=crs,
            properties={"source": "bounds", "bounds": bounds}
        )
        
        return cls(
            geometries=[study_geom],
            source_type="bounds",
            metadata={"original_bounds": bounds, "crs": crs}
        )


@dataclass
class StudyAreaConfig:
    """
    Configuration for study area loading and processing.
    
    Attributes
    ----------
    shapefile_path : Path, optional
        Path to shapefile/geopackage (preferred method)
    bounds : Tuple[float, float, float, float], optional
        Fallback bounding box (min_lon, min_lat, max_lon, max_lat)
    name_column : str, optional
        Column to use for naming subcatchments
    buffer_distance : float
        Buffer distance in meters for geometry operations
    simplify_tolerance : float
        Tolerance for geometry simplification
    validate_geometry : bool
        Whether to validate and fix geometries
    target_crs : str, optional
        Target CRS for reprojection
    """
    shapefile_path: Optional[Path] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    name_column: Optional[str] = None
    buffer_distance: float = 0.0
    simplify_tolerance: float = 0.0
    validate_geometry: bool = True
    target_crs: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.shapefile_path is None and self.bounds is None:
            raise ValueError("Must specify either shapefile_path or bounds")
        
        if self.shapefile_path is not None:
            self.shapefile_path = Path(self.shapefile_path)
    
    def has_shapefile(self) -> bool:
        """Check if shapefile path is specified and exists."""
        return (self.shapefile_path is not None and 
                self.shapefile_path.exists())
    
    def has_bounds(self) -> bool:
        """Check if bounds are specified."""
        return (self.bounds is not None and 
                len(self.bounds) == 4)
    
    def get_preferred_source(self) -> str:
        """Get the preferred data source."""
        if self.has_shapefile():
            return "shapefile"
        elif self.has_bounds():
            return "bounds"
        else:
            return "none"