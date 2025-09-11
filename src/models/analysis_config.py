"""
Analysis Configuration Domain Models
====================================

Domain models for EXZECO analysis configuration parameters.
These models provide structured, validated configuration for different
aspects of the flood risk analysis workflow.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml


@dataclass
class ExzecoConfig:
    """
    Core EXZECO analysis configuration parameters.
    
    Attributes
    ----------
    noise_levels : List[float]
        DEM noise levels in meters for Monte Carlo analysis
    iterations : int
        Number of Monte Carlo iterations per noise level
    min_drainage_area : float
        Minimum drainage area threshold in km²
    drainage_classes : List[float]
        Drainage area classification thresholds in km²
    n_jobs : int
        Number of parallel jobs (-1 for all available cores)
    chunk_size : int
        Processing chunk size for memory management
    seed : int, optional
        Random seed for reproducibility
    """
    noise_levels: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    iterations: int = 100
    min_drainage_area: float = 0.01
    drainage_classes: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50])
    n_jobs: int = -1
    chunk_size: int = 1000
    seed: Optional[int] = 42
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.noise_levels or any(level <= 0 for level in self.noise_levels):
            raise ValueError("Noise levels must be positive numbers")
        
        if self.iterations <= 0:
            raise ValueError("Iterations must be positive")
        
        if self.min_drainage_area <= 0:
            raise ValueError("Minimum drainage area must be positive")
        
        if not self.drainage_classes or any(cls <= 0 for cls in self.drainage_classes):
            raise ValueError("Drainage classes must be positive numbers")
    
    def get_noise_level_labels(self) -> List[str]:
        """Get human-readable labels for noise levels."""
        return [f"exzeco_{int(level*100)}cm" for level in self.noise_levels]
    
    def get_max_noise_level(self) -> float:
        """Get maximum noise level."""
        return max(self.noise_levels)
    
    def get_min_noise_level(self) -> float:
        """Get minimum noise level."""
        return min(self.noise_levels)


@dataclass
class StudyAreaConfig:
    """
    Study area configuration parameters.
    
    Attributes
    ----------
    shapefile_path : Path, optional
        Path to shapefile/geopackage (preferred method)
    bounds : Tuple[float, float, float, float], optional
        Fallback bounding box (min_lon, min_lat, max_lon, max_lat)
    """
    shapefile_path: Optional[Path] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    
    def __post_init__(self):
        """Validate and convert paths."""
        if self.shapefile_path is not None:
            self.shapefile_path = Path(self.shapefile_path)
        
        if self.shapefile_path is None and self.bounds is None:
            raise ValueError("Must specify either shapefile_path or bounds")
        
        if self.bounds is not None and len(self.bounds) != 4:
            raise ValueError("Bounds must be a 4-element tuple (min_lon, min_lat, max_lon, max_lat)")
    
    def has_shapefile(self) -> bool:
        """Check if shapefile is specified and exists."""
        return (self.shapefile_path is not None and 
                self.shapefile_path.exists())
    
    def has_valid_bounds(self) -> bool:
        """Check if bounds are specified and valid."""
        if self.bounds is None:
            return False
        min_lon, min_lat, max_lon, max_lat = self.bounds
        return min_lon < max_lon and min_lat < max_lat
    
    def get_preferred_source(self) -> str:
        """Get the preferred data source type."""
        if self.has_shapefile():
            return "shapefile"
        elif self.has_valid_bounds():
            return "bounds"
        else:
            return "none"


@dataclass
class DEMConfig:
    """
    DEM configuration parameters.
    
    Attributes
    ----------
    resolution : int
        Target DEM resolution in meters
    source : str
        DEM source ('srtm', 'copernicus', 'local')
    cache_dir : Path
        Directory for caching downloaded DEMs
    """
    resolution: int = 30
    source: str = "srtm"
    cache_dir: Path = Path("./data/dem")
    
    def __post_init__(self):
        """Validate and convert paths."""
        self.cache_dir = Path(self.cache_dir)
        
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
        
        valid_sources = ["srtm", "copernicus", "local"]
        if self.source not in valid_sources:
            raise ValueError(f"Source must be one of: {valid_sources}")


@dataclass
class ProcessingConfig:
    """
    Processing configuration parameters.
    
    Attributes
    ----------
    n_jobs : int
        Number of parallel processing jobs
    chunk_size : int
        Memory chunk size for large datasets
    use_gpu : bool
        Whether to use GPU acceleration (if available)
    memory_limit_gb : float, optional
        Memory usage limit in GB
    """
    n_jobs: int = -1
    chunk_size: int = 100000
    use_gpu: bool = False
    memory_limit_gb: Optional[float] = None
    
    def __post_init__(self):
        """Validate processing parameters."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")


@dataclass
class VisualizationConfig:
    """
    Visualization configuration parameters.
    
    Attributes
    ----------
    cmap : str
        Colormap for visualizations
    interactive : bool
        Whether to generate interactive visualizations
    export_format : List[str]
        Export formats for visualizations
    dpi : int
        DPI for static image exports
    figsize : Tuple[int, int]
        Figure size for static plots
    """
    cmap: str = "Blues"
    interactive: bool = True
    export_format: List[str] = field(default_factory=lambda: ["html", "png", "geojson"])
    dpi: int = 300
    figsize: Tuple[int, int] = (12, 8)
    
    def __post_init__(self):
        """Validate visualization parameters."""
        valid_formats = ["html", "png", "pdf", "svg", "geojson", "shapefile"]
        for fmt in self.export_format:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format '{fmt}'. Valid formats: {valid_formats}")
        
        if self.dpi <= 0:
            raise ValueError("DPI must be positive")


@dataclass
class AnalysisConfig:
    """
    Complete analysis configuration containing all parameter groups.
    
    Attributes
    ----------
    exzeco : ExzecoConfig
        Core EXZECO analysis parameters
    study_area : StudyAreaConfig
        Study area definition parameters
    dem : DEMConfig
        DEM processing parameters
    processing : ProcessingConfig
        Computational processing parameters
    visualization : VisualizationConfig
        Visualization and export parameters
    metadata : Dict[str, Any]
        Additional metadata
    """
    exzeco: ExzecoConfig = field(default_factory=ExzecoConfig)
    study_area: StudyAreaConfig = field(default_factory=StudyAreaConfig)
    dem: DEMConfig = field(default_factory=DEMConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "AnalysisConfig":
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to YAML configuration file
            
        Returns
        -------
        AnalysisConfig
            Loaded configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisConfig":
        """
        Create configuration from dictionary.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        AnalysisConfig
            Created configuration
        """
        # Extract section dictionaries with defaults
        exzeco_dict = config_dict.get('exzeco', {})
        study_area_dict = config_dict.get('study_area', {})
        dem_dict = config_dict.get('dem', {})
        processing_dict = config_dict.get('processing', {})
        visualization_dict = config_dict.get('visualization', {})
        
        # Create configuration objects
        exzeco_config = ExzecoConfig(**exzeco_dict)
        study_area_config = StudyAreaConfig(**study_area_dict)
        dem_config = DEMConfig(**dem_dict)
        processing_config = ProcessingConfig(**processing_dict)
        visualization_config = VisualizationConfig(**visualization_dict)
        
        # Extract metadata
        metadata = {k: v for k, v in config_dict.items() 
                   if k not in ['exzeco', 'study_area', 'dem', 'processing', 'visualization']}
        
        return cls(
            exzeco=exzeco_config,
            study_area=study_area_config,
            dem=dem_config,
            processing=processing_config,
            visualization=visualization_config,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Configuration as dictionary
        """
        from dataclasses import asdict
        
        result = {
            'exzeco': asdict(self.exzeco),
            'study_area': asdict(self.study_area),
            'dem': asdict(self.dem),
            'processing': asdict(self.processing),
            'visualization': asdict(self.visualization)
        }
        
        # Add metadata
        result.update(self.metadata)
        
        # Convert Path objects to strings
        def _convert_paths(obj):
            if isinstance(obj, dict):
                return {k: _convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        return _convert_paths(result)
    
    def to_yaml(self, output_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Parameters
        ----------
        output_path : str or Path
            Output file path
        """
        output_path = Path(output_path)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate the complete configuration.
        
        Returns
        -------
        List[str]
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check study area configuration
        if not self.study_area.has_shapefile() and not self.study_area.has_valid_bounds():
            errors.append("Study area configuration invalid: no valid shapefile or bounds")
        
        # Check DEM cache directory
        if not self.dem.cache_dir.parent.exists():
            errors.append(f"DEM cache parent directory does not exist: {self.dem.cache_dir.parent}")
        
        # Check processing parameters compatibility
        if self.processing.use_gpu and self.processing.n_jobs > 1:
            errors.append("GPU processing and multi-processing are not compatible")
        
        # Check visualization export formats
        if not self.visualization.export_format:
            errors.append("At least one export format must be specified")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def get_analysis_id(self) -> str:
        """Generate unique analysis identifier."""
        import hashlib
        import json
        
        # Create hash from key configuration parameters
        key_params = {
            'noise_levels': self.exzeco.noise_levels,
            'iterations': self.exzeco.iterations,
            'min_drainage_area': self.exzeco.min_drainage_area,
            'dem_source': self.dem.source,
            'dem_resolution': self.dem.resolution,
        }
        
        # Add study area info
        if self.study_area.shapefile_path:
            key_params['shapefile'] = str(self.study_area.shapefile_path)
        elif self.study_area.bounds:
            key_params['bounds'] = self.study_area.bounds
        
        # Generate hash
        params_str = json.dumps(key_params, sort_keys=True)
        analysis_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return f"exzeco_{analysis_hash}"