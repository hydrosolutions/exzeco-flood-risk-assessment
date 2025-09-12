"""
Flood Risk Domain Models
========================

Domain models for flood risk analysis results and metrics in EXZECO analysis.
These models provide structured representations of flood probability maps,
risk assessments, and statistical summaries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon


@dataclass
class FloodProbabilityMap:
    """
    Container for flood probability map data.
    
    Attributes
    ----------
    probability_data : np.ndarray
        2D array with flood probability values (0.0 to 1.0)
    noise_level : float
        DEM noise level in meters used for this analysis
    threshold : float
        Default probability threshold for binary flood classification
    resolution : float
        Spatial resolution in meters per pixel
    transform : Any
        Rasterio affine transformation matrix
    crs : str
        Coordinate reference system
    nodata : float
        No data value
    """
    probability_data: np.ndarray
    noise_level: float
    threshold: float = 0.5
    resolution: float = 30.0
    transform: Optional[Any] = None
    crs: str = "EPSG:4326"
    nodata: float = np.nan
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the probability array (height, width)."""
        return self.probability_data.shape
    
    @property
    def max_probability(self) -> float:
        """Maximum probability value in the map."""
        valid_data = self.probability_data[~np.isnan(self.probability_data)]
        return float(np.max(valid_data)) if len(valid_data) > 0 else 0.0
    
    @property
    def mean_probability(self) -> float:
        """Mean probability value in the map."""
        valid_data = self.probability_data[~np.isnan(self.probability_data)]
        return float(np.mean(valid_data)) if len(valid_data) > 0 else 0.0
    
    @property
    def flooded_pixels(self) -> int:
        """Number of pixels above threshold."""
        return int(np.sum(self.probability_data > self.threshold))
    
    @property
    def pixel_area_km2(self) -> float:
        """Area of each pixel in square kilometers."""
        return (self.resolution ** 2) / 1e6
    
    def get_binary_flood_map(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get binary flood map using threshold.
        
        Parameters
        ----------
        threshold : float, optional
            Probability threshold (uses default if not provided)
            
        Returns
        -------
        np.ndarray
            Binary array (1 for flooded, 0 for not flooded)
        """
        thresh = threshold if threshold is not None else self.threshold
        return (self.probability_data > thresh).astype(int)
    
    def calculate_flood_area_km2(self, threshold: Optional[float] = None) -> float:
        """
        Calculate flooded area in square kilometers.
        
        Parameters
        ----------
        threshold : float, optional
            Probability threshold (uses default if not provided)
            
        Returns
        -------
        float
            Flooded area in square kilometers
        """
        thresh = threshold if threshold is not None else self.threshold
        flood_pixels = np.sum(self.probability_data > thresh)
        return flood_pixels * self.pixel_area_km2


@dataclass
class FloodRiskMetrics:
    """
    Comprehensive flood risk metrics for a specific noise level.
    
    Attributes
    ----------
    noise_level : float
        DEM noise level in meters
    pixel_area_km2 : float
        Area of each pixel in square kilometers
    total_pixels : int
        Total number of pixels in the analysis area
    very_high_risk : RiskLevelMetrics
        Very high risk metrics (probability > 0.8)
    high_risk : RiskLevelMetrics
        High risk metrics (probability 0.6-0.8)
    medium_risk : RiskLevelMetrics
        Medium risk metrics (probability 0.4-0.6)
    low_risk : RiskLevelMetrics
        Low risk metrics (probability 0.2-0.4)
    very_low_risk : RiskLevelMetrics
        Very low risk metrics (probability 0.01-0.2)
    flood_threshold_metrics : RiskLevelMetrics
        Standard flood threshold metrics (probability > 0.5)
    statistical_metrics : StatisticalMetrics
        Statistical measures of probability distribution
    """
    noise_level: float
    pixel_area_km2: float
    total_pixels: int
    very_high_risk: 'RiskLevelMetrics'
    high_risk: 'RiskLevelMetrics'
    medium_risk: 'RiskLevelMetrics'
    low_risk: 'RiskLevelMetrics'
    very_low_risk: 'RiskLevelMetrics'
    flood_threshold_metrics: 'RiskLevelMetrics'
    statistical_metrics: 'StatisticalMetrics'
    
    @property
    def critical_risk_area_km2(self) -> float:
        """Combined area of high and very high risk zones."""
        return self.very_high_risk.area_km2 + self.high_risk.area_km2
    
    @property
    def critical_risk_percentage(self) -> float:
        """Percentage of area in high and very high risk zones."""
        return self.very_high_risk.percentage + self.high_risk.percentage
    
    @property
    def total_risk_pixels(self) -> int:
        """Total pixels with any level of risk."""
        return (self.very_high_risk.pixel_count + 
                self.high_risk.pixel_count + 
                self.medium_risk.pixel_count + 
                self.low_risk.pixel_count + 
                self.very_low_risk.pixel_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'noise_level': self.noise_level,
            'pixel_area_km2': self.pixel_area_km2,
            'total_pixels': self.total_pixels,
            
            # Risk level counts
            'very_high_count': self.very_high_risk.pixel_count,
            'high_count': self.high_risk.pixel_count,
            'medium_count': self.medium_risk.pixel_count,
            'low_count': self.low_risk.pixel_count,
            'very_low_count': self.very_low_risk.pixel_count,
            'flood_count': self.flood_threshold_metrics.pixel_count,
            'risk_pixels_total': self.total_risk_pixels,
            
            # Areas in km²
            'very_high_area_km2': self.very_high_risk.area_km2,
            'high_area_km2': self.high_risk.area_km2,
            'medium_area_km2': self.medium_risk.area_km2,
            'low_area_km2': self.low_risk.area_km2,
            'very_low_area_km2': self.very_low_risk.area_km2,
            'total_flood_area_km2': self.flood_threshold_metrics.area_km2,
            'high_and_very_high_area_km2': self.critical_risk_area_km2,
            
            # Percentages
            'very_high_pct': self.very_high_risk.percentage,
            'high_pct': self.high_risk.percentage,
            'medium_pct': self.medium_risk.percentage,
            'low_pct': self.low_risk.percentage,
            'very_low_pct': self.very_low_risk.percentage,
            'flood_pct': self.flood_threshold_metrics.percentage,
            'high_and_very_high_pct': self.critical_risk_percentage,
            
            # Statistical measures
            'max_probability': self.statistical_metrics.max_probability,
            'mean_probability': self.statistical_metrics.mean_probability,
            'median_probability': self.statistical_metrics.median_probability,
            'std_probability': self.statistical_metrics.std_probability
        }


@dataclass
class RiskLevelMetrics:
    """
    Metrics for a specific risk level.
    
    Attributes
    ----------
    pixel_count : int
        Number of pixels in this risk level
    area_km2 : float
        Area in square kilometers
    percentage : float
        Percentage of total area
    min_threshold : float
        Minimum probability threshold for this level
    max_threshold : float
        Maximum probability threshold for this level
    """
    pixel_count: int
    area_km2: float
    percentage: float
    min_threshold: float
    max_threshold: float


@dataclass
class StatisticalMetrics:
    """
    Statistical metrics for probability distribution.
    
    Attributes
    ----------
    max_probability : float
        Maximum probability value
    mean_probability : float
        Mean probability value
    median_probability : float
        Median probability value
    std_probability : float
        Standard deviation of probability values
    """
    max_probability: float
    mean_probability: float
    median_probability: float
    std_probability: float


@dataclass
class SubcatchmentResult:
    """
    Flood risk results for a specific subcatchment.
    
    Attributes
    ----------
    name : str
        Name or identifier of the subcatchment
    probability_map : FloodProbabilityMap
        Masked probability map for this subcatchment
    geometry : Union[Polygon, MultiPolygon]
        Subcatchment geometry
    properties : Dict[str, Any]
        Additional properties from source data
    area_km2 : float, optional
        Subcatchment area in square kilometers
    """
    name: str
    probability_map: FloodProbabilityMap
    geometry: Union[Polygon, MultiPolygon]
    properties: Dict[str, Any] = field(default_factory=dict)
    area_km2: Optional[float] = None


@dataclass
class FloodRiskResult:
    """
    Complete flood risk analysis result for a single noise level.
    
    Attributes
    ----------
    noise_level : float
        DEM noise level in meters
    probability_map : FloodProbabilityMap
        Main probability map for the entire domain
    is_total_domain : bool
        Whether this represents the total study area
    subcatchments : Dict[str, SubcatchmentResult]
        Individual subcatchment results (if applicable)
    processing_metadata : Dict[str, Any]
        Metadata about the analysis processing
    """
    noise_level: float
    probability_map: FloodProbabilityMap
    is_total_domain: bool = True
    subcatchments: Dict[str, SubcatchmentResult] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def noise_level_label(self) -> str:
        """Human-readable label for the noise level."""
        return f"exzeco_{int(self.noise_level*100)}cm"
    
    def has_subcatchments(self) -> bool:
        """Check if result includes subcatchment analysis."""
        return len(self.subcatchments) > 0
    
    def get_subcatchment_names(self) -> List[str]:
        """Get list of subcatchment names."""
        return list(self.subcatchments.keys())
    
    def calculate_metrics(self) -> FloodRiskMetrics:
        """
        Calculate comprehensive risk metrics for this result.
        
        Returns
        -------
        FloodRiskMetrics
            Calculated risk metrics
        """
        prob_map = self.probability_map.probability_data
        pixel_area_km2 = self.probability_map.pixel_area_km2
        
        # Define risk thresholds
        very_high_mask = prob_map > 0.8
        high_mask = (prob_map > 0.6) & (prob_map <= 0.8)
        medium_mask = (prob_map > 0.4) & (prob_map <= 0.6)
        low_mask = (prob_map > 0.2) & (prob_map <= 0.4)
        very_low_mask = (prob_map > 0.01) & (prob_map <= 0.2)
        flood_mask = prob_map > 0.5
        
        total_pixels = prob_map.size
        
        def _create_risk_level_metrics(mask: np.ndarray, min_thresh: float, max_thresh: float) -> RiskLevelMetrics:
            count = int(np.sum(mask))
            area = count * pixel_area_km2
            percentage = (count / total_pixels) * 100
            return RiskLevelMetrics(count, area, percentage, min_thresh, max_thresh)
        
        # Calculate metrics for each risk level
        very_high_risk = _create_risk_level_metrics(very_high_mask, 0.8, 1.0)
        high_risk = _create_risk_level_metrics(high_mask, 0.6, 0.8)
        medium_risk = _create_risk_level_metrics(medium_mask, 0.4, 0.6)
        low_risk = _create_risk_level_metrics(low_mask, 0.2, 0.4)
        very_low_risk = _create_risk_level_metrics(very_low_mask, 0.01, 0.2)
        flood_threshold = _create_risk_level_metrics(flood_mask, 0.5, 1.0)
        
        # Statistical measures
        valid_data = prob_map[~np.isnan(prob_map)]
        if len(valid_data) > 0:
            statistical_metrics = StatisticalMetrics(
                max_probability=float(np.max(valid_data)),
                mean_probability=float(np.mean(valid_data)),
                median_probability=float(np.median(valid_data)),
                std_probability=float(np.std(valid_data))
            )
        else:
            statistical_metrics = StatisticalMetrics(0.0, 0.0, 0.0, 0.0)
        
        return FloodRiskMetrics(
            noise_level=self.noise_level,
            pixel_area_km2=pixel_area_km2,
            total_pixels=total_pixels,
            very_high_risk=very_high_risk,
            high_risk=high_risk,
            medium_risk=medium_risk,
            low_risk=low_risk,
            very_low_risk=very_low_risk,
            flood_threshold_metrics=flood_threshold,
            statistical_metrics=statistical_metrics
        )


@dataclass
class FloodRiskAnalysisResults:
    """
    Complete flood risk analysis results for all noise levels.
    
    Attributes
    ----------
    results : Dict[str, FloodRiskResult]
        Results for each noise level (keyed by noise level label)
    analysis_metadata : Dict[str, Any]
        Metadata about the overall analysis
    configuration : Any
        Analysis configuration used
    """
    results: Dict[str, FloodRiskResult] = field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    configuration: Optional[Any] = None
    
    def get_noise_levels(self) -> List[float]:
        """Get list of noise levels analyzed."""
        return [result.noise_level for result in self.results.values()]
    
    def get_result_by_noise_level(self, noise_level: float) -> Optional[FloodRiskResult]:
        """Get result for specific noise level."""
        label = f"exzeco_{int(noise_level*100)}cm"
        return self.results.get(label)
    
    def calculate_all_metrics(self) -> Dict[str, FloodRiskMetrics]:
        """Calculate metrics for all results."""
        return {label: result.calculate_metrics() 
                for label, result in self.results.items()}
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Create summary DataFrame of all results.
        
        Returns
        -------
        pd.DataFrame
            Summary DataFrame with key metrics for each noise level
        """
        summary_data = []
        
        for label, result in self.results.items():
            metrics = result.calculate_metrics()
            summary_data.append({
                'Noise Level': result.noise_level,
                'Very High Risk (km²)': metrics.very_high_risk.area_km2,
                'High Risk (km²)': metrics.high_risk.area_km2,
                'Medium Risk (km²)': metrics.medium_risk.area_km2,
                'Low Risk (km²)': metrics.low_risk.area_km2,
                'Very Low Risk (km²)': metrics.very_low_risk.area_km2,
                'Total Flood Area (km²)': metrics.flood_threshold_metrics.area_km2,
                'Critical Risk Area (km²)': metrics.critical_risk_area_km2,
                'Total Risk Pixels': metrics.total_risk_pixels,
                'Flood Coverage (%)': metrics.flood_threshold_metrics.percentage,
                'Critical Risk Coverage (%)': metrics.critical_risk_percentage,
                'Max Probability': metrics.statistical_metrics.max_probability,
                'Mean Probability': metrics.statistical_metrics.mean_probability,
                'Median Probability': metrics.statistical_metrics.median_probability,
                'Std Probability': metrics.statistical_metrics.std_probability
            })
        
        return pd.DataFrame(summary_data)
    
    @classmethod
    def from_legacy_results(cls, legacy_results: Dict, metadata: Optional[Dict] = None) -> "FloodRiskAnalysisResults":
        """
        Convert from legacy results dictionary format.
        
        Parameters
        ----------
        legacy_results : Dict
            Results in the original EXZECO format
        metadata : Dict, optional
            Additional metadata
            
        Returns
        -------
        FloodRiskAnalysisResults
            Converted results
        """
        results = {}
        
        for label, data in legacy_results.items():
            # Extract noise level from label (e.g., "exzeco_20cm" -> 0.2)
            noise_level = float(label.split('_')[1].replace('cm', '')) / 100.0
            
            # Create probability map
            prob_map = FloodProbabilityMap(
                probability_data=data['probability_map'],
                noise_level=noise_level,
                threshold=data.get('threshold', 0.5)
            )
            
            # Create subcatchment results if present
            subcatchments = {}
            if 'subcatchments' in data:
                for sub_name, sub_data in data['subcatchments'].items():
                    subcatchments[sub_name] = SubcatchmentResult(
                        name=sub_name,
                        probability_map=FloodProbabilityMap(
                            probability_data=sub_data['probability_map'],
                            noise_level=noise_level
                        ),
                        geometry=sub_data['geometry'],
                        properties=sub_data.get('original_data', {})
                    )
            
            # Create main result
            results[label] = FloodRiskResult(
                noise_level=noise_level,
                probability_map=prob_map,
                is_total_domain=data.get('total_domain', True),
                subcatchments=subcatchments
            )
        
        return cls(
            results=results,
            analysis_metadata=metadata or {},
        )


@dataclass
class FloodRiskSummary:
    """
    High-level summary of flood risk analysis.
    
    Attributes
    ----------
    total_area_analyzed_km2 : float
        Total area analyzed in square kilometers
    noise_levels_analyzed : List[float]
        List of noise levels analyzed
    max_flood_area_km2 : float
        Maximum flood area across all noise levels
    min_flood_area_km2 : float
        Minimum flood area across all noise levels
    mean_flood_area_km2 : float
        Average flood area across all noise levels
    critical_risk_evolution : List[float]
        Evolution of critical risk area across noise levels
    """
    total_area_analyzed_km2: float
    noise_levels_analyzed: List[float]
    max_flood_area_km2: float
    min_flood_area_km2: float
    mean_flood_area_km2: float
    critical_risk_evolution: List[float] = field(default_factory=list)
    
    @classmethod
    def from_results(cls, results: FloodRiskAnalysisResults) -> "FloodRiskSummary":
        """Create summary from analysis results."""
        all_metrics = results.calculate_all_metrics()
        
        flood_areas = [m.flood_threshold_metrics.area_km2 for m in all_metrics.values()]
        critical_areas = [m.critical_risk_area_km2 for m in all_metrics.values()]
        noise_levels = sorted(results.get_noise_levels())
        
        # Estimate total area from first result
        first_result = next(iter(results.results.values()))
        total_pixels = first_result.probability_map.shape[0] * first_result.probability_map.shape[1]
        pixel_area = first_result.probability_map.pixel_area_km2
        total_area = total_pixels * pixel_area
        
        return cls(
            total_area_analyzed_km2=total_area,
            noise_levels_analyzed=noise_levels,
            max_flood_area_km2=max(flood_areas) if flood_areas else 0.0,
            min_flood_area_km2=min(flood_areas) if flood_areas else 0.0,
            mean_flood_area_km2=np.mean(flood_areas) if flood_areas else 0.0,
            critical_risk_evolution=critical_areas
        )