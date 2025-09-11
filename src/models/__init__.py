"""
EXZECO Domain Models
===================

This package contains domain models for the EXZECO flood risk assessment system.
These models provide structured data representations for:
- DEM data and metadata
- Study area definitions
- Flood risk analysis results
- Analysis configuration parameters

The models use dataclasses for clean, type-safe data structures while maintaining
backward compatibility with the existing codebase.
"""

from .dem import DEMInfo, DEMData, DEMBounds, DEMDownloadConfig
from .study_area import StudyArea, StudyAreaBounds, StudyAreaConfig
from .flood_risk import FloodRiskResult, FloodRiskMetrics, FloodRiskSummary
from .analysis_config import ExzecoConfig, ProcessingConfig, VisualizationConfig

__all__ = [
    # DEM models
    'DEMInfo',
    'DEMData', 
    'DEMBounds',
    'DEMDownloadConfig',
    
    # Study area models
    'StudyArea',
    'StudyAreaBounds',
    'StudyAreaConfig',
    
    # Flood risk models
    'FloodRiskResult',
    'FloodRiskMetrics',
    'FloodRiskSummary',
    
    # Configuration models
    'ExzecoConfig',
    'ProcessingConfig',
    'VisualizationConfig',
]