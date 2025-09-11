"""
EXZECO (Extraction des Zones d'Écoulement) Package
==================================================

A Python implementation of flood risk assessment using Monte Carlo simulation
on Digital Elevation Models (DEMs), implementing the CEREMA methodology.
"""

# Import main classes and functions
from .exzeco import ExzecoAnalysis, ExzecoConfig, load_config, run_exzeco_with_config
from .dem_utils import DEMDownloader, StudyArea
from .visualization import ExzecoVisualizer, StudyAreaVisualizer, DEMVisualizer
from .visualization import create_study_area_visualization, create_dem_visualization
from .risk_metrics import compute_risk_metrics, create_risk_summary_dataframe, analyze_risk_evolution
from .risk_metrics import check_risk_significance, create_risk_visualization, export_risk_analysis

# Import domain models
from .models import (
    # DEM models
    DEMInfo,
    DEMData, 
    DEMBounds,
    DEMDownloadConfig,
    
    # Study area models
    StudyArea as StudyAreaModel,  # Avoid name conflict
    StudyAreaBounds,
    StudyAreaConfig,
    
    # Flood risk models
    FloodRiskResult,
    FloodRiskMetrics,
    FloodRiskSummary,
    FloodProbabilityMap,
    FloodRiskAnalysisResults,
    
    # Configuration models
    ExzecoConfig as NewExzecoConfig,  # Avoid name conflict
    ProcessingConfig,
    VisualizationConfig,
    AnalysisConfig,
)

__version__ = "1.0.0"
__author__ = "EXZECO Implementation Team"

__all__ = [
    # Main classes
    'ExzecoAnalysis',
    'ExzecoConfig', 
    'DEMDownloader',
    'StudyArea',
    'ExzecoVisualizer',
    'StudyAreaVisualizer',
    'DEMVisualizer',
    
    # Main functions
    'load_config',
    'run_exzeco_with_config',
    'create_study_area_visualization',
    'create_dem_visualization',
    'compute_risk_metrics',
    'create_risk_summary_dataframe',
    'analyze_risk_evolution',
    'check_risk_significance',
    'create_risk_visualization',
    'export_risk_analysis',
    
    # Domain models - DEM
    'DEMInfo',
    'DEMData',
    'DEMBounds',
    'DEMDownloadConfig',
    
    # Domain models - Study Area
    'StudyAreaModel',
    'StudyAreaBounds', 
    'StudyAreaConfig',
    
    # Domain models - Flood Risk
    'FloodRiskResult',
    'FloodRiskMetrics',
    'FloodRiskSummary',
    'FloodProbabilityMap',
    'FloodRiskAnalysisResults',
    
    # Domain models - Configuration
    'NewExzecoConfig',
    'ProcessingConfig',
    'VisualizationConfig',
    'AnalysisConfig',
]