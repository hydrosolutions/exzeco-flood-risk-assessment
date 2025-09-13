"""
Core modules for EXZECO flood risk assessment.

This package contains the core functionality split from the monolithic
exzeco.py file into specialized modules for better maintainability.
"""

from .flow_analysis import FlowAnalyzer
from .monte_carlo import MonteCarloSimulator
from .geometry_processing import GeometryProcessor
from .drainage_classification import DrainageClassifier, ClassificationThresholds, ClassificationStats
from .export_utils import ResultExporter

__all__ = [
    'FlowAnalyzer',
    'MonteCarloSimulator', 
    'GeometryProcessor',
    'DrainageClassifier',
    'ClassificationThresholds',
    'ClassificationStats',
    'ResultExporter'
]