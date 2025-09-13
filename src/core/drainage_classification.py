#!/usr/bin/env python
"""
Drainage Classification Module for EXZECO
=========================================

This module provides drainage classification functionality including:
- Drainage area classification based on thresholds
- Statistical analysis of classified areas
- Risk level assignment
- Classification validation

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationThresholds:
    """Drainage area classification thresholds."""
    very_low: float = 0.001  # km²
    low: float = 0.01       # km²
    medium: float = 0.1     # km²
    high: float = 1.0       # km²
    very_high: float = 10.0 # km²


@dataclass
class ClassificationStats:
    """Statistics for a drainage classification."""
    area_km2: float
    pixel_count: int
    percentage: float
    mean_probability: float
    max_probability: float
    min_probability: float


class DrainageClassifier:
    """
    Handles drainage area classification and statistical analysis.
    
    This class provides functionality for classifying drainage areas
    based on configurable thresholds and computing associated statistics.
    """
    
    def __init__(self, thresholds: Optional[ClassificationThresholds] = None):
        """
        Initialize drainage classifier.
        
        Parameters
        ----------
        thresholds : Optional[ClassificationThresholds]
            Classification thresholds, uses defaults if None
        """
        self.thresholds = thresholds or ClassificationThresholds()
        
        # Define classification labels
        self.class_labels = {
            0: 'No Flow',
            1: 'Very Low Risk',
            2: 'Low Risk', 
            3: 'Medium Risk',
            4: 'High Risk',
            5: 'Very High Risk'
        }
        
        # Define classification colors for visualization
        self.class_colors = {
            0: '#ffffff',  # White - No flow
            1: '#d4edda',  # Light green - Very low
            2: '#fff3cd',  # Light yellow - Low
            3: '#ffeaa7',  # Yellow - Medium
            4: '#fab1a0',  # Orange - High
            5: '#e17055'   # Red - Very high
        }
    
    def classify_drainage_areas(self, 
                               drainage_area: np.ndarray,
                               probability_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Classify drainage areas based on thresholds.
        
        Parameters
        ----------
        drainage_area : np.ndarray
            Drainage area array in km²
        probability_map : Optional[np.ndarray]
            Probability map for enhanced classification
            
        Returns
        -------
        np.ndarray
            Classification array with integer class labels
        """
        logger.info("Classifying drainage areas")
        
        # Initialize classification array
        classification = np.zeros_like(drainage_area, dtype=np.uint8)
        
        # Apply thresholds
        classification[drainage_area >= self.thresholds.very_low] = 1  # Very Low
        classification[drainage_area >= self.thresholds.low] = 2       # Low
        classification[drainage_area >= self.thresholds.medium] = 3    # Medium
        classification[drainage_area >= self.thresholds.high] = 4      # High
        classification[drainage_area >= self.thresholds.very_high] = 5 # Very High
        
        # If probability map is provided, enhance classification
        if probability_map is not None:
            # Areas with very high probability (>0.8) get elevated one class
            high_prob_mask = probability_map > 0.8
            classification[high_prob_mask & (classification > 0) & (classification < 5)] += 1
            
            # Areas with very low probability (<0.1) get reduced one class
            low_prob_mask = probability_map < 0.1
            classification[low_prob_mask & (classification > 1)] -= 1
        
        return classification
    
    def compute_class_statistics(self, 
                                classification: np.ndarray,
                                probability_map: np.ndarray,
                                pixel_area_km2: float) -> Dict[int, ClassificationStats]:
        """
        Compute statistics for each classification class.
        
        Parameters
        ----------
        classification : np.ndarray
            Classification array
        probability_map : np.ndarray
            Probability map
        pixel_area_km2 : float
            Area of each pixel in km²
            
        Returns
        -------
        Dict[int, ClassificationStats]
            Statistics for each class
        """
        logger.info("Computing classification statistics")
        
        stats = {}
        total_pixels = classification.size
        
        for class_id in range(6):  # 0 to 5
            # Get mask for this class
            mask = classification == class_id
            pixel_count = np.sum(mask)
            
            if pixel_count > 0:
                # Calculate area
                area_km2 = pixel_count * pixel_area_km2
                percentage = (pixel_count / total_pixels) * 100
                
                # Calculate probability statistics
                class_probabilities = probability_map[mask]
                mean_prob = np.mean(class_probabilities)
                max_prob = np.max(class_probabilities)
                min_prob = np.min(class_probabilities)
                
                stats[class_id] = ClassificationStats(
                    area_km2=area_km2,
                    pixel_count=pixel_count,
                    percentage=percentage,
                    mean_probability=mean_prob,
                    max_probability=max_prob,
                    min_probability=min_prob
                )
            else:
                # Empty class
                stats[class_id] = ClassificationStats(
                    area_km2=0.0,
                    pixel_count=0,
                    percentage=0.0,
                    mean_probability=0.0,
                    max_probability=0.0,
                    min_probability=0.0
                )
        
        return stats
    
    def create_summary_dataframe(self, 
                                stats: Dict[int, ClassificationStats]) -> pd.DataFrame:
        """
        Create summary DataFrame from classification statistics.
        
        Parameters
        ----------
        stats : Dict[int, ClassificationStats]
            Classification statistics
            
        Returns
        -------
        pd.DataFrame
            Summary statistics DataFrame
        """
        data = []
        
        for class_id, class_stats in stats.items():
            data.append({
                'Class': class_id,
                'Label': self.class_labels[class_id],
                'Area_km2': class_stats.area_km2,
                'Pixel_Count': class_stats.pixel_count,
                'Percentage': class_stats.percentage,
                'Mean_Probability': class_stats.mean_probability,
                'Max_Probability': class_stats.max_probability,
                'Min_Probability': class_stats.min_probability
            })
        
        df = pd.DataFrame(data)
        return df
    
    def validate_classification(self, 
                               classification: np.ndarray,
                               drainage_area: np.ndarray) -> bool:
        """
        Validate classification results.
        
        Parameters
        ----------
        classification : np.ndarray
            Classification array
        drainage_area : np.ndarray
            Drainage area array
            
        Returns
        -------
        bool
            True if classification is valid
        """
        # Check for valid class range
        unique_classes = np.unique(classification)
        valid_classes = set(range(6))
        
        if not set(unique_classes).issubset(valid_classes):
            logger.error(f"Invalid classes found: {set(unique_classes) - valid_classes}")
            return False
        
        # Check classification consistency with drainage areas
        for class_id in unique_classes:
            if class_id == 0:
                continue  # Skip no-flow class
            
            mask = classification == class_id
            class_drainage = drainage_area[mask]
            
            # Get expected threshold range for this class
            if class_id == 1:
                min_thresh = self.thresholds.very_low
                max_thresh = self.thresholds.low
            elif class_id == 2:
                min_thresh = self.thresholds.low
                max_thresh = self.thresholds.medium
            elif class_id == 3:
                min_thresh = self.thresholds.medium
                max_thresh = self.thresholds.high
            elif class_id == 4:
                min_thresh = self.thresholds.high
                max_thresh = self.thresholds.very_high
            elif class_id == 5:
                min_thresh = self.thresholds.very_high
                max_thresh = np.inf
            
            # Check if most values are in expected range
            in_range = np.sum((class_drainage >= min_thresh) & (class_drainage < max_thresh))
            total = len(class_drainage)
            
            if total > 0 and (in_range / total) < 0.8:  # 80% threshold
                logger.warning(f"Class {class_id} has {(in_range/total)*100:.1f}% values in expected range")
        
        logger.info("Classification validation completed")
        return True
    
    def get_risk_summary(self, stats: Dict[int, ClassificationStats]) -> Dict[str, float]:
        """
        Generate risk summary from classification statistics.
        
        Parameters
        ----------
        stats : Dict[int, ClassificationStats]
            Classification statistics
            
        Returns
        -------
        Dict[str, float]
            Risk summary metrics
        """
        # Calculate total risk area (classes 1-5)
        total_risk_area = sum(stats[i].area_km2 for i in range(1, 6))
        
        # Calculate high risk area (classes 4-5)
        high_risk_area = sum(stats[i].area_km2 for i in range(4, 6))
        
        # Calculate total study area
        total_area = sum(stats[i].area_km2 for i in range(6))
        
        # Calculate weighted risk score
        weights = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        weighted_score = sum(stats[i].area_km2 * weights[i] for i in range(6))
        avg_risk_score = weighted_score / total_area if total_area > 0 else 0
        
        return {
            'total_area_km2': total_area,
            'total_risk_area_km2': total_risk_area,
            'high_risk_area_km2': high_risk_area,
            'risk_area_percentage': (total_risk_area / total_area * 100) if total_area > 0 else 0,
            'high_risk_percentage': (high_risk_area / total_area * 100) if total_area > 0 else 0,
            'average_risk_score': avg_risk_score
        }