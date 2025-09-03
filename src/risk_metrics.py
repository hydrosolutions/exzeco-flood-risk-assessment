#!/usr/bin/env python
"""
Risk Metrics Analysis Module for EXZECO Results
===============================================

This module provides comprehensive risk metrics computation and analysis tools
for EXZECO flood risk assessment results.

Key Functions:
- compute_risk_metrics: Calculate comprehensive risk statistics for each noise level
- create_risk_summary_dataframe: Generate structured risk summary DataFrame
- analyze_risk_evolution: Compare risk metrics across different noise levels
- export_risk_analysis: Save risk analysis results in multiple formats

Usage:
    from risk_metrics import compute_risk_metrics, create_risk_summary_dataframe
    
    # Compute risk metrics for all noise levels
    risk_data = compute_risk_metrics(results, analyzer)
    
    # Create summary DataFrame
    risk_df = create_risk_summary_dataframe(risk_data)

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def compute_risk_metrics(results: Dict, analyzer: Any) -> Dict:
    """
    Compute comprehensive risk metrics for all noise levels.
    
    Parameters
    ----------
    results : Dict
        EXZECO results dictionary with noise levels as keys
    analyzer : ExzecoAnalysis
        EXZECO analyzer instance with configuration and metadata
        
    Returns
    -------
    Dict
        Dictionary containing risk metrics for each noise level
    """
    risk_data = {}
    pixel_area_km2 = (analyzer.resolution ** 2) / 1e6
    
    for level, data in results.items():
        prob_map = data['probability_map']
        
        # Define risk thresholds
        very_high_risk = prob_map > 0.8
        high_risk = (prob_map > 0.6) & (prob_map <= 0.8)
        medium_risk = (prob_map > 0.4) & (prob_map <= 0.6)
        low_risk = (prob_map > 0.2) & (prob_map <= 0.4)
        very_low_risk = (prob_map > 0.01) & (prob_map <= 0.2)
        
        # Calculate pixel counts
        very_high_count = np.sum(very_high_risk)
        high_count = np.sum(high_risk)
        medium_count = np.sum(medium_risk)
        low_count = np.sum(low_risk)
        very_low_count = np.sum(very_low_risk)
        flood_count = np.sum(prob_map > 0.5)  # Standard flood threshold
        
        # Calculate areas in km²
        very_high_area = very_high_count * pixel_area_km2
        high_area = high_count * pixel_area_km2
        medium_area = medium_count * pixel_area_km2
        low_area = low_count * pixel_area_km2
        very_low_area = very_low_count * pixel_area_km2
        total_flood_area = flood_count * pixel_area_km2
        
        # Calculate percentages
        total_pixels = prob_map.size
        very_high_pct = (very_high_count / total_pixels) * 100
        high_pct = (high_count / total_pixels) * 100
        medium_pct = (medium_count / total_pixels) * 100
        low_pct = (low_count / total_pixels) * 100
        very_low_pct = (very_low_count / total_pixels) * 100
        flood_pct = (flood_count / total_pixels) * 100
        
        # Statistical measures
        max_prob = np.max(prob_map)
        mean_prob = np.mean(prob_map)
        median_prob = np.median(prob_map)
        std_prob = np.std(prob_map)
        
        # Additional risk metrics
        risk_pixels_total = very_high_count + high_count + medium_count + low_count + very_low_count
        high_and_very_high_area = very_high_area + high_area
        high_and_very_high_pct = (very_high_count + high_count) / total_pixels * 100
        
        risk_data[level] = {
            # Pixel counts
            'total_pixels': total_pixels,
            'very_high_count': very_high_count,
            'high_count': high_count,
            'medium_count': medium_count,
            'low_count': low_count,
            'very_low_count': very_low_count,
            'flood_count': flood_count,
            'risk_pixels_total': risk_pixels_total,
            
            # Areas in km²
            'very_high_area_km2': very_high_area,
            'high_area_km2': high_area,
            'medium_area_km2': medium_area,
            'low_area_km2': low_area,
            'very_low_area_km2': very_low_area,
            'total_flood_area_km2': total_flood_area,
            'high_and_very_high_area_km2': high_and_very_high_area,
            
            # Percentages
            'very_high_pct': very_high_pct,
            'high_pct': high_pct,
            'medium_pct': medium_pct,
            'low_pct': low_pct,
            'very_low_pct': very_low_pct,
            'flood_pct': flood_pct,
            'high_and_very_high_pct': high_and_very_high_pct,
            
            # Statistical measures
            'max_probability': max_prob,
            'mean_probability': mean_prob,
            'median_probability': median_prob,
            'std_probability': std_prob,
            
            # Metadata
            'pixel_area_km2': pixel_area_km2,
            'noise_level': level
        }
    
    logger.info(f"Computed risk metrics for {len(risk_data)} noise levels")
    return risk_data


def create_risk_summary_dataframe(risk_data: Dict) -> pd.DataFrame:
    """
    Create a structured DataFrame from risk metrics data.
    
    Parameters
    ----------
    risk_data : Dict
        Risk metrics data from compute_risk_metrics()
        
    Returns
    -------
    pd.DataFrame
        Structured DataFrame with risk metrics
    """
    summary_data = []
    
    for level, metrics in risk_data.items():
        summary_data.append({
            'Noise Level': level,
            'Very High Risk (km²)': metrics['very_high_area_km2'],
            'High Risk (km²)': metrics['high_area_km2'],
            'Medium Risk (km²)': metrics['medium_area_km2'],
            'Low Risk (km²)': metrics['low_area_km2'],
            'Very Low Risk (km²)': metrics['very_low_area_km2'],
            'Total Flood Area (km²)': metrics['total_flood_area_km2'],
            'Critical Risk Area (km²)': metrics['high_and_very_high_area_km2'],
            'Total Risk Pixels': metrics['risk_pixels_total'],
            'Flood Coverage (%)': metrics['flood_pct'],
            'Critical Risk Coverage (%)': metrics['high_and_very_high_pct'],
            'Max Probability': metrics['max_probability'],
            'Mean Probability': metrics['mean_probability'],
            'Median Probability': metrics['median_probability'],
            'Std Probability': metrics['std_probability']
        })
    
    risk_df = pd.DataFrame(summary_data)
    logger.info(f"Created risk summary DataFrame with {len(risk_df)} rows")
    return risk_df


def analyze_risk_evolution(risk_data: Dict) -> Dict:
    """
    Analyze how risk metrics evolve across different noise levels.
    
    Parameters
    ----------
    risk_data : Dict
        Risk metrics data from compute_risk_metrics()
        
    Returns
    -------
    Dict
        Analysis of risk evolution trends
    """
    if len(risk_data) < 2:
        logger.warning("Need at least 2 noise levels for evolution analysis")
        return {'has_evolution': False, 'message': 'Insufficient data for evolution analysis'}
    
    # Extract noise level values for sorting
    noise_levels = []
    for level in risk_data.keys():
        try:
            # Extract numeric value from level name (e.g., 'exzeco_100cm' -> 1.0)
            if 'cm' in level:
                cm_value = float(level.split('_')[-1].replace('cm', ''))
                noise_levels.append((level, cm_value / 100))  # Convert cm to m
            else:
                noise_levels.append((level, 0))
        except (ValueError, IndexError):
            noise_levels.append((level, 0))
    
    # Sort by noise level
    noise_levels.sort(key=lambda x: x[1])
    sorted_levels = [x[0] for x in noise_levels]
    
    # Calculate trends
    flood_areas = [risk_data[level]['total_flood_area_km2'] for level in sorted_levels]
    critical_areas = [risk_data[level]['high_and_very_high_area_km2'] for level in sorted_levels]
    max_probs = [risk_data[level]['max_probability'] for level in sorted_levels]
    mean_probs = [risk_data[level]['mean_probability'] for level in sorted_levels]
    
    evolution_analysis = {
        'has_evolution': True,
        'sorted_levels': sorted_levels,
        'noise_values': [x[1] for x in noise_levels],
        'trends': {
            'flood_area_trend': 'increasing' if flood_areas[-1] > flood_areas[0] else 'decreasing',
            'critical_area_trend': 'increasing' if critical_areas[-1] > critical_areas[0] else 'decreasing',
            'max_prob_trend': 'increasing' if max_probs[-1] > max_probs[0] else 'decreasing',
            'mean_prob_trend': 'increasing' if mean_probs[-1] > mean_probs[0] else 'decreasing'
        },
        'changes': {
            'flood_area_change': flood_areas[-1] - flood_areas[0],
            'critical_area_change': critical_areas[-1] - critical_areas[0],
            'max_prob_change': max_probs[-1] - max_probs[0],
            'mean_prob_change': mean_probs[-1] - mean_probs[0]
        },
        'values': {
            'flood_areas': flood_areas,
            'critical_areas': critical_areas,
            'max_probabilities': max_probs,
            'mean_probabilities': mean_probs
        }
    }
    
    logger.info("Completed risk evolution analysis")
    return evolution_analysis


def check_risk_significance(risk_data: Dict, min_threshold: float = 0.01) -> Dict:
    """
    Check if the analysis detected significant flood risk.
    
    Parameters
    ----------
    risk_data : Dict
        Risk metrics data from compute_risk_metrics()
    min_threshold : float
        Minimum area threshold (km²) to consider risk significant
        
    Returns
    -------
    Dict
        Risk significance analysis
    """
    has_significant_risk = False
    max_flood_area = 0
    max_critical_area = 0
    significant_levels = []
    
    for level, metrics in risk_data.items():
        flood_area = metrics['total_flood_area_km2']
        critical_area = metrics['high_and_very_high_area_km2']
        
        max_flood_area = max(max_flood_area, flood_area)
        max_critical_area = max(max_critical_area, critical_area)
        
        if flood_area >= min_threshold or critical_area >= min_threshold:
            has_significant_risk = True
            significant_levels.append(level)
    
    significance = {
        'has_significant_risk': has_significant_risk,
        'max_flood_area_km2': max_flood_area,
        'max_critical_area_km2': max_critical_area,
        'significant_levels': significant_levels,
        'threshold_used': min_threshold,
        'recommendation': 'detailed_analysis' if has_significant_risk else 'low_priority'
    }
    
    if not has_significant_risk:
        significance['possible_reasons'] = [
            "The study area has low flood susceptibility",
            "The noise levels tested were too low",
            "The DEM resolution or quality needs improvement",
            "The EXZECO parameters need adjustment"
        ]
    
    logger.info(f"Risk significance analysis: {'Significant' if has_significant_risk else 'No significant'} risk detected")
    return significance


def create_risk_visualization(risk_df: pd.DataFrame, 
                            risk_data: Dict,
                            figsize: Tuple[int, int] = (15, 10),
                            save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create comprehensive risk visualization plots.
    
    Parameters
    ----------
    risk_df : pd.DataFrame
        Risk summary DataFrame
    risk_data : Dict
        Detailed risk metrics data
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Path, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Generated matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Check if we have meaningful risk data
    has_risk_data = risk_df[['Very High Risk (km²)', 'High Risk (km²)', 
                            'Medium Risk (km²)', 'Low Risk (km²)']].sum().sum() > 0
    
    if has_risk_data:
        # Risk area by category (top-left)
        ax = axes[0, 0]
        x = range(len(risk_df))
        width = 0.15
        
        risk_categories = ['Very High Risk (km²)', 'High Risk (km²)', 
                          'Medium Risk (km²)', 'Low Risk (km²)']
        colors = ['#8B0000', '#FF4500', '#FFD700', '#90EE90']
        
        for i, (category, color) in enumerate(zip(risk_categories, colors)):
            offset = (i - 1.5) * width
            values = risk_df[category].values
            ax.bar([xi + offset for xi in x], values, 
                  width, label=category.replace(' (km²)', ''), color=color)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Area (km²)')
        ax.set_title('Flood Risk Area by Category and Noise Level')
        ax.set_xticks(x)
        ax.set_xticklabels([level.replace('exzeco_', '') for level in risk_df['Noise Level']])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Probability evolution (top-right)
        ax = axes[0, 1]
        noise_levels = [level.replace('exzeco_', '').replace('cm', '') for level in risk_df['Noise Level']]
        ax.plot(noise_levels, risk_df['Max Probability'], 'ro-', label='Max Probability', linewidth=2)
        ax.plot(noise_levels, risk_df['Mean Probability'], 'bo-', label='Mean Probability', linewidth=2)
        ax.set_xlabel('Noise Level (cm)')
        ax.set_ylabel('Probability')
        ax.set_title('Probability Evolution by Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coverage percentage (bottom-left)
        ax = axes[1, 0]
        ax.bar(noise_levels, risk_df['Flood Coverage (%)'], alpha=0.7, color='blue', label='Flood Coverage')
        ax.bar(noise_levels, risk_df['Critical Risk Coverage (%)'], alpha=0.7, color='red', label='Critical Risk')
        ax.set_xlabel('Noise Level (cm)')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('Area Coverage by Risk Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Total flood area trend (bottom-right)
        ax = axes[1, 1]
        ax.plot(noise_levels, risk_df['Total Flood Area (km²)'], 'go-', linewidth=3, markersize=8)
        ax.fill_between(noise_levels, risk_df['Total Flood Area (km²)'], alpha=0.3, color='green')
        ax.set_xlabel('Noise Level (cm)')
        ax.set_ylabel('Flood Area (km²)')
        ax.set_title('Total Flood Area Evolution')
        ax.grid(True, alpha=0.3)
        
    else:
        # No significant risk detected - show probability summary
        for i, ax in enumerate(axes.flat):
            ax.clear()
            
        ax = axes[0, 0]
        noise_levels = [level.replace('exzeco_', '').replace('cm', '') for level in risk_df['Noise Level']]
        max_probs = risk_df['Max Probability'].values
        mean_probs = risk_df['Mean Probability'].values
        
        x = range(len(noise_levels))
        ax.bar([xi - 0.2 for xi in x], max_probs, 0.4, label='Max Probability', color='red', alpha=0.7)
        ax.bar([xi + 0.2 for xi in x], mean_probs, 0.4, label='Mean Probability', color='blue', alpha=0.7)
        
        ax.set_xlabel('Noise Level (cm)')
        ax.set_ylabel('Probability')
        ax.set_title('Flood Probability by Noise Level')
        ax.set_xticks(x)
        ax.set_xticklabels(noise_levels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add warning text to other subplots
        for ax in axes.flat[1:]:
            ax.text(0.5, 0.5, '⚠️ No Significant\nFlood Risk Detected', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                   fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('EXZECO Risk Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Risk visualization saved to {save_path}")
    
    return fig


def export_risk_analysis(risk_df: pd.DataFrame,
                        risk_data: Dict,
                        output_dir: Path,
                        config: Any,
                        formats: List[str] = ['csv', 'excel']) -> Dict:
    """
    Export risk analysis results in multiple formats.
    
    Parameters
    ----------
    risk_df : pd.DataFrame
        Risk summary DataFrame
    risk_data : Dict
        Detailed risk metrics data
    output_dir : Path
        Output directory for saved files
    config : ExzecoConfig
        Configuration object for naming
    formats : List[str]
        Export formats ('csv', 'excel', 'json')
        
    Returns
    -------
    Dict
        Export status and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive base name
    drainage_threshold_str = str(config.min_drainage_area).replace('.', 'p')
    base_name = f"risk_analysis_{config.iterations}_{drainage_threshold_str}km2"
    
    exported_files = {}
    
    try:
        # Export summary DataFrame
        if 'csv' in formats:
            csv_path = output_dir / f"{base_name}.csv"
            risk_df.to_csv(csv_path, index=False)
            exported_files['csv'] = csv_path
            logger.info(f"Risk summary exported to CSV: {csv_path}")
        
        if 'excel' in formats:
            excel_path = output_dir / f"{base_name}.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                risk_df.to_excel(writer, sheet_name='Risk Summary', index=False)
                
                # Add detailed metrics sheet
                detailed_data = []
                for level, metrics in risk_data.items():
                    detailed_data.append(metrics)
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
            
            exported_files['excel'] = excel_path
            logger.info(f"Risk analysis exported to Excel: {excel_path}")
        
        if 'json' in formats:
            import json
            json_path = output_dir / f"{base_name}.json"
            export_data = {
                'summary': risk_df.to_dict(orient='records'),
                'detailed_metrics': risk_data,
                'metadata': {
                    'iterations': config.iterations,
                    'min_drainage_area': config.min_drainage_area,
                    'export_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            exported_files['json'] = json_path
            logger.info(f"Risk analysis exported to JSON: {json_path}")
        
        return {
            'success': True,
            'exported_files': exported_files,
            'base_name': base_name
        }
        
    except Exception as e:
        logger.error(f"Error exporting risk analysis: {e}")
        return {
            'success': False,
            'error': str(e),
            'exported_files': exported_files
        }
