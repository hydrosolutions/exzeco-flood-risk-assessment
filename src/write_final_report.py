"""
EXZECO Final Report Generator

This module generates a comprehensive HTML report for EXZECO flood risk assessment results.
The report includes study area information, analysis parameters, risk assessment results,
and recommendations.

Author: Tobias Siegfried, hydrosolutions GmbH
Date: 2025
License: MIT
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np


def generate_html_report(
    risk_df: pd.DataFrame,
    study_bounds: Tuple[float, float, float, float],
    area_km2: float,
    dem_stats: Dict[str, Any],
    config: Any,
    output_path: Path
) -> bool:
    """
    Generate a comprehensive HTML report for EXZECO flood risk assessment.
    
    Args:
        risk_df: DataFrame containing risk assessment results
        study_bounds: Tuple of (min_lon, min_lat, max_lon, max_lat)
        area_km2: Study area in square kilometers
        dem_stats: Dictionary with DEM statistics (min_elevation, max_elevation, resolution)
        config: EXZECO configuration object with analysis parameters
        output_path: Path where the HTML report should be saved
        
    Returns:
        bool: True if report was generated successfully, False otherwise
    """
    try:
        # Round the risk_df to 2 decimal places for professional presentation
        risk_df_rounded = risk_df.round(2)
        
        # Round coordinates and DEM stats for consistent formatting
        study_bounds_rounded = [round(coord, 2) for coord in study_bounds]
        dem_min_elev = round(dem_stats['min_elevation'], 2)
        dem_max_elev = round(dem_stats['max_elevation'], 2)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EXZECO Flood Risk Assessment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .risk-high {{ color: #e74c3c; font-weight: bold; }}
        .risk-medium {{ color: #f39c12; font-weight: bold; }}
        .risk-low {{ color: #27ae60; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>EXZECO Flood Risk Assessment Report</h1>
    
    <div class="summary-box">
        <h2>Study Area Summary</h2>
        <ul>
            <li><strong>Location:</strong> {study_bounds_rounded[1]:.2f}&deg;N-{study_bounds_rounded[3]:.2f}&deg;N, {study_bounds_rounded[0]:.2f}&deg;E-{study_bounds_rounded[2]:.2f}&deg;E</li>
            <li><strong>Area:</strong> {area_km2:.2f} km&#178;</li>
            <li><strong>DEM Resolution:</strong> {dem_stats['resolution']} m</li>
            <li><strong>Elevation Range:</strong> {dem_min_elev:.2f} - {dem_max_elev:.2f} m</li>
        </ul>
    </div>
    
    <div class="summary-box">
        <h2>Analysis Parameters</h2>
        <ul>
            <li><strong>Noise Levels:</strong> {', '.join([f'{x}m' for x in config.noise_levels])}</li>
            <li><strong>Monte Carlo Iterations:</strong> {config.iterations}</li>
            <li><strong>Minimum Drainage Area:</strong> {config.min_drainage_area} km&#178;</li>
            <li><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</li>
        </ul>
    </div>
    
    <h2>Risk Assessment Results</h2>
    {risk_df_rounded.to_html(index=False, classes='risk-table')}
    
    <div class="summary-box">
        <h2>Key Findings</h2>
        <ul>
            <li class="risk-high">High Risk Area (>60% probability): {risk_df_rounded['High Risk (km²)'].iloc[-1]:.2f} km&sup2;</li>
            <li class="risk-medium">Medium Risk Area (40-60% probability): {risk_df_rounded['Medium Risk (km²)'].iloc[-1]:.2f} km&sup2;</li>
            <li class="risk-low">Low Risk Area (20-40% probability): {risk_df_rounded['Low Risk (km²)'].iloc[-1]:.2f} km&sup2;</li>
            <li><strong>Total Flood-Prone Area:</strong> {risk_df_rounded['Total Flood Area (km²)'].iloc[-1]:.2f} km&sup2;</li>
        </ul>
    </div>
    
    <h2>Recommendations</h2>
    <ol>
        <li>Areas identified as high risk should be prioritized for detailed hydraulic modeling</li>
        <li>Consider implementing flood mitigation measures in zones with >60% flood probability</li>
        <li>Regular monitoring of drainage patterns in medium-risk areas is recommended</li>
        <li>Update the assessment periodically with higher resolution DEM data when available</li>
    </ol>
    
    <hr>
    <p><em>Report generated using EXZECO methodology (CEREMA) - {pd.Timestamp.now()}</em></p>
</body>
</html>
"""
        
        # Save HTML report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return True
        
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        return False


def write_final_report(
    risk_df: pd.DataFrame,
    study_bounds: Tuple[float, float, float, float],
    area_km2: float,
    dem_stats: Dict[str, Any],
    config: Any,
    output_dir: Path,
    filename: str = 'exzeco_final_report.html'
) -> Dict[str, Any]:
    """
    Write the final EXZECO HTML report to the specified directory.
    
    Args:
        risk_df: DataFrame containing risk assessment results
        study_bounds: Tuple of (min_lon, min_lat, max_lon, max_lat)
        area_km2: Study area in square kilometers
        dem_stats: Dictionary with DEM statistics
        config: EXZECO configuration object
        output_dir: Directory where the report should be saved
        filename: Name of the output HTML file
        
    Returns:
        Dict containing success status and report path information
    """
    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate full report path
        report_path = output_dir / filename
        
        # Generate the HTML report
        success = generate_html_report(
            risk_df=risk_df,
            study_bounds=study_bounds,
            area_km2=area_km2,
            dem_stats=dem_stats,
            config=config,
            output_path=report_path
        )
        
        if success:
            file_size_kb = report_path.stat().st_size / 1024
            return {
                'success': True,
                'report_path': report_path,
                'file_size_kb': file_size_kb,
                'message': f"Final HTML report saved to {report_path}"
            }
        else:
            return {
                'success': False,
                'error': 'Failed to generate HTML content',
                'report_path': None
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Error writing final report: {e}",
            'report_path': None
        }


if __name__ == "__main__":
    print("EXZECO Final Report Generator")
    print("This module is designed to be imported and used within the EXZECO workflow.")
