#!/usr/bin/env python
"""
Regenerate the final EXZECO report with corrected DEM resolution
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.dem_utils import DEMDownloader
from src.exzeco import ExzecoConfig

def regenerate_final_report():
    """Regenerate the final HTML report with corrected DEM resolution"""
    
    # Get current directory and paths
    current_dir = Path(os.getcwd())
    cache_dir = current_dir / "data" / "dem" / "cache"
    output_dir = current_dir / "data" / "outputs"
    
    # Initialize DEM downloader with corrected resolution calculation
    dem_downloader = DEMDownloader(cache_dir)
    dem_path = cache_dir / "study_area_dem.tif"
    
    # Get corrected DEM stats
    print("Getting corrected DEM statistics...")
    dem_stats = dem_downloader.get_dem_stats(dem_path)
    print(f"✅ DEM Resolution: {dem_stats['resolution']:.1f} m (corrected)")
    
    # Study area parameters (from the original analysis)
    STUDY_BOUNDS = (74.3, 42.3, 74.9, 43.2)
    area_km2 = 9113.00
    
    # Load the risk analysis results from the existing CSV
    risk_df = pd.read_csv(output_dir / "risk_summary.csv")
    
    # Configuration (from the original analysis)
    config = ExzecoConfig()
    
    # Generate the corrected HTML report
    html_report = f"""<!DOCTYPE html>
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
            <li><strong>Location:</strong> {STUDY_BOUNDS}</li>
            <li><strong>Area:</strong> {area_km2:.2f} km&sup2;</li>
            <li><strong>DEM Resolution:</strong> {dem_stats['resolution']:.1f} m</li>
            <li><strong>Elevation Range:</strong> {dem_stats['min_elevation']:.0f} - {dem_stats['max_elevation']:.0f} m</li>
        </ul>
    </div>

    <div class="summary-box">
        <h2>Analysis Parameters</h2>
        <ul>
            <li><strong>Noise Levels:</strong> {', '.join([f'{x}m' for x in config.noise_levels])}</li>
            <li><strong>Monte Carlo Iterations:</strong> {config.iterations}</li>
            <li><strong>Minimum Drainage Area:</strong> {config.min_drainage_area} km&sup2;</li>
            <li><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</li>
        </ul>
    </div>

    <h2>Risk Assessment Results</h2>
    {risk_df.to_html(index=False, classes='risk-table')}

    <div class="summary-box">
        <h2>Key Findings</h2>
        <ul>
            <li class="risk-high">High Risk Area (>60% probability): {risk_df.iloc[-1]['High Risk (km²)']:.2f} km&sup2;</li>
            <li class="risk-medium">Medium Risk Area (40-60% probability): {risk_df.iloc[-1]['Medium Risk (km²)']:.2f} km&sup2;</li>
            <li class="risk-low">Low Risk Area (20-40% probability): {risk_df.iloc[-1]['Low Risk (km²)']:.2f} km&sup2;</li>
            <li><strong>Total Flood-Prone Area:</strong> {risk_df.iloc[-1]['Total Flood Area (km²)']:.2f} km&sup2;</li>
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
    <p><em>Report generated using EXZECO methodology (CEREMA) - {datetime.now()}</em></p>
</body>
</html>"""

    # Save the corrected report
    report_path = output_dir / 'exzeco_final_report_corrected.html'
    
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    print(f"✅ Corrected final HTML report saved to {report_path}")
    
    # Also update the original file
    original_report_path = output_dir / 'exzeco_final_report.html'
    with open(original_report_path, 'w') as f:
        f.write(html_report)
    
    print(f"✅ Original report updated: {original_report_path}")
    
    return report_path, dem_stats

if __name__ == "__main__":
    report_path, stats = regenerate_final_report()
    print(f"\nSummary:")
    print(f"- DEM Resolution corrected from 0.000278 m to {stats['resolution']:.1f} m")
    print(f"- Report saved to: {report_path}")
