#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXZECO Flood Risk Assessment - Complete Analysis Script

This script performs the complete EXZECO workflow for preliminary
flood risk assessment using Monte Carlo simulation on Digital Elevation Models.

Based on the methodology from CEREMA:
https://www.cerema.fr/system/files/documents/2020/07/methode_exzeco_25mai2020.pdf

Author: Tobias Siegfried, hydrosolutions GmbH
Date: 2025
License: MIT

USAGE INSTRUCTIONS:
==================

QUICK START:
-----------
1. Activate the conda environment:
   ```bash
   conda activate exzeco
   ```

2. Navigate to the notebooks directory:
   ```bash
   cd /path/to/exzeco-flood-risk-assessment/notebooks/
   ```

3. Open this script in your preferred IDE:
   - VS Code: `code exzeco_pfra_script.py`
   - Spyder: `spyder exzeco_pfra_script.py`
   - PyCharm: Open file directly

4. Execute step-by-step using cell dividers (# %%)

EXECUTION METHODS:
==================

METHOD 1: Interactive Cell Execution (Recommended)
--------------------------------------------------
- In VS Code: Use "Python Interactive" extension
  * Press Ctrl+Shift+P → "Python: Run Selection/Line in Python Terminal"
  * Or click "Run Cell" above each # %% comment
  * Allows inspection of variables and results between steps

- In Spyder: 
  * Use F9 to run current cell (between # %% markers)
  * Or use "Run → Run cell" from menu
  * View variables in Variable Explorer panel

METHOD 2: Full Script Execution
-------------------------------
Run the complete script at once:
```bash
python exzeco_pfra_script.py
```
Note: This method runs all analysis steps sequentially without interruption.

METHOD 3: Command Line with Python -i (Interactive)
---------------------------------------------------
```bash
python -i exzeco_pfra_script.py
```
Runs script and drops into interactive Python shell for post-analysis exploration.

CONFIGURATION SETUP:
====================

STEP 1: Verify Configuration File
---------------------------------
Ensure ../config/config.yml exists and contains:
```yaml
study_area:
  shapefile_path: "./data/doi/your_study_area.gpkg"  # Path to your shapefile
  bounds: [min_lon, min_lat, max_lon, max_lat]       # Fallback coordinates

analysis:
  noise_levels: [0.2, 0.4, 0.6, 0.8, 1.0]          # DEM noise levels (meters)
  iterations: 10                                     # Monte Carlo iterations
  min_drainage_area: 0.1                           # Minimum drainage area (km²)
  n_jobs: -1                                        # Parallel processing cores
```

STEP 2: Prepare Study Area Data
-------------------------------
Option A - Use Shapefile (Recommended):
  - Place your study area shapefile in data/doi/ directory
  - Supported formats: .shp, .gpkg, .geojson
  - Ensure CRS is properly defined (will auto-convert to EPSG:4326)

Option B - Use Bounding Box:
  - Define coordinates in config.yml under study_area.bounds
  - Format: [min_longitude, min_latitude, max_longitude, max_latitude]
  - Example: [74.3, 42.3, 74.9, 43.2] for a region in Central Asia

TROUBLESHOOTING:
================

Common Issues and Solutions:
---------------------------

1. Module Import Errors:
   Problem: "ModuleNotFoundError: No module named 'exzeco'"
   Solution: Ensure you're in the correct conda environment and src/ is accessible

2. Configuration File Not Found:
   Problem: "FileNotFoundError: config.yml not found"
   Solution: Verify path ../config/config.yml exists relative to notebooks/ directory

3. Shapefile Loading Issues:
   Problem: "Error loading shapefile"
   Solution: Check file path and format, script will fall back to bounding box

4. DEM Download Failures:
   Problem: "Failed to download DEM"
   Solution: Check internet connection, script has multiple fallback sources

5. Memory Issues:
   Problem: "MemoryError during analysis"
   Solution: Reduce iterations or study area size in config.yml

6. Permission Errors:
   Problem: "Permission denied" when saving outputs
   Solution: Ensure write permissions to ../data/outputs/ directory

MONITORING PROGRESS:
===================
- Each cell prints progress messages with ✅ success or ⚠️ warnings
- Key checkpoints include DEM download, analysis completion, and file exports
- Estimated runtime: 5-30 minutes depending on study area size and iterations

UNDERSTANDING OUTPUTS:
=====================
The script generates comprehensive outputs in ../data/outputs/:

Raster Outputs:
- exzeco_[noise_level]_[iterations]_[threshold].tif (flood probability maps)
- drainage_classification.tif (drainage area classification)

Vector Outputs:
- Shapefiles with flood zones and risk classifications

Interactive Outputs:
- exzeco_map.html (interactive web map)
- exzeco_3d.html (3D terrain visualization)
- exzeco_interactive_dashboard.html (complete analysis dashboard)

Reports:
- exzeco_final_report.html (comprehensive analysis report)
- risk_analysis_*.xlsx (statistical summaries)
- exzeco_report_*.csv (detailed results tables)

CUSTOMIZATION OPTIONS:
=====================
Modify analysis parameters by editing ../config/config.yml:

- noise_levels: Adjust DEM uncertainty levels (typically 0.2-1.0 meters)
- iterations: Increase for more robust results (10-100)
- min_drainage_area: Threshold for flow accumulation (0.01-1.0 km²)
- n_jobs: Parallel processing cores (-1 for all available cores)

CELL STRUCTURE:
===============
1. Setup and Initialization - Import libraries and configure plotting
2. Configuration - Load config.yml settings
3. Define Study Area - Load shapefile or use bounding box
4. Download and Process DEM - Download and validate elevation data
5. Run EXZECO Analysis - Core Monte Carlo flood risk assessment
6. Visualization of Results - Basic flood probability visualizations
7. Detect Spatial Features - Endorheic basins and drainage analysis
8. Interactive Map - Create interactive web map
9. 3D Visualization - Generate 3D terrain views
10. Statistical Analysis - Comparison plots and summary statistics
11. Interactive Dashboard - Complete analysis dashboard
12. Detailed Statistical Analysis - Advanced risk metrics
13. Export Results - Save all outputs in multiple formats
14. Create Final Report - Generate HTML summary report
15. Conclusion - Final summary and next steps

REQUIREMENTS:
=============
- Python environment as specified in environment.yml
- All dependencies from requirements.txt
- Access to ../config/config.yml configuration file
- Study area shapefile or bounding box coordinates in config
- Internet connection for DEM download (cached after first run)

OUTPUT DIRECTORY:
================
All outputs are saved to: ../data/outputs/
"""


# %%
# =============================================================================
# ENVIRONMENT ACTIVATION (Optional - for reference only)
# =============================================================================

# Note: If running this script, ensure you're already in the 'exzeco' conda environment
# You can activate it before running the script with:
# conda activate exzeco

# To check if you're in the correct environment:
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Check if we're in a conda environment
if 'CONDA_DEFAULT_ENV' in os.environ:
    print(f"Active conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
else:
    print("No conda environment detected")

# Verify key packages are available
try:
    import numpy, pandas, geopandas, rasterio, matplotlib
    print("✅ Core packages available")
except ImportError as e:
    print(f"❌ Missing packages: {e}")
    print("Please ensure you're in the 'exzeco' conda environment")


# %%
# =============================================================================
# 1. SETUP AND INITIALIZATION
# =============================================================================

# 1.1 Import required libraries
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('../src')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from typing import Tuple, Dict, List
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import rasterio
from shapely.geometry import box

# Import EXZECO modules
from exzeco import ExzecoAnalysis, ExzecoConfig, load_config
from dem_utils import DEMDownloader, StudyArea
from visualization import ExzecoVisualizer

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define output directory (used throughout the script)
output_dir = Path("../data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

print("✅ All modules loaded successfully!")
print(f"📁 Output directory: {output_dir.absolute()}")

# %%
# =============================================================================
# 1.2 CONFIGURE HIGH-RESOLUTION PLOTS
# =============================================================================

# Configure matplotlib for high-resolution plots
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300  # High DPI for crisp plots
mpl.rcParams['savefig.dpi'] = 300  # High DPI for saved figures
mpl.rcParams['figure.figsize'] = [12, 8]  # Default figure size [inches]
mpl.rcParams['font.size'] = 12  # Readable font size
mpl.rcParams['axes.labelsize'] = 12  # Axis label font size
mpl.rcParams['axes.titlesize'] = 14  # Title font size
mpl.rcParams['xtick.labelsize'] = 10  # X-tick font size
mpl.rcParams['ytick.labelsize'] = 10  # Y-tick font size
mpl.rcParams['legend.fontsize'] = 11  # Legend font size

# Configure Plotly
import plotly.io as pio
try:
    # Try to configure kaleido if available
    if hasattr(pio, 'kaleido') and pio.kaleido.scope is not None:
        pio.kaleido.scope.mathjax = None  # Faster rendering
except (AttributeError, TypeError):
    # Skip if kaleido is not available or configured
    pass
pio.renderers.default = "browser"  # Use browser for script execution

print("✅ High-resolution plot configuration applied!")
print(f"   • Figure DPI: {mpl.rcParams['figure.dpi']}")
print(f"   • Save DPI: {mpl.rcParams['savefig.dpi']}")
print(f"   • Default figure size: {mpl.rcParams['figure.figsize']}")
print(f"   • Plotly renderer: {pio.renderers.default}")

# %%
# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# Load configuration from config file
config = load_config('../config/config.yml')

print("Configuration:")
print(f"  Noise levels: {config.noise_levels}")
print(f"  Iterations: {config.iterations}")
print(f"  Min drainage area: {config.min_drainage_area} km²")
print(f"  Parallel jobs: {config.n_jobs}")

# %%
# =============================================================================
# 3. DEFINE STUDY AREA
# =============================================================================

# Construct path to shapefile from config
# The path in config.yml is relative to the project root.
# The notebook is in notebooks/, so we need to go up one level.
shapefile_path = Path('../') / config.shapefile_path

# Load from shapefile or GeoJSON with error handling
try:
    # Check if the shapefile path exists
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found at: {shapefile_path}")
    
    # Try to load the shapefile
    study_gdf = gpd.read_file(shapefile_path)
    
    if study_gdf.empty:
        raise ValueError(f"Shapefile is empty: {shapefile_path}")
    
    print(f"✅ Shapefile loaded successfully: {shapefile_path.name}")
    print(f"   • Features: {len(study_gdf)}")
    print(f"   • Geometry types: {study_gdf.geometry.type.unique()}")
    
    # Ensure the GeoDataFrame is in lat/lon coordinates (EPSG:4326)
    original_crs = study_gdf.crs
    needs_reprojection = study_gdf.crs != 'EPSG:4326'

    if needs_reprojection:
        study_gdf = study_gdf.to_crs('EPSG:4326')
        
        # Save the transformed shapefile to avoid coordinate system issues in analysis
        transformed_shapefile_path = shapefile_path.parent / f"{shapefile_path.stem}_wgs84{shapefile_path.suffix}"
        study_gdf.to_file(transformed_shapefile_path, driver='GPKG')
        print(f"🗺️  Shapefile: {shapefile_path.name} | {original_crs} → EPSG:4326 | Saved as: {transformed_shapefile_path.name}")
    else:
        transformed_shapefile_path = shapefile_path
        print(f"🗺️  Shapefile: {shapefile_path.name} | CRS: {original_crs} ✓")

    # Extract bounds and create study area
    STUDY_BOUNDS = tuple(study_gdf.total_bounds)
    
except (FileNotFoundError, ValueError, Exception) as e:
    print(f"⚠️  Error loading shapefile: {e}")
    print(f"🔄 Falling back to bounding box coordinates from config...")
    
    # Fallback to bounding box from config
    if hasattr(config, 'bounds') and config.bounds:
        STUDY_BOUNDS = tuple(config.bounds)
        print(f"📦 Using bounding box from config: {STUDY_BOUNDS}")
        
        # Create a simple rectangular polygon for the bounding box
        from shapely.geometry import box
        bbox_polygon = box(*STUDY_BOUNDS)
        study_gdf = gpd.GeoDataFrame([1], geometry=[bbox_polygon], crs='EPSG:4326')
        transformed_shapefile_path = None  # No transformed file for bounding box
        
        print(f"✅ Created study area from bounding box")
    else:
        print(f"❌ No fallback bounding box found in config!")
        raise ValueError("Neither shapefile nor bounding box coordinates are available. Please check your config.yml file.")

# Create study area and calculate area
study_area = StudyArea(STUDY_BOUNDS)
area_km2 = study_area.get_area_km2()

print(f"📍 Study Area: {area_km2:.1f} km² | Bounds: {STUDY_BOUNDS[1]:.4f}°N-{STUDY_BOUNDS[3]:.4f}°N, {STUDY_BOUNDS[0]:.4f}°E-{STUDY_BOUNDS[2]:.4f}°E")

# Create comprehensive study area visualization (static + interactive)
import contextlib
import io
import importlib
import visualization
importlib.reload(visualization)
from visualization import create_study_area_visualization

# Suppress verbose output from visualization function
print("📊 Creating study area visualizations...")
with contextlib.redirect_stdout(io.StringIO()):
    visualization_results = create_study_area_visualization(
        study_area=study_area,
        area_km2=area_km2,
        output_dir=output_dir,
        show_static=True,
        show_interactive=True,
        polygon_gdf=study_gdf
    )

print("✅ Study area visualizations created")

# %%
# =============================================================================
# 4. DOWNLOAD AND PROCESS DIGITAL ELEVATION MODEL (DEM)
# =============================================================================

# Reload DEM utils module to get the latest changes
import importlib
import dem_utils
importlib.reload(dem_utils)
from dem_utils import DEMDownloader

# Initialize DEM downloader
dem_downloader = DEMDownloader()

# Get absolute path for cache directory
current_dir = os.getcwd()
cache_dir = Path(current_dir).parent / "data" / "dem" / "cache"

# Generate expected DEM filename to check if it exists
expected_filename = "study_area_dem.tif"  # Default
if shapefile_path is not None:
    shapefile_stem = Path(shapefile_path).stem
    # Apply same logic as in DEM downloader
    if shapefile_stem.endswith('_wgs84'):
        shapefile_stem = shapefile_stem[:-6]
    for suffix in ['_wgs84', '_4326', '_reprojected']:
        if shapefile_stem.endswith(suffix):
            shapefile_stem = shapefile_stem[:shapefile_stem.rfind(suffix)]
            break
    expected_filename = f"{shapefile_stem}_dem.tif"

expected_dem_path = cache_dir / expected_filename
if expected_dem_path.exists():
    status = "📁 Cached"
else:
    status = "🌍 Download"

print(f"🎯 DEM Processing: {expected_filename} | Status: {status}")

# Download DEM using the refactored method with shapefile-based naming
dem_path, dem_stats = dem_downloader.download_dem_with_fallback(
    bounds=STUDY_BOUNDS,
    cache_dir=cache_dir,
    output_filename="study_area_dem.tif",  # Will be overridden if shapefile_path is provided
    product='SRTM1',  # Trying with 30m resolution
    shapefile_path=shapefile_path  # Pass shapefile path for automatic naming
)

# Read DEM data for shape information
with rasterio.open(dem_path) as src:
    dem_data = src.read(1)

print(f"✅ DEM Ready: {dem_data.shape} | {dem_stats['min_elevation']:.0f}-{dem_stats['max_elevation']:.0f}m | {dem_stats['resolution']}m")

# Visualize DEM using the visualization module
# Reload visualization module to get the latest DEMVisualizer class
import importlib
import visualization
importlib.reload(visualization)
from visualization import create_dem_visualization

# Create comprehensive DEM analysis visualization
dem_fig = create_dem_visualization(
    dem_path=dem_path,
    show_plot=True,
    save_individual=True,
    save_path=output_dir / 'dem_analysis.png',
    shapefile_path=transformed_shapefile_path
)

# %%
# =============================================================================
# 5. RUN EXZECO ANALYSIS
# =============================================================================

print("🎯 FINDING APPROPRIATE DRAINAGE AREA THRESHOLD")
print("=" * 55)

# Create temporary analyzer to analyze flow accumulation
temp_config = ExzecoConfig(
    noise_levels=[0.2, 0.4],  
    iterations=5,  
    min_drainage_area=0.001,  
    n_jobs=1  
)

temp_analyzer = ExzecoAnalysis(temp_config)
temp_analyzer.load_dem(dem_path)

# Calculate flow accumulation for original DEM
flow_dir, slopes = temp_analyzer._compute_flow_direction_d8(temp_analyzer.dem_data)
flow_acc = temp_analyzer._compute_flow_accumulation(flow_dir)

# Convert to drainage area
pixel_area_km2 = (temp_analyzer.resolution_x * temp_analyzer.resolution_y) / 1e6
drainage_area = flow_acc * pixel_area_km2

print(f"Drainage area statistics:")
print(f"  Single pixel area: {pixel_area_km2:.6f} km²")
print(f"  Min drainage: {np.min(drainage_area):.6f} km²")
print(f"  Max drainage: {np.max(drainage_area):.6f} km²")
print(f"  90th percentile: {np.percentile(drainage_area, 90):.6f} km²")
print(f"  95th percentile: {np.percentile(drainage_area, 95):.6f} km²")
print(f"  99th percentile: {np.percentile(drainage_area, 99):.6f} km²")

# Test different thresholds to find appropriate one
thresholds_to_test = config.noise_levels  # Use config values
print(f"\n📊 Testing different drainage area thresholds:")
for thresh in thresholds_to_test:
    pixels_above = np.sum(drainage_area >= thresh)
    percent_above = 100 * pixels_above / drainage_area.size
    print(f"  {thresh:0.2f} km²: {pixels_above:,} pixels ({percent_above:.1f}%)")

# Choose threshold that includes 10-30% of study area (typical for EXZECO)
appropriate_threshold = 0.05  # km²
print(f"\n🎯 Selected threshold: {appropriate_threshold} km²")
print(f"   This represents {appropriate_threshold/pixel_area_km2:.0f} pixels minimum")

# Select hand-picked threshold and override automatically determined one (uncomment if needed)
appropriate_threshold = 0.1 # km²

# Configure final EXZECO analysis
final_config = ExzecoConfig(
    noise_levels=config.noise_levels,  # Use config values
    iterations=config.iterations,  # Use config values  
    min_drainage_area=appropriate_threshold,
    n_jobs=config.n_jobs  # Use config values  
)

print(f"\n🚀 Running EXZECO Analysis")
print(f"   Noise levels: {final_config.noise_levels}")
print(f"   Iterations: {final_config.iterations}")  
print(f"   Min drainage area: {final_config.min_drainage_area} km²")
print(f"   Using shapefile: {transformed_shapefile_path}")

# Run complete EXZECO analysis with transformed shapefile
analyzer = ExzecoAnalysis(final_config)

try:
    results = analyzer.run_full_analysis(
        dem_path=dem_path,
        bounds=STUDY_BOUNDS,
        shapefile_path=transformed_shapefile_path  # Use the transformed shapefile
    )
    
    print(f"\n✅ EXZECO ANALYSIS COMPLETED!")
    
    # Generate summary report
    report = analyzer.generate_report()
    print(f"\n📈 ANALYSIS RESULTS:")
    print(report)
    print(f"\n🎉 EXZECO ANALYSIS SUCCESSFUL!")
    
except Exception as e:
    print(f"❌ Error in EXZECO analysis: {e}")
    import traceback
    traceback.print_exc()

# %%
# =============================================================================
# 6. VISUALIZATION OF RESULTS
# =============================================================================

# Reload EXZECO and visualization module to get the fix
import importlib
import exzeco
importlib.reload(exzeco)
from exzeco import ExzecoAnalysis, ExzecoConfig, load_config
importlib.reload(visualization)
from visualization import ExzecoVisualizer

# Create visualizer instance
visualizer = ExzecoVisualizer(results, dem_path)

# Create flood probability visualization with shapefile overlay
fig = visualizer.plot_flood_probability_with_shapefile(
    noise_level='exzeco_200cm',
    shapefile_gdf=study_gdf,
    threshold=0.5,
    figsize=(15, 8),
    show_plot=True,
    save_path=output_dir / 'flood_probability_with_boundaries.png'
)

# %%
# =============================================================================
# 7. DETECT SPATIAL FEATURES
# =============================================================================

importlib.reload(visualization) # just for development
visualizer = ExzecoVisualizer(results, dem_path)

# Perform comprehensive spatial features analysis
spatial_figures = visualizer.plot_spatial_features_analysis(
    analyzer=analyzer,
    results=results,
    figsize_endorheic=(10, 8),
    figsize_drainage=(12, 8),
    shapefile_gdf=study_gdf,
    show_plots=True,
    save_paths={
        'endorheic': output_dir / 'endorheic_basins_analysis.png',
        'drainage': output_dir / 'drainage_classification.png'
    },
    save_tiff=True
)

print(f"✅ Spatial features analysis completed")

# %%
# =============================================================================
# 8. INTERACTIVE VISUALIZATION - MAP
# =============================================================================

# Initialize visualizer
import importlib
import visualization
importlib.reload(visualization)
from visualization import ExzecoVisualizer

visualizer = ExzecoVisualizer(results, dem_path)

# Create interactive map
print("Creating interactive map...")

try:
    # Create map with basic features first
    interactive_map = visualizer.create_interactive_map(
        noise_level='exzeco_200cm',
        include_layers=[]  # Start with no additional layers
    )
    
    # Create outputs directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save map
    interactive_map.save(str(output_dir / 'exzeco_map.html'))
    print(f"✅ Interactive map saved to {output_dir / 'exzeco_map.html'}")
    
except Exception as e:
    print(f"Error creating interactive map: {e}")
    print("Creating a simple fallback map...")
    
    # Create a simple folium map as fallback
    import folium
    
    # Get center coordinates
    center_lat = (STUDY_BOUNDS[1] + STUDY_BOUNDS[3]) / 2
    center_lon = (STUDY_BOUNDS[0] + STUDY_BOUNDS[2]) / 2
    
    # Create simple map
    fallback_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add study area boundary
    folium.Rectangle(
        bounds=[[STUDY_BOUNDS[1], STUDY_BOUNDS[0]], 
                [STUDY_BOUNDS[3], STUDY_BOUNDS[2]]],
        color='red',
        fill=True,
        fillOpacity=0.1,
        popup='Study Area'
    ).add_to(fallback_map)
    
    fallback_map.save(str(output_dir / 'exzeco_map_simple.html'))
    print(f"✅ Simple map saved to {output_dir / 'exzeco_map_simple.html'}")

# %%
# =============================================================================
# 9. INTERACTIVE VISUALIZATION - 3D
# =============================================================================

# Create 3D visualization
print("Creating 3D visualization...")

# Reload EXZECO and visualization module to get the fix
import importlib
import exzeco
importlib.reload(exzeco)
from exzeco import ExzecoAnalysis, ExzecoConfig, load_config
importlib.reload(visualization)
from visualization import ExzecoVisualizer
visualizer = ExzecoVisualizer(results, dem_path)

try:
    fig_3d = visualizer.create_3d_visualization('exzeco_100cm')
    
    # Save 3D visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_3d.write_html(str(output_dir / 'exzeco_3d.html'))
    print(f"✅ 3D visualization saved to {output_dir / 'exzeco_3d.html'}")
    
except Exception as e:
    print(f"Error creating 3D visualization: {e}")

# %%
# =============================================================================
# 10. STATISTICAL ANALYSIS - COMPARISON
# =============================================================================

# Create comparison plots
print("Creating comparison analysis...")

try:
    fig_comparison = visualizer.create_comparison_plot()
    
    # Export the EXZECO Multi-Level Comparison plot
    print("\n📊 Exporting EXZECO Multi-Level Comparison plot...")
    
    # Create outputs directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as HTML (interactive)
    comparison_html_path = output_dir / "exzeco_multi_level_comparison.html"
    fig_comparison.write_html(str(comparison_html_path))
    print(f"✅ Multi-Level Comparison plot saved as HTML: {comparison_html_path}")
    
    # Save as PNG (static image) - requires kaleido
    try:
        comparison_png_path = output_dir / "exzeco_multi_level_comparison.png"
        fig_comparison.write_image(str(comparison_png_path), width=1200, height=800, scale=2)
        print(f"✅ Multi-Level Comparison plot saved as PNG: {comparison_png_path}")
    except Exception as png_error:
        print(f"⚠️  PNG export failed (requires kaleido package): {png_error}")
        print("   HTML version saved successfully - you can open it in a browser")
        
    # Optional: Save as SVG (vector format)
    try:
        comparison_svg_path = output_dir / "exzeco_multi_level_comparison.svg"
        fig_comparison.write_image(str(comparison_svg_path), format='svg', width=1200, height=800)
        print(f"✅ Multi-Level Comparison plot saved as SVG: {comparison_svg_path}")
    except Exception as svg_error:
        print(f"⚠️  SVG export failed: {svg_error}")
        
except Exception as e:
    print(f"Error creating comparison plot: {e}")

# Simple results summary
print("\n" + "="*50)
print("EXZECO ANALYSIS SUMMARY")
print("="*50)

for level, result in results.items():
    prob_map = result['probability_map']
    flooded_cells = np.sum(prob_map > 0.5)  # Cells with >50% flood probability
    total_cells = prob_map.size
    flooded_percentage = (flooded_cells / total_cells) * 100
    
    print(f"\n{level.replace('_', ' ').title()}:")
    print(f"  - Flooded area (>50% probability): {flooded_percentage:.1f}%")
    print(f"  - Max probability: {np.max(prob_map):.3f}")
    print(f"  - Mean probability: {np.mean(prob_map):.3f}")

print(f"\nAnalysis completed for study area: {STUDY_BOUNDS}")
print(f"DEM resolution: {dem_data.shape}")
print("="*50)

# %%
# =============================================================================
# 11. INTERACTIVE DASHBOARD
# =============================================================================

# Create an interactive dashboard for exploring results.
print("Creating interactive dashboard...")

dashboard = visualizer.create_interactive_dashboard()

# Export dashboard as HTML file
from dashboard_exporter import export_dashboard

# Export the dashboard with a single clean command
export_info = export_dashboard(
    dashboard=dashboard,
    output_dir=output_dir,
    filename="exzeco_interactive_dashboard.html",
    study_bounds=STUDY_BOUNDS,
    config=config,
    title="EXZECO Interactive Dashboard"
)

print(f"📊 Dashboard export completed: {export_info['export_method']}")
if export_info['success']:
    print(f"✅ Dashboard saved: {export_info['html_path']}")
    print(f"📂 File size: {export_info['file_size_kb']:.1f} KB")
else:
    print(f"⚠️ Export issues: {export_info.get('error', 'Unknown error')}")

# %%
# =============================================================================
# 12. DETAILED STATISTICAL ANALYSIS
# =============================================================================

# Reload modules for latest updates
import importlib
import visualization
importlib.reload(visualization)
from visualization import ExzecoVisualizer

# Import risk metrics module
import sys
sys.path.append('../src')
import risk_metrics
importlib.reload(risk_metrics)
from risk_metrics import (
    compute_risk_metrics, 
    create_risk_summary_dataframe,
    analyze_risk_evolution,
    check_risk_significance,
    create_risk_visualization,
    export_risk_analysis
)

# Statistical analysis - Individual analysis for each noise level
print("📊 Generating individual statistical analysis for each noise level...")
print(f"   • Configuration: {config.iterations:,} iterations, {config.min_drainage_area} km² drainage threshold")
print("")

# Create a visualizer for the statistical analysis
visualizer = ExzecoVisualizer(results, dem_path=dem_path)

for level in results.keys():
    print(f"Analyzing {level}...")
    # Include config parameter and enable file saving with output directory
    visualizer.plot_statistics(
        noise_level=level, 
        config=config,  # Include config to show iterations and drainage threshold
        save_files=True,  # Enable file saving 
        output_dir=output_dir  # Save to outputs directory
    )

print(f"\n✅ Statistical analysis completed for {len(results)} noise levels")
print(f"📁 Individual analysis files saved to: {output_dir}")

# Comprehensive risk assessment using risk_metrics module
print(f"\n{'='*60}")
print("COMPREHENSIVE RISK ASSESSMENT")
print('='*60)

# Compute detailed risk metrics for all noise levels
print("Computing comprehensive risk metrics...")
risk_data = compute_risk_metrics(results, analyzer)

# Create structured risk summary DataFrame
risk_df = create_risk_summary_dataframe(risk_data)

# Display the risk summary
print("\n📊 Risk Assessment Summary:")
print(risk_df)

# Risk significance analysis
print(f"\n{'='*40}")
print("RISK SIGNIFICANCE ANALYSIS")
print('='*40)

significance = check_risk_significance(risk_data, min_threshold=0.01)

if significance['has_significant_risk']:
    print("✅ SIGNIFICANT FLOOD RISK DETECTED")
    print(f"   • Maximum flood area: {significance['max_flood_area_km2']:.3f} km²")
    print(f"   • Maximum critical risk area: {significance['max_critical_area_km2']:.3f} km²")
    print(f"   • Significant risk levels: {len(significance['significant_levels'])}/{len(results)}")
    print(f"   • Recommendation: {significance['recommendation'].replace('_', ' ').title()}")
else:
    print("⚠️  NO SIGNIFICANT FLOOD RISK DETECTED")
    print(f"   • Maximum flood area: {significance['max_flood_area_km2']:.6f} km²")
    print(f"   • Threshold used: {significance['threshold_used']} km²")
    print(f"   • Recommendation: {significance['recommendation'].replace('_', ' ').title()}")
    print("\n   Possible reasons:")
    for reason in significance['possible_reasons']:
        print(f"     - {reason}")

# Risk evolution analysis (if multiple noise levels)
if len(results) > 1:
    print(f"\n{'='*40}")
    print("RISK EVOLUTION ANALYSIS")
    print('='*40)
    
    evolution = analyze_risk_evolution(risk_data)
    
    if evolution['has_evolution']:
        print("📈 Risk Evolution Trends:")
        for metric, trend in evolution['trends'].items():
            change_value = evolution['changes'][metric.replace('_trend', '_change')]
            print(f"   • {metric.replace('_', ' ').title()}: {trend} (Δ {change_value:+.3f})")
        
        print(f"\n📊 Noise Level Progression:")
        for i, (level, noise_val) in enumerate(zip(evolution['sorted_levels'], evolution['noise_values'])):
            flood_area = evolution['values']['flood_areas'][i]
            print(f"   {i+1}. {level}: {noise_val:.1f}m → {flood_area:.3f} km² flood area")

# Create comprehensive risk visualization
print(f"\n{'='*40}")
print("GENERATING RISK VISUALIZATIONS")
print('='*40)

try:
    # Create comprehensive risk visualization
    risk_fig = create_risk_visualization(
        risk_df=risk_df,
        risk_data=risk_data,
        figsize=(15, 10),
        save_path=output_dir / 'comprehensive_risk_analysis.png'
    )
    plt.show()
    print("✅ Risk visualization created and saved")
    
except Exception as e:
    print(f"⚠️  Error creating risk visualization: {e}")

# Export risk analysis results
print(f"\n{'='*40}")
print("EXPORTING RISK ANALYSIS RESULTS")
print('='*40)

export_result = export_risk_analysis(
    risk_df=risk_df,
    risk_data=risk_data,
    output_dir=output_dir,
    config=config,
    formats=['csv', 'excel', 'json']
)

if export_result['success']:
    print("✅ Risk analysis results exported successfully:")
    for format_type, file_path in export_result['exported_files'].items():
        print(f"   • {format_type.upper()}: {file_path.name}")
else:
    print(f"❌ Export failed: {export_result.get('error', 'Unknown error')}")

print(f"\n✅ Comprehensive statistical analysis completed for {len(results)} noise levels")
print(f"📂 All analysis files saved to: {output_dir}")
print(f"🔧 Analysis used {config.iterations:,} iterations with {config.min_drainage_area} km² drainage threshold")

# %%
# =============================================================================
# 13. EXPORT RESULTS
# =============================================================================

# Export analysis results in various formats with descriptive naming convention.
# New naming format: {file_type}_{noise_level_cm}_{iterations}_{drainage_threshold_km2}.{extension}

# Import the export function from our dedicated export module
import sys
sys.path.append('../src')  # Add src directory to path
from export import export_exzeco_results

# Export all results using the dedicated export function
export_info = export_exzeco_results(
    analyzer=analyzer,
    results=results,
    dem_path=dem_path,
    output_dir=output_dir,
    report=report,
    risk_df=risk_df,
    study_bounds=STUDY_BOUNDS,
    dem_data=dem_data
)

# Display export summary
if export_info['success']:
    print(f"\n🎉 Export completed successfully!")
    print(f"   • Total files exported: {export_info['total_files']}")
    print(f"   • Total size: {export_info['total_size_mb']:.1f} MB")
    print(f"   • Files with descriptive naming: {len(export_info['descriptive_files'])}")
    
    if export_info['errors']:
        print(f"\n⚠️  Warnings/Errors encountered:")
        for error in export_info['errors']:
            print(f"   • {error}")
else:
    print(f"\n❌ Export failed with errors:")
    for error in export_info['errors']:
        print(f"   • {error}")

# %%
# =============================================================================
# 14. CREATE FINAL REPORT
# =============================================================================

# Generate a comprehensive HTML report using the dedicated script.

# Import the final report generation function
import sys
sys.path.append('../src')  # Add src directory to path
from write_final_report import write_final_report

# Generate the final HTML report using the dedicated script
report_result = write_final_report(
    risk_df=risk_df,
    study_bounds=STUDY_BOUNDS,
    area_km2=area_km2,
    dem_stats=dem_stats,
    config=config,
    output_dir=output_dir,
    filename='exzeco_final_report.html'
)

# Display results
if report_result['success']:
    print(f"✅ {report_result['message']}")
    print(f"📁 File size: {report_result['file_size_kb']:.1f} KB")
else:
    print(f"❌ Failed to generate report: {report_result['error']}")

# %%
# =============================================================================
# 15. CONCLUSION
# =============================================================================

# Save notebook state
print("\n" + "="*60)
print("EXZECO Analysis Complete!")
print("="*60)
print(f"\nAll results saved to: {output_dir.absolute()}")
print("\nSummary of Outputs:")
print("1. **Raster Results**: GeoTIFF files with flood probability maps for each noise level")
print("2. **Vector Results**: Shapefiles/GeoJSON with flood zones")
print("3. **Interactive Map**: HTML map with multiple layers")
print("4. **3D Visualization**: Interactive 3D terrain with flood zones")
print("5. **Statistical Reports**: CSV/Excel files with detailed statistics")
print("6. **Visualizations**: PNG/HTML charts and graphs")

print("\nNext Steps:")
print("1. Review the interactive map to identify critical flood zones")
print("2. Examine the statistical reports for risk quantification")
print("3. Use the vector outputs for further GIS analysis")
print("4. Share the HTML report with stakeholders")

print("\nImportant Notes:")
print("- This is a preliminary assessment based on topography only")
print("- Results should be validated with field observations and historical flood data")
print("- For detailed planning, complement with hydraulic modeling")
print("- Consider local infrastructure and drainage systems not captured in the DEM")

print("\nThank you for using the EXZECO Flood Risk Assessment Tool!")