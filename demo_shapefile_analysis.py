#!/usr/bin/env python
"""
Demonstration script for shapefile-based EXZECO analysis
"""

import sys
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.append('src')

from exzeco import ExzecoAnalysis, load_config, run_exzeco_with_config
import geopandas as gpd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_shapefile_analysis():
    """Demonstrate the complete shapefile-based analysis workflow"""
    print("🌊 EXZECO Shapefile-based Flood Risk Assessment")
    print("=" * 60)
    
    # Configuration paths
    config_path = 'config/config.yml'
    dem_path = 'data/dem/cache/study_area_dem.tif'
    output_dir = 'data/outputs'
    
    # Check if files exist
    config_file = Path(config_path)
    dem_file = Path(dem_path)
    
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    if not dem_file.exists():
        print(f"❌ DEM file not found: {dem_path}")
        print("   Please run the DEM download process first")
        return
    
    print(f"📁 Using files:")
    print(f"   Config: {config_path}")
    print(f"   DEM: {dem_path}")
    print(f"   Output: {output_dir}")
    
    # Load and display configuration
    print(f"\n📋 Loading configuration...")
    config = load_config(config_path)
    
    print(f"   Shapefile path: {config.shapefile_path}")
    print(f"   Fallback bounds: {config.bounds}")
    print(f"   Noise levels: {config.noise_levels}")
    print(f"   Iterations: {config.iterations}")
    
    # Load and display study area information
    print(f"\n🗺️  Loading study areas...")
    try:
        analyzer = ExzecoAnalysis(config)
        study_areas_gdf, total_bounds = analyzer.load_study_areas(
            config.shapefile_path, 
            config.bounds
        )
        
        print(f"   ✅ Successfully loaded {len(study_areas_gdf)} subcatchment(s)")
        print(f"   📍 Total bounds: {total_bounds}")
        print(f"   🗺️  CRS: {study_areas_gdf.crs}")
        
        # Display subcatchment details
        print(f"\n📊 Subcatchment Details:")
        for idx, row in study_areas_gdf.iterrows():
            name = row.get('NAME_EN', row.get('name', f'feature_{idx}'))
            area_crs_units = row.geometry.area
            
            # Try to convert to km² if in projected CRS
            if hasattr(study_areas_gdf.crs, 'is_projected') and study_areas_gdf.crs.is_projected:
                area_km2 = area_crs_units / 1e6  # Convert m² to km²
                print(f"   • {name}: {area_km2:.1f} km²")
            else:
                print(f"   • {name}: {area_crs_units:.0f} (CRS units)")
        
    except Exception as e:
        print(f"   ❌ Failed to load study areas: {e}")
        return
    
    # Ask user if they want to proceed with full analysis
    print(f"\n🚀 Analysis Setup Complete!")
    print(f"   This will perform EXZECO analysis for {len(config.noise_levels)} noise levels")
    print(f"   with {config.iterations} Monte Carlo iterations each.")
    print(f"   Results will include statistics for:")
    print(f"   • Total domain (entire area)")
    print(f"   • Individual subcatchments ({len(study_areas_gdf)} areas)")
    
    # For demo purposes, let's run a quick test with reduced iterations
    print(f"\n⚡ Running demonstration analysis (reduced iterations)...")
    
    # Create a reduced-iteration config for demo
    demo_config = config
    demo_config.iterations = 10  # Much faster for demo
    demo_config.noise_levels = [0.2, 0.6, 1.0]  # Fewer levels for demo
    
    try:
        demo_analyzer = ExzecoAnalysis(demo_config)
        
        # Load study areas and DEM
        study_areas_gdf, total_bounds = demo_analyzer.load_study_areas(
            demo_config.shapefile_path, 
            demo_config.bounds
        )
        
        # Check if we can load the DEM successfully
        demo_analyzer.load_dem(dem_path, total_bounds)
        
        if demo_analyzer.dem_data.size == 0:
            print("   ⚠️  DEM appears to be empty for the study area bounds.")
            print("   This might be due to coordinate system mismatch.")
            print("   The analysis framework is working, but DEM preprocessing may be needed.")
            return
        
        print(f"   ✅ DEM loaded successfully:")
        print(f"     Shape: {demo_analyzer.dem_data.shape}")
        print(f"     Resolution: {demo_analyzer.resolution:.1f}m")
        
        # Run a single noise level for demonstration
        print(f"\n🔄 Running Monte Carlo simulation for 20cm noise level...")
        prob_map = demo_analyzer.run_monte_carlo(0.2, progress_bar=True)
        
        print(f"   ✅ Monte Carlo completed!")
        print(f"     Probability map shape: {prob_map.shape}")
        print(f"     Max probability: {prob_map.max():.3f}")
        print(f"     Mean probability: {prob_map.mean():.3f}")
        
        # Generate a sample report
        print(f"\n📄 Generating sample statistics...")
        
        # Basic statistics for total domain
        flood_mask = prob_map > 0.5
        pixel_area_km2 = (demo_analyzer.resolution_x * demo_analyzer.resolution_y) / 1e6
        
        total_area = prob_map.size * pixel_area_km2
        flood_area = flood_mask.sum() * pixel_area_km2
        flood_percentage = 100 * flood_mask.sum() / prob_map.size if prob_map.size > 0 else 0
        
        print(f"   📊 Total Domain Results (20cm noise):")
        print(f"     • Total area: {total_area:.2f} km²")
        print(f"     • Flood-prone area: {flood_area:.2f} km²")
        print(f"     • Flood percentage: {flood_percentage:.1f}%")
        
        # Subcatchment statistics
        if len(study_areas_gdf) > 1:
            print(f"\n   📊 Subcatchment Results (20cm noise):")
            for idx, row in study_areas_gdf.iterrows():
                name = row.get('NAME_EN', row.get('name', f'subcatchment_{idx}'))
                
                # Transform geometry to raster CRS if needed
                geom = row.geometry
                if study_areas_gdf.crs != demo_analyzer.crs:
                    geom_gdf = gpd.GeoDataFrame([row], crs=study_areas_gdf.crs)
                    geom_gdf = geom_gdf.to_crs(demo_analyzer.crs)
                    geom = geom_gdf.geometry.iloc[0]
                
                # Mask probability map by subcatchment
                try:
                    masked_prob = demo_analyzer.mask_raster_by_geometry(
                        prob_map, geom, demo_analyzer.transform
                    )
                    
                    subcatch_flood_mask = masked_prob > 0.5
                    subcatch_valid_pixels = (~masked_prob.isnan()).sum() if hasattr(masked_prob, 'isnan') else (masked_prob == masked_prob).sum()
                    subcatch_flood_pixels = (subcatch_flood_mask & ~masked_prob.isnan()).sum() if hasattr(masked_prob, 'isnan') else subcatch_flood_mask.sum()
                    
                    if subcatch_valid_pixels > 0:
                        subcatch_area = subcatch_valid_pixels * pixel_area_km2
                        subcatch_flood_area = subcatch_flood_pixels * pixel_area_km2
                        subcatch_flood_pct = 100 * subcatch_flood_pixels / subcatch_valid_pixels
                        
                        print(f"     • {name}: {subcatch_flood_area:.2f} km² flood area ({subcatch_flood_pct:.1f}% of {subcatch_area:.2f} km²)")
                    else:
                        print(f"     • {name}: No valid data within subcatchment")
                        
                except Exception as e:
                    print(f"     • {name}: Error processing subcatchment: {e}")
        
        print(f"\n🎉 Demonstration completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Demo analysis failed: {e}")
        import traceback
        traceback.print_exc()

def show_full_analysis_example():
    """Show example of how to run the full analysis"""
    print(f"\n" + "=" * 60)
    print(f"📖 Full Analysis Example")
    print(f"=" * 60)
    
    print(f"""
To run the complete EXZECO analysis with all noise levels and full iterations:

1. Using the built-in function:
   ```python
   from src.exzeco import run_exzeco_with_config
   
   analyzer, results, report = run_exzeco_with_config(
       config_path='config/config.yml',
       dem_path='data/dem/cache/study_area_dem.tif',
       output_dir='data/outputs'
   )
   
   # Print the report
   print(report)
   ```

2. Using the class directly:
   ```python
   from src.exzeco import ExzecoAnalysis, load_config
   
   # Load configuration
   config = load_config('config/config.yml')
   
   # Initialize analyzer
   analyzer = ExzecoAnalysis(config)
   
   # Run full analysis
   results = analyzer.run_full_analysis(
       dem_path='data/dem/cache/study_area_dem.tif',
       bounds=config.bounds,
       shapefile_path=config.shapefile_path
   )
   
   # Export results
   analyzer.export_results('data/outputs', format='geotiff')
   analyzer.export_results('data/outputs', format='geojson')
   
   # Generate report
   report = analyzer.generate_report()
   report.to_csv('data/outputs/exzeco_report.csv', index=False)
   ```

3. From command line:
   ```bash
   cd /path/to/exzeco_project
   python src/exzeco.py
   ```

The analysis will automatically:
• Load subcatchments from the shapefile (data/doi/qmp_doi.gpkg)
• Fall back to bounding box if shapefile is not found
• Perform analysis for the entire domain AND individual subcatchments
• Export results as both rasters and vector files
• Generate comprehensive statistics for all areas
""")

if __name__ == "__main__":
    demonstrate_shapefile_analysis()
    show_full_analysis_example()
