#!/usr/bin/env python
"""
Test script for shapefile-based EXZECO analysis
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from exzeco import ExzecoAnalysis, load_config, run_exzeco_with_config
import geopandas as gpd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_shapefile_loading():
    """Test loading of study areas from shapefile"""
    print("=== Testing Shapefile Loading ===")
    
    # Load configuration
    config = load_config('config/config.yml')
    print(f"Configuration loaded:")
    print(f"  Shapefile path: {config.shapefile_path}")
    print(f"  Fallback bounds: {config.bounds}")
    
    # Initialize analyzer
    analyzer = ExzecoAnalysis(config)
    
    # Test loading study areas
    try:
        study_areas_gdf, total_bounds = analyzer.load_study_areas(
            config.shapefile_path, 
            config.bounds
        )
        
        print(f"\n✅ Successfully loaded study areas:")
        print(f"  Number of subcatchments: {len(study_areas_gdf)}")
        print(f"  Total bounds: {total_bounds}")
        print(f"  CRS: {study_areas_gdf.crs}")
        
        print(f"\nSubcatchment details:")
        for idx, row in study_areas_gdf.iterrows():
            name = row.get('NAME_EN', row.get('name', f'feature_{idx}'))
            area = row.geometry.area
            print(f"  - {name}: Area = {area:.0f} (CRS units)")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load study areas: {e}")
        return False

def test_fallback_to_bounds():
    """Test fallback to bounding box when shapefile is not available"""
    print("\n=== Testing Fallback to Bounds ===")
    
    config = load_config('config/config.yml')
    analyzer = ExzecoAnalysis(config)
    
    # Test with non-existent shapefile
    try:
        study_areas_gdf, total_bounds = analyzer.load_study_areas(
            "non_existent_file.gpkg", 
            config.bounds
        )
        
        print(f"✅ Successfully fell back to bounds:")
        print(f"  Number of features: {len(study_areas_gdf)}")
        print(f"  Total bounds: {total_bounds}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fall back to bounds: {e}")
        return False

def test_error_handling():
    """Test error handling when neither shapefile nor bounds are provided"""
    print("\n=== Testing Error Handling ===")
    
    config = load_config('config/config.yml')
    analyzer = ExzecoAnalysis(config)
    
    # Test with neither shapefile nor bounds
    try:
        study_areas_gdf, total_bounds = analyzer.load_study_areas(
            None,  # No shapefile
            None   # No bounds
        )
        print("❌ Should have raised an error!")
        return False
        
    except ValueError as e:
        print(f"✅ Correctly raised error: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dem_analysis_preview():
    """Test a minimal analysis to ensure the workflow works"""
    print("\n=== Testing DEM Analysis Preview ===")
    
    # Check if DEM file exists
    dem_path = Path('data/dem/cache/study_area_dem.tif')
    if not dem_path.exists():
        print(f"❌ DEM file not found: {dem_path}")
        print("   Please ensure you have run the DEM download first")
        return False
    
    try:
        # Load configuration with reduced iterations for testing
        config = load_config('config/config.yml')
        config.iterations = 5  # Reduced for quick testing
        
        analyzer = ExzecoAnalysis(config)
        
        # Test loading study areas and DEM
        study_areas_gdf, total_bounds = analyzer.load_study_areas(
            config.shapefile_path, 
            config.bounds
        )
        
        analyzer.load_dem(dem_path, total_bounds)
        
        print(f"✅ Successfully loaded DEM:")
        print(f"  DEM shape: {analyzer.dem_data.shape}")
        print(f"  DEM resolution: {analyzer.resolution:.1f}m")
        print(f"  CRS: {analyzer.crs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed DEM loading test: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Shapefile-based EXZECO Analysis")
    print("=" * 50)
    
    tests = [
        test_shapefile_loading,
        test_fallback_to_bounds,
        test_error_handling,
        test_dem_analysis_preview
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print(f"\n{'=' * 50}")
    print(f"📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! The shapefile analysis feature is ready to use.")
        print(f"\nTo run a full analysis, use:")
        print(f"  python src/exzeco.py")
        print(f"or")
        print(f"  from src.exzeco import run_exzeco_with_config")
        print(f"  run_exzeco_with_config('config/config.yml', 'data/dem/cache/study_area_dem.tif', 'data/outputs')")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
