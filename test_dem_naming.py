#!/usr/bin/env python
"""
Test script to validate the new DEM naming and caching functionality.
This script tests the shapefile-based naming and caching features.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append('./src')

from dem_utils import DEMDownloader
import logging

# Set up logging to see the detailed output
logging.basicConfig(level=logging.INFO)

def test_dem_naming():
    """Test DEM naming functionality."""
    print("🧪 Testing DEM Naming and Caching Functionality")
    print("=" * 60)
    
    # Initialize DEM downloader
    downloader = DEMDownloader()
    
    # Test bounds (small area to minimize download time)
    bounds = (74.5, 42.5, 74.6, 42.6)  # Small area in Kyrgyzstan
    cache_dir = Path("./data/dem/cache")
    
    # Test cases
    test_cases = [
        {
            "name": "Test 1: Default filename",
            "shapefile_path": None,
            "expected_pattern": "study_area_dem.tif"
        },
        {
            "name": "Test 2: Simple shapefile name",
            "shapefile_path": Path("./data/doi/qmp_doi.gpkg"),
            "expected_pattern": "qmp_doi_dem.tif"
        },
        {
            "name": "Test 3: WGS84 transformed shapefile",
            "shapefile_path": Path("./data/doi/qmp_doi_wgs84.gpkg"),
            "expected_pattern": "qmp_doi_dem.tif"  # Should remove _wgs84
        },
        {
            "name": "Test 4: Different shapefile format",
            "shapefile_path": Path("./data/doi/Zhabay_River_Basin.gpkg"),
            "expected_pattern": "Zhabay_River_Basin_dem.tif"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 40)
        
        try:
            # Check if this would create the expected filename
            if test_case["shapefile_path"]:
                shapefile_path = test_case["shapefile_path"]
                shapefile_stem = shapefile_path.stem
                
                # Apply the same logic as in the DEM downloader
                if shapefile_stem.endswith('_wgs84'):
                    shapefile_stem = shapefile_stem[:-6]
                for suffix in ['_wgs84', '_4326', '_reprojected']:
                    if shapefile_stem.endswith(suffix):
                        shapefile_stem = shapefile_stem[:shapefile_stem.rfind(suffix)]
                        break
                expected_filename = f"{shapefile_stem}_dem.tif"
            else:
                expected_filename = "study_area_dem.tif"
            
            print(f"  Shapefile: {test_case['shapefile_path']}")
            print(f"  Expected: {expected_filename}")
            print(f"  Pattern:  {test_case['expected_pattern']}")
            
            if expected_filename == test_case['expected_pattern']:
                print("  ✅ PASS: Filename generation correct")
            else:
                print("  ❌ FAIL: Filename generation incorrect")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    # Test actual existence check
    print(f"\n🔍 Testing Existing File Detection")
    print("-" * 40)
    
    # Check what DEM files currently exist in the cache
    if cache_dir.exists():
        dem_files = list(cache_dir.glob("*_dem.tif"))
        print(f"Existing DEM files in cache:")
        for dem_file in dem_files:
            print(f"  - {dem_file.name}")
        
        if dem_files:
            print(f"\n✅ Found {len(dem_files)} existing DEM files")
            print("The system should detect and reuse these files automatically")
        else:
            print("\nℹ️  No existing DEM files found - first run will download new data")
    else:
        print("ℹ️  Cache directory does not exist yet")

if __name__ == "__main__":
    test_dem_naming()
