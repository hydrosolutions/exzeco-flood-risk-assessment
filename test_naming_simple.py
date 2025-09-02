#!/usr/bin/env python
"""
Simple test to demonstrate DEM naming logic without dependencies.
"""

from pathlib import Path

def test_dem_naming_logic():
    """Test DEM naming functionality without imports."""
    print("🧪 Testing DEM Naming Logic")
    print("=" * 40)
    
    # Test cases matching your actual data
    test_cases = [
        {
            "name": "qmp_doi.gpkg",
            "expected": "qmp_doi_dem.tif"
        },
        {
            "name": "qmp_doi_wgs84.gpkg", 
            "expected": "qmp_doi_dem.tif"  # Should remove _wgs84
        },
        {
            "name": "Zhabay_River_Basin.gpkg",
            "expected": "Zhabay_River_Basin_dem.tif"
        },
        {
            "name": "test_file_4326.shp",
            "expected": "test_file_dem.tif"  # Should remove _4326
        },
        {
            "name": "area_reprojected.gpkg",
            "expected": "area_dem.tif"  # Should remove _reprojected
        }
    ]
    
    for test in test_cases:
        shapefile_path = Path(test["name"])
        shapefile_stem = shapefile_path.stem
        
        # Apply the same logic as in the DEM downloader
        if shapefile_stem.endswith('_wgs84'):
            shapefile_stem = shapefile_stem[:-6]
        for suffix in ['_wgs84', '_4326', '_reprojected']:
            if shapefile_stem.endswith(suffix):
                shapefile_stem = shapefile_stem[:shapefile_stem.rfind(suffix)]
                break
        
        result = f"{shapefile_stem}_dem.tif"
        
        print(f"Input:    {test['name']}")
        print(f"Expected: {test['expected']}")
        print(f"Result:   {result}")
        
        if result == test['expected']:
            print("✅ PASS\n")
        else:
            print("❌ FAIL\n")
    
    # Show current cache contents
    cache_dir = Path("./data/dem/cache")
    print("Current DEM Cache Contents:")
    print("-" * 30)
    if cache_dir.exists():
        dem_files = list(cache_dir.glob("*.tif"))
        for dem_file in dem_files:
            print(f"  📁 {dem_file.name}")
    else:
        print("  (Cache directory not found)")
    
    print(f"\n🎯 With your current shapefiles, the expected DEM names will be:")
    print("   qmp_doi.gpkg → qmp_doi_dem.tif")
    print("   qmp_doi_wgs84.gpkg → qmp_doi_dem.tif (same as above)")
    print("   Zhabay_River_Basin.gpkg → Zhabay_River_Basin_dem.tif")

if __name__ == "__main__":
    test_dem_naming_logic()
