#!/usr/bin/env python
"""Test script to verify EXZECO environment setup."""

import sys
import importlib

def test_imports():
    """Test all required imports."""
    required_packages = [
        'numpy', 'scipy', 'pandas', 'geopandas',
        'rasterio', 'xarray', 'rioxarray', 'shapely',
        'matplotlib', 'plotly', 'folium', 'joblib',
        'numba', 'skimage', 'elevation', 'whitebox'
    ]
    
    failed = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed imports: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n✅ All packages imported successfully!")

if __name__ == "__main__":
    test_imports()