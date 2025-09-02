# DEM Naming and Caching Enhancement

## Overview

The DEM download functionality has been enhanced to automatically name DEM files based on the shapefile name and implement intelligent caching to avoid redundant downloads.

## Key Features

### 1. Automatic Naming
- DEM files are now automatically named based on the input shapefile
- Pattern: `{shapefile_name}_dem.tif`
- Examples:
  - `qmp_doi.gpkg` → `qmp_doi_dem.tif`
  - `Zhabay_River_Basin.gpkg` → `Zhabay_River_Basin_dem.tif`

### 2. Smart Suffix Removal
The system automatically removes common suffixes from shapefile names:
- `_wgs84` (from reprojected files)
- `_4326` (from coordinate system conversions)
- `_reprojected` (from transformed files)

Examples:
- `qmp_doi_wgs84.gpkg` → `qmp_doi_dem.tif` (removes `_wgs84`)
- `area_4326.shp` → `area_dem.tif` (removes `_4326`)

### 3. Intelligent Caching
- Before downloading, the system checks if a DEM with the same name already exists
- If found and valid, it reuses the existing file
- If corrupted, it downloads a fresh copy
- Provides detailed logging about cache hits and misses

### 4. Validation and Error Handling
- Validates existing DEM files before reusing
- Falls back to fresh download if cached file is corrupted
- Comprehensive logging for troubleshooting

## Usage

### In Notebook
```python
# The shapefile path is automatically passed to the DEM downloader
dem_path, dem_stats = dem_downloader.download_dem_with_fallback(
    bounds=STUDY_BOUNDS,
    cache_dir=cache_dir,
    output_filename="study_area_dem.tif",  # Will be overridden
    product='SRTM1',
    shapefile_path=shapefile_path  # New parameter
)
```

### In Code
```python
from dem_utils import DEMDownloader

downloader = DEMDownloader()
dem_path, stats = downloader.download_dem_with_fallback(
    bounds=(74.3, 42.3, 74.9, 43.2),
    cache_dir="./data/dem/cache",
    shapefile_path="./data/doi/qmp_doi.gpkg"
)
# Results in: ./data/dem/cache/qmp_doi_dem.tif
```

## Benefits

1. **Organized Storage**: DEM files are clearly associated with their source shapefiles
2. **Avoid Redundancy**: No duplicate downloads for the same study area
3. **Faster Workflow**: Subsequent runs skip DEM download if file exists
4. **Clear Naming**: Easy to identify which DEM belongs to which study area
5. **Robust Caching**: Handles corrupted files gracefully

## File Structure

```
data/
└── dem/
    └── cache/
        ├── qmp_doi_dem.tif                    # For qmp_doi.gpkg
        ├── Zhabay_River_Basin_dem.tif         # For Zhabay_River_Basin.gpkg
        └── study_area_dem.tif                 # Default/legacy files
```

## Backward Compatibility

- Existing workflows continue to work unchanged
- If no shapefile path is provided, uses default naming
- Legacy DEM files are preserved and usable

## Implementation Details

The enhancement modifies the `download_dem_with_fallback()` method in `DEMDownloader` class:

1. **Parameter Addition**: New optional `shapefile_path` parameter
2. **Name Generation**: Extract base name from shapefile path
3. **Suffix Cleaning**: Remove common transformation suffixes
4. **Cache Check**: Verify if matching DEM already exists
5. **Validation**: Ensure cached file is valid before reuse
6. **Fallback**: Download fresh copy if needed

## Testing

Run the test script to verify functionality:
```bash
python test_naming_simple.py
```

This validates the naming logic and shows current cache contents.
