# Shapefile-based EXZECO Analysis

This document describes the new shapefile-based analysis functionality added to the EXZECO project.

## Overview

The EXZECO analysis now supports user-defined shapefiles for study area definition. This allows analysis of:
- **Multiple subcatchments** within a single analysis run
- **Individual basin statistics** alongside total domain statistics
- **Automatic fallback** to bounding box if shapefile is not available
- **Flexible geometry support** for any number of features

## Key Features

### 1. Multi-Scale Analysis
- **Total Domain**: Statistics for the entire study area (union of all subcatchments)
- **Individual Subcatchments**: Separate statistics for each feature in the shapefile
- **Comparative Analysis**: Easy comparison between different basins/catchments

### 2. Flexible Input Options
- **Primary**: Shapefile/GeoPackage with multiple features
- **Fallback**: Bounding box coordinates
- **Error Handling**: Clear error messages if neither option is available

### 3. Enhanced Output
- **Comprehensive Reports**: Statistics for all areas and subcatchments
- **Separate Exports**: Individual raster and vector files for each subcatchment
- **Organized Structure**: Results organized in subdirectories for clarity

## Configuration

Update your `config/config.yml` to include study area parameters:

```yaml
# EXZECO Configuration
exzeco:
  noise_levels: [0.2, 0.4, 0.6, 0.8, 1.0]  # meters
  iterations: 100  # Monte Carlo iterations
  min_drainage_area: 0.001  # km²
  drainage_classes: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]  # km²

study_area:
  # Option 1: Use shapefile (preferred if available)
  shapefile_path: "./data/doi/qmp_doi.gpkg"  # Path to shapefile/geopackage
  
  # Option 2: Use bounding box (fallback)
  bounds: [74.3, 42.3, 74.9, 43.2]  # [minx, miny, maxx, maxy]

dem:
  resolution: 30  # meters
  source: "srtm"
  cache_dir: "./data/dem"

processing:
  n_jobs: -1  # Use all available cores
  chunk_size: 1000  # pixels per chunk
  use_gpu: false

visualization:
  cmap: "Blues"
  interactive: true
  export_format: ["html", "png", "geojson"]
```

## Usage Examples

### 1. Simple Configuration-Based Analysis

```python
from src.exzeco import run_exzeco_with_config

# Run analysis using configuration file
analyzer, results, report = run_exzeco_with_config(
    config_path='config/config.yml',
    dem_path='data/dem/cache/study_area_dem.tif',
    output_dir='data/outputs'
)

# Print comprehensive report
print(report)
```

### 2. Direct Class Usage

```python
from src.exzeco import ExzecoAnalysis, load_config

# Load configuration
config = load_config('config/config.yml')

# Initialize analyzer
analyzer = ExzecoAnalysis(config)

# Run full analysis with automatic shapefile/bounds handling
results = analyzer.run_full_analysis(
    dem_path='data/dem/cache/study_area_dem.tif',
    bounds=config.bounds,
    shapefile_path=config.shapefile_path
)

# Export results
analyzer.export_results('data/outputs', format='geotiff')
analyzer.export_results('data/outputs', format='geojson')

# Generate and save report
report = analyzer.generate_report()
report.to_csv('data/outputs/exzeco_report.csv', index=False)
report.to_excel('data/outputs/exzeco_report.xlsx', index=False)
```

### 3. Custom Shapefile Analysis

```python
from src.exzeco import ExzecoAnalysis, ExzecoConfig

# Create custom configuration
config = ExzecoConfig(
    noise_levels=[0.2, 0.4, 0.6, 0.8, 1.0],
    iterations=100,
    shapefile_path='path/to/your/shapefile.shp',
    bounds=[xmin, ymin, xmax, ymax]  # fallback bounds
)

# Run analysis
analyzer = ExzecoAnalysis(config)
results = analyzer.run_full_analysis(
    dem_path='path/to/your/dem.tif'
)

# Process results
report = analyzer.generate_report()
```

## Output Structure

The analysis generates comprehensive outputs:

```
data/outputs/
├── exzeco_20cm.tif                    # Total domain raster (20cm noise)
├── exzeco_40cm.tif                    # Total domain raster (40cm noise)
├── ...                                # Other noise levels
├── exzeco_20cm.geojson               # Total domain vectors (20cm noise)
├── ...                                # Other noise levels
├── subcatchments/                     # Individual subcatchment results
│   ├── exzeco_20cm_Ala_Archa.tif     # Ala Archa basin (20cm noise)
│   ├── exzeco_20cm_Alamedin.tif      # Alamedin basin (20cm noise)
│   ├── exzeco_40cm_Ala_Archa.tif     # Ala Archa basin (40cm noise)
│   └── ...                           # Other combinations
├── exzeco_report.csv                 # Comprehensive statistics
└── exzeco_report.xlsx                # Excel version of report
```

## Report Structure

The generated report includes detailed statistics:

| Column | Description |
|--------|-------------|
| Analysis | Analysis identifier (e.g., "exzeco_20cm") |
| Area_Type | "Total Domain" or "Subcatchment" |
| Area_Name | Name of the area (from shapefile attributes) |
| Noise Level (m) | Noise level used in analysis |
| Total Area (km²) | Total area of the region |
| Flood Area (km²) | Area identified as flood-prone |
| Flood Area (%) | Percentage of area that is flood-prone |
| Mean Probability | Average flood probability |
| Max Probability | Maximum flood probability |
| Pixels > 0.8 Prob | Number of high-confidence flood pixels |

## Shapefile Requirements

### Supported Formats
- **Shapefile** (.shp)
- **GeoPackage** (.gpkg)
- **GeoJSON** (.geojson)
- Any format supported by GeoPandas

### Attribute Requirements
The shapefile should have meaningful names in one of these columns:
- `NAME_EN` (English name)
- `name` (generic name)
- Or the system will use `feature_N` as fallback

### Geometry Requirements
- **Valid geometries** only (system filters invalid ones)
- **Polygon or MultiPolygon** features
- **Any CRS** (system handles coordinate transformations)

## Error Handling

The system provides clear error messages for common issues:

1. **Missing shapefile**: Falls back to bounding box
2. **Invalid shapefile**: Falls back to bounding box
3. **No bounds provided**: Raises descriptive error
4. **Empty geometries**: Filters and continues with valid ones
5. **CRS mismatches**: Automatic coordinate transformation

## Testing

Run the test suite to verify functionality:

```bash
# Test all functionality
python test_shapefile_analysis.py

# Run demonstration
python demo_shapefile_analysis.py
```

## Implementation Notes

### Key Changes Made

1. **Configuration Updates**:
   - Added `study_area` section to config.yml
   - New `ExzecoConfig` parameters for shapefile and bounds

2. **Core Analysis Updates**:
   - `load_study_areas()` method for flexible area loading
   - `mask_raster_by_geometry()` for subcatchment masking
   - Enhanced `run_full_analysis()` with subcatchment support
   - Updated reporting and export functions

3. **Workflow Enhancements**:
   - Automatic fallback from shapefile to bounds
   - Individual subcatchment result processing
   - Organized output directory structure

### Performance Considerations

- **Parallel Processing**: Monte Carlo iterations run in parallel
- **Memory Efficient**: Subcatchment masking done on-demand
- **Optimized I/O**: Compressed output formats
- **Progress Tracking**: Progress bars for long-running operations

## Troubleshooting

### Common Issues

1. **Empty DEM Results**:
   - Check coordinate system compatibility between DEM and shapefile
   - Verify bounds overlap with DEM extent
   - Ensure DEM covers the study area

2. **Slow Performance**:
   - Reduce number of iterations for testing
   - Use fewer noise levels during development
   - Consider reducing DEM resolution

3. **Memory Issues**:
   - Reduce chunk_size in configuration
   - Lower number of parallel jobs (n_jobs)
   - Process smaller study areas

### Debug Information

The system provides detailed logging. Enable debug mode:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements for future versions:

1. **Interactive Selection**: GUI for shapefile attribute selection
2. **Advanced Filtering**: Filter subcatchments by area/properties
3. **Nested Hierarchies**: Support for nested basin structures
4. **Performance Optimization**: GPU acceleration for large areas
5. **Visualization**: Interactive maps showing subcatchment results

---

For questions or issues, please refer to the main project documentation or contact the development team.
