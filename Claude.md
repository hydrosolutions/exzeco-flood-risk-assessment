# Claude Code Refactoring Instructions for EXZECO Flood Risk Assessment

## Project Overview
EXZECO (Extraction des Zones d'Écoulement) is a Python implementation of flood risk assessment using Monte Carlo simulation on Digital Elevation Models (DEMs). The project implements the CEREMA methodology for preliminary flood risk assessment.

## Current Architecture
- **Core Module**: `src/exzeco.py` - Main EXZECO analysis class with Monte Carlo simulation
- **DEM Utils**: `src/dem_utils.py` - DEM downloading and processing utilities
- **Visualization**: `src/visualization.py` - Comprehensive visualization tools
- **Risk Metrics**: `src/risk_metrics.py` - Risk analysis and metrics computation
- **Main Notebook**: `notebooks/exzeco_pfra.ipynb` - Primary workflow orchestration
- **Configuration**: `config/config.yml` - Analysis parameters

## Refactoring Goals

### 1. Code Organization and Structure
- **Separate Concerns**: Break down large modules into smaller, focused components
- **Create Domain Models**: Extract data classes for DEM, StudyArea, FloodRisk, etc.
- **Implement Repository Pattern**: Create data access layer for DEM and results storage
- **Add Service Layer**: Business logic services for analysis workflows

### 2. Performance Optimizations
- **Memory Management**: Implement chunked processing for large DEMs
- **Parallel Processing**: Optimize Monte Carlo iterations with better parallelization
- **Caching Strategy**: Implement intelligent caching for DEM tiles and intermediate results
- **GPU Acceleration**: Add optional CUDA support for flow accumulation calculations

### 3. Code Quality Improvements
- **Type Hints**: Add comprehensive type hints throughout the codebase
- **Error Handling**: Implement structured exception hierarchy
- **Logging**: Standardize logging with proper levels and formatters
- **Documentation**: Add detailed docstrings with numpy style
- **Testing**: Create comprehensive test suite with pytest

### 4. API Design
- **Clean Interfaces**: Design clear public APIs for each module
- **Builder Pattern**: Implement builders for complex configurations
- **Factory Pattern**: Create factories for different DEM sources
- **Strategy Pattern**: For different flood calculation algorithms

## Specific Refactoring Tasks

### Task 1: Extract Core Domain Models
```python
# Create these new files:
# src/models/dem.py
# src/models/study_area.py
# src/models/flood_risk.py
# src/models/analysis_config.py
```

### Task 2: Refactor ExzecoAnalysis Class
The current `ExzecoAnalysis` class is doing too much. Break it down into:
- `MonteCarloSimulator` - Handles Monte Carlo iterations
- `FlowAnalyzer` - Computes flow direction and accumulation
- `RiskCalculator` - Calculates flood risk probabilities
- `ExzecoOrchestrator` - Coordinates the analysis workflow

### Task 3: Improve DEM Management
Current `DEMDownloader` should be split into:
- `DEMSourceFactory` - Creates appropriate downloader for each source
- `DEMCache` - Manages local DEM cache
- `DEMProcessor` - Handles DEM preprocessing (pit filling, resampling)
- `TerrainAnalyzer` - Calculates hillshade, slope, aspect

### Task 4: Modernize Visualization Module
The visualization module is too large. Refactor into:
- `visualizers/base.py` - Abstract base visualizer
- `visualizers/map_visualizer.py` - Folium-based maps
- `visualizers/plot_visualizer.py` - Matplotlib plots
- `visualizers/interactive_visualizer.py` - Plotly visualizations
- `visualizers/report_generator.py` - HTML/PDF report generation

### Task 5: Implement Proper Configuration Management
Replace the current YAML-based config with:
- Pydantic models for configuration validation
- Environment variable support
- Configuration profiles (development, production)
- Runtime configuration overrides

### Task 6: Add Comprehensive Testing
Create test structure:
```
tests/
├── unit/
│   ├── test_flow_analyzer.py
│   ├── test_monte_carlo.py
│   └── test_risk_calculator.py
├── integration/
│   ├── test_analysis_workflow.py
│   └── test_dem_pipeline.py
└── fixtures/
    ├── sample_dems.py
    └── test_configs.py
```

### Task 7: Implement Async/Await for I/O Operations
- Make DEM downloading async
- Implement async file operations
- Add progress callbacks for long-running operations

### Task 8: Create CLI Interface
Build a proper CLI using Click or Typer:
```bash
exzeco analyze --config config.yml --dem path/to/dem.tif
exzeco download-dem --bounds "74.3,42.3,74.9,43.2" --source srtm
exzeco visualize --results path/to/results --type interactive
```

### Task 9: Add Data Validation
- Validate DEM data integrity
- Check study area boundaries
- Verify configuration parameters
- Implement result validation

### Task 10: Improve Export Functionality
Current export is scattered. Consolidate into:
- `exporters/base.py` - Abstract exporter
- `exporters/geotiff_exporter.py` - Raster exports
- `exporters/vector_exporter.py` - Shapefile/GeoJSON exports
- `exporters/report_exporter.py` - PDF/HTML reports

## Code Style Guidelines
- Follow PEP 8 strictly
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Implement pre-commit hooks
- Use meaningful variable names (no single letters except in loops)
- Prefer composition over inheritance
- Keep functions under 20 lines when possible
- Keep classes under 200 lines when possible

## Performance Targets
- Reduce memory usage by 40% for large DEMs
- Improve Monte Carlo simulation speed by 3x
- Enable processing of 10GB+ DEMs
- Support real-time visualization updates

## Migration Strategy
1. **Phase 1**: Extract models and add type hints (no breaking changes)
2. **Phase 2**: Refactor core modules with backward compatibility
3. **Phase 3**: Add new features (async, CLI, GPU support)
4. **Phase 4**: Deprecate old interfaces
5. **Phase 5**: Remove deprecated code

## Testing Requirements
- Minimum 80% code coverage
- All public APIs must have tests
- Integration tests for complete workflows
- Performance benchmarks for critical paths
- Property-based testing for numerical algorithms

## Documentation Updates
- Update README with new architecture
- Create API documentation with Sphinx
- Add usage examples for each module
- Include performance tuning guide
- Document migration from old to new API

## Dependencies to Add
```toml
[project.dependencies]
pydantic = ">=2.0"
click = ">=8.0"
pytest = ">=7.0"
pytest-cov = ">=4.0"
pytest-asyncio = ">=0.21"
black = ">=23.0"
isort = ">=5.12"
mypy = ">=1.0"
sphinx = ">=6.0"
```

## Backward Compatibility
- Maintain old API with deprecation warnings
- Provide migration scripts for existing configs
- Keep notebook compatibility
- Support old result format reading

## Priority Order
1. Extract domain models and add type hints
2. Refactor ExzecoAnalysis class
3. Add comprehensive testing
4. Improve DEM management
5. Modernize visualization
6. Implement CLI
7. Add async support
8. GPU acceleration (optional)

## Success Metrics
- Code complexity reduced by 50% (measured by cyclomatic complexity)
- Test coverage > 80%
- Documentation coverage 100% for public APIs
- Performance improvements validated by benchmarks
- Zero breaking changes in Phase 1-2
- All existing notebooks still functional

## Notes for Claude Code
- Start with non-breaking refactors
- Maintain the scientific accuracy of calculations
- Preserve all existing functionality
- Focus on code clarity and maintainability
- Consider memory constraints for large datasets
- Ensure Windows/Linux/Mac compatibility
- Keep dependencies manageable
- Make GPU support optional
