# EXZECO Flood Risk Assessment

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Ready-orange.svg)](https://jupyter.org/)

A comprehensive Python implementation of the **EXZECO** (Extraction des Zones d'Écoulement) methodology for preliminary flood risk assessment using Monte Carlo simulation on Digital Elevation Models (DEMs).

## 🌊 Overview

EXZECO is a probabilistic flood risk assessment method that uses uncertainty analysis on topographic data to identify potentially flooded areas. By applying Monte Carlo simulation with controlled DEM perturbation, the method generates flood probability maps that account for terrain uncertainty and provides robust flood risk estimates.

### Key Features

- **Monte Carlo Simulation**: Robust uncertainty quantification through multiple DEM realizations
- **Multi-scale Analysis**: Configurable noise levels from 20cm to 100cm+ elevation uncertainty
- **Interactive Visualizations**: Dynamic 3D maps, heatmaps, and statistical dashboards
- **Scalable Processing**: Parallel computation support for large study areas
- **Professional Outputs**: Publication-ready maps, reports, and statistical summaries

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Conda or pip package manager
- At least 4GB RAM (8GB+ recommended for large areas)

### Installation

#### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/hydrosolutions/exzeco-flood-risk-assessment.git
cd exzeco-flood-risk-assessment

# Create and activate conda environment
conda env create -f environment.yml
conda activate exzeco

# Install additional dependencies
pip install -r requirements.txt
```

#### Option 2: Pip Installation

```bash
# Clone the repository
git clone https://github.com/hydrosolutions/exzeco-flood-risk-assessment.git
cd exzeco-flood-risk-assessment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 3: Development Installation

```bash
# For contributors and developers
git clone https://github.com/hydrosolutions/exzeco-flood-risk-assessment.git
cd exzeco-flood-risk-assessment

# Install in development mode
pip install -e .
```

### Quick Test

Verify your installation by running:

```python
from src.exzeco import ExzecoAnalysis
print("✅ EXZECO successfully installed!")
```

---

## 📊 Methodology

### The EXZECO Approach

EXZECO (Extraction des Zones d'Écoulement) is a probabilistic flood risk assessment methodology developed by CEREMA that addresses the inherent uncertainty in topographic data for flood modeling.

#### Core Principles

1. **Uncertainty Quantification**: Recognizes that elevation data contains measurement errors and uncertainties
2. **Monte Carlo Framework**: Uses repeated simulations with perturbed DEMs to capture uncertainty propagation
3. **Flow Accumulation Analysis**: Identifies drainage networks and accumulation patterns under uncertainty
4. **Probabilistic Risk Assessment**: Generates flood probability maps instead of deterministic flood zones

#### Mathematical Foundation

The method applies controlled noise to elevation data:

```
DEM_perturbed = DEM_original + N(0, σ²)
```

Where:
- `N(0, σ²)` is normally distributed noise with mean 0 and standard deviation σ
- `σ` represents the elevation uncertainty (noise level)

#### Workflow Steps

1. **DEM Acquisition**: Download and preprocess elevation data for the study area
2. **Noise Application**: Apply multiple noise levels (e.g., 20cm, 40cm, 60cm, 80cm, 100cm)
3. **Monte Carlo Simulation**: Generate hundreds of perturbed DEM realizations
4. **Flow Analysis**: Calculate flow accumulation for each realization
5. **Probability Calculation**: Aggregate results to determine flood probability at each pixel
6. **Risk Classification**: Categorize areas by flood risk level based on probability thresholds

#### Risk Categories

- **Very High Risk**: >80% probability of flooding
- **High Risk**: 60-80% probability
- **Moderate Risk**: 40-60% probability  
- **Low Risk**: 20-40% probability
- **Very Low Risk**: <20% probability

#### Scientific Basis

The methodology is based on:
- **Terrain Uncertainty Modeling**: Accounts for systematic and random errors in elevation data
- **Hydrological Process Simulation**: Models surface water flow under uncertainty
- **Statistical Aggregation**: Combines multiple realizations for robust probability estimation
- **Spatial Risk Assessment**: Provides spatially explicit flood risk maps

#### Validation and Applications

EXZECO has been validated for:
- **Preliminary Flood Risk Assessment**: First-pass identification of flood-prone areas
- **Planning Applications**: Urban planning and infrastructure development
- **Comparative Analysis**: Relative risk assessment between different areas
- **Uncertainty Communication**: Transparent representation of modeling uncertainties

*Reference: [CEREMA EXZECO Methodology](https://www.cerema.fr/system/files/documents/2020/07/methode_exzeco_25mai2020.pdf)*


## 🔬 What EXZECO Produces

### Primary Outputs

#### 1. Flood Probability Rasters
- **Format**: GeoTIFF files for each noise level
- **Content**: Pixel-wise flood probability values (0-1)
- **Files**: `exzeco_20cm.tif`, `exzeco_40cm.tif`, etc.
- **Use**: Direct input to GIS software, further modeling

#### 2. Interactive Maps
- **3D Visualizations**: Three-dimensional terrain with flood overlays
  - `3d_exzeco_[noise_level].html` - Individual noise level 3D maps
  - `exzeco_3d.html` - Comprehensive 3D comparison
- **2D Maps**: Standard map projections with flood zones
  - `map_exzeco_[noise_level].html` - Individual risk maps
  - `exzeco_map.html` - Combined visualization
- **Heatmaps**: Probability density visualizations
  - `heatmap_exzeco_[noise_level].html` - Statistical heat maps

#### 3. Statistical Reports
- **Risk Summary**: `risk_summary.csv`
  - Area statistics by risk category
  - Percentage coverage by noise level
  - Comparative analysis between scenarios
- **Detailed Report**: `exzeco_report.csv`
  - Pixel-level statistics
  - Probability distributions
  - Spatial metrics

#### 4. Visual Documentation
- **Study Area Overview**: `study_area_map.html`
- **DEM Analysis**: `dem_analysis.png`
- **Comparison Dashboard**: `comparison.html`
- **Final Report**: `exzeco_final_report.html`

### Secondary Outputs

#### Configuration Files
- **Analysis Configuration**: `analysis_config.yml`
  - Records analysis parameters
  - Ensures reproducibility
  - Documents methodology choices

#### Intermediate Data
- **DEM Cache**: Processed elevation data
- **Flow Networks**: Computed drainage patterns
- **Probability Matrices**: Statistical aggregations

### Output Specifications

#### Spatial Data
- **Coordinate System**: Preserves input DEM projection
- **Resolution**: Matches source DEM resolution
- **Extent**: Covers full study area
- **Format**: Standard GeoTIFF with metadata

#### Interactive Content
- **Technology**: HTML5 with Plotly/Folium
- **Compatibility**: Modern web browsers
- **Interactivity**: Zoom, pan, layer control, data queries
- **Export**: PNG, PDF, SVG formats supported

#### Statistical Content
- **Metrics**: Mean, median, standard deviation, percentiles
- **Coverage**: Area calculations in km² and percentages
- **Uncertainty**: Confidence intervals and uncertainty bounds
- **Validation**: Cross-validation statistics where applicable

### Data Integration

All outputs are designed for integration with:
- **GIS Software**: QGIS, ArcGIS, GRASS GIS
- **Modeling Platforms**: Python, R, MATLAB
- **Web Platforms**: Leaflet, OpenLayers, MapBox
- **Reporting Tools**: Jupyter notebooks, R Markdown

---

## 🛠️ Usage

### Basic Usage

1. **Configure your analysis** by editing `config/config.yml`:

```yaml
noise_levels: [0.2, 0.4, 0.6, 0.8, 1.0]  # meters
iterations: 100
min_drainage_area: 0.01  # km²
n_jobs: -1  # Use all available cores
```

2. **Run the complete analysis** using the Jupyter notebook:

```bash
jupyter notebook notebooks/exzeco_pfra.ipynb
```

3. **Or use the Python API directly**:

```python
from src.exzeco import ExzecoAnalysis, load_config
from src.dem_utils import StudyArea

# Define study area (bbox: min_lon, min_lat, max_lon, max_lat)
study_area = StudyArea((74.3, 42.3, 74.9, 43.2))

# Load configuration
config = load_config('config/config.yml')

# Run analysis
analysis = ExzecoAnalysis(study_area, config)
results = analysis.run_complete_analysis()
```

### Advanced Usage

#### Custom Study Areas

```python
# From shapefile
import geopandas as gpd
study_gdf = gpd.read_file('path/to/study_area.shp')
study_area = StudyArea(tuple(study_gdf.total_bounds))

# From coordinates list
custom_bounds = (longitude_min, latitude_min, longitude_max, latitude_max)
study_area = StudyArea(custom_bounds)
```

#### Parallel Processing

```python
# Configure parallel execution
config.n_jobs = 8  # Use 8 cores
config.iterations = 500  # More iterations for better statistics
```

#### Custom Visualization

```python
from src.visualization import ExzecoVisualizer

visualizer = ExzecoVisualizer(results)
custom_map = visualizer.create_risk_map(
    noise_level=0.6,
    color_scheme='viridis',
    include_topography=True
)
```

---

## 📁 Project Structure

```
exzeco-flood-risk-assessment/
├── 📁 config/                  # Configuration files
│   └── config.yml              # Main analysis parameters
├── 📁 data/                    # Data directory
│   ├── 📁 cache/              # Temporary processing files  
│   ├── 📁 dem/                # DEM data and cache
│   ├── 📁 outputs/            # Analysis results
│   └── 📁 results/            # Final processed outputs
├── 📁 notebooks/              # Jupyter notebooks
│   └── exzeco_pfra.ipynb      # Main analysis notebook
├── 📁 src/                    # Source code
│   ├── exzeco.py              # Core EXZECO implementation
│   ├── dem_utils.py           # DEM processing utilities
│   ├── visualization.py       # Visualization functions
│   └── test_setup.py          # Testing utilities
├── 📁 .github/                # GitHub configuration
│   ├── 📁 workflows/          # CI/CD workflows
│   └── 📁 ISSUE_TEMPLATE/     # Issue templates
├── environment.yml            # Conda environment
├── requirements.txt           # Python dependencies
├── setup.py                  # Package installation
└── README.md                 # This file
```

---

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing use cases, your input helps make EXZECO better for everyone.

### How to Contribute

#### 🐛 Report Bugs
- Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include system information and reproducible examples
- Check existing issues before creating new ones

#### 💡 Suggest Features
- Use our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the problem and proposed solution
- Consider implementation approaches

#### 📝 Improve Documentation
- Fix typos and clarify explanations
- Add examples and use cases
- Translate documentation to other languages

#### 🔧 Contribute Code
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Submit** a Pull Request

### Development Guidelines

#### Code Standards
- Follow [PEP 8](https://pep8.org/) style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new functionality

#### Testing
```bash
# Run tests
python -m pytest tests/

# Check code coverage
pytest --cov=src tests/
```

#### Documentation
- Update README.md for new features
- Add docstrings following NumPy style
- Include examples in docstrings

### Community Guidelines

We are committed to fostering an open and welcoming environment. Please read our [Code of Conduct](CONTRIBUTING.md) and follow these principles:

- **Be Respectful**: Treat all community members with respect
- **Be Collaborative**: Work together constructively
- **Be Patient**: Help newcomers learn and grow
- **Be Inclusive**: Welcome diverse perspectives and backgrounds

### Recognition

Contributors are recognized in:
- [CHANGELOG.md](CHANGELOG.md) for each release
- Repository contributors list
- Academic publications (for significant contributions)

### Getting Help

- **Discord**: Join our community chat (coming soon)
- **GitHub Discussions**: Ask questions and share ideas
- **Email**: Contact maintainers for sensitive issues

### Research Collaboration

We're particularly interested in:
- **Validation Studies**: Real-world case studies and comparisons
- **Methodological Improvements**: Algorithm enhancements and optimizations
- **New Applications**: Novel use cases and domain applications
- **Educational Materials**: Tutorials, workshops, and teaching resources

#### Academic Collaboration
- Co-authorship opportunities for significant contributions
- Conference presentations and workshop participation
- Joint grant applications for further development

#### Industry Partnership
- Case study development with real-world applications
- Performance testing on large-scale datasets
- Integration with existing flood risk management systems

---

## 📊 Example Results

Check out example outputs in the `data/outputs/` directory:
- Interactive 3D flood risk maps
- Statistical analysis dashboards  
- Comparative visualizations across noise levels
- Comprehensive risk assessment reports

---

## 📚 References

- [CEREMA EXZECO Methodology](https://www.cerema.fr/system/files/documents/2020/07/methode_exzeco_25mai2020.pdf)
- Probabilistic Flood Risk Assessment in Ungauged Basins
- Monte Carlo Methods for Hydrological Uncertainty Analysis

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CEREMA** for developing the original EXZECO methodology
- **Open Source Community** for the excellent libraries that make this possible
- **Contributors** who help improve and maintain this implementation

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/hydrosolutions/exzeco-flood-risk-assessment/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hydrosolutions/exzeco-flood-risk-assessment/discussions)
- **Email**: [Contact the maintainers](mailto:info@hydrosolutions.ch)

---

*Made with ❤️ for the flood risk assessment community*
