#!/usr/bin/env python3
"""
Setup script for EXZECO (Extraction des Zones d'Écoulement) Package
====================================================================

A Python implementation of flood risk assessment using Monte Carlo simulation
on Digital Elevation Models (DEMs), implementing the CEREMA methodology.
"""

import os
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Remove inline comments
                    req = line.split('#')[0].strip()
                    if req:
                        requirements.append(req)
            return requirements
    return []

# Read version from src/__init__.py
def get_version():
    """Extract version from __init__.py file."""
    init_path = this_directory / "src" / "__init__.py"
    if init_path.exists():
        with open(init_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    # Extract version string between quotes
                    return line.split('=')[1].strip().strip('"\'')
    return "1.0.0"

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "twine>=4.0.0",
    "wheel>=0.40.0",
    "build>=0.10.0",
]

# Documentation dependencies
docs_requirements = [
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "sphinx-autodoc-typehints>=1.20.0",
    "nbsphinx>=0.9.0",
    "pandoc>=2.3.0",
]

# Entry points for command-line tools
entry_points = {
    'console_scripts': [
        'exzeco-analysis=src.exzeco:main',
        'exzeco-dem-download=src.dem_utils:main',
        'exzeco-visualize=src.visualization:main',
        'exzeco-dashboard=src.dashboard_web_viewer:main',
        'exzeco-export=src.export:main',
    ],
}

# Package data to include
package_data = {
    'src': [
        'config/*.yml',
        'config/*.yaml',
        'templates/*.html',
        'templates/*.jinja2',
        'static/*.css',
        'static/*.js',
        'static/images/*',
    ],
}

# Data files to include in the distribution
data_files = [
    ('config', ['config/config.yml']),
    ('examples', ['notebooks/exzeco_pfra.ipynb']),
    ('docs', ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md']),
]

setup(
    # Basic package information
    name="exzeco",
    version=get_version(),
    
    # Author and contact information
    author="EXZECO Implementation Team",
    author_email="siegfried@hydrosolutions.ch",
    maintainer="Tobias Siegfried",
    maintainer_email="siegfried@hydrosolutions.ch",
    
    # Package description
    description="Flood risk assessment using Monte Carlo simulation on DEMs (CEREMA EXZECO methodology)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Project URLs
    url="https://github.com/hydrosolutions/exzeco-flood-risk-assessment",
    project_urls={
        "Homepage": "https://github.com/hydrosolutions/exzeco-flood-risk-assessment",
        "Documentation": "https://github.com/hydrosolutions/exzeco-flood-risk-assessment/wiki",
        "Repository": "https://github.com/hydrosolutions/exzeco-flood-risk-assessment",
        "Issues": "https://github.com/hydrosolutions/exzeco-flood-risk-assessment/issues",
        "Changelog": "https://github.com/hydrosolutions/exzeco-flood-risk-assessment/blob/main/CHANGELOG.md",
        "Contributing": "https://github.com/hydrosolutions/exzeco-flood-risk-assessment/blob/main/CONTRIBUTING.md",
    },
    
    # Package discovery and structure
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    package_data=package_data,
    data_files=data_files,
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': dev_requirements,
        'docs': docs_requirements,
        'all': dev_requirements + docs_requirements,
    },
    
    # Entry points for command-line tools
    entry_points=entry_points,
    
    # Classification metadata
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        
        # Topic classification
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating systems
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        
        # Natural language
        "Natural Language :: English",
        
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
        
        # Framework
        "Framework :: Jupyter",
        "Framework :: Matplotlib",
    ],
    
    # Keywords for package discovery
    keywords=[
        "flood risk assessment",
        "monte carlo simulation", 
        "digital elevation model",
        "dem",
        "hydrology",
        "gis",
        "spatial analysis",
        "uncertainty quantification",
        "exzeco",
        "cerema",
        "flood modeling",
        "probabilistic assessment",
        "risk mapping",
        "geospatial",
        "raster analysis",
        "visualization",
        "jupyter",
    ],
    
    # Additional metadata
    license="MIT",
    platforms=["any"],
    zip_safe=False,  # Due to data files and templates
    
    # Test configuration
    test_suite="tests",
    tests_require=["pytest>=7.0.0"],
    
    # Options for bdist_wheel
    options={
        'bdist_wheel': {
            'universal': False,  # Package is not universal (has C extensions or version-specific code)
        },
    },
)