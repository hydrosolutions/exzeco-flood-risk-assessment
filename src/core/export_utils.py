#!/usr/bin/env python
"""
Export Utilities Module for EXZECO
==================================

This module provides export functionality including:
- GeoTIFF export with proper georeferencing
- CSV report generation
- Configuration file export
- Metadata handling

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import os
import yaml
import pandas as pd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ResultExporter:
    """
    Handles export of analysis results to various formats.
    
    This class provides functionality for exporting raster data,
    statistical summaries, and configuration files.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize result exporter.
        
        Parameters
        ----------
        output_dir : Union[str, Path]
            Output directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ResultExporter initialized with output directory: {self.output_dir}")
    
    def export_geotiff(self, 
                      data: np.ndarray,
                      filename: str,
                      transform: Affine,
                      crs: CRS,
                      nodata: Optional[float] = None,
                      dtype: str = 'float32',
                      compress: str = 'lzw') -> Path:
        """
        Export numpy array as GeoTIFF.
        
        Parameters
        ----------
        data : np.ndarray
            Data array to export
        filename : str
            Output filename (without path)
        transform : Affine
            Geospatial transformation matrix
        crs : CRS
            Coordinate reference system
        nodata : Optional[float]
            NoData value
        dtype : str
            Output data type
        compress : str
            Compression method
            
        Returns
        -------
        Path
            Path to exported file
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting GeoTIFF: {output_path}")
        
        # Ensure data is 2D
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        elif data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress=compress,
            tiled=True,
            blockxsize=512,
            blockysize=512
        ) as dst:
            dst.write(data.astype(dtype), 1)
            
            # Add metadata
            dst.update_tags(
                EXZECO_VERSION="1.0",
                CREATION_DATE=pd.Timestamp.now().isoformat(),
                DESCRIPTION=f"EXZECO flood risk assessment result"
            )
        
        return output_path
    
    def export_probability_map(self,
                              probability_map: np.ndarray,
                              noise_level: float,
                              iterations: int,
                              min_drainage_area: float,
                              transform: Affine,
                              crs: CRS) -> Path:
        """
        Export probability map with standardized naming.
        
        Parameters
        ----------
        probability_map : np.ndarray
            Probability map (0-1)
        noise_level : float
            Noise level in cm
        iterations : int
            Number of iterations
        min_drainage_area : float
            Minimum drainage area in km²
        transform : Affine
            Geospatial transformation
        crs : CRS
            Coordinate reference system
            
        Returns
        -------
        Path
            Path to exported file
        """
        # Create standardized filename
        noise_cm = int(noise_level * 100)
        area_str = f"{min_drainage_area:.3f}".replace(".", "p")
        filename = f"exzeco_{noise_cm}cm_{iterations}_{area_str}km2.tif"
        
        return self.export_geotiff(
            data=probability_map,
            filename=filename,
            transform=transform,
            crs=crs,
            nodata=None,
            dtype='float32'
        )
    
    def export_classification(self,
                             classification: np.ndarray,
                             noise_level: float,
                             iterations: int,
                             min_drainage_area: float,
                             transform: Affine,
                             crs: CRS) -> Path:
        """
        Export drainage classification with standardized naming.
        
        Parameters
        ----------
        classification : np.ndarray
            Classification array
        noise_level : float
            Noise level in cm
        iterations : int
            Number of iterations
        min_drainage_area : float
            Minimum drainage area in km²
        transform : Affine
            Geospatial transformation
        crs : CRS
            Coordinate reference system
            
        Returns
        -------
        Path
            Path to exported file
        """
        # Create standardized filename
        filename = "drainage_classification.tif"
        
        return self.export_geotiff(
            data=classification,
            filename=filename,
            transform=transform,
            crs=crs,
            nodata=0,
            dtype='uint8'
        )
    
    def export_csv_report(self,
                         summary_df: pd.DataFrame,
                         noise_level: float,
                         iterations: int,
                         min_drainage_area: float,
                         additional_data: Optional[Dict[str, Any]] = None) -> Path:
        """
        Export analysis summary as CSV.
        
        Parameters
        ----------
        summary_df : pd.DataFrame
            Summary statistics DataFrame
        noise_level : float
            Noise level in cm
        iterations : int
            Number of iterations
        min_drainage_area : float
            Minimum drainage area in km²
        additional_data : Optional[Dict[str, Any]]
            Additional data to include
            
        Returns
        -------
        Path
            Path to exported CSV file
        """
        # Create standardized filename
        noise_cm = int(noise_level * 100)
        area_str = f"{min_drainage_area:.3f}".replace(".", "p")
        filename = f"exzeco_report_{iterations}_{area_str}km2.csv"
        
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting CSV report: {output_path}")
        
        # Prepare data for export
        export_data = summary_df.copy()
        
        # Add metadata columns
        export_data['noise_level_cm'] = int(noise_level * 100)
        export_data['iterations'] = iterations
        export_data['min_drainage_area_km2'] = min_drainage_area
        export_data['export_date'] = pd.Timestamp.now().isoformat()
        
        # Add additional data if provided
        if additional_data:
            for key, value in additional_data.items():
                export_data[key] = value
        
        # Export to CSV
        export_data.to_csv(output_path, index=False)
        
        return output_path
    
    def export_risk_summary(self,
                           risk_summary: Dict[str, float],
                           noise_level: float,
                           iterations: int,
                           min_drainage_area: float) -> Path:
        """
        Export risk summary as CSV.
        
        Parameters
        ----------
        risk_summary : Dict[str, float]
            Risk summary metrics
        noise_level : float
            Noise level in cm
        iterations : int
            Number of iterations
        min_drainage_area : float
            Minimum drainage area in km²
            
        Returns
        -------
        Path
            Path to exported CSV file
        """
        # Create standardized filename
        area_str = f"{min_drainage_area:.3f}".replace(".", "p")
        filename = f"risk_summary_{iterations}_{area_str}km2.csv"
        
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting risk summary: {output_path}")
        
        # Convert to DataFrame
        data = []
        for metric, value in risk_summary.items():
            data.append({
                'metric': metric,
                'value': value,
                'noise_level_cm': int(noise_level * 100),
                'iterations': iterations,
                'min_drainage_area_km2': min_drainage_area,
                'export_date': pd.Timestamp.now().isoformat()
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def export_config(self,
                     config: Dict[str, Any],
                     filename: str = "analysis_config.yml") -> Path:
        """
        Export analysis configuration to YAML file.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        filename : str
            Output filename
            
        Returns
        -------
        Path
            Path to exported config file
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting configuration: {output_path}")
        
        # Add export metadata
        export_config = config.copy()
        export_config['export_metadata'] = {
            'export_date': pd.Timestamp.now().isoformat(),
            'exzeco_version': '1.0'
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(export_config, f, default_flow_style=False, sort_keys=False)
        
        return output_path
    
    def create_file_inventory(self) -> Path:
        """
        Create inventory of all files in output directory.
        
        Returns
        -------
        Path
            Path to inventory file
        """
        inventory_path = self.output_dir / "file_inventory.csv"
        
        logger.info(f"Creating file inventory: {inventory_path}")
        
        # Collect file information
        files_data = []
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and file_path.name != "file_inventory.csv":
                stat = file_path.stat()
                files_data.append({
                    'filename': file_path.name,
                    'size_bytes': stat.st_size,
                    'modification_time': pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
                    'file_type': file_path.suffix
                })
        
        # Create DataFrame and export
        df = pd.DataFrame(files_data)
        df['inventory_date'] = pd.Timestamp.now().isoformat()
        df.to_csv(inventory_path, index=False)
        
        return inventory_path
    
    def validate_exports(self) -> Dict[str, bool]:
        """
        Validate exported files.
        
        Returns
        -------
        Dict[str, bool]
            Validation results for each file type
        """
        results = {}
        
        # Check for GeoTIFF files
        tiff_files = list(self.output_dir.glob("*.tif"))
        results['geotiff_files'] = len(tiff_files) > 0
        
        # Check for CSV files
        csv_files = list(self.output_dir.glob("*.csv"))
        results['csv_files'] = len(csv_files) > 0
        
        # Check for config files
        config_files = list(self.output_dir.glob("*.yml")) + list(self.output_dir.glob("*.yaml"))
        results['config_files'] = len(config_files) > 0
        
        # Validate GeoTIFF files can be opened
        valid_tiffs = 0
        for tiff_file in tiff_files:
            try:
                with rasterio.open(tiff_file) as src:
                    # Basic checks
                    if src.crs is not None and src.transform is not None:
                        valid_tiffs += 1
            except Exception as e:
                logger.error(f"Invalid GeoTIFF {tiff_file}: {e}")
        
        results['valid_geotiff_files'] = valid_tiffs == len(tiff_files) if tiff_files else False
        
        logger.info(f"Export validation results: {results}")
        
        return results