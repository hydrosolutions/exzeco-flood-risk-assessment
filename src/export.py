"""
Export module for EXZECO flood risk assessment results.

This module handles exporting analysis results in various formats with
descriptive naming conventions that include analysis parameters.

The main function `export_exzeco_results` was refactored from section 10 
of the main notebook to reduce code duplication and improve maintainability.

Key Features:
- Descriptive file naming that includes analysis parameters
- Support for multiple export formats (GeoTIFF, HTML, PNG, CSV, Excel, YAML)
- Error handling and fallback mechanisms
- Comprehensive export summary and file organization
"""

import importlib
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from exzeco import ExzecoAnalysis


def export_exzeco_results(
    analyzer: 'ExzecoAnalysis',
    results: Dict[str, Any], 
    dem_path: Path,
    output_dir: Path,
    report: Optional[pd.DataFrame] = None,
    risk_df: Optional[pd.DataFrame] = None,
    study_bounds: Optional[tuple] = None,
    dem_data: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Export EXZECO analysis results with descriptive naming convention.
    
    Args:
        analyzer: ExzecoAnalysis instance
        results: Analysis results dictionary
        dem_path: Path to DEM file
        output_dir: Output directory path
        report: Analysis report DataFrame (optional)
        risk_df: Risk summary DataFrame (optional)
        study_bounds: Study area bounds tuple (optional)
        dem_data: DEM data array (optional)
    
    Returns:
        Dictionary with export status and information
    """
    # Import modules with reload to get updated export functions
    import exzeco
    import visualization
    importlib.reload(exzeco)
    importlib.reload(visualization)
    from exzeco import ExzecoAnalysis
    from visualization import ExzecoVisualizer
    
    export_info = {
        'success': True,
        'exported_files': [],
        'errors': [],
        'total_files': 0,
        'total_size_mb': 0.0
    }
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive base name for non-noise-level-specific files
    drainage_threshold_str = str(analyzer.config.min_drainage_area).replace('.', 'p')
    base_descriptive = f"{analyzer.config.iterations}_{drainage_threshold_str}km2"
    
    print("=== EXPORTING RESULTS WITH DESCRIPTIVE NAMING ===")
    print(f"Configuration used:")
    print(f"  - Noise levels: {analyzer.config.noise_levels}")
    print(f"  - Iterations: {analyzer.config.iterations}")
    print(f"  - Min drainage area: {analyzer.config.min_drainage_area} km²")
    print(f"  - Results available: {list(results.keys())}")
    
    # Preview new naming convention
    print(f"\nNew naming convention preview:")
    print(f"📊 GeoTIFF files:")
    for name, data in results.items():
        noise_level_cm = name.split('_')[-1] if '_' in name else '0cm'
        descriptive_name = f"exzeco_{noise_level_cm}_{analyzer.config.iterations}_{drainage_threshold_str}km2"
        print(f"  {name}.tif → {descriptive_name}.tif")
    
    # Export raster results
    try:
        new_analyzer = ExzecoAnalysis(analyzer.config)
        new_analyzer.results = results
        new_analyzer.crs = analyzer.crs
        new_analyzer.transform = analyzer.transform
        new_analyzer.resolution = analyzer.resolution
        if hasattr(analyzer, 'resolution_x'):
            new_analyzer.resolution_x = analyzer.resolution_x
            new_analyzer.resolution_y = analyzer.resolution_y
        
        new_analyzer.export_results(
            output_dir=output_dir,
            format='geotiff'
        )
        print("✅ Raster results exported successfully with descriptive naming")
        
    except Exception as e:
        print(f"❌ Error exporting raster results: {e}")
        export_info['errors'].append(f"Raster export error: {e}")
        
        # Fallback to original export if new method fails
        print("\nFalling back to original export method...")
        try:
            analyzer.export_results(
                output_dir=output_dir,
                format='geotiff'
            )
            print("✅ Raster results exported with original naming (fallback)")
        except Exception as e2:
            print(f"❌ Fallback export also failed: {e2}")
            export_info['errors'].append(f"Fallback raster export error: {e2}")
            export_info['success'] = False
    
    # Export visualizations
    try:
        new_visualizer = ExzecoVisualizer(results, dem_path)
        
        # Try with PNG format first
        new_visualizer.export_visualizations(
            output_dir=output_dir,
            formats=['html', 'png'],
            config=analyzer.config
        )
        print("✅ Visualizations exported successfully with descriptive naming (HTML + PNG)")
    except Exception as e:
        print(f"⚠️  PNG export failed ({e})")
        export_info['errors'].append(f"PNG visualization error: {e}")
        try:
            # Fallback to HTML only
            new_visualizer.export_visualizations(
                output_dir=output_dir,
                formats=['html'],
                config=analyzer.config
            )
            print("✅ Visualizations exported successfully with descriptive naming (HTML only)")
        except Exception as e2:
            print(f"❌ Error exporting visualizations with descriptive naming: {e2}")
            export_info['errors'].append(f"HTML visualization error: {e2}")
            # Final fallback to original method
            try:
                from visualization import ExzecoVisualizer
                visualizer = ExzecoVisualizer(results, dem_path)
                visualizer.export_visualizations(
                    output_dir=output_dir,
                    formats=['html']
                )
                print("✅ Visualizations exported with original naming (fallback)")
            except Exception as e3:
                print(f"❌ All visualization export methods failed: {e3}")
                export_info['errors'].append(f"All visualization exports failed: {e3}")
    
    # Export reports if provided
    if report is not None:
        try:
            report_filename = f'exzeco_report_{base_descriptive}'
            report.to_csv(output_dir / f'{report_filename}.csv', index=False)
            report.to_excel(output_dir / f'{report_filename}.xlsx', index=False)
            print(f"✅ Analysis report exported successfully with descriptive naming")
            export_info['exported_files'].extend([f'{report_filename}.csv', f'{report_filename}.xlsx'])
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            export_info['errors'].append(f"Report export error: {e}")
    
    # Export risk summary if provided
    if risk_df is not None:
        try:
            risk_filename = f'risk_summary_{base_descriptive}'
            risk_df.to_csv(output_dir / f'{risk_filename}.csv', index=False)
            risk_df.to_excel(output_dir / f'{risk_filename}.xlsx', index=False)
            print("✅ Risk summary exported successfully with descriptive naming")
            export_info['exported_files'].extend([f'{risk_filename}.csv', f'{risk_filename}.xlsx'])
        except Exception as e:
            print(f"❌ Error exporting risk summary: {e}")
            export_info['errors'].append(f"Risk summary export error: {e}")
    
    # Export configuration for reproducibility
    try:
        config_dict = {
            'noise_levels': analyzer.config.noise_levels,
            'iterations': analyzer.config.iterations,
            'min_drainage_area': analyzer.config.min_drainage_area,
            'n_jobs': analyzer.config.n_jobs,
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'naming_convention': 'exzeco_{noise_level_cm}_{iterations}_{drainage_threshold_km2}.tif'
        }
        
        if study_bounds:
            config_dict['study_bounds'] = study_bounds
        if dem_data is not None:
            config_dict['dem_resolution'] = dem_data.shape
        
        config_filename = f'analysis_config_{base_descriptive}.yml'
        with open(output_dir / config_filename, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print("✅ Analysis configuration exported successfully with descriptive naming")
        export_info['exported_files'].append(config_filename)
    except Exception as e:
        print(f"❌ Error exporting configuration: {e}")
        export_info['errors'].append(f"Configuration export error: {e}")
    
    # Generate summary
    export_summary = _generate_export_summary(
        output_dir, analyzer, drainage_threshold_str, base_descriptive
    )
    
    export_info.update(export_summary)
    
    return export_info


def _generate_export_summary(
    output_dir: Path, 
    analyzer: 'ExzecoAnalysis', 
    drainage_threshold_str: str,
    base_descriptive: str
) -> Dict[str, Any]:
    """Generate a summary of exported files."""
    
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY WITH DESCRIPTIVE NAMING CONVENTION")
    print('='*60)
    print(f"Output directory: {output_dir}")
    
    summary_info = {
        'total_files': 0,
        'total_size_mb': 0.0,
        'file_types': {},
        'descriptive_files': []
    }
    
    # List all exported files
    if output_dir.exists():
        exported_files = list(output_dir.glob('*'))
        if exported_files:
            print(f"\nExported {len(exported_files)} files:")
            
            # Group files by type
            file_groups = {
                'geotiffs': [f for f in exported_files if f.suffix == '.tif'],
                'htmls': [f for f in exported_files if f.suffix == '.html'],
                'pngs': [f for f in exported_files if f.suffix == '.png'],
                'csvs': [f for f in exported_files if f.suffix == '.csv'],
                'excels': [f for f in exported_files if f.suffix == '.xlsx'],
                'configs': [f for f in exported_files if f.suffix == '.yml']
            }
            
            file_type_labels = {
                'geotiffs': '📊 GeoTIFF Rasters',
                'htmls': '🌐 Interactive HTML',
                'pngs': '🖼️  PNG Images',
                'csvs': '📈 CSV Reports',
                'excels': '📊 Excel Reports',
                'configs': '⚙️  Configuration'
            }
            
            for file_type, files in file_groups.items():
                if files:
                    print(f"\n{file_type_labels[file_type]} ({len(files)}):")
                    summary_info['file_types'][file_type] = len(files)
                    
                    for f in sorted(files):
                        # Highlight files with new descriptive naming
                        is_descriptive = any(pattern in f.name for pattern in 
                                           [f'_{analyzer.config.iterations}_', f'_{drainage_threshold_str}'])
                        if is_descriptive:
                            print(f"  - {f.name} ✨ (DESCRIPTIVE NAMING)")
                            summary_info['descriptive_files'].append(f.name)
                        else:
                            print(f"  - {f.name}")
            
            # Calculate total size
            total_size_mb = sum(f.stat().st_size for f in exported_files) / (1024 * 1024)
            print(f"\nTotal exported data: {total_size_mb:.1f} MB")
            
            summary_info['total_files'] = len(exported_files)
            summary_info['total_size_mb'] = total_size_mb
            
            # Show naming convention explanation
            if summary_info['descriptive_files']:
                print(f"\n🆕 Files with New Descriptive Naming Convention ({len(summary_info['descriptive_files'])}):")
                for f in sorted(summary_info['descriptive_files']):
                    print(f"  - {f}")
                
                print(f"\n📝 Naming Convention Explanation:")
                print(f"    🗺️  GeoTIFF Format: exzeco_{{noise_level_cm}}_{{iterations}}_{{drainage_threshold}}.tif")
                print(f"    🌐 HTML Format: {{type}}_{{noise_level_cm}}_{{iterations}}_{{drainage_threshold}}.html")
                print(f"    📊 Report Format: {{name}}_{{iterations}}_{{drainage_threshold}}.{{ext}}")
                print(f"    Example meanings:")
                print(f"      - exzeco_100cm_10_0p1km2.tif")
                print(f"        • exzeco_: EXZECO flood probability raster")
                print(f"        • 100cm: Noise level (1.0m = 100cm)")
                print(f"        • 10: Number of Monte Carlo iterations")
                print(f"        • 0p1km2: Min drainage area (0.001 km², 'p' replaces decimal point)")
                print(f"      - map_200cm_10_0p1km2.html")
                print(f"        • Interactive map for 200cm noise level with same parameters")
                print(f"      - risk_summary_10_0p1km2.csv")
                print(f"        • Risk summary report for 10 iterations, 0.001 km² threshold")
        else:
            print("\n⚠️  No files were exported")
    else:
        print(f"\n❌ Output directory does not exist: {output_dir}")
    
    print(f"\n✅ Export process completed with comprehensive descriptive naming convention!")
    print(f"💡 All file types now include analysis parameters in their names for easy identification")
    
    return summary_info
