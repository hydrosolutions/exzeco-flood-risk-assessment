"""
Dashboard Exporter Module for EXZECO Flood Risk Assessment

This module provides functionality to export interactive dashboards as HTML files
with professional styling and comprehensive error handling.

Author: Tobias Siegfried, hydrosolutions GmbH
Date: 2025
License: MIT
"""

import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd


class DashboardExporter:
    """
    A class to handle exporting interactive dashboards to HTML files.
    
    This class provides methods to export various types of dashboard objects
    (Plotly figures, ipywidgets, etc.) as standalone HTML files with
    professional styling and metadata.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the DashboardExporter.
        
        Args:
            output_dir (Path): Directory where exported files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_dashboard(
        self,
        dashboard: Any,
        filename: str = "exzeco_interactive_dashboard.html",
        study_bounds: Optional[tuple] = None,
        config: Optional[Any] = None,
        title: str = "EXZECO Interactive Dashboard",
        subtitle: str = "Preliminary Flood Risk Assessment using Monte Carlo Simulation"
    ) -> Dict[str, Any]:
        """
        Export a dashboard object as an HTML file.
        
        Args:
            dashboard: The dashboard object to export
            filename: Name of the output HTML file
            study_bounds: Study area bounds tuple (min_lon, min_lat, max_lon, max_lat)
            config: Configuration object with analysis parameters
            title: Main title for the HTML document
            subtitle: Subtitle for the HTML document
            
        Returns:
            Dict with export results and metadata
        """
        print(f"\n📊 Exporting {title} to HTML...")
        
        dashboard_html_path = self.output_dir / filename
        export_info = {
            'success': False,
            'html_path': str(dashboard_html_path),
            'file_size_kb': 0,
            'export_method': 'unknown',
            'error': None
        }
        
        try:
            # Method 1: Direct export for Plotly figures
            if hasattr(dashboard, 'write_html'):
                dashboard.write_html(str(dashboard_html_path))
                export_info['export_method'] = 'plotly_write_html'
                export_info['success'] = True
                print(f"✅ Dashboard exported using Plotly write_html: {dashboard_html_path}")
            
            # Method 2: Generic save method
            elif hasattr(dashboard, 'save'):
                dashboard.save(str(dashboard_html_path))
                export_info['export_method'] = 'generic_save'
                export_info['success'] = True
                print(f"✅ Dashboard exported using save method: {dashboard_html_path}")
            
            # Method 3: Custom HTML wrapper
            else:
                print("📝 Creating custom HTML wrapper for dashboard...")
                self._create_custom_html_export(
                    dashboard, dashboard_html_path, title, subtitle, study_bounds, config
                )
                export_info['export_method'] = 'custom_html_wrapper'
                export_info['success'] = True
                print(f"✅ Custom HTML dashboard created: {dashboard_html_path}")
            
            # Get file size if export was successful
            if export_info['success'] and dashboard_html_path.exists():
                export_info['file_size_kb'] = dashboard_html_path.stat().st_size / 1024
                print(f"📂 Dashboard file size: {export_info['file_size_kb']:.1f} KB")
            
            # Create metadata file
            self._create_metadata_file(filename, study_bounds, config, export_info)
            
        except Exception as e:
            export_info['error'] = str(e)
            print(f"❌ Error exporting dashboard: {e}")
            traceback.print_exc()
            
            # Create fallback HTML
            fallback_info = self._create_fallback_dashboard(
                filename.replace('.html', '_fallback.html'),
                title, subtitle, study_bounds, config
            )
            if fallback_info['success']:
                export_info.update(fallback_info)
                export_info['export_method'] = 'fallback_html'
        
        print(f"\n📂 Dashboard export completed!")
        return export_info
    
    def _create_custom_html_export(
        self,
        dashboard: Any,
        output_path: Path,
        title: str,
        subtitle: str,
        study_bounds: Optional[tuple],
        config: Optional[Any]
    ) -> None:
        """Create a custom HTML export with embedded dashboard content."""
        
        # Get HTML representation of the dashboard
        dashboard_html_content = self._get_dashboard_html_content(dashboard)
        
        # Create the complete HTML document
        html_template = self._create_html_template(
            dashboard_html_content, title, subtitle, study_bounds, config
        )
        
        # Save the HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
    
    def _get_dashboard_html_content(self, dashboard: Any) -> str:
        """Extract HTML content from dashboard object."""
        
        if hasattr(dashboard, '_repr_html_'):
            return dashboard._repr_html_()
        elif hasattr(dashboard, 'to_html'):
            return dashboard.to_html()
        else:
            # Fallback content
            return f"""
            <div style="width: 100%; height: 600px; display: flex; justify-content: center; align-items: center; 
                        background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px;">
                <div style="text-align: center; color: #6c757d;">
                    <h3>📊 Interactive Dashboard</h3>
                    <p>Dashboard content would be displayed here.</p>
                    <p><small>Dashboard type: {type(dashboard).__name__}</small></p>
                    <p><small>Note: This dashboard is best viewed in the Jupyter notebook environment.</small></p>
                </div>
            </div>
            """
    
    def _create_html_template(
        self,
        dashboard_content: str,
        title: str,
        subtitle: str,
        study_bounds: Optional[tuple],
        config: Optional[Any]
    ) -> str:
        """Create the complete HTML document template."""
        
        # Build study area info
        study_area_info = ""
        if study_bounds:
            study_area_info = f"""
            <p><strong>Study Area:</strong> {study_bounds[1]:.4f}°N-{study_bounds[3]:.4f}°N, 
               {study_bounds[0]:.4f}°E-{study_bounds[2]:.4f}°E</p>
            """
        
        # Build analysis parameters info
        analysis_info = ""
        if config:
            try:
                analysis_info = f"""
                <p><strong>Analysis Parameters:</strong></p>
                <ul>
                    <li>Noise Levels: {', '.join([f'{x}m' for x in config.noise_levels])}</li>
                    <li>Iterations: {config.iterations}</li>
                    <li>Min Drainage Area: {config.min_drainage_area} km²</li>
                </ul>
                """
            except AttributeError:
                analysis_info = "<p>Analysis parameters available in the source notebook.</p>"
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header h2 {{
            font-size: 1.2rem;
            font-weight: 400;
            opacity: 0.9;
        }}
        
        .info-panel {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }}
        
        .info-panel h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }}
        
        .dashboard-container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-bottom: 30px;
            min-height: 600px;
        }}
        
        .footer {{
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
            padding: 20px;
            border-top: 1px solid #dee2e6;
            margin-top: 30px;
        }}
        
        .badge {{
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 2px;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .dashboard-container {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 {title}</h1>
            <h2>{subtitle}</h2>
        </div>
        
        <div class="info-panel">
            <h3>📋 Analysis Overview</h3>
            {study_area_info}
            {analysis_info}
            <p><span class="badge">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
        </div>
        
        <div class="dashboard-container">
            {dashboard_content}
        </div>
        
        <div class="footer">
            <p><strong>EXZECO Flood Risk Assessment</strong></p>
            <p>Generated using EXZECO methodology (CEREMA) | Author: Tobias Siegfried, hydrosolutions GmbH</p>
            <p><em>This is a preliminary assessment for planning purposes only. Validate with field observations and detailed hydraulic modeling.</em></p>
        </div>
    </div>
</body>
</html>
        """
    
    def _create_fallback_dashboard(
        self,
        filename: str,
        title: str,
        subtitle: str,
        study_bounds: Optional[tuple],
        config: Optional[Any]
    ) -> Dict[str, Any]:
        """Create a simple fallback HTML dashboard."""
        
        print(f"\n📝 Creating fallback dashboard: {filename}")
        
        fallback_path = self.output_dir / filename
        fallback_info = {
            'success': False,
            'html_path': str(fallback_path),
            'file_size_kb': 0,
            'export_method': 'fallback_html',
            'error': None
        }
        
        try:
            study_info = ""
            if study_bounds:
                study_info = f"<p><strong>Study Area:</strong> {study_bounds}</p>"
            
            config_info = ""
            if config:
                try:
                    config_info = f"""
                    <p><strong>Configuration:</strong></p>
                    <ul>
                        <li>Noise Levels: {', '.join([f'{x}m' for x in config.noise_levels])}</li>
                        <li>Iterations: {config.iterations}</li>
                        <li>Min Drainage Area: {config.min_drainage_area} km²</li>
                    </ul>
                    """
                except AttributeError:
                    config_info = "<p>Configuration details available in the source notebook.</p>"
            
            fallback_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Summary</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
            margin: 40px; 
            line-height: 1.6; 
            background: #f8f9fa;
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 30px; 
            text-align: center; 
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .content {{ 
            background: white;
            padding: 30px; 
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-box {{ 
            background: #e8f4f8; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px; 
            border-left: 4px solid #2196F3;
        }}
        .note {{ 
            background: #fff3cd; 
            padding: 15px; 
            border-radius: 8px; 
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
        ul {{ padding-left: 20px; }}
        .badge {{ 
            background: #e3f2fd; 
            color: #1976d2; 
            padding: 4px 12px; 
            border-radius: 15px; 
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 {title}</h1>
        <p>{subtitle}</p>
        <p><span class="badge">Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
    </div>
    
    <div class="content">
        <div class="summary-box">
            <h2>📊 Analysis Summary</h2>
            {study_info}
            {config_info}
        </div>
        
        <div class="summary-box">
            <h2>🗺️ Available Outputs</h2>
            <ul>
                <li><strong>Interactive Maps:</strong> Flood risk visualization with multiple layers</li>
                <li><strong>3D Terrain Views:</strong> Interactive elevation models with flood zones</li>
                <li><strong>Statistical Plots:</strong> Multi-level comparison and risk distribution</li>
                <li><strong>GeoTIFF Rasters:</strong> Downloadable spatial data for GIS analysis</li>
                <li><strong>Reports:</strong> Comprehensive analysis results in CSV/Excel format</li>
            </ul>
        </div>
        
        <div class="note">
            <h3>📝 Note</h3>
            <p>This is a summary dashboard. The full interactive components are available in the Jupyter notebook environment.</p>
            <p>Check the <code>data/outputs</code> directory for all exported visualization files and analysis results.</p>
        </div>
        
        <div class="summary-box">
            <h2>🔗 Next Steps</h2>
            <ol>
                <li>Review interactive visualizations in the notebook</li>
                <li>Examine exported GeoTIFF files in GIS software</li>
                <li>Analyze statistical reports for risk quantification</li>
                <li>Validate results with field observations</li>
                <li>Consider detailed hydraulic modeling for critical areas</li>
            </ol>
        </div>
    </div>
</body>
</html>
            """
            
            with open(fallback_path, 'w', encoding='utf-8') as f:
                f.write(fallback_html)
            
            fallback_info['success'] = True
            fallback_info['file_size_kb'] = fallback_path.stat().st_size / 1024
            print(f"✅ Fallback dashboard created: {fallback_path}")
            
        except Exception as e:
            fallback_info['error'] = str(e)
            print(f"❌ Failed to create fallback dashboard: {e}")
        
        return fallback_info
    
    def _create_metadata_file(
        self,
        dashboard_filename: str,
        study_bounds: Optional[tuple],
        config: Optional[Any],
        export_info: Dict[str, Any]
    ) -> None:
        """Create a JSON metadata file with dashboard information."""
        
        try:
            # Build metadata
            metadata = {
                'title': 'EXZECO Interactive Dashboard',
                'created': pd.Timestamp.now().isoformat(),
                'dashboard_file': dashboard_filename,
                'export_info': export_info
            }
            
            if study_bounds:
                metadata['study_area'] = {
                    'bounds': study_bounds,
                    'bounds_format': '(min_lon, min_lat, max_lon, max_lat)'
                }
            
            if config:
                try:
                    metadata['analysis_parameters'] = {
                        'noise_levels': list(config.noise_levels),
                        'iterations': config.iterations,
                        'min_drainage_area': config.min_drainage_area,
                        'n_jobs': getattr(config, 'n_jobs', 'not_specified')
                    }
                except AttributeError:
                    metadata['analysis_parameters'] = 'Configuration object not fully accessible'
            
            # Save metadata
            metadata_path = self.output_dir / "dashboard_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Dashboard metadata saved: {metadata_path}")
            
        except Exception as e:
            print(f"⚠️ Could not create metadata file: {e}")


def export_dashboard(
    dashboard: Any,
    output_dir: Path,
    filename: str = "exzeco_interactive_dashboard.html",
    study_bounds: Optional[tuple] = None,
    config: Optional[Any] = None,
    title: str = "EXZECO Interactive Dashboard"
) -> Dict[str, Any]:
    """
    Convenience function to export a dashboard as HTML.
    
    Args:
        dashboard: The dashboard object to export
        output_dir: Directory where files will be saved
        filename: Name of the output HTML file
        study_bounds: Study area bounds tuple
        config: Configuration object with analysis parameters
        title: Title for the HTML document
        
    Returns:
        Dict with export results and metadata
    """
    exporter = DashboardExporter(output_dir)
    return exporter.export_dashboard(
        dashboard=dashboard,
        filename=filename,
        study_bounds=study_bounds,
        config=config,
        title=title
    )
