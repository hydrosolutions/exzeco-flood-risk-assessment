#!/usr/bin/env python
"""
Interactive Visualization Module for EXZECO Results
===================================================

This module provides interactive and static visualization tools for
EXZECO flood risk assessment results.

Windows Compatibility Notes:
- Uses pathlib.Path for all file operations
- Specifies encoding for text file operations  
- Avoids Unix-specific dependencies
- Uses cross-platform file handling

Key Components:
- ExzecoVisualizer: Main class for EXZECO analysis visualization
- StudyAreaVisualizer: Specialized class for study area visualization
- DEMVisualizer: Specialized class for DEM analysis and visualization
- create_study_area_visualization: Convenience function for study area display
- create_dem_visualization: Convenience function for DEM analysis

Usage:
    # For study area visualization
    from visualization import create_study_area_visualization
    results = create_study_area_visualization(study_area, area_km2)
    
    # For DEM visualization
    from visualization import create_dem_visualization
    fig = create_dem_visualization(dem_path)
    
    # For EXZECO results visualization
    from visualization import ExzecoVisualizer
    viz = ExzecoVisualizer(results, dem_path)
    interactive_map = viz.create_interactive_map()

Author: EXZECO Implementation
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
import rasterio
from rasterio.plot import show as rshow
from rasterio.warp import reproject, Resampling, calculate_default_transform
import ipywidgets as widgets
from IPython.display import display, HTML
import contextily as ctx
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import base64
from io import BytesIO
import logging

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def create_study_area_visualization(study_area, area_km2: float, 
                                   output_dir: Optional[Path] = None,
                                   show_static: bool = True, 
                                   show_interactive: bool = True) -> Dict:
    """
    Convenience function to create study area visualization.
    
    Parameters
    ----------
    study_area : StudyArea
        Study area object
    area_km2 : float
        Area in square kilometers
    output_dir : Path, optional
        Output directory to save maps
    show_static : bool
        Whether to show static matplotlib plot
    show_interactive : bool
        Whether to create interactive map
        
    Returns
    -------
    dict
        Dictionary containing created visualizations
    """
    visualizer = StudyAreaVisualizer(study_area, area_km2)
    return visualizer.create_complete_visualization(
        output_dir=output_dir,
        show_static=show_static,
        show_interactive=show_interactive
    )


class ExzecoVisualizer:
    """
    Comprehensive visualization class for EXZECO results.
    
    Provides:
    - Interactive maps with Folium
    - 3D visualizations with Plotly
    - Statistical plots
    - Comparative analysis
    - Animation capabilities
    """
    
    def __init__(self, results: Dict, dem_path: Optional[str] = None):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        results : dict
            EXZECO analysis results
        dem_path : str, optional
            Path to DEM file for background
        """
        self.results = results
        self.dem_path = dem_path
        self.dem_data = None
        self.transform = None
        self.crs = None
        
        if dem_path:
            self._load_dem()
        
        # Color schemes
        self.flood_colors = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', 
                             '#2c7fb8', '#253494']
        self.risk_colors = ['#00ff00', '#ffff00', '#ff8c00', '#ff0000']
        
    def _load_dem(self):
        """Load DEM data for visualization."""
        with rasterio.open(self.dem_path) as src:
            self.dem_data = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            self.resolution = src.res[0]
    
    def create_interactive_map(self, 
                              noise_level: str = 'exzeco_100cm',
                              include_layers: List[str] = None) -> folium.Map:
        """
        Create interactive Folium map with EXZECO results.
        
        Parameters
        ----------
        noise_level : str
            EXZECO result level to display
        include_layers : list
            Additional layers to include
            
        Returns
        -------
        folium.Map
            Interactive map
        """
        if noise_level not in self.results:
            raise ValueError(f"Noise level {noise_level} not in results")
        
        # Get center coordinates
        if self.bounds:
            center_lat = (self.bounds[1] + self.bounds[3]) / 2
            center_lon = (self.bounds[0] + self.bounds[2]) / 2
        else:
            center_lat, center_lon = 0, 0
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            control_scale=True,
            tiles=None
        )
        
        # Add base layers
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        
        # Add Stamen Terrain with proper attribution
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
            name='Terrain',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add CartoDB Dark Matter with proper attribution
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name='Dark',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add satellite imagery
        esri_sat = folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        )
        esri_sat.add_to(m)
        
        # Add DEM hillshade if available
        if self.dem_data is not None:
            self._add_hillshade_layer(m)
        
        # Add EXZECO flood zones
        self._add_flood_zones(m, noise_level)
        
        # Add additional layers if specified
        if include_layers:
            for layer in include_layers:
                if layer == 'drainage':
                    self._add_drainage_network(m)
                elif layer == 'contours':
                    self._add_contour_lines(m)
                elif layer == 'risk_zones':
                    self._add_risk_classification(m)
        
        # Add drawing tools
        draw = plugins.Draw(
            export=True,
            filename='exzeco_analysis.geojson',
            position='topleft',
            draw_options={
                'polygon': True,
                'polyline': True,
                'rectangle': True,
                'circle': False,
                'marker': True,
                'circlemarker': False
            }
        )
        draw.add_to(m)
        
        # Add measurement tool
        plugins.MeasureControl(position='topright').add_to(m)
        
        # Add minimap
        minimap = plugins.MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # Add fullscreen button
        plugins.Fullscreen(position='topright').add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # Add legend
        self._add_legend(m, noise_level)
        
        return m
    
    def _add_flood_zones(self, m: folium.Map, noise_level: str):
        """Add flood zone layer to map."""
        data = self.results[noise_level]
        prob_map = data['probability_map']
        
        # Convert probability map to RGB image
        rgba = self._probability_to_rgba(prob_map)
        
        # Convert to base64 for overlay
        img = self._array_to_image(rgba)
        
        # Add as image overlay
        folium.raster_layers.ImageOverlay(
            image=img,
            bounds=[[self.bounds[1], self.bounds[0]], 
                   [self.bounds[3], self.bounds[2]]],
            opacity=0.7,
            name=f'Flood Zones ({noise_level})',
            overlay=True,
            control=True,
            zindex=1
        ).add_to(m)
    
    def _add_hillshade_layer(self, m: folium.Map):
        """Add hillshade layer from DEM."""
        from scipy import ndimage
        
        # Calculate hillshade
        dy, dx = np.gradient(self.dem_data)
        slope = np.pi/2. - np.arctan(np.sqrt(dx*dx + dy*dy))
        aspect = np.arctan2(-dx, dy)
        
        azimuth = 315 * np.pi / 180.
        altitude = 45 * np.pi / 180.
        
        shaded = np.sin(altitude) * np.sin(slope) + \
                 np.cos(altitude) * np.cos(slope) * \
                 np.cos((azimuth - np.pi/2.) - aspect)
        
        # Normalize to 0-255
        hillshade = ((shaded + 1) * 127.5).astype(np.uint8)
        
        # Convert to RGBA
        rgba = np.zeros((*hillshade.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = hillshade[:, :, np.newaxis]
        rgba[:, :, 3] = 200  # Semi-transparent
        
        # Add to map
        img = self._array_to_image(rgba)
        
        folium.raster_layers.ImageOverlay(
            image=img,
            bounds=[[self.bounds[1], self.bounds[0]], 
                   [self.bounds[3], self.bounds[2]]],
            opacity=0.5,
            name='Hillshade',
            overlay=True,
            control=True,
            zindex=0
        ).add_to(m)
    
    def _add_legend(self, m: folium.Map, noise_level: str):
        """Add legend to map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 250px; height: auto;
                    background-color: white; z-index: 9999; font-size: 14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <h4 style="margin-top: 0;">EXZECO Flood Risk</h4>
        <p style="margin: 5px;">{}</p>
        <div style="margin: 5px;">
            <div style="display: flex; align-items: center; margin: 2px;">
                <div style="width: 20px; height: 20px; background: #253494; margin-right: 5px;"></div>
                <span>Very High Risk (>80%)</span>
            </div>
            <div style="display: flex; align-items: center; margin: 2px;">
                <div style="width: 20px; height: 20px; background: #2c7fb8; margin-right: 5px;"></div>
                <span>High Risk (60-80%)</span>
            </div>
            <div style="display: flex; align-items: center; margin: 2px;">
                <div style="width: 20px; height: 20px; background: #41b6c4; margin-right: 5px;"></div>
                <span>Medium Risk (40-60%)</span>
            </div>
            <div style="display: flex; align-items: center; margin: 2px;">
                <div style="width: 20px; height: 20px; background: #7fcdbb; margin-right: 5px;"></div>
                <span>Low Risk (20-40%)</span>
            </div>
            <div style="display: flex; align-items: center; margin: 2px;">
                <div style="width: 20px; height: 20px; background: #c7e9b4; margin-right: 5px;"></div>
                <span>Very Low Risk (<20%)</span>
            </div>
        </div>
        </div>
        '''.format(noise_level.replace('_', ' ').title())
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_3d_visualization(self, noise_level: str = 'exzeco_100cm') -> go.Figure:
        """
        Create 3D surface plot with EXZECO results.
        
        Parameters
        ----------
        noise_level : str
            EXZECO result level to display
            
        Returns
        -------
        plotly.graph_objects.Figure
            3D visualization
        """
        if self.dem_data is None:
            raise ValueError("DEM data required for 3D visualization")
        
        data = self.results[noise_level]
        prob_map = data['probability_map']
        
        # Downsample for performance
        step = max(1, min(self.dem_data.shape) // 100)
        dem_ds = self.dem_data[::step, ::step]
        prob_ds = prob_map[::step, ::step]
        
        # Create coordinate grids
        ny, nx = dem_ds.shape
        x = np.arange(nx) * step * self.resolution
        y = np.arange(ny) * step * self.resolution
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Add DEM surface
        fig.add_trace(go.Surface(
            x=x, y=y, z=dem_ds,
            colorscale='earth',
            showscale=False,
            name='Terrain',
            opacity=0.9,
            contours={
                "z": {"show": True, "start": dem_ds.min(), "end": dem_ds.max(), 
                      "size": 50, "color": "white", "width": 1}
            }
        ))
        
        # Add flood risk overlay
        flood_surface = np.where(prob_ds > 0.5, dem_ds + 5, np.nan)
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=flood_surface,
            colorscale=[[0, 'blue'], [0.5, 'cyan'], [1, 'red']],
            showscale=True,
            name='Flood Risk',
            opacity=0.7,
            colorbar=dict(
                title="Flood<br>Probability",
                x=1.0,
                xpad=20
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Flood Risk Visualization - {noise_level}',
            scene=dict(
                xaxis_title='Distance East (m)',
                yaxis_title='Distance North (m)',
                zaxis_title='Elevation (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.3)
            ),
            height=700,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    
    def create_comparison_plot(self, levels: Optional[List[str]] = None) -> go.Figure:
        """
        Create comparison plot for different noise levels.
        
        Parameters
        ----------
        levels : list, optional
            Noise levels to compare
            
        Returns
        -------
        plotly.graph_objects.Figure
            Comparison visualization
        """
        if levels is None:
            levels = list(self.results.keys())
        
        # Calculate statistics for each level
        stats = []
        for level in levels:
            data = self.results[level]
            prob_map = data['probability_map']
            
            stats.append({
                'Level': level.replace('exzeco_', '').replace('cm', ''),
                'Mean Probability': np.nanmean(prob_map),
                'Max Probability': np.nanmax(prob_map),
                'Flood Area (%)': 100 * np.sum(prob_map > 0.5) / prob_map.size,
                'High Risk Area (%)': 100 * np.sum(prob_map > 0.8) / prob_map.size
            })
        
        df = pd.DataFrame(stats)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Flood Probability', 'Flood Area Coverage',
                          'Risk Distribution', 'Cumulative Risk'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                  [{'type': 'box'}, {'type': 'scatter'}]]
        )
        
        # Mean probability trend
        fig.add_trace(
            go.Scatter(x=df['Level'], y=df['Mean Probability'],
                      mode='lines+markers', name='Mean Probability',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Flood area bar chart
        fig.add_trace(
            go.Bar(x=df['Level'], y=df['Flood Area (%)'],
                  name='Flood Area', marker_color='lightblue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=df['Level'], y=df['High Risk Area (%)'],
                  name='High Risk Area', marker_color='red'),
            row=1, col=2
        )
        
        # Risk distribution box plots
        for level in levels:
            prob_map = self.results[level]['probability_map']
            flood_probs = prob_map[prob_map > 0.1].flatten()
            
            fig.add_trace(
                go.Box(y=flood_probs, name=level.replace('exzeco_', ''),
                      boxpoints='outliers'),
                row=2, col=1
            )
        
        # Cumulative risk
        fig.add_trace(
            go.Scatter(x=df['Level'], y=df['Flood Area (%)'].cumsum(),
                      mode='lines+markers', name='Cumulative Area',
                      line=dict(color='green', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='EXZECO Multi-Level Comparison',
            showlegend=True,
            height=800,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text='Noise Level', row=1, col=1)
        fig.update_xaxes(title_text='Noise Level', row=1, col=2)
        fig.update_xaxes(title_text='Noise Level', row=2, col=1)
        fig.update_xaxes(title_text='Noise Level', row=2, col=2)
        
        fig.update_yaxes(title_text='Probability', row=1, col=1)
        fig.update_yaxes(title_text='Area (%)', row=1, col=2)
        fig.update_yaxes(title_text='Probability', row=2, col=1)
        fig.update_yaxes(title_text='Cumulative Area (%)', row=2, col=2)
        
        return fig
    
    def create_risk_heatmap(self, noise_level: str = 'exzeco_100cm') -> go.Figure:
        """
        Create risk heatmap visualization.
        
        Parameters
        ----------
        noise_level : str
            EXZECO result level
            
        Returns
        -------
        plotly.graph_objects.Figure
            Heatmap visualization
        """
        data = self.results[noise_level]
        prob_map = data['probability_map']
        
        # Downsample for visualization
        step = max(1, min(prob_map.shape) // 200)
        prob_ds = prob_map[::step, ::step]
        
        fig = go.Figure(data=go.Heatmap(
            z=prob_ds,
            colorscale='RdYlBu_r',
            colorbar=dict(title='Flood Probability'),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Probability: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Flood Risk Heatmap - {noise_level}',
            xaxis_title='East (pixels)',
            yaxis_title='North (pixels)',
            height=600,
            yaxis=dict(scaleanchor='x', scaleratio=1)
        )
        
        return fig
    
    def create_interactive_dashboard(self) -> widgets.VBox:
        """
        Create interactive Jupyter dashboard.
        
        Returns
        -------
        ipywidgets.VBox
            Interactive dashboard
        """
        # Create widgets
        level_dropdown = widgets.Dropdown(
            options=list(self.results.keys()),
            value=list(self.results.keys())[-1],
            description='Noise Level:',
            style={'description_width': 'initial'}
        )
        
        viz_type = widgets.RadioButtons(
            options=['2D Map', '3D Surface', 'Heatmap', 'Statistics'],
            value='2D Map',
            description='Visualization:',
            style={'description_width': 'initial'}
        )
        
        threshold_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Risk Threshold:',
            style={'description_width': 'initial'}
        )
        
        opacity_slider = widgets.FloatSlider(
            value=0.7,
            min=0.0,
            max=1.0,
            step=0.1,
            description='Opacity:',
            style={'description_width': 'initial'}
        )
        
        output = widgets.Output()
        
        def update_viz(*args):
            with output:
                output.clear_output(wait=True)
                
                if viz_type.value == '2D Map':
                    m = self.create_interactive_map(level_dropdown.value)
                    display(m)
                elif viz_type.value == '3D Surface':
                    fig = self.create_3d_visualization(level_dropdown.value)
                    fig.show()
                elif viz_type.value == 'Heatmap':
                    fig = self.create_risk_heatmap(level_dropdown.value)
                    fig.show()
                else:  # Statistics
                    self.plot_statistics(level_dropdown.value)
        
        # Link widgets
        level_dropdown.observe(update_viz, 'value')
        viz_type.observe(update_viz, 'value')
        threshold_slider.observe(update_viz, 'value')
        
        # Initial display
        update_viz()
        
        # Create layout
        controls = widgets.VBox([
            widgets.HTML('<h3>EXZECO Analysis Dashboard</h3>'),
            level_dropdown,
            viz_type,
            threshold_slider,
            opacity_slider
        ])
        
        dashboard = widgets.HBox([controls, output])
        
        return widgets.VBox([
            widgets.HTML('<h2>Interactive EXZECO Visualization</h2>'),
            dashboard
        ])
    
    def plot_statistics(self, noise_level: str):
        """Plot statistical analysis."""
        data = self.results[noise_level]
        prob_map = data['probability_map']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Check if we have any meaningful flood probabilities
        has_floods = np.any(prob_map > 0.01)
        
        # Probability distribution
        ax = axes[0, 0]
        if has_floods:
            flood_probs = prob_map[prob_map > 0.01].flatten()
            ax.hist(flood_probs, bins=50, edgecolor='black', alpha=0.7)
        else:
            # Show all probabilities if no significant floods
            flood_probs = prob_map.flatten()
            ax.hist(flood_probs, bins=50, edgecolor='black', alpha=0.7)
            ax.text(0.5, 0.5, 'No significant\nflood probabilities\ndetected', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Flood Probability')
        ax.set_ylabel('Frequency')
        ax.set_title('Probability Distribution')
        ax.grid(True, alpha=0.3)
        
        # Cumulative distribution
        ax = axes[0, 1]
        if has_floods and len(flood_probs) > 0:
            sorted_probs = np.sort(flood_probs)
            cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
            ax.plot(sorted_probs, cumulative, 'b-', linewidth=2)
        else:
            ax.text(0.5, 0.5, 'No flood data\nfor cumulative\ndistribution', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Flood Probability')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Cumulative Distribution')
        ax.grid(True, alpha=0.3)
        
        # Risk categories pie chart
        ax = axes[1, 0]
        categories = ['Very Low (<0.2)', 'Low (0.2-0.4)', 'Medium (0.4-0.6)', 
                     'High (0.6-0.8)', 'Very High (>0.8)']
        counts = [
            np.sum((prob_map > 0.01) & (prob_map < 0.2)),
            np.sum((prob_map >= 0.2) & (prob_map < 0.4)),
            np.sum((prob_map >= 0.4) & (prob_map < 0.6)),
            np.sum((prob_map >= 0.6) & (prob_map < 0.8)),
            np.sum(prob_map >= 0.8)
        ]
        colors = ['#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494']
        
        # Check if we have any non-zero counts
        total_count = sum(counts)
        if total_count > 0:
            # Filter out zero counts for pie chart
            non_zero_counts = []
            non_zero_categories = []
            non_zero_colors = []
            for i, (count, cat, color) in enumerate(zip(counts, categories, colors)):
                if count > 0:
                    non_zero_counts.append(count)
                    non_zero_categories.append(cat)
                    non_zero_colors.append(color)
            
            if non_zero_counts:
                ax.pie(non_zero_counts, labels=non_zero_categories, colors=non_zero_colors, autopct='%1.1f%%')
            else:
                ax.text(0.5, 0.5, 'No risk\ncategories\ndetected', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No flood risk\ndetected', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        ax.set_title('Risk Category Distribution')
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
        Summary Statistics - {noise_level}
        {'='*40}
        Total Pixels: {prob_map.size:,}
        Flood Pixels: {np.sum(prob_map > 0.5):,}
        
        Mean Probability: {np.nanmean(prob_map):.3f}
        Median Probability: {np.nanmedian(prob_map):.3f}
        Max Probability: {np.nanmax(prob_map):.3f}
        Std Deviation: {np.nanstd(prob_map):.3f}
        
        Area Statistics:
        Total Area: {prob_map.size * (self.resolution**2) / 1e6:.2f} km²
        Flood Area: {np.sum(prob_map > 0.5) * (self.resolution**2) / 1e6:.2f} km²
        High Risk Area: {np.sum(prob_map > 0.8) * (self.resolution**2) / 1e6:.2f} km²
        """
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Statistical Analysis - {noise_level}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _probability_to_rgba(self, prob_map: np.ndarray) -> np.ndarray:
        """Convert probability map to RGBA image."""
        # Create custom colormap
        colors = ['#ffffff00', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('flood', colors, N=n_bins)
        
        # Normalize and apply colormap
        norm = mcolors.Normalize(vmin=0, vmax=1)
        rgba = cmap(norm(prob_map))
        
        # Set transparency for low values
        rgba[:, :, 3] = np.where(prob_map > 0.1, prob_map, 0)
        
        return (rgba * 255).astype(np.uint8)
    
    def _array_to_image(self, array: np.ndarray) -> str:
        """Convert numpy array to base64 image string."""
        from PIL import Image
        
        img = Image.fromarray(array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _add_drainage_network(self, m: folium.Map):
        """Add drainage network layer to map (placeholder)."""
        # This is a placeholder - in a real implementation, you would
        # compute and add actual drainage networks
        logger.info("Drainage network layer not implemented - placeholder added")
        
    def _add_contour_lines(self, m: folium.Map):
        """Add contour lines layer to map (placeholder)."""
        # This is a placeholder - in a real implementation, you would
        # generate contour lines from the DEM
        logger.info("Contour lines layer not implemented - placeholder added")
        
    def _add_risk_classification(self, m: folium.Map):
        """Add risk classification layer to map (placeholder)."""
        # This is a placeholder - in a real implementation, you would
        # add classified risk zones
        logger.info("Risk classification layer not implemented - placeholder added")
    
    def export_visualizations(self, output_dir: Union[str, Path], formats: List[str] = None):
        """
        Export all visualizations to files.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        formats : list
            Export formats ('html', 'png', 'pdf')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = ['html', 'png']
        
        logger.info(f"Exporting visualizations to {output_dir}")
        
        # Export for each noise level
        for level in self.results.keys():
            # Interactive map
            if 'html' in formats:
                m = self.create_interactive_map(level)
                m.save(str(output_dir / f"map_{level}.html"))
            
            # 3D visualization
            if self.dem_data is not None:
                fig_3d = self.create_3d_visualization(level)
                if 'html' in formats:
                    fig_3d.write_html(str(output_dir / f"3d_{level}.html"))
                if 'png' in formats:
                    fig_3d.write_image(str(output_dir / f"3d_{level}.png"))
            
            # Heatmap
            fig_heat = self.create_risk_heatmap(level)
            if 'html' in formats:
                fig_heat.write_html(str(output_dir / f"heatmap_{level}.html"))
            if 'png' in formats:
                fig_heat.write_image(str(output_dir / f"heatmap_{level}.png"))
        
        # Comparison plot
        fig_comp = self.create_comparison_plot()
        if 'html' in formats:
            fig_comp.write_html(str(output_dir / "comparison.html"))
        if 'png' in formats:
            fig_comp.write_image(str(output_dir / "comparison.png"))
        
        logger.info(f"Visualizations exported successfully")


class DEMVisualizer:
    """
    Specialized visualizer for Digital Elevation Model (DEM) data.
    
    Handles DEM analysis and visualization including hillshade, slope, and aspect calculations.
    """
    
    def __init__(self, dem_path: Union[str, Path]):
        """
        Initialize DEM visualizer.
        
        Parameters
        ----------
        dem_path : str or Path
            Path to the DEM file
        """
        self.dem_path = Path(dem_path)
        self.dem_data = None
        self.dem_stats = None
        self.hillshade = None
        self.slope = None
        self.aspect = None
        
        if self.dem_path.exists():
            self._load_dem_data()
    
    def _load_dem_data(self):
        """Load DEM data and calculate basic statistics."""
        with rasterio.open(self.dem_path) as src:
            self.dem_data = src.read(1).astype(float)
            # Handle nodata values
            if src.nodata is not None:
                self.dem_data[self.dem_data == src.nodata] = np.nan
            
            self.dem_stats = {
                'min_elevation': np.nanmin(self.dem_data),
                'max_elevation': np.nanmax(self.dem_data),
                'mean_elevation': np.nanmean(self.dem_data),
                'std_elevation': np.nanstd(self.dem_data),
                'shape': self.dem_data.shape,
                'crs': str(src.crs),
                'bounds': src.bounds
            }
    
    def calculate_terrain_derivatives(self):
        """
        Calculate hillshade, slope, and aspect from DEM data.
        
        Returns
        -------
        dict
            Dictionary containing hillshade, slope, and aspect arrays
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded")
        
        try:
            # Try using earthpy if available
            import earthpy.spatial as es
            self.hillshade = es.hillshade(self.dem_data, azimuth=315, altitude=45)
            self.slope = es.slope(self.dem_data)
            self.aspect = es.aspect(self.dem_data)
            print("✅ Terrain derivatives calculated using earthpy")
            
        except ImportError:
            print("⚠️  earthpy not available, using numpy gradient method")
            # Fallback to simple gradient method
            self._calculate_simple_derivatives()
        except Exception as e:
            print(f"⚠️  earthpy calculation failed: {e}")
            print("Falling back to simple gradient method...")
            self._calculate_simple_derivatives()
        
        return {
            'hillshade': self.hillshade,
            'slope': self.slope,
            'aspect': self.aspect
        }
    
    def _calculate_simple_derivatives(self):
        """Calculate simple terrain derivatives using numpy gradients."""
        # Simple hillshade calculation
        dy, dx = np.gradient(self.dem_data)
        slope = np.pi/2. - np.arctan(np.sqrt(dx*dx + dy*dy))
        aspect = np.arctan2(-dx, dy)
        
        # Calculate hillshade
        azimuth = 315 * np.pi / 180.
        altitude = 45 * np.pi / 180.
        
        self.hillshade = (np.sin(altitude) * np.sin(slope) + 
                         np.cos(altitude) * np.cos(slope) * 
                         np.cos((azimuth - np.pi/2.) - aspect))
        
        # Normalize hillshade to 0-1
        self.hillshade = (self.hillshade + 1) / 2
        
        # Calculate slope and aspect
        self.slope = np.sqrt(dx*dx + dy*dy)
        self.aspect = np.arctan2(-dx, dy)
    
    def create_dem_analysis_plot(self, figsize: Tuple[int, int] = (15, 15)) -> plt.Figure:
        """
        Create comprehensive DEM analysis plot with hillshade, elevation, slope, and aspect.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
            
        Returns
        -------
        matplotlib.figure.Figure
            DEM analysis figure
        """
        if self.dem_data is None:
            raise ValueError("DEM data not loaded")
        
        # Calculate terrain derivatives if not already done
        if self.hillshade is None:
            self.calculate_terrain_derivatives()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # DEM Hillshade
        ax = axes[0, 0]
        im1 = ax.imshow(self.hillshade, cmap='gray', alpha=0.8)
        ax.set_title('DEM Hillshade')
        ax.set_xlabel('Pixels East')
        ax.set_ylabel('Pixels North')
        ax.grid(True, alpha=0.3)
        
        # Elevation
        ax = axes[0, 1]
        im2 = ax.imshow(self.dem_data, cmap='terrain')
        ax.set_title(f'Elevation ({self.dem_stats["min_elevation"]:.0f} - {self.dem_stats["max_elevation"]:.0f} m)')
        plt.colorbar(im2, ax=ax, label='Elevation (m)')
        ax.grid(True, alpha=0.3)
        
        # Slope
        ax = axes[1, 0]
        im3 = ax.imshow(self.slope, cmap='YlOrRd', vmax=np.nanpercentile(self.slope, 95))
        ax.set_title('Slope')
        plt.colorbar(im3, ax=ax, label='Slope')
        ax.grid(True, alpha=0.3)
        
        # Aspect
        ax = axes[1, 1]
        im4 = ax.imshow(self.aspect, cmap='hsv')
        ax.set_title('Aspect')
        plt.colorbar(im4, ax=ax, label='Aspect')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('DEM Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def print_dem_statistics(self):
        """Print DEM statistics to console."""
        if self.dem_stats is None:
            print("❌ DEM statistics not available")
            return
        
        print(f"DEM file confirmed at: {self.dem_path}")
        print("\nDEM Statistics:")
        for key, value in self.dem_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    def visualize_dem(self, show_plot: bool = True, save_path: Optional[Path] = None) -> Optional[plt.Figure]:
        """
        Complete DEM visualization workflow.
        
        Parameters
        ----------
        show_plot : bool
            Whether to display the plot
        save_path : Path, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure or None
            DEM analysis figure if successful
        """
        if not self.dem_path.exists():
            print("❌ Cannot visualize DEM - file does not exist")
            return None
        
        try:
            # Print statistics
            self.print_dem_statistics()
            
            # Create analysis plot
            fig = self.create_dem_analysis_plot()
            
            if show_plot:
                plt.show()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ DEM analysis plot saved to {save_path}")
            
            print(f"\n✅ DEM visualization complete!")
            print(f"DEM file saved at: {self.dem_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Error during DEM visualization: {e}")
            return None


def create_dem_visualization(dem_path: Union[str, Path], 
                           show_plot: bool = True,
                           save_path: Optional[Path] = None) -> Optional[plt.Figure]:
    """
    Convenience function to create DEM visualization.
    
    Parameters
    ----------
    dem_path : str or Path
        Path to the DEM file
    show_plot : bool
        Whether to display the plot
    save_path : Path, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure or None
        DEM analysis figure if successful
    """
    visualizer = DEMVisualizer(dem_path)
    return visualizer.visualize_dem(show_plot=show_plot, save_path=save_path)


class StudyAreaVisualizer:
    """
    Specialized visualizer for study area display and analysis.
    """
    
    def __init__(self, study_area, area_km2: float):
        """
        Initialize study area visualizer.
        
        Parameters
        ----------
        study_area : StudyArea
            Study area object
        area_km2 : float
            Area in square kilometers
        """
        self.study_area = study_area
        self.area_km2 = area_km2
        self.study_bounds = study_area.bounds
    
    def create_static_visualization(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create static matplotlib visualization of study area with topographic context.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
            
        Returns
        -------
        matplotlib.figure.Figure
            Study area visualization
        """
        import contextily as ctx
        from pyproj import Transformer
        import matplotlib.ticker as ticker
        
        print(f"Study Area Configuration:")
        print(f"  Bounds: {self.study_bounds}")
        print(f"  Area: {self.area_km2:.2f} km²")
        print(f"  Center: ({np.mean(self.study_bounds[::2]):.4f}, {np.mean(self.study_bounds[1::2]):.4f})")
        
        # Create study area geodataframe
        study_gdf = self.study_area.to_geopandas()
        
        # Convert to Web Mercator for contextily
        study_gdf_web = study_gdf.to_crs(epsg=3857)
        
        # Create the plot with background map
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        try:
            # Use very transparent fill for the study area
            study_gdf_web.plot(ax=ax, alpha=0.15, edgecolor='red', facecolor='yellow', linewidth=3)
            
            # Add topographic background map
            ctx.add_basemap(ax, crs=study_gdf_web.crs.to_string(), 
                            source=ctx.providers.Esri.WorldTopoMap,
                            attribution='© Esri, USGS, NOAA')
            
            ax.set_title('Study Area - Topographic Context', fontsize=14, fontweight='bold')
            
            # Convert Web Mercator bounds back to lat/lon for axis labels
            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            
            # Get current axis limits in Web Mercator
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Create custom tick labels showing lat/lon
            def format_longitude(x, pos):
                lon, _ = transformer.transform(x, (ylim[0] + ylim[1])/2)
                return f'{lon:.3f}°E'
            
            def format_latitude(y, pos):
                _, lat = transformer.transform((xlim[0] + xlim[1])/2, y)
                return f'{lat:.3f}°N'
            
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_longitude))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_latitude))
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Add area information in a nice box
            info_text = f'Area: {self.area_km2:.2f} km²\nBounds: {self.study_bounds}\nCenter: ({np.mean(self.study_bounds[::2]):.4f}°E, {np.mean(self.study_bounds[1::2]):.4f}°N)'
            ax.text(0.02, 0.98, info_text, 
                     transform=ax.transAxes, ha='left', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'),
                     fontsize=10)
            
            print("✅ Study area plotted with topographic background")
            
        except Exception as e:
            print(f"⚠️  Could not add topographic map, trying OpenStreetMap: {e}")
            try:
                # Fallback to OpenStreetMap
                study_gdf_web.plot(ax=ax, alpha=0.15, edgecolor='red', facecolor='yellow', linewidth=3)
                
                ctx.add_basemap(ax, crs=study_gdf_web.crs.to_string(), 
                                source=ctx.providers.OpenStreetMap.Mapnik,
                                attribution='© OpenStreetMap contributors')
                
                ax.set_title('Study Area - Geographic Context', fontsize=14, fontweight='bold')
                
                # Convert Web Mercator bounds back to lat/lon for axis labels
                transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
                
                # Get current axis limits in Web Mercator
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # Create custom tick labels showing lat/lon
                def format_longitude(x, pos):
                    lon, _ = transformer.transform(x, (ylim[0] + ylim[1])/2)
                    return f'{lon:.3f}°E'
                
                def format_latitude(y, pos):
                    _, lat = transformer.transform((xlim[0] + xlim[1])/2, y)
                    return f'{lat:.3f}°N'
                
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_longitude))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_latitude))
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                
                # Add area information
                info_text = f'Area: {self.area_km2:.2f} km²\nBounds: {self.study_bounds}\nCenter: ({np.mean(self.study_bounds[::2]):.4f}°E, {np.mean(self.study_bounds[1::2]):.4f}°N)'
                ax.text(0.02, 0.98, info_text, 
                         transform=ax.transAxes, ha='left', va='top',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'),
                         fontsize=10)
                
                print("✅ Study area plotted with OpenStreetMap background")
                
            except Exception as e2:
                print(f"⚠️  Could not add any background map: {e2}")
                print("Falling back to simple plot...")
                
                # Fallback to simple plot with lat/lon coordinates
                study_gdf.plot(ax=ax, alpha=0.15, edgecolor='red', facecolor='lightblue', linewidth=2)
                ax.set_title('Study Area - Simple View', fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude (°E)', fontsize=12)
                ax.set_ylabel('Latitude (°N)', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add coordinate labels for fallback
                ax.text(self.study_bounds[0], self.study_bounds[1], f'SW\n({self.study_bounds[0]:.2f}°E, {self.study_bounds[1]:.2f}°N)', 
                         ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(self.study_bounds[2], self.study_bounds[3], f'NE\n({self.study_bounds[2]:.2f}°E, {self.study_bounds[3]:.2f}°N)', 
                         ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def create_interactive_map(self, output_dir: Optional[Path] = None) -> folium.Map:
        """
        Create interactive Folium map of study area.
        
        Parameters
        ----------
        output_dir : Path, optional
            Directory to save the map
            
        Returns
        -------
        folium.Map
            Interactive study area map
        """
        print("\nCreating interactive study area map...")
        
        try:
            # Calculate center
            center_lat = (self.study_bounds[1] + self.study_bounds[3]) / 2
            center_lon = (self.study_bounds[0] + self.study_bounds[2]) / 2
            
            # Create map with topographic tiles
            study_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles=None  # We'll add custom tiles
            )
            
            # Add topographic tile layer
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
                attr='© Esri, USGS, NOAA',
                name='Topographic',
                overlay=False,
                control=True
            ).add_to(study_map)
            
            # Add OpenStreetMap as alternative
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='OpenStreetMap',
                overlay=False,
                control=True
            ).add_to(study_map)
            
            # Add study area as rectangle with more transparency
            folium.Rectangle(
                bounds=[[self.study_bounds[1], self.study_bounds[0]], 
                        [self.study_bounds[3], self.study_bounds[2]]],
                color='red',
                weight=3,
                fill=True,
                fillColor='yellow',
                fillOpacity=0.1,  # Much more transparent
                popup=f'Study Area<br>Area: {self.area_km2:.2f} km²<br>Bounds: {self.study_bounds[0]:.4f}°E to {self.study_bounds[2]:.4f}°E<br>{self.study_bounds[1]:.4f}°N to {self.study_bounds[3]:.4f}°N'
            ).add_to(study_map)
            
            # Add center marker
            folium.Marker(
                [center_lat, center_lon],
                popup=f'Study Area Center<br>({center_lon:.4f}°E, {center_lat:.4f}°N)',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(study_map)
            
            # Add corner markers with lat/lon labels
            corners = [
                ([self.study_bounds[1], self.study_bounds[0]], f'SW Corner<br>({self.study_bounds[0]:.4f}°E, {self.study_bounds[1]:.4f}°N)'),
                ([self.study_bounds[1], self.study_bounds[2]], f'SE Corner<br>({self.study_bounds[2]:.4f}°E, {self.study_bounds[1]:.4f}°N)'),
                ([self.study_bounds[3], self.study_bounds[0]], f'NW Corner<br>({self.study_bounds[0]:.4f}°E, {self.study_bounds[3]:.4f}°N)'),
                ([self.study_bounds[3], self.study_bounds[2]], f'NE Corner<br>({self.study_bounds[2]:.4f}°E, {self.study_bounds[3]:.4f}°N)')
            ]
            
            for (lat, lon), label in corners:
                folium.CircleMarker(
                    [lat, lon],
                    radius=5,
                    popup=label,
                    color='red',
                    fill=True,
                    fillColor='red'
                ).add_to(study_map)
            
            # Add layer control
            folium.LayerControl().add_to(study_map)
            
            # Save the map if output directory provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                map_path = output_dir / 'study_area_map.html'
                study_map.save(str(map_path))  # Convert Path to string for Windows compatibility
                print(f"✅ Interactive study area map saved to {map_path}")
            
            return study_map
            
        except Exception as e:
            print(f"⚠️  Could not create interactive map: {e}")
            return None
    
    def display_trust_instructions(self, output_dir: Path):
        """
        Display instructions for trusting the notebook to show interactive content.
        
        Parameters
        ----------
        output_dir : Path
            Output directory where map is saved
        """
        print("\n" + "="*60)
        print("INTERACTIVE MAP DISPLAY INSTRUCTIONS")
        print("="*60)
        print("If the folium map above doesn't show properly:")
        print()
        print("📋 VS Code Trust Methods:")
        print("   1. Press Cmd+Shift+P → search 'Notebook: Trust'")
        print("   2. Look for a 🛡️ shield icon in the notebook toolbar")
        print("   3. Check for yellow/orange trust notification bar")
        print()
        print("📁 Alternative - Open Saved Map:")
        print(f"   File saved at: {output_dir / 'study_area_map.html'}")
        print("   Double-click to open in your browser")
        print()
        print("⚙️ VS Code Settings:")
        print("   - Open Settings (Cmd+,)")
        print("   - Search 'notebook trust'")
        print("   - Enable 'notebook.trustNotebooks'")
        print("="*60)
    
    def create_complete_visualization(self, output_dir: Optional[Path] = None, 
                                    show_static: bool = True, 
                                    show_interactive: bool = True) -> Dict:
        """
        Create complete study area visualization with both static and interactive components.
        
        Parameters
        ----------
        output_dir : Path, optional
            Output directory to save maps
        show_static : bool
            Whether to show static matplotlib plot
        show_interactive : bool
            Whether to create interactive map
            
        Returns
        -------
        dict
            Dictionary containing created visualizations
        """
        results = {}
        
        if show_static:
            # Create and show static plot
            fig = self.create_static_visualization()
            plt.show()
            results['static_figure'] = fig
        
        if show_interactive:
            # Create interactive map
            if output_dir is None:
                output_dir = Path('../data/outputs')
            
            interactive_map = self.create_interactive_map(output_dir)
            if interactive_map:
                # Display the map
                from IPython.display import display
                display(interactive_map)
                results['interactive_map'] = interactive_map
                
                # Show trust instructions
                self.display_trust_instructions(output_dir)
        
        # Print summary
        print(f"\n✅ Study area visualization completed!")
        print(f"   Total area: {self.area_km2:.2f} km²")
        print(f"   Bounding box: {self.study_bounds[0]:.4f}°E to {self.study_bounds[2]:.4f}°E, {self.study_bounds[1]:.4f}°N to {self.study_bounds[3]:.4f}°N")
        
        return results


if __name__ == "__main__":
    print("Visualization module loaded successfully")