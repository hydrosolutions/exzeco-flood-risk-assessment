#!/usr/bin/env python3
"""
Dashboard Info Web Viewer

Creates a simple web page to view dashboard metadata in a browser.
Usage: python dashboard_web_viewer.py [port]
"""

import json
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
import webbrowser
import threading
import time


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for dashboard metadata."""
    
    def do_GET(self):
        """Handle GET requests."""
        
        if self.path == '/' or self.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Generate HTML page
            html_content = self.generate_dashboard_page()
            self.wfile.write(html_content.encode())
            
        elif self.path == '/metadata.json':
            # Serve the JSON file
            json_file = Path("data/outputs/dashboard_metadata.json")
            if json_file.exists():
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                with open(json_file, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Metadata file not found")
        else:
            super().do_GET()
    
    def generate_dashboard_page(self):
        """Generate HTML page with dashboard metadata."""
        
        # Try to read metadata
        json_file = Path("data/outputs/dashboard_metadata.json")
        metadata = {}
        
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                metadata = {"error": f"Could not read metadata: {e}"}
        
        # Build content sections
        basic_info = self.build_basic_info(metadata)
        export_info = self.build_export_info(metadata)
        study_area_info = self.build_study_area_info(metadata)
        analysis_info = self.build_analysis_info(metadata)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EXZECO Dashboard Metadata Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .card h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        
        .info-item {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            color: #6c757d;
        }}
        
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        
        .json-viewer {{
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }}
        
        .refresh-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            margin: 10px 0;
        }}
        
        .refresh-btn:hover {{
            background: #5a6fd8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 EXZECO Dashboard Metadata</h1>
            <p>Analysis Information & Export Details</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        </div>
        
        <div class="content">
            {basic_info}
            {export_info}
            {study_area_info}
            {analysis_info}
            
            <div class="card">
                <h3>📄 Raw JSON Data</h3>
                <div class="json-viewer">
                    <pre>{json.dumps(metadata, indent=2)}</pre>
                </div>
            </div>
            
            <div class="card">
                <h3>🔗 Quick Actions</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">View Dashboard</div>
                        <div class="info-value">
                            <a href="../data/outputs/exzeco_interactive_dashboard.html" target="_blank">
                                Open HTML Dashboard
                            </a>
                        </div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Download Metadata</div>
                        <div class="info-value">
                            <a href="/metadata.json" download="dashboard_metadata.json">
                                Download JSON File
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def build_basic_info(self, metadata):
        """Build basic information section."""
        
        title = metadata.get('title', 'Unknown')
        created = metadata.get('created', 'Unknown')
        dashboard_file = metadata.get('dashboard_file', 'Unknown')
        
        if created != 'Unknown':
            try:
                created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                created = created_dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        return f"""
        <div class="card">
            <h3>📊 Basic Information</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Title</div>
                    <div class="info-value">{title}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Created</div>
                    <div class="info-value">{created}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Dashboard File</div>
                    <div class="info-value">{dashboard_file}</div>
                </div>
            </div>
        </div>
        """
    
    def build_export_info(self, metadata):
        """Build export information section."""
        
        export_info = metadata.get('export_info', {})
        success = export_info.get('success', False)
        method = export_info.get('export_method', 'Unknown')
        file_size = export_info.get('file_size_kb', 0)
        error = export_info.get('error', '')
        
        status_class = 'success' if success else 'error'
        status_text = '✅ Success' if success else '❌ Failed'
        
        error_section = ""
        if error:
            error_section = f"""
                <div class="info-item">
                    <div class="info-label">Error</div>
                    <div class="info-value error">{error}</div>
                </div>
            """
        
        return f"""
        <div class="card">
            <h3>📤 Export Information</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Status</div>
                    <div class="info-value {status_class}">{status_text}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Export Method</div>
                    <div class="info-value">{method}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">File Size</div>
                    <div class="info-value">{file_size:.1f} KB</div>
                </div>
                {error_section}
            </div>
        </div>
        """
    
    def build_study_area_info(self, metadata):
        """Build study area information section."""
        
        study_area = metadata.get('study_area', {})
        if not study_area:
            return ""
        
        bounds = study_area.get('bounds', [])
        bounds_format = study_area.get('bounds_format', 'Unknown')
        
        if len(bounds) >= 4:
            lon_range = f"{bounds[0]:.4f}° to {bounds[2]:.4f}°"
            lat_range = f"{bounds[1]:.4f}° to {bounds[3]:.4f}°"
        else:
            lon_range = "Unknown"
            lat_range = "Unknown"
        
        return f"""
        <div class="card">
            <h3>🗺️ Study Area</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Longitude Range</div>
                    <div class="info-value">{lon_range}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Latitude Range</div>
                    <div class="info-value">{lat_range}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Bounds Format</div>
                    <div class="info-value">{bounds_format}</div>
                </div>
            </div>
        </div>
        """
    
    def build_analysis_info(self, metadata):
        """Build analysis parameters section."""
        
        params = metadata.get('analysis_parameters', {})
        if not params or not isinstance(params, dict):
            return ""
        
        noise_levels = params.get('noise_levels', 'Unknown')
        iterations = params.get('iterations', 'Unknown')
        min_drainage = params.get('min_drainage_area', 'Unknown')
        n_jobs = params.get('n_jobs', 'Unknown')
        
        if isinstance(noise_levels, list):
            noise_levels = ', '.join(map(str, noise_levels))
        
        return f"""
        <div class="card">
            <h3>⚙️ Analysis Parameters</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Noise Levels</div>
                    <div class="info-value">{noise_levels}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Iterations</div>
                    <div class="info-value">{iterations}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Min Drainage Area</div>
                    <div class="info-value">{min_drainage} km²</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Parallel Jobs</div>
                    <div class="info-value">{n_jobs}</div>
                </div>
            </div>
        </div>
        """


def start_web_server(port=8080):
    """Start the web server."""
    
    server = HTTPServer(('localhost', port), DashboardHandler)
    
    def run_server():
        print(f"🌐 Starting web server at http://localhost:{port}")
        print(f"🚀 Opening browser in 2 seconds...")
        server.serve_forever()
    
    # Start server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Open browser after a short delay
    time.sleep(2)
    webbrowser.open(f'http://localhost:{port}')
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down web server...")
        server.shutdown()


def main():
    """Main function."""
    
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid port number: {sys.argv[1]}")
            return
    
    print(f"🌊 EXZECO Dashboard Metadata Web Viewer")
    print(f"📁 Looking for metadata in: data/outputs/dashboard_metadata.json")
    
    metadata_file = Path("data/outputs/dashboard_metadata.json")
    if not metadata_file.exists():
        print(f"⚠️ Metadata file not found: {metadata_file}")
        print(f"💡 Run the EXZECO analysis first to generate the metadata file.")
        return
    
    start_web_server(port)


if __name__ == "__main__":
    main()
