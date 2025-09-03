#!/usr/bin/env python3
"""
JSON Metadata Reader for EXZECO Dashboard

This script reads and displays the dashboard metadata in a user-friendly format.
Usage: python read_dashboard_metadata.py [path_to_json_file]
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def read_dashboard_metadata(json_file_path):
    """Read and display dashboard metadata."""
    
    try:
        with open(json_file_path, 'r') as f:
            metadata = json.load(f)
        
        print("🌊 EXZECO Dashboard Metadata")
        print("=" * 50)
        
        # Basic info
        if 'title' in metadata:
            print(f"📊 Title: {metadata['title']}")
        
        if 'created' in metadata:
            created_time = datetime.fromisoformat(metadata['created'].replace('Z', '+00:00'))
            print(f"📅 Created: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'dashboard_file' in metadata:
            print(f"📁 Dashboard File: {metadata['dashboard_file']}")
        
        # Export information
        if 'export_info' in metadata:
            export_info = metadata['export_info']
            print(f"\n📤 Export Information:")
            print(f"   • Success: {'✅ Yes' if export_info.get('success') else '❌ No'}")
            print(f"   • Method: {export_info.get('export_method', 'Unknown')}")
            print(f"   • File Size: {export_info.get('file_size_kb', 0):.1f} KB")
            if export_info.get('error'):
                print(f"   • Error: {export_info['error']}")
        
        # Study area
        if 'study_area' in metadata:
            study_area = metadata['study_area']
            bounds = study_area.get('bounds', [])
            if len(bounds) >= 4:
                print(f"\n🗺️  Study Area:")
                print(f"   • Longitude: {bounds[0]:.4f}° to {bounds[2]:.4f}°")
                print(f"   • Latitude: {bounds[1]:.4f}° to {bounds[3]:.4f}°")
                print(f"   • Bounds Format: {study_area.get('bounds_format', 'Unknown')}")
        
        # Analysis parameters
        if 'analysis_parameters' in metadata:
            params = metadata['analysis_parameters']
            print(f"\n⚙️  Analysis Parameters:")
            if isinstance(params, dict):
                if 'noise_levels' in params:
                    print(f"   • Noise Levels: {params['noise_levels']}")
                if 'iterations' in params:
                    print(f"   • Iterations: {params['iterations']}")
                if 'min_drainage_area' in params:
                    print(f"   • Min Drainage Area: {params['min_drainage_area']} km²")
                if 'n_jobs' in params:
                    print(f"   • Parallel Jobs: {params['n_jobs']}")
            else:
                print(f"   • {params}")
        
        print(f"\n✅ Metadata read successfully from: {json_file_path}")
        
        return metadata
        
    except FileNotFoundError:
        print(f"❌ File not found: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None


def main():
    """Main function to handle command line usage."""
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Default path
        json_file = "../data/outputs/dashboard_metadata.json"
    
    json_path = Path(json_file)
    
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        print(f"💡 Usage: python {sys.argv[0]} [path_to_json_file]")
        print(f"💡 Default path: ../data/outputs/dashboard_metadata.json")
        return
    
    metadata = read_dashboard_metadata(json_path)
    
    if metadata:
        print(f"\n💾 Raw JSON data available as Python dict")
        print(f"🐍 To use in Python: metadata = {metadata}")


if __name__ == "__main__":
    main()
