#!/usr/bin/env python3
"""
Simple test to launch the semantic visualizer
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from semantic_visualizer import create_app

def main():
    """Launch the visualizer with test data"""
    print("Starting Semantic Visualizer...")
    print("=" * 50)
    
    # Check if we have existing extraction results
    results_file = Path(__file__).parent.parent / "output" / "extractions" / "construction_robotics_extraction.json"
    
    if results_file.exists():
        print(f"Found existing results: {results_file}")
        app = create_app(str(results_file))
    else:
        print("No existing results found, starting with empty visualizer")
        app = create_app()
    
    print("\nLaunching visualizer at http://127.0.0.1:8050")
    print("Check the following features:")
    print("1. Domain filter dropdown (Robotics/Building)")
    print("2. Component styling differences:")
    print("   - Robotics: Red ellipse components")
    print("   - Building: Orange round-rectangle components")
    print("3. Domain filtering works on all tabs")
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)

if __name__ == "__main__":
    main()
