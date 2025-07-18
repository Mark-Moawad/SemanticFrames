#!/usr/bin/env python3
"""
Quick Setup and Test Script for Semantic Extraction System

Runs the complete workflow: extraction → visualization in one command
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def check_ollama():
    """Check if Ollama is running"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Run the complete semantic extraction and visualization workflow"""
    
    print("🤖🏗️ Semantic Extraction System - Quick Test")
    print("=" * 60)
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    if not check_ollama():
        print("❌ Ollama is not running!")
        print("💡 Start Ollama first:")
        print("   1. Open a terminal and run: ollama serve")
        print("   2. In another terminal: ollama pull llama3.1:8b")
        return False
    
    print("✅ Ollama is running")
    
    # Check if extraction results exist
    extraction_file = Path("output/extractions/construction_robotics_extraction.json")
    
    if not extraction_file.exists():
        print("🚀 Running semantic extraction demo...")
        success = run_command("python examples/construction_robotics_demo.py", 
                             "Semantic extraction demo")
        if not success:
            print("❌ Demo failed. Check the error messages above.")
            return False
    else:
        print("✅ Extraction results already exist")
    
    # Install visualization requirements if needed
    print("📦 Installing visualization requirements...")
    run_command("pip install dash dash-bootstrap-components dash-cytoscape plotly", 
               "Installing visualization dependencies")
    
    # Start the visualizer
    print("\n🎯 Starting interactive visualizer...")
    print("📱 The visualizer will open at: http://localhost:8050")
    print("🛑 Press Ctrl+C to stop the visualizer")
    
    try:
        # Import and run visualizer
        from semantic_visualizer import SemanticVisualizationSystem
        
        visualizer = SemanticVisualizationSystem(str(extraction_file))
        visualizer.run(debug=False, port=8050)
        
    except KeyboardInterrupt:
        print("\n👋 Visualizer stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing missing dependencies:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")

if __name__ == "__main__":
    main()
