#!/usr/bin/env python3
"""
Test script to verify the annotation tool is working correctly
"""

import sys
import os
import json
import requests
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_web_tool():
    """Test if the web annotation tool is running and responsive"""
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✅ Web annotation tool is running successfully!")
            print(f"   Status: {response.status_code}")
            print(f"   Content length: {len(response.text)} characters")
            return True
        else:
            print(f"❌ Web tool returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web annotation tool at http://localhost:5000")
        print("   Make sure the web server is running with: python src/web_annotation_tool.py")
        return False
    except Exception as e:
        print(f"❌ Error testing web tool: {e}")
        return False

def test_command_line_tool():
    """Test if the command line annotation tool can be imported"""
    try:
        from annotation_tool import FrameAnnotationTool
        tool = FrameAnnotationTool()
        print("✅ Command-line annotation tool loads successfully!")
        print(f"   Loaded {len(tool.robotics_lexicon)} lexicon entries")
        return True
    except Exception as e:
        print(f"❌ Error loading command-line tool: {e}")
        return False

def test_test_data():
    """Test if test sentences are available"""
    try:
        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_dataset', 'robotics_test_sentences.json')
        with open(test_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        print(f"✅ Test dataset loaded successfully!")
        print(f"   Found {len(sentences)} test sentences")
        return True
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Semantic Frame Annotation Tools")
    print("=" * 50)
    
    results = []
    
    print("\n1. Testing test dataset...")
    results.append(test_test_data())
    
    print("\n2. Testing command-line tool...")
    results.append(test_command_line_tool())
    
    print("\n3. Testing web annotation tool...")
    results.append(test_web_tool())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Your annotation tools are ready to use.")
        print("\nQuick start:")
        print("1. Web interface: http://localhost:5000")
        print("2. Command line: python src/annotation_tool.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
