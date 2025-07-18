#!/usr/bin/env python3
"""
Test Script for Semantic Extraction System

Quick test to verify that the LLM + GraphRAG workflow is working correctly
with your Ollama installation and semantic frames.
"""

import sys
import requests
from pathlib import Path

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama is running!")
            print(f"📋 Available models: {[m['name'] for m in models]}")
            return True
        else:
            print("❌ Ollama is not responding correctly")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: `ollama serve`")
        return False

def test_frame_files():
    """Test if semantic frame files exist"""
    workspace = Path("c:/Users/markm/Documents/GitHub/SemanticFrames")
    
    building_frames = workspace / "output" / "frames" / "building"
    robotics_frames = workspace / "output" / "frames" / "robotics"
    
    building_count = len(list(building_frames.glob("*.json"))) if building_frames.exists() else 0
    robotics_count = len(list(robotics_frames.glob("*.json"))) if robotics_frames.exists() else 0
    
    print(f"📁 Building frames found: {building_count}")
    print(f"🤖 Robotics frames found: {robotics_count}")
    
    if building_count > 0 and robotics_count > 0:
        print("✅ Semantic frames are available")
        return True
    else:
        print("❌ Missing semantic frame files")
        return False

def test_extraction_system():
    """Test the semantic extraction system"""
    try:
        from src.semantic_extraction_system import SemanticExtractionSystem
        
        # Initialize system
        extractor = SemanticExtractionSystem()
        print("✅ Semantic extraction system initialized")
        
        # Test simple extraction
        test_text = "A mobile robot with a camera sensor for navigation"
        result = extractor.extract_semantic_knowledge(test_text)
        
        print(f"✅ Extraction test completed")
        print(f"🎯 Domain detected: {result.domain}")
        print(f"📊 Frames extracted: {len(result.extracted_frames)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import extraction system: {e}")
        return False
    except Exception as e:
        print(f"❌ Extraction system error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Semantic Extraction System Setup")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Semantic Frame Files", test_frame_files),
        ("Extraction System", test_extraction_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    
    all_passed = all(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} {test_name}")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Your system is ready!")
        print("🚀 You can now use the semantic extraction system for construction robotics!")
    else:
        print("\n⚠️  Some tests failed. Please check the setup:")
        print("1. Ensure Ollama is running: `ollama serve`")
        print("2. Install a model: `ollama pull llama3.1:8b`")
        print("3. Check that semantic frame files exist in output/frames/")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
