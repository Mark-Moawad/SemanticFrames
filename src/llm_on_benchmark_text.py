#!/usr/bin/env python3
"""
LLM Extraction on Benchmark Text

This script runs the LLM + GraphRAG pipeline on the EXACT SAME text 
from the benchmark test data for proper comparison.

Benchmark Texts:
- Robotics: "The mobile robot has a manipulator arm and camera sensor"
- Building: "The office building has HVAC system and electrical panels"
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_extraction_system import SemanticExtractionSystem

def main():
    """Run LLM extraction on the exact benchmark texts"""
    
    print("🎯 LLM Extraction on Benchmark Text")
    print("=" * 45)
    print("Running LLM + GraphRAG on the SAME text from benchmark\n")
    
    # Initialize the extraction system
    extractor = SemanticExtractionSystem()
    
    # THE EXACT SAME TEXT FROM YOUR BENCHMARK
    benchmark_robotics_text = "The mobile robot has a manipulator arm and camera sensor"
    benchmark_building_text = "The office building has HVAC system and electrical panels"
    
    print("📝 Benchmark Texts:")
    print(f"   🤖 Robotics: '{benchmark_robotics_text}'")
    print(f"   🏗️ Building: '{benchmark_building_text}'")
    print()
    
    try:
        # Extract from the exact same robotics text
        print("🤖 Running LLM extraction on robotics benchmark text...")
        robotics_result = extractor.extract_semantic_knowledge(benchmark_robotics_text, "robotics")
        print("✅ Robotics extraction completed")
        
        # Extract from the exact same building text  
        print("🏗️ Running LLM extraction on building benchmark text...")
        building_result = extractor.extract_semantic_knowledge(benchmark_building_text, "building")
        print("✅ Building extraction completed")
        
        # Save results for comparison
        print("\n💾 Saving LLM results on benchmark texts...")
        extractor.save_extraction_results([robotics_result], "llm_on_benchmark_robotics.json")
        extractor.save_extraction_results([building_result], "llm_on_benchmark_building.json")
        
        # Display robotics lexical units for immediate comparison
        print("\n🔍 ROBOTICS LEXICAL UNITS COMPARISON:")
        print("=" * 50)
        print(f"📊 Benchmark text: '{benchmark_robotics_text}'")
        print(f"🤖 LLM extracted lexical units:")
        
        for term, mapping in robotics_result.lexical_units.items():
            frame = mapping.get("frame", "unknown")
            element = mapping.get("element", "unknown")
            print(f"   • '{term}' → {frame}.{element}")
        
        print(f"\n🎯 Confidence scores:")
        for frame, score in robotics_result.confidence_scores.items():
            print(f"   • {frame}: {score:.2f}")
        
        print(f"\n📁 Output files:")
        print(f"   ✅ llm_on_benchmark_robotics.json")
        print(f"   ✅ llm_on_benchmark_building.json")
        
        print(f"\n💡 Now you can compare:")
        print(f"   📊 Benchmark: test_extraction_results.json (corrected)")
        print(f"   🤖 LLM Output: llm_on_benchmark_robotics.json (same text)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
