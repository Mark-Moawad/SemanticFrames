#!/usr/bin/env python3
"""
Proper Comparison: LLM vs Benchmark on Same Text

This script compares:
1. LLM + GraphRAG output on benchmark text
2. Your corrected benchmark data
Both using the EXACT SAME input text for fair comparison.
"""

import json
import sys
from pathlib import Path

def load_json_file(file_path: str):
    """Load JSON extraction results"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def main():
    """Compare LLM vs benchmark on the same text"""
    
    print("🔍 PROPER COMPARISON: LLM vs Benchmark")
    print("=" * 50)
    print("Same text: 'The mobile robot has a manipulator arm and camera sensor'\n")
    
    # Load the data
    llm_data = load_json_file("output/extractions/llm_on_benchmark_robotics.json")
    benchmark_data = load_json_file("output/extractions/test_extraction_results.json")
    
    if not llm_data or not benchmark_data:
        print("❌ Could not load comparison data")
        return False
    
    # Extract robotics data
    llm_result = llm_data[0]  # First (only) result
    benchmark_result = None
    
    # Find the robotics scenario in benchmark
    for scenario in benchmark_data:
        if scenario.get('domain') == 'robotics':
            benchmark_result = scenario
            break
    
    if not benchmark_result:
        print("❌ No robotics benchmark found")
        return False
    
    # Compare lexical units
    llm_lexical = llm_result.get('lexical_units', {})
    benchmark_lexical = benchmark_result.get('lexical_units', {})
    
    print("📊 LEXICAL UNITS COMPARISON")
    print("-" * 35)
    print(f"🤖 LLM found: {len(llm_lexical)} terms")
    print(f"📋 Benchmark has: {len(benchmark_lexical)} terms")
    
    # Terms comparison
    llm_terms = set(llm_lexical.keys())
    benchmark_terms = set(benchmark_lexical.keys())
    
    common_terms = llm_terms.intersection(benchmark_terms)
    llm_only = llm_terms - benchmark_terms
    benchmark_only = benchmark_terms - llm_terms
    
    print(f"\n✅ Common terms: {len(common_terms)}")
    for term in sorted(common_terms):
        llm_map = llm_lexical[term]
        bench_map = benchmark_lexical[term]
        llm_frame_elem = f"{llm_map.get('frame', '?')}.{llm_map.get('element', '?')}"
        bench_frame_elem = f"{bench_map.get('frame', '?')}.{bench_map.get('element', '?')}"
        
        match_status = "✓" if llm_frame_elem == bench_frame_elem else "✗"
        print(f"   {match_status} '{term}':")
        print(f"     🤖 LLM:       {llm_frame_elem}")
        print(f"     📋 Benchmark: {bench_frame_elem}")
        
        # Check implicit frames
        llm_implicit = len(llm_map.get('implicit_frames', []))
        bench_implicit = len(bench_map.get('implicit_frames', []))
        if llm_implicit != bench_implicit:
            print(f"     🔗 Implicit:   LLM={llm_implicit}, Benchmark={bench_implicit}")
    
    if llm_only:
        print(f"\n🤖 LLM-only terms ({len(llm_only)}):")
        for term in sorted(llm_only):
            mapping = llm_lexical[term]
            frame_elem = f"{mapping.get('frame', '?')}.{mapping.get('element', '?')}"
            print(f"   • '{term}' → {frame_elem}")
    
    if benchmark_only:
        print(f"\n📋 Benchmark-only terms ({len(benchmark_only)}):")
        for term in sorted(benchmark_only):
            mapping = benchmark_lexical[term]
            frame_elem = f"{mapping.get('frame', '?')}.{mapping.get('element', '?')}"
            implicit_count = len(mapping.get('implicit_frames', []))
            implicit_info = f" + {implicit_count} implicit" if implicit_count > 0 else ""
            print(f"   • '{term}' → {frame_elem}{implicit_info}")
    
    # Calculate accuracy
    perfect_matches = 0
    for term in common_terms:
        llm_map = llm_lexical[term]
        bench_map = benchmark_lexical[term]
        if (llm_map.get('frame') == bench_map.get('frame') and 
            llm_map.get('element') == bench_map.get('element')):
            perfect_matches += 1
    
    accuracy = (perfect_matches / len(common_terms) * 100) if common_terms else 0
    
    print(f"\n📈 ACCURACY METRICS")
    print("-" * 25)
    print(f"   Perfect matches: {perfect_matches}/{len(common_terms)}")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Term coverage: {len(common_terms)}/{len(benchmark_terms)} ({len(common_terms)/len(benchmark_terms)*100:.1f}%)")
    
    # Key issues
    print(f"\n⚠️  KEY ISSUES IDENTIFIED")
    print("-" * 30)
    
    core_element_violations = 0
    for term, mapping in llm_lexical.items():
        element = mapping.get('element', '')
        # Check if using instance-specific elements instead of core elements
        if any(specific in element.lower() for specific in ['rgb', 'robotic', 'camera', 'type']):
            core_element_violations += 1
    
    print(f"   1. Core element violations: {core_element_violations}/{len(llm_lexical)}")
    print(f"   2. Missing compound terms: {len(benchmark_only)} crucial terms")
    print(f"   3. Implicit frame detection: {sum(len(m.get('implicit_frames', [])) for m in llm_lexical.values())} vs {sum(len(m.get('implicit_frames', [])) for m in benchmark_lexical.values())}")
    
    print(f"\n🎯 CONCLUSION")
    print("-" * 15)
    if accuracy < 50:
        print("   ❌ LLM extraction needs significant improvement")
    elif accuracy < 80:
        print("   ⚠️  LLM extraction shows moderate performance")
    else:
        print("   ✅ LLM extraction performs well")
    
    print(f"\n💡 This is now a fair comparison using the same input text!")
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ COMPARISON COMPLETE' if success else '❌ COMPARISON FAILED'}")
    sys.exit(0 if success else 1)
