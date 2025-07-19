#!/usr/bin/env python3
"""
Test Scenarios for Semantic Frame Extraction

Tests the LLM + GraphRAG pipeline with real robot and building descriptions
to verify knowledge structuring according to semantic frames.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_robot_extraction():
    """Test robot description extraction"""
    
    robot_description = """
    The KUKA KR 6 R900 sixx is a 6-axis industrial robot designed for precise assembly operations 
    in automotive manufacturing. It features a maximum payload of 6 kg and a reach of 900 mm. 
    The robot is equipped with a force-torque sensor for delicate handling tasks and can operate 
    at speeds up to 2.2 m/s. It includes safety features like collision detection and emergency 
    stop functionality. The robot controller uses advanced path planning algorithms for optimal 
    trajectory execution in confined workspaces.
    """
    
    print("🤖 TESTING ROBOT EXTRACTION")
    print("=" * 50)
    print(f"Input text: {robot_description}")
    print("\n" + "🔍 EXTRACTING SEMANTIC KNOWLEDGE...")
    
    try:
        from semantic_extraction_system import SemanticExtractionSystem
        
        extractor = SemanticExtractionSystem()
        result = extractor.extract_semantic_knowledge(robot_description)
        
        print(f"✅ Extraction completed!")
        print(f"🎯 Domain detected: {result.domain}")
        print(f"📊 Frames extracted: {list(result.extracted_frames.keys())}")
        print(f"🔗 Relationships found: {len(result.relationships)}")
        print(f"💬 Lexical units: {len(result.lexical_units)}")
        
        # Save results
        save_path = Path("output/extractions/robot_test_result.json")
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "source_text": result.source_text,
                "domain": result.domain,
                "extracted_frames": result.extracted_frames,
                "confidence_scores": result.confidence_scores,
                "relationships": result.relationships,
                "lexical_units": result.lexical_units,
                "timestamp": result.timestamp
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {save_path}")
        return result
        
    except Exception as e:
        print(f"❌ Error during robot extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_building_extraction():
    """Test building description extraction"""
    
    building_description = """
    The office complex is a 12-story steel frame building with curtain wall facade located in 
    downtown Berlin. It features an integrated HVAC system with variable air volume controls, 
    LED lighting with occupancy sensors, and a building management system for energy optimization. 
    The structure includes fire safety systems with sprinklers and smoke detection, elevator 
    systems with regenerative drives, and solar panels on the roof for renewable energy generation. 
    The building is designed to LEED Gold standards with green roof areas and rainwater collection systems.
    """
    
    print("\n🏢 TESTING BUILDING EXTRACTION")
    print("=" * 50)
    print(f"Input text: {building_description}")
    print("\n" + "🔍 EXTRACTING SEMANTIC KNOWLEDGE...")
    
    try:
        from semantic_extraction_system import SemanticExtractionSystem
        
        extractor = SemanticExtractionSystem()
        result = extractor.extract_semantic_knowledge(building_description)
        
        print(f"✅ Extraction completed!")
        print(f"🎯 Domain detected: {result.domain}")
        print(f"📊 Frames extracted: {list(result.extracted_frames.keys())}")
        print(f"🔗 Relationships found: {len(result.relationships)}")
        print(f"💬 Lexical units: {len(result.lexical_units)}")
        
        # Save results
        save_path = Path("output/extractions/building_test_result.json")
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "source_text": result.source_text,
                "domain": result.domain,
                "extracted_frames": result.extracted_frames,
                "confidence_scores": result.confidence_scores,
                "relationships": result.relationships,
                "lexical_units": result.lexical_units,
                "timestamp": result.timestamp
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {save_path}")
        return result
        
    except Exception as e:
        print(f"❌ Error during building extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_cross_domain_extraction():
    """Test extraction from text that involves both robots and buildings"""
    
    cross_domain_description = """
    A construction robot equipped with 3D scanning capabilities is inspecting the concrete 
    structure of a high-rise building. The robot uses LiDAR sensors to detect structural 
    defects and cracks in the building's facade. It autonomously navigates along the 
    building's exterior using a rail system mounted on the scaffold. The robot's inspection 
    data is integrated with the building's BIM model to update maintenance schedules and 
    structural health monitoring systems.
    """
    
    print("\n🏗️ TESTING CROSS-DOMAIN EXTRACTION")
    print("=" * 50)
    print(f"Input text: {cross_domain_description}")
    print("\n" + "🔍 EXTRACTING SEMANTIC KNOWLEDGE...")
    
    try:
        from semantic_extraction_system import SemanticExtractionSystem
        
        extractor = SemanticExtractionSystem()
        result = extractor.extract_semantic_knowledge(cross_domain_description)
        
        print(f"✅ Extraction completed!")
        print(f"🎯 Domain detected: {result.domain}")
        print(f"📊 Frames extracted: {list(result.extracted_frames.keys())}")
        print(f"🔗 Relationships found: {len(result.relationships)}")
        print(f"💬 Lexical units: {len(result.lexical_units)}")
        
        # Save results
        save_path = Path("output/extractions/cross_domain_test_result.json")
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "source_text": result.source_text,
                "domain": result.domain,
                "extracted_frames": result.extracted_frames,
                "confidence_scores": result.confidence_scores,
                "relationships": result.relationships,
                "lexical_units": result.lexical_units,
                "timestamp": result.timestamp
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {save_path}")
        return result
        
    except Exception as e:
        print(f"❌ Error during cross-domain extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_frame_structure(result):
    """Analyze the structure of extracted frames"""
    if not result:
        return
    
    print("\n📊 FRAME STRUCTURE ANALYSIS")
    print("=" * 50)
    
    for frame_name, frame_data in result.extracted_frames.items():
        print(f"\n🔸 Frame: {frame_name}")
        
        if isinstance(frame_data, dict):
            for element, value in frame_data.items():
                if element != 'frame':
                    print(f"  ├─ {element}: {type(value).__name__}")
                    if isinstance(value, dict):
                        for sub_key in value.keys():
                            print(f"  │  └─ {sub_key}")
    
    print(f"\n🔗 RELATIONSHIPS ({len(result.relationships)}):")
    for rel in result.relationships[:5]:  # Show first 5
        print(f"  └─ {rel[0]} → {rel[1]} → {rel[2]}")
    
    print(f"\n💬 LEXICAL UNITS ({len(result.lexical_units)}):")
    for word, data in list(result.lexical_units.items())[:5]:  # Show first 5
        print(f"  └─ '{word}' evokes {data.get('frame', 'unknown')} frame")

def main():
    """Run all test scenarios"""
    print("🧪 SEMANTIC FRAME EXTRACTION TESTING")
    print("=" * 60)
    
    # Test robot extraction
    robot_result = test_robot_extraction()
    if robot_result:
        analyze_frame_structure(robot_result)
    
    # Test building extraction
    building_result = test_building_extraction()
    if building_result:
        analyze_frame_structure(building_result)
    
    # Test cross-domain extraction
    cross_result = test_cross_domain_extraction()
    if cross_result:
        analyze_frame_structure(cross_result)
    
    print("\n🎯 TESTING COMPLETE")
    print("Check output/extractions/ for detailed results")

if __name__ == "__main__":
    main()
