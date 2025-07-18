    #!/usr/bin/env python3
"""
Construction Robotics Semantic Extraction Example

Demonstrates how to use the LLM + GraphRAG system to extract semantic knowledge
from construction robotics scenarios and organize it into semantic frames.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from semantic_extraction_system import SemanticExtractionSystem
import json

def construction_robotics_example():
    """
    Complete example of extracting semantic knowledge for construction robotics
    """
    
    print("Construction Robotics Semantic Extraction Demo")
    print("=" * 60)
    
    # Initialize the extraction system
    print("Initializing semantic extraction system...")
    extractor = SemanticExtractionSystem(ollama_model="llama3.1:8b")
    
    # Real-world construction robotics scenarios
    scenarios = [
        {
            "title": "Mobile Inspection Robot",
            "description": """
            The autonomous inspection robot is designed for structural health monitoring of buildings.
            It features a 6-DOF robotic arm with an integrated force/torque sensor for contact-based inspection.
            The robot is equipped with multiple sensors: a high-resolution camera for visual inspection,
            an ultrasonic thickness gauge for material assessment, and a thermal imaging camera for
            detecting heat anomalies. It uses LiDAR for autonomous navigation through building corridors
            and can detect cracks, corrosion, and structural defects in concrete and steel structures.
            The robot communicates with the building's IoT network to access real-time environmental data.
            """,
            "domain": "robotics"
        },
        {
            "title": "Smart Office Building",
            "description": """
            The 15-story smart office building features an integrated building management system
            that controls HVAC, lighting, and security systems. The HVAC system includes variable
            air volume units with occupancy sensors for energy optimization. The electrical system
            has smart outlets with individual monitoring capabilities and emergency backup power.
            Fire safety systems include smoke detectors, sprinkler systems, and automated evacuation
            procedures. The building structure uses reinforced concrete with steel frame construction.
            IoT sensors throughout the building monitor temperature, humidity, air quality, and occupancy
            levels. The building management system can interface with external robots for maintenance tasks.
            """,
            "domain": "building"
        },
        {
            "title": "Robot-Building Integration Scenario",
            "description": """
            The maintenance robot system is deployed in a smart manufacturing facility to perform
            automated inspection and maintenance tasks. The robot fleet includes specialized units:
            ceiling-mounted robots for overhead system inspection, mobile ground robots for floor-level
            equipment monitoring, and wall-climbing robots for vertical surface inspection.
            The building's HVAC system has accessible maintenance points that robots can interface with
            using standardized connectors. Smart sensors in electrical panels allow robots to perform
            safety checks without human intervention. The building's fire suppression system can be
            triggered by robot safety protocols. Integration between robot navigation systems and
            building access control enables autonomous facility-wide operations.
            """,
            "domain": "general"  # Mixed robot-building scenario
        }
    ]
    
    # Process each scenario
    all_results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📝 Processing Scenario {i}: {scenario['title']}")
        print("-" * 40)
        
        # Extract semantic knowledge
        result = extractor.extract_semantic_knowledge(
            scenario["description"], 
            scenario["domain"] if scenario["domain"] != "general" else None
        )
        
        print(f"🎯 Domain detected: {result.domain}")
        print(f"📊 Frames extracted: {len(result.extracted_frames)}")
        print(f"🔗 Relationships found: {len(result.relationships)}")
        
        # Display confidence scores
        avg_confidence = sum(result.confidence_scores.values()) / len(result.confidence_scores) if result.confidence_scores else 0
        print(f"📈 Average confidence: {avg_confidence:.2f}")
        
        all_results.append(result)
    
    # Build unified knowledge graph
    print(f"\n🧠 Building unified knowledge graph...")
    knowledge_graph = extractor.build_knowledge_graph(all_results)
    
    print(f"📊 Knowledge Graph Statistics:")
    print(f"   - Nodes: {knowledge_graph.number_of_nodes()}")
    print(f"   - Edges: {knowledge_graph.number_of_edges()}")
    # Use correct NetworkX method for connected components
    import networkx as nx
    if knowledge_graph.is_directed():
        components = list(nx.weakly_connected_components(knowledge_graph))
    else:
        components = list(nx.connected_components(knowledge_graph))
    print(f"   - Connected components: {len(components)}")
    
    # Save results
    extractor.save_extraction_results(all_results, "construction_robotics_extraction.json")
    
    print(f"\n🎯 Results saved! You can now run the visualizer:")
    print(f"   python tools/semantic_visualizer.py")
    print(f"   or")
    print(f"   python tools/quick_test.py")
    
    # Display sample extracted knowledge
    print(f"\n📋 Sample Extracted Knowledge:")
    print("-" * 40)
    
    for result in all_results:
        print(f"\n🏷️ {result.domain.upper()} DOMAIN:")
        for frame_name, frame_data in list(result.extracted_frames.items())[:2]:  # Show first 2 frames
            print(f"   {frame_name}: {str(frame_data)[:100]}...")
    
    # Show cross-domain relationships for construction robotics
    print(f"\n🔄 Cross-Domain Integration Opportunities:")
    print("-" * 40)
    
    robotics_frames = [r for r in all_results if r.domain == "robotics"]
    building_frames = [r for r in all_results if r.domain == "building"] 
    
    if robotics_frames and building_frames:
        print("🤖 Robot capabilities that can support building processes:")
        print("   - Inspection actions → Building monitoring processes")
        print("   - Navigation capabilities → Building access systems")
        print("   - Sensor data → Building IoT integration")
        print("   - Maintenance actions → Building system upkeep")
    
    print(f"\n🎉 Semantic extraction completed successfully!")
    print(f"💾 Results saved to: output/extractions/construction_robotics_extraction.json")
    
    return all_results, knowledge_graph

def analyze_construction_semantics(results, graph):
    """
    Analyze the extracted semantic knowledge for construction robotics insights
    """
    print(f"\n🔍 Construction Robotics Semantic Analysis")
    print("=" * 50)
    
    # Analyze domains
    domain_counts = {}
    for result in results:
        domain_counts[result.domain] = domain_counts.get(result.domain, 0) + 1
    
    print(f"📊 Domain Distribution:")
    for domain, count in domain_counts.items():
        print(f"   {domain}: {count} scenarios")
    
    # Analyze frame types
    all_frames = {}
    for result in results:
        for frame_name in result.extracted_frames.keys():
            all_frames[frame_name] = all_frames.get(frame_name, 0) + 1
    
    print(f"\n🏗️ Frame Type Analysis:")
    for frame_name, count in sorted(all_frames.items(), key=lambda x: x[1], reverse=True):
        print(f"   {frame_name}: {count} occurrences")
    
    # Analyze confidence levels
    all_confidences = []
    for result in results:
        all_confidences.extend(result.confidence_scores.values())
    
    if all_confidences:
        avg_confidence = sum(all_confidences) / len(all_confidences)
        print(f"\n📈 Extraction Quality:")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   High confidence (>0.7): {sum(1 for c in all_confidences if c > 0.7)}/{len(all_confidences)}")
    
    print(f"\n🎯 Construction Robotics Integration Insights:")
    print("   ✅ System can understand both robot and building descriptions")
    print("   ✅ Semantic frames provide structured knowledge representation")
    print("   ✅ Cross-domain relationships enable intelligent coordination")
    print("   ✅ Knowledge graph supports multi-robot reasoning")

def main():
    """Run the complete construction robotics semantic extraction demo"""
    try:
        # Run extraction example
        results, graph = construction_robotics_example()
        
        # Analyze results
        analyze_construction_semantics(results, graph)
        
        print(f"\n🚀 Demo completed successfully!")
        print(f"   Your LLM + GraphRAG system is ready for construction robotics!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print(f"💡 Make sure:")
        print(f"   1. Ollama is running: `ollama serve`")
        print(f"   2. Model is installed: `ollama pull llama3.1:8b`")
        print(f"   3. Dependencies are installed: `pip install -r requirements.txt`")

if __name__ == "__main__":
    main()
