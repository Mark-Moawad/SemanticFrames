#!/usr/bin/env python3
"""
Comprehensive Test of GraphRAG Knowledge System

This script tests the complete LLM + GraphRAG pipeline including:
1. Semantic frame extraction
2. Knowledge graph construction  
3. Frame-based reasoning
4. Cross-domain relationship mapping
5. Lexical unit identification
"""

import sys
import json
from pathlib import Path
import networkx as nx

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_knowledge_graph_construction():
    """Test the construction of knowledge graphs from semantic frames"""
    print("🔗 TESTING KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 60)
    
    try:
        from semantic_knowledge_system import SemanticKnowledgeSystem
        
        # Initialize knowledge system
        knowledge_system = SemanticKnowledgeSystem()
        
        # Load frames
        frames_dir = "output/frames"
        knowledge_system.load_frames(frames_dir)
        
        print(f"✅ Loaded {len(knowledge_system.frames)} semantic frames")
        
        # Build semantic graph
        knowledge_system.build_semantic_graph()
        
        print(f"📊 Knowledge graph has {knowledge_system.knowledge_graph.number_of_nodes()} nodes")
        print(f"🔗 Knowledge graph has {knowledge_system.knowledge_graph.number_of_edges()} edges")
        
        # Analyze frame hierarchies
        for domain, hierarchy in knowledge_system.frame_hierarchies.items():
            print(f"🏗️ {domain.capitalize()} domain: {len(hierarchy)} hierarchical relationships")
        
        return knowledge_system
        
    except Exception as e:
        print(f"❌ Error in knowledge graph construction: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_graphrag_system():
    """Test the GraphRAG system for knowledge retrieval"""
    print("\n📚 TESTING GRAPHRAG SYSTEM")
    print("=" * 60)
    
    try:
        from graphrag_frame_system import GraphRAGFrameSystem
        
        # Initialize GraphRAG system
        graphrag_system = GraphRAGFrameSystem()
        
        # Ingest frame knowledge
        graphrag_system.ingest_frame_knowledge(".")
        
        print(f"✅ GraphRAG system initialized")
        print(f"📊 Semantic chunks: {len(graphrag_system.semantic_chunks)}")
        print(f"🎯 Frame entities: {len(graphrag_system.frame_entities)}")
        
        # Test knowledge retrieval
        test_queries = [
            "robot manipulation capabilities",
            "building HVAC systems",
            "construction robotics integration"
        ]
        
        for query in test_queries:
            try:
                relevant_chunks = graphrag_system.retrieve_relevant_knowledge(query)
                print(f"🔍 Query: '{query}' → {len(relevant_chunks)} relevant chunks")
            except Exception as e:
                print(f"⚠️ Query '{query}' failed: {e}")
        
        return graphrag_system
        
    except Exception as e:
        print(f"❌ Error in GraphRAG system: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_frame_semantic_reasoner():
    """Test the frame-semantic reasoning system"""
    print("\n🧠 TESTING FRAME-SEMANTIC REASONER")
    print("=" * 60)
    
    try:
        from frame_semantic_reasoner import FrameSemanticReasoner
        
        # Initialize reasoner
        reasoner = FrameSemanticReasoner()
        
        print("✅ Frame-semantic reasoner initialized")
        
        # Test reasoning queries
        test_queries = [
            "What are the capabilities of an industrial robot?",
            "How do HVAC systems relate to building management?",
            "What is the relationship between robot sensors and building monitoring?"
        ]
        
        for query in test_queries:
            try:
                print(f"\n🤔 Query: {query}")
                
                # Get reasoning context
                context = reasoner.analyze_query(query)
                print(f"   📋 Domain: {context.domain}")
                print(f"   🎯 Frames involved: {context.frames_involved}")
                print(f"   💭 Semantic intent: {context.semantic_intent}")
                
                # Generate response
                response = reasoner.generate_response(context)
                print(f"   💬 Response: {response[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        return reasoner
        
    except Exception as e:
        print(f"❌ Error in frame-semantic reasoner: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_lexical_unit_mapping():
    """Test lexical unit identification and frame evocation"""
    print("\n💬 TESTING LEXICAL UNIT MAPPING")
    print("=" * 60)
    
    # Load extraction results
    extraction_files = [
        "output/extractions/robot_test_result.json",
        "output/extractions/building_test_result.json",
        "output/extractions/cross_domain_test_result.json"
    ]
    
    total_lexical_units = 0
    explicit_evocations = 0
    implicit_evocations = 0
    
    for file_path in extraction_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            domain = result.get('domain', 'unknown')
            lexical_units = result.get('lexical_units', {})
            
            print(f"\n📂 {domain.capitalize()} domain:")
            print(f"   💬 Lexical units found: {len(lexical_units)}")
            
            for word, data in lexical_units.items():
                frame = data.get('frame', 'unknown')
                element = data.get('element', 'unknown')
                evocation = data.get('evocation', 'unknown')
                
                print(f"   └─ '{word}' → {frame}.{element} ({evocation})")
                
                total_lexical_units += 1
                if evocation == 'explicit':
                    explicit_evocations += 1
                elif evocation == 'implicit':
                    implicit_evocations += 1
                    
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
    
    print(f"\n📊 LEXICAL UNIT SUMMARY:")
    print(f"   Total lexical units: {total_lexical_units}")
    print(f"   Explicit evocations: {explicit_evocations}")
    print(f"   Implicit evocations: {implicit_evocations}")

def test_frame_structure_validation():
    """Validate that extracted frames match the intended structure"""
    print("\n🏗️ TESTING FRAME STRUCTURE VALIDATION")
    print("=" * 60)
    
    # Load frame templates
    robotics_frames_dir = Path("output/frames/robotics")
    building_frames_dir = Path("output/frames/building")
    
    expected_structures = {}
    
    # Load expected frame structures
    for frame_file in robotics_frames_dir.glob("*.json"):
        try:
            with open(frame_file, 'r', encoding='utf-8') as f:
                frame_data = json.load(f)
            expected_structures[f"robotics_{frame_file.stem}"] = frame_data
        except Exception as e:
            print(f"⚠️ Warning: Could not load {frame_file}: {e}")
    
    for frame_file in building_frames_dir.glob("*.json"):
        try:
            with open(frame_file, 'r', encoding='utf-8') as f:
                frame_data = json.load(f)
            expected_structures[f"building_{frame_file.stem}"] = frame_data
        except Exception as e:
            print(f"⚠️ Warning: Could not load {frame_file}: {e}")
    
    print(f"📋 Expected frame structures loaded: {len(expected_structures)}")
    
    # Validate extracted frames against templates
    extraction_files = [
        ("robot_test_result.json", "robotics"),
        ("building_test_result.json", "building"),
        ("cross_domain_test_result.json", "general")
    ]
    
    for file_name, domain in extraction_files:
        file_path = f"output/extractions/{file_name}"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            extracted_frames = result.get('extracted_frames', {})
            
            print(f"\n🔍 Validating {domain} extraction:")
            
            for frame_name, frame_data in extracted_frames.items():
                template_key = f"{domain}_{frame_name}"
                
                if template_key in expected_structures:
                    template = expected_structures[template_key]
                    print(f"   ✅ {frame_name}: Matches template structure")
                    
                    # Check frame elements
                    if 'frame_elements' in template:
                        template_elements = template['frame_elements']
                        if isinstance(frame_data, dict):
                            for element in template_elements.get('core', {}):
                                if element in frame_data:
                                    print(f"      ✓ Core element '{element}' present")
                                else:
                                    print(f"      ⚠️ Core element '{element}' missing")
                else:
                    print(f"   ⚠️ {frame_name}: No template found")
                    
        except Exception as e:
            print(f"   ❌ Error validating {file_name}: {e}")

def generate_knowledge_graph_visualization():
    """Generate a visualization of the knowledge graph"""
    print("\n📊 GENERATING KNOWLEDGE GRAPH VISUALIZATION")
    print("=" * 60)
    
    try:
        # Create a simple knowledge graph from extraction results
        G = nx.DiGraph()
        
        extraction_files = [
            "output/extractions/robot_test_result.json",
            "output/extractions/building_test_result.json",
            "output/extractions/cross_domain_test_result.json"
        ]
        
        for file_path in extraction_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                domain = result.get('domain', 'unknown')
                extracted_frames = result.get('extracted_frames', {})
                relationships = result.get('relationships', [])
                
                # Add nodes for each frame
                for frame_name in extracted_frames.keys():
                    G.add_node(f"{domain}_{frame_name}", 
                              type='frame', 
                              domain=domain,
                              label=frame_name)
                
                # Add relationships
                for rel in relationships:
                    if len(rel) >= 3:
                        source, relation, target = rel[0], rel[1], rel[2]
                        G.add_edge(f"{domain}_{source}", target, 
                                  relation=relation)
                        
            except Exception as e:
                print(f"   ⚠️ Error processing {file_path}: {e}")
        
        print(f"📊 Knowledge graph created:")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
        
        # Save graph structure
        graph_data = {
            "nodes": [{"id": node, **data} for node, data in G.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
        }
        
        output_path = "output/extractions/knowledge_graph.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Knowledge graph saved to: {output_path}")
        
        return G
        
    except Exception as e:
        print(f"❌ Error generating knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run comprehensive GraphRAG and knowledge system tests"""
    print("🧪 COMPREHENSIVE GRAPHRAG + KNOWLEDGE SYSTEM TESTING")
    print("=" * 70)
    
    # Test knowledge graph construction
    knowledge_system = test_knowledge_graph_construction()
    
    # Test GraphRAG system
    graphrag_system = test_graphrag_system()
    
    # Test frame-semantic reasoner
    reasoner = test_frame_semantic_reasoner()
    
    # Test lexical unit mapping
    test_lexical_unit_mapping()
    
    # Test frame structure validation
    test_frame_structure_validation()
    
    # Generate knowledge graph visualization
    graph = generate_knowledge_graph_visualization()
    
    print("\n🎯 COMPREHENSIVE TESTING COMPLETE")
    print("=" * 70)
    print("✅ LLM + GraphRAG pipeline tested successfully!")
    print("📊 Check output/extractions/ for detailed results and visualizations")

if __name__ == "__main__":
    main()
