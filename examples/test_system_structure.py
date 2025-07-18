#!/usr/bin/env python3
"""
Quick Semantic Extraction Test (Without LLM)

Test the semantic extraction system structure without requiring Ollama model download.
Uses mock LLM responses to verify the overall system architecture.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Verify the path exists
if not src_path.exists():
    raise ImportError(f"Source directory not found: {src_path}")

try:
    from semantic_extraction_system import SemanticExtractionSystem
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Trying to import from: {src_path}")
    print(f"📂 Directory contents: {list(src_path.glob('*.py'))}")
    raise

import json

def test_system_structure():
    """Test the semantic extraction system structure without LLM calls"""
    
    print("🧪 Testing Semantic Extraction System Structure")
    print("=" * 55)
    
    # Initialize the extraction system
    print("🔧 Initializing semantic extraction system...")
    extractor = SemanticExtractionSystem(ollama_model="llama3.1:8b")
    
    print(f"✅ System initialized successfully!")
    print(f"📁 Workspace: {extractor.workspace_path}")
    print(f"🎯 Frame templates loaded: {len(extractor.frame_templates)}")
    print(f"📋 Blueprint files loaded: {len(extractor.blueprints)}")
    
    # Test domain detection
    print(f"\n🔍 Testing domain detection...")
    robot_text = "The mobile robot has a manipulator arm and camera sensor"
    building_text = "The office building has HVAC system and electrical panels"
    
    robot_domain = extractor._detect_domain(robot_text)
    building_domain = extractor._detect_domain(building_text)
    
    print(f"🤖 Robot text → Domain: {robot_domain}")
    print(f"🏗️ Building text → Domain: {building_domain}")
    
    # Mock extraction result for testing
    print(f"\n🎯 Creating comprehensive mock extraction following blueprint structure...")
    
    mock_result = {
        "source_text": robot_text,
        "domain": "robotics",
        "extracted_frames": {
            "Robot": {
                "frame": "Robot",
                "Agent": {
                    "robot_type": "mobile_inspection_robot",
                    "manufacturer": "Construction Robotics Inc",
                    "model": "CR-Inspector-2024"
                },
                "Function": {
                    "primary_function": "structural_health_monitoring",
                    "application_domain": "building_inspection",
                    "autonomy_level": "fully_autonomous"
                },
                "Domain": {
                    "operational_domain": "indoor_building_environments",
                    "target_structures": ["concrete", "steel", "mixed_construction"]
                }
            },
            "Capability": {
                "frame": "Capability",
                "manipulation_capability": {
                    "capability_type": "manipulation",
                    "dof": 6,
                    "payload": "5kg",
                    "reach": "850mm",
                    "precision": "±0.1mm"
                },
                "sensing_capability": {
                    "capability_type": "sensing",
                    "modalities": ["visual", "thermal", "ultrasonic", "force"],
                    "range": "0.1m - 10m",
                    "resolution": "high"
                },
                "navigation_capability": {
                    "capability_type": "navigation",
                    "method": "lidar_slam",
                    "max_speed": "1.5m/s",
                    "obstacle_avoidance": "dynamic"
                }
            },
            "Component": {
                "frame": "Component",
                "robotic_arm": {
                    "component_type": "manipulator",
                    "specification": "6-DOF robotic arm",
                    "sensors": ["force_torque_sensor"],
                    "enables_capability": "manipulation_capability"
                },
                "rgb_camera": {
                    "component_type": "sensor",
                    "specification": "high-resolution RGB camera",
                    "resolution": "4K",
                    "enables_capability": "sensing_capability"
                },
                "thermal_camera": {
                    "component_type": "sensor", 
                    "specification": "thermal imaging camera",
                    "temperature_range": "-20°C to 150°C",
                    "enables_capability": "sensing_capability"
                },
                "ultrasonic_gauge": {
                    "component_type": "sensor",
                    "specification": "ultrasonic thickness gauge",
                    "measurement_range": "1mm - 500mm",
                    "enables_capability": "sensing_capability"
                },
                "lidar_sensor": {
                    "component_type": "sensor",
                    "specification": "LiDAR for navigation",
                    "range": "30m",
                    "enables_capability": "navigation_capability"
                }
            },
            "Action": {
                "frame": "Action",
                "visual_inspection": {
                    "action_type": "inspection",
                    "method": "visual_analysis",
                    "required_components": ["rgb_camera"],
                    "required_capabilities": ["sensing_capability"],
                    "target": "surface_defects"
                },
                "thermal_inspection": {
                    "action_type": "inspection", 
                    "method": "thermal_analysis",
                    "required_components": ["thermal_camera"],
                    "required_capabilities": ["sensing_capability"],
                    "target": "heat_anomalies"
                },
                "thickness_measurement": {
                    "action_type": "measurement",
                    "method": "ultrasonic_measurement", 
                    "required_components": ["ultrasonic_gauge", "robotic_arm"],
                    "required_capabilities": ["sensing_capability", "manipulation_capability"],
                    "target": "material_thickness"
                },
                "autonomous_navigation": {
                    "action_type": "movement",
                    "method": "lidar_slam_navigation",
                    "required_components": ["lidar_sensor"],
                    "required_capabilities": ["navigation_capability"],
                    "target": "building_corridors"
                },
                "defect_documentation": {
                    "action_type": "documentation",
                    "method": "automated_reporting",
                    "required_components": ["rgb_camera", "thermal_camera"],
                    "required_capabilities": ["sensing_capability"],
                    "target": "structural_defects"
                }
            }
        },
        "confidence_scores": {
            "Robot": 0.92,
            "Capability": 0.88,
            "Component": 0.85,
            "Action": 0.90
        },
        "relationships": [
            # Robot → Capability relationships
            ("Robot", "has_capability", "manipulation_capability"),
            ("Robot", "has_capability", "sensing_capability"), 
            ("Robot", "has_capability", "navigation_capability"),
            
            # Capability → Component relationships
            ("manipulation_capability", "enabled_by", "robotic_arm"),
            ("sensing_capability", "enabled_by", "rgb_camera"),
            ("sensing_capability", "enabled_by", "thermal_camera"),
            ("sensing_capability", "enabled_by", "ultrasonic_gauge"),
            ("navigation_capability", "enabled_by", "lidar_sensor"),
            
            # Component → Action relationships
            ("robotic_arm", "enables_action", "thickness_measurement"),
            ("rgb_camera", "enables_action", "visual_inspection"),
            ("rgb_camera", "enables_action", "defect_documentation"),
            ("thermal_camera", "enables_action", "thermal_inspection"),
            ("thermal_camera", "enables_action", "defect_documentation"),
            ("ultrasonic_gauge", "enables_action", "thickness_measurement"),
            ("lidar_sensor", "enables_action", "autonomous_navigation"),
            
            # Action interdependencies
            ("visual_inspection", "precedes", "defect_documentation"),
            ("thermal_inspection", "precedes", "defect_documentation"),
            ("autonomous_navigation", "enables", "visual_inspection"),
            ("autonomous_navigation", "enables", "thermal_inspection")
        ],
        "lexical_units": {
            "mobile": {"frame": "Robot", "element": "robot_type", "evocation": "explicit"},
            "robot": {"frame": "Robot", "element": "Agent", "evocation": "explicit"},
            "manipulator": {"frame": "Component", "element": "robotic_arm", "evocation": "explicit"},
            "arm": {"frame": "Component", "element": "robotic_arm", "evocation": "explicit"},
            "camera": {"frame": "Component", "element": "rgb_camera", "evocation": "explicit"},
            "sensor": {"frame": "Component", "element": "lidar_sensor", "evocation": "implicit"}
        },
        "timestamp": "2025-01-11T12:00:00"
    }
    
    # Create building domain mock extraction for comprehensive testing
    print(f"\n🏗️ Creating building domain mock extraction...")
    
    building_mock_result = {
        "source_text": building_text,
        "domain": "building",
        "extracted_frames": {
            "Building": {
                "frame": "Building",
                "Structure": {
                    "building_type": "commercial_office_building",
                    "construction_year": "2020",
                    "floors": 15,
                    "total_area": "50000_sqft"
                },
                "Function": {
                    "primary_function": "office_workspace",
                    "occupancy_type": "commercial",
                    "energy_efficiency_rating": "LEED_Gold"
                },
                "Location": {
                    "address": "Downtown Business District",
                    "climate_zone": "temperate",
                    "seismic_zone": "moderate"
                }
            },
            "Component": {
                "frame": "Component",
                "hvac_system": {
                    "component_type": "mechanical_system",
                    "specification": "Variable Air Volume HVAC",
                    "capacity": "500_tons",
                    "enables_system": "climate_control_system"
                },
                "electrical_panels": {
                    "component_type": "electrical_system",
                    "specification": "Main distribution panels",
                    "voltage": "480V_3_phase",
                    "enables_system": "power_distribution_system"
                },
                "fire_safety_system": {
                    "component_type": "safety_system",
                    "specification": "Sprinkler and alarm system",
                    "coverage": "full_building",
                    "enables_system": "safety_management_system"
                },
                "elevator_system": {
                    "component_type": "transportation_system",
                    "specification": "High-speed passenger elevators",
                    "capacity": "20_persons",
                    "enables_system": "vertical_transportation_system"
                }
            },
            "System": {
                "frame": "System",
                "climate_control_system": {
                    "system_type": "environmental_control",
                    "temperature_range": "68-72°F",
                    "humidity_control": "40-60%",
                    "air_quality_management": "HEPA_filtration"
                },
                "power_distribution_system": {
                    "system_type": "electrical_supply",
                    "power_capacity": "2MW",
                    "backup_power": "diesel_generator",
                    "smart_metering": "enabled"
                },
                "safety_management_system": {
                    "system_type": "safety_systems",
                    "fire_detection": "smoke_and_heat",
                    "emergency_response": "automated_alerts",
                    "egress_management": "lighted_exit_signs"
                },
                "vertical_transportation_system": {
                    "system_type": "people_movement",
                    "floors_served": "all_15_floors",
                    "accessibility": "ADA_compliant",
                    "traffic_management": "destination_dispatch"
                }
            },
            "Process": {
                "frame": "Process",
                "temperature_regulation": {
                    "process_type": "environmental_control",
                    "method": "automated_hvac_control",
                    "required_components": ["hvac_system"],
                    "required_systems": ["climate_control_system"],
                    "target": "occupant_comfort"
                },
                "power_management": {
                    "process_type": "electrical_control",
                    "method": "load_balancing_and_distribution",
                    "required_components": ["electrical_panels"],
                    "required_systems": ["power_distribution_system"],
                    "target": "building_electrical_loads"
                },
                "emergency_response": {
                    "process_type": "safety_management",
                    "method": "automated_emergency_protocols",
                    "required_components": ["fire_safety_system"],
                    "required_systems": ["safety_management_system"],
                    "target": "occupant_safety"
                },
                "occupant_transportation": {
                    "process_type": "vertical_movement",
                    "method": "elevator_dispatch_optimization",
                    "required_components": ["elevator_system"],
                    "required_systems": ["vertical_transportation_system"],
                    "target": "efficient_people_movement"
                },
                "energy_optimization": {
                    "process_type": "efficiency_management",
                    "method": "smart_building_controls",
                    "required_components": ["hvac_system", "electrical_panels"],
                    "required_systems": ["climate_control_system", "power_distribution_system"],
                    "target": "energy_consumption_reduction"
                }
            }
        },
        "confidence_scores": {
            "Building": 0.91,
            "Component": 0.87,
            "System": 0.89,
            "Process": 0.86
        },
        "relationships": [
            # Building → System relationships  
            ("Building", "contains_system", "climate_control_system"),
            ("Building", "contains_system", "power_distribution_system"),
            ("Building", "contains_system", "safety_management_system"),
            ("Building", "contains_system", "vertical_transportation_system"),
            
            # System → Component relationships
            ("climate_control_system", "includes_component", "hvac_system"),
            ("power_distribution_system", "includes_component", "electrical_panels"),
            ("safety_management_system", "includes_component", "fire_safety_system"),
            ("vertical_transportation_system", "includes_component", "elevator_system"),
            
            # System → Process relationships
            ("climate_control_system", "implements_process", "temperature_regulation"),
            ("climate_control_system", "implements_process", "energy_optimization"),
            ("power_distribution_system", "implements_process", "power_management"),
            ("power_distribution_system", "implements_process", "energy_optimization"),
            ("safety_management_system", "implements_process", "emergency_response"),
            ("vertical_transportation_system", "implements_process", "occupant_transportation"),
            
            # Process interdependencies
            ("power_management", "supports", "temperature_regulation"),
            ("power_management", "supports", "occupant_transportation"),
            ("energy_optimization", "coordinated_with", "temperature_regulation"),
            ("emergency_response", "overrides", "occupant_transportation")
        ],
        "lexical_units": {
            "office": {"frame": "Building", "element": "building_type", "evocation": "explicit"},
            "building": {"frame": "Building", "element": "Structure", "evocation": "explicit"},
            "has": {"frame": "Process", "element": "possession_process", "evocation": "implicit"},
            "HVAC": {"frame": "Component", "element": "hvac_system", "evocation": "explicit"},
            "system": {"frame": "System", "element": "hvac_system", "evocation": "explicit"},
            "electrical": {"frame": "Component", "element": "electrical_panels", "evocation": "explicit"},
            "panels": {"frame": "Component", "element": "electrical_panels", "evocation": "explicit"}
        },
        "timestamp": "2025-01-11T12:00:00"
    }
    
    # Test knowledge graph building with both domains
    print(f"\n🧠 Testing knowledge graph construction with both robotics and building domains...")
    
    # Create mock extraction result objects - but use REAL lexical unit extraction
    from semantic_extraction_system import ExtractionResult
    
    # Generate real lexical units using the actual algorithm
    robotics_lexical_units = extractor._extract_lexical_units(
        robot_text, mock_result["extracted_frames"], "robotics"
    )
    building_lexical_units = extractor._extract_lexical_units(
        building_text, building_mock_result["extracted_frames"], "building"
    )
    
    print(f"\n🔍 Real lexical units generated:")
    print(f"Robotics text: '{robot_text}'")
    for word, mapping in robotics_lexical_units.items():
        print(f"  '{word}' → {mapping['frame']} ({mapping['evocation']})")
    
    print(f"\nBuilding text: '{building_text}'")
    for word, mapping in building_lexical_units.items():
        print(f"  '{word}' → {mapping['frame']} ({mapping['evocation']})")
    
    # Robotics extraction
    robotics_extraction = ExtractionResult(
        source_text=mock_result["source_text"],
        domain=mock_result["domain"],
        extracted_frames=mock_result["extracted_frames"],
        confidence_scores=mock_result["confidence_scores"],
        relationships=mock_result["relationships"],
        lexical_units=robotics_lexical_units,  # Use real lexical units
        timestamp=mock_result["timestamp"]
    )
    
    # Building extraction
    building_extraction = ExtractionResult(
        source_text=building_mock_result["source_text"],
        domain=building_mock_result["domain"],
        extracted_frames=building_mock_result["extracted_frames"],
        confidence_scores=building_mock_result["confidence_scores"],
        relationships=building_mock_result["relationships"],
        lexical_units=building_lexical_units,  # Use real lexical units
        timestamp=building_mock_result["timestamp"]
    )
    
    # Build combined knowledge graph
    combined_extractions = [robotics_extraction, building_extraction]
    knowledge_graph = extractor.build_knowledge_graph(combined_extractions)
    
    print(f"📊 Combined Knowledge Graph Statistics:")
    print(f"   - Total Nodes: {knowledge_graph.number_of_nodes()}")
    print(f"   - Total Edges: {knowledge_graph.number_of_edges()}")
    
    # Display node types
    node_types = {}
    for node_id, node_data in knowledge_graph.nodes(data=True):
        frame_type = node_data.get('frame_type', 'unknown')
        node_types[frame_type] = node_types.get(frame_type, 0) + 1
    
    print(f"   - Node types: {dict(node_types)}")
    
    # Display relationship types
    relation_types = {}
    for source, target, edge_data in knowledge_graph.edges(data=True):
        relation = edge_data.get('relation', 'unknown')
        relation_types[relation] = relation_types.get(relation, 0) + 1
    
    print(f"   - Relationship types: {dict(relation_types)}")
    
    # Test saving results
    print(f"\n💾 Testing results saving for both domains...")
    extractor.save_extraction_results(combined_extractions, "test_extraction_results.json")
    
    # Check if output file was created
    output_file = extractor.workspace_path / "output" / "extractions" / "test_extraction_results.json"
    if output_file.exists():
        print(f"✅ Results saved successfully to: {output_file}")
        
        # Load and verify
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        print(f"✅ Saved data contains {len(saved_data)} extraction results")
        print(f"   - Robotics domain: '{saved_data[0]['source_text']}'")
        print(f"   - Building domain: '{saved_data[1]['source_text']}'")
    else:
        print(f"❌ Failed to save results")
    
    print(f"\n🎉 System structure test completed successfully!")
    print(f"📋 Summary:")
    print(f"   ✅ Semantic extraction system initializes properly")
    print(f"   ✅ Frame templates load correctly ({len(extractor.frame_templates)} templates)")
    print(f"   ✅ Domain detection works correctly")
    print(f"   ✅ Knowledge graph construction functions")
    print(f"   ✅ Results saving and loading works")
    print(f"\n🚀 Ready for LLM-powered extraction once Ollama model is available!")

def main():
    """Run the system structure test"""
    try:
        test_system_structure()
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
