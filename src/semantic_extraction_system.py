#!/usr/bin/env python3
"""
Semantic Knowledge Extraction System for Construction Robotics

An LLM + GraphRAG workflow that automatically extracts semantic knowledge from 
text descriptions of robots and buildings, organizing it into the semantic 
frames we've created. Uses Ollama for local LLM processing.

Key Features:
- Text-to-semantic-frame extraction
- Robot and building knowledge parsing  
- Automatic frame population from descriptions
- GraphRAG-powered knowledge organization
- Construction robotics domain expertise
"""

import json
import os
import requests
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
from pathlib import Path

# Import our frame systems
from semantic_knowledge_system import SemanticKnowledgeSystem
from graphrag_frame_system import GraphRAGFrameSystem

@dataclass
class ExtractionResult:
    """Result of semantic extraction from text"""
    source_text: str
    domain: str  # 'robotics' or 'building'
    extracted_frames: Dict[str, Any]
    confidence_scores: Dict[str, float]
    relationships: List[Tuple[str, str, str]]  # (source, relation, target)
    lexical_units: Dict[str, Dict[str, str]]  # word -> {frame, element, evocation}
    timestamp: str

class SemanticExtractionSystem:
    """
    LLM + GraphRAG system for extracting semantic knowledge from text descriptions
    and organizing it into our semantic frame structure
    """
    
    def __init__(self, ollama_model: str = "llama3.1:8b"):
        """
        Initialize the semantic extraction system
        
        Args:
            ollama_model: Ollama model to use (default: llama3.1:8b for good performance/resource balance)
        """
        self.ollama_base_url = "http://localhost:11434"
        self.model = ollama_model
        self.workspace_path = Path("c:/Users/markm/Documents/GitHub/SemanticFrames")
        
        # Load our semantic frames as templates
        self.frame_templates = self._load_frame_templates()
        self.blueprints = self._load_blueprints()
        
        # Initialize our knowledge systems
        self.knowledge_system = SemanticKnowledgeSystem()
        self.graphrag_system = GraphRAGFrameSystem()
        
        # Extraction prompts for different domains
        self.extraction_prompts = self._initialize_prompts()
        
        print(f"🤖 Semantic Extraction System initialized with {self.model}")
        print(f"📁 Workspace: {self.workspace_path}")
        print(f"🎯 Frame templates loaded: {len(self.frame_templates)}")
    
    def _load_frame_templates(self) -> Dict[str, Any]:
        """Load all semantic frame templates from the workspace"""
        templates = {}
        
        # Load building frames
        building_frames_path = self.workspace_path / "output" / "frames" / "building"
        if building_frames_path.exists():
            for frame_file in building_frames_path.glob("*.json"):
                try:
                    with open(frame_file, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                        frame_name = f"building_{frame_file.stem.lower()}"
                        templates[frame_name] = frame_data
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {frame_file}: {e}")
        
        # Load robotics frames  
        robotics_frames_path = self.workspace_path / "output" / "frames" / "robotics"
        if robotics_frames_path.exists():
            for frame_file in robotics_frames_path.glob("*.json"):
                try:
                    with open(frame_file, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                        frame_name = f"robotics_{frame_file.stem.lower()}"
                        templates[frame_name] = frame_data
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {frame_file}: {e}")
        
        return templates
    
    def _load_blueprints(self) -> Dict[str, Any]:
        """Load blueprint files for reference"""
        blueprints = {}
        
        blueprints_path = self.workspace_path / "output" / "blueprints"
        if blueprints_path.exists():
            for blueprint_file in blueprints_path.rglob("*.json"):
                try:
                    with open(blueprint_file, 'r', encoding='utf-8') as f:
                        blueprint_data = json.load(f)
                        blueprint_name = blueprint_file.stem
                        blueprints[blueprint_name] = blueprint_data
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {blueprint_file}: {e}")
        
        return blueprints
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize extraction prompts for different domains"""
        return {
            "robotics": """
You are an expert in robotics and semantic frame analysis. Extract semantic knowledge from the robot description and organize it according to our semantic frame structure.

ROBOTICS SEMANTIC FRAMES:
- Robot: Core robot entity with Agent/Function/Domain elements
- Component: Physical components like sensors, actuators, manipulators  
- Capability: Functional capabilities like manipulation, sensing, locomotion
- Action: Specific actions like pick, place, navigate, inspect

EXTRACT AND STRUCTURE:
1. Identify the main robot entity and its properties
2. List all physical components mentioned
3. Determine capabilities and what actions they enable
4. Map relationships between frames (Robot uses Components, has Capabilities, performs Actions)

Return ONLY a valid JSON structure matching our semantic frames. Do not include any markdown formatting or explanation text.

ROBOT DESCRIPTION:
{text}

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "Robot": {{
    "Agent": "robot_name",
    "Function": "primary_purpose",
    "Domain": "application_area",
    "Properties": {{"payload": "value", "reach": "value"}}
  }},
  "Component": [
    {{"name": "component_name", "type": "component_type"}}
  ],
  "Capability": [
    {{"name": "capability_name", "enabled_actions": ["action1", "action2"]}}
  ],
  "Action": [
    {{"name": "action_name"}}
  ]
}}
""",
            
            "building": """
You are an expert in building systems and semantic frame analysis. Extract semantic knowledge from the building description and organize it according to our semantic frame structure.

BUILDING SEMANTIC FRAMES:
- Building: Core building entity with Asset/Function/Location elements  
- System: Building systems like HVAC, electrical, structural, security
- Component: Physical components like ducts, sensors, beams, doors
- Process: Operational processes like heating, monitoring, maintenance

EXTRACT AND STRUCTURE:
1. Identify the main building entity and its properties
2. List all building systems mentioned
3. Determine components within each system
4. Identify operational processes and procedures  
5. Map relationships between frames (Building contains Systems, Systems contain Components, Systems enable Processes)

Return ONLY a valid JSON structure matching our semantic frames. Do not include any markdown formatting or explanation text.

BUILDING DESCRIPTION:
{text}

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "Building": {{
    "Asset": "building_name",
    "Function": "building_purpose", 
    "Location": "building_location",
    "Properties": {{"height": "value", "area": "value"}}
  }},
  "System": [
    {{"name": "system_name", "type": "system_type"}}
  ],
  "Component": [
    {{"name": "component_name", "system": "parent_system"}}
  ],
  "Process": [
    {{"name": "process_name", "system": "related_system"}}
  ]
}}
""",
            
            "general": """
You are an expert in semantic analysis for construction robotics. Analyze the text and determine if it describes robots, buildings, or both. Extract semantic knowledge accordingly.

SEMANTIC FRAME STRUCTURE:
ROBOTICS: Robot → Capability → Component → Action
BUILDINGS: Building → System → Component → Process  

CROSS-DOMAIN MAPPING:
- Robot Actions ↔ Building Processes (for construction robotics coordination)
- Robot Components ↔ Building Components (for interaction understanding)

Extract and organize information into appropriate semantic frames, noting any cross-domain relationships.

Return ONLY a valid JSON structure. Do not include any markdown formatting or explanation text.

TEXT DESCRIPTION:
{text}

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "Robot": {{"Agent": "robot_name", "Function": "robot_purpose"}},
  "Building": {{"Asset": "building_name", "Function": "building_purpose"}},
  "Cross_Domain_Relations": [
    {{"robot_element": "element", "building_element": "element", "relationship": "type"}}
  ]
}}
"""
        }
    
    def _call_ollama(self, prompt: str, temperature: float = 0.3) -> str:
        """Call Ollama API with the given prompt"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error calling Ollama: {e}")
            return ""
    
    def _detect_domain(self, text: str) -> str:
        """Detect if text describes robots, buildings, or both"""
        robot_keywords = [
            'robot', 'manipulator', 'sensor', 'actuator', 'gripper', 'arm', 'mobile',
            'autonomous', 'navigation', 'pick', 'place', 'grasp', 'dof', 'joint'
        ]
        
        building_keywords = [
            'building', 'structure', 'hvac', 'electrical', 'plumbing', 'foundation',
            'floor', 'wall', 'roof', 'system', 'duct', 'pipe', 'beam', 'column'
        ]
        
        text_lower = text.lower()
        robot_score = sum(1 for keyword in robot_keywords if keyword in text_lower)
        building_score = sum(1 for keyword in building_keywords if keyword in text_lower)
        
        if robot_score > building_score:
            return "robotics"
        elif building_score > robot_score:
            return "building"
        else:
            return "general"
    
    def extract_semantic_knowledge(self, text: str, domain: Optional[str] = None) -> ExtractionResult:
        """
        Extract semantic knowledge from text description
        
        Args:
            text: Input text describing robots, buildings, or both
            domain: Optional domain hint ('robotics', 'building', or None for auto-detection)
        
        Returns:
            ExtractionResult with extracted semantic frames
        """
        # Detect domain if not provided
        if domain is None:
            domain = self._detect_domain(text)
        
        print(f"🔍 Extracting semantic knowledge from {domain} description...")
        
        # Select appropriate prompt
        prompt_template = self.extraction_prompts.get(domain, self.extraction_prompts["general"])
        prompt = prompt_template.format(text=text)
        
        # Call LLM for extraction
        llm_response = self._call_ollama(prompt)
        
        # Parse LLM response into structured format
        extracted_frames = self._parse_llm_response(llm_response, domain)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(text, extracted_frames)
        
        # Extract relationships
        relationships = self._extract_relationships(extracted_frames)
        
        # Extract lexical units mapping
        lexical_units = self._extract_lexical_units(text, extracted_frames, domain)
        
        return ExtractionResult(
            source_text=text,
            domain=domain,
            extracted_frames=extracted_frames,
            confidence_scores=confidence_scores,
            relationships=relationships,
            lexical_units=lexical_units,
            timestamp=datetime.now().isoformat()
        )
    
    def _parse_llm_response(self, response: str, domain: str) -> Dict[str, Any]:
        """Parse LLM response into structured semantic frames"""
        try:
            # Remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                extracted_data = json.loads(json_str)
                return extracted_data
            else:
                # Fallback: parse structured text response
                return self._parse_text_response(response, domain)
        except Exception as e:
            print(f"⚠️ Warning: Could not parse LLM response as JSON: {e}")
            return self._parse_text_response(response, domain)
    
    def _parse_text_response(self, response: str, domain: str) -> Dict[str, Any]:
        """Fallback parser for text-based LLM responses"""
        # Simple text parsing - can be enhanced based on actual LLM output patterns
        result = {
            "domain": domain,
            "raw_response": response,
            "parsed_entities": []
        }
        
        # Extract common patterns
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and len(line) > 5:
                key, value = line.split(':', 1)
                result[key.strip().lower().replace(' ', '_')] = value.strip()
        
        return result
    
    def _calculate_confidence_scores(self, text: str, extracted_frames: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for extracted information"""
        scores = {}
        
        # Simple confidence calculation based on keyword matching
        text_lower = text.lower()
        
        for frame_name, frame_data in extracted_frames.items():
            if isinstance(frame_data, dict):
                confidence = 0.5  # Base confidence
                
                # Increase confidence if frame data contains relevant keywords
                frame_str = json.dumps(frame_data).lower()
                common_words = set(text_lower.split()) & set(frame_str.split())
                confidence += min(len(common_words) * 0.05, 0.4)
                
                scores[frame_name] = min(confidence, 1.0)
            else:
                scores[frame_name] = 0.3  # Low confidence for non-dict data
        
        return scores
    
    def _extract_relationships(self, extracted_frames: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Extract relationships between frame elements"""
        relationships = []
        
        # Extract relationships based on frame structure
        for frame_name, frame_data in extracted_frames.items():
            if isinstance(frame_data, dict):
                for key, value in frame_data.items():
                    if isinstance(value, list):
                        for item in value:
                            relationships.append((frame_name, "contains", str(item)))
                    elif isinstance(value, str) and value:
                        relationships.append((frame_name, "has_property", f"{key}:{value}"))
        
        return relationships
    
    def _extract_lexical_units(self, text: str, extracted_frames: Dict[str, Any], domain: str) -> Dict[str, Dict[str, str]]:
        """Extract lexical unit mappings from text to semantic frames"""
        lexical_units = {}
        text_words = text.lower().split()
        
        # Domain-specific keyword mappings
        if domain == "robotics":
            keyword_mappings = {
                "robot": {"frame": "Robot", "element": "Agent", "evocation": "explicit"},
                "mobile": {"frame": "Robot", "element": "robot_type", "evocation": "explicit"},
                "manipulator": {"frame": "Component", "element": "robotic_arm", "evocation": "explicit"},
                "arm": {"frame": "Component", "element": "robotic_arm", "evocation": "explicit"},
                "camera": {"frame": "Component", "element": "rgb_camera", "evocation": "explicit"},
                "sensor": {"frame": "Component", "element": "sensor", "evocation": "implicit"},
                "gripper": {"frame": "Component", "element": "gripper", "evocation": "explicit"},
                "navigate": {"frame": "Action", "element": "navigation", "evocation": "explicit"},
                "pick": {"frame": "Action", "element": "pick_action", "evocation": "explicit"},
                "place": {"frame": "Action", "element": "place_action", "evocation": "explicit"}
            }
        elif domain == "building":
            keyword_mappings = {
                "building": {"frame": "Building", "element": "Structure", "evocation": "explicit"},
                "office": {"frame": "Building", "element": "building_type", "evocation": "explicit"},
                "hvac": {"frame": "Component", "element": "hvac_system", "evocation": "explicit"},
                "electrical": {"frame": "Component", "element": "electrical_panels", "evocation": "explicit"},
                "panels": {"frame": "Component", "element": "electrical_panels", "evocation": "explicit"},
                "system": {"frame": "System", "element": "system", "evocation": "explicit"},
                # Removed "has" - it's a relation, not a process
                "control": {"frame": "Process", "element": "control_process", "evocation": "explicit"},
                "management": {"frame": "Process", "element": "management_process", "evocation": "explicit"},
                "monitoring": {"frame": "Process", "element": "monitoring_process", "evocation": "explicit"},
                "regulation": {"frame": "Process", "element": "regulation_process", "evocation": "explicit"},
                "distribution": {"frame": "Process", "element": "distribution_process", "evocation": "explicit"},
                "heating": {"frame": "Process", "element": "heating_process", "evocation": "explicit"},
                "cooling": {"frame": "Process", "element": "cooling_process", "evocation": "explicit"},
                "maintenance": {"frame": "Process", "element": "maintenance_process", "evocation": "explicit"}
            }
        else:
            keyword_mappings = {}
        
        # Enhanced mapping with multi-word and context awareness
        skip_next = False
        for i, word in enumerate(text_words):
            if skip_next:
                skip_next = False
                continue
                
            clean_word = word.strip('.,!?').lower()
            
            # Check for multi-word patterns first
            if domain == "building" and i < len(text_words) - 1:
                next_word = text_words[i+1].strip('.,!?').lower()
                two_word = f"{clean_word} {next_word}"
                if two_word == "hvac system":
                    lexical_units["hvac"] = {"frame": "Component", "element": "hvac_system", "evocation": "explicit"}
                    lexical_units["system"] = {"frame": "System", "element": "hvac_system", "evocation": "explicit"}
                    skip_next = True  # Skip processing "system" as a single word
                    continue
            
            # Single word mappings
            if clean_word in keyword_mappings:
                lexical_units[clean_word] = keyword_mappings[clean_word]
        
        return lexical_units
    
    def process_multiple_descriptions(self, descriptions: List[str]) -> List[ExtractionResult]:
        """Process multiple text descriptions and extract semantic knowledge"""
        results = []
        
        print(f"🔄 Processing {len(descriptions)} descriptions...")
        
        for i, description in enumerate(descriptions, 1):
            print(f"📝 Processing description {i}/{len(descriptions)}")
            result = self.extract_semantic_knowledge(description)
            results.append(result)
        
        return results
    
    def process_robotics_descriptions(self, descriptions: List[str]) -> List[ExtractionResult]:
        """Process robotics descriptions specifically"""
        results = []
        
        print(f"🤖 Processing {len(descriptions)} robotics descriptions...")
        
        for i, description in enumerate(descriptions, 1):
            print(f"📝 Processing robotics description {i}/{len(descriptions)}")
            result = self.extract_semantic_knowledge(description, domain="robotics")
            results.append(result)
        
        return results
    
    def process_building_descriptions(self, descriptions: List[str]) -> List[ExtractionResult]:
        """Process building descriptions specifically"""
        results = []
        
        print(f"🏗️ Processing {len(descriptions)} building descriptions...")
        
        for i, description in enumerate(descriptions, 1):
            print(f"📝 Processing building description {i}/{len(descriptions)}")
            result = self.extract_semantic_knowledge(description, domain="building")
            results.append(result)
        
        return results
    
    def build_knowledge_graph(self, extraction_results: List[ExtractionResult]) -> nx.DiGraph:
        """Build a knowledge graph from extraction results (domain-specific, no cross-domain interactions)"""
        graph = nx.DiGraph()
        
        for result in extraction_results:
            # Add nodes for each frame within the same domain
            for frame_name, frame_data in result.extracted_frames.items():
                node_id = f"{result.domain}_{frame_name}_{id(result)}"  # Unique ID per result
                graph.add_node(
                    node_id,
                    frame_type=frame_name,
                    domain=result.domain,
                    data=frame_data,
                    source_text=result.source_text[:100] + "..." if len(result.source_text) > 100 else result.source_text,
                    confidence=result.confidence_scores.get(frame_name, 0.0)
                )
            
            # Add edges for relationships within the same domain only
            for source, relation, target in result.relationships:
                source_id = f"{result.domain}_{source}_{id(result)}"
                target_id = f"{result.domain}_{target}_{id(result)}"
                graph.add_edge(source_id, target_id, relation=relation)
        
        return graph
    
    def build_robotics_knowledge_graph(self, robotics_results: List[ExtractionResult]) -> nx.DiGraph:
        """Build a knowledge graph specifically for robotics domain"""
        # Filter to ensure only robotics results
        robotics_only = [r for r in robotics_results if r.domain == "robotics"]
        return self.build_knowledge_graph(robotics_only)
    
    def build_building_knowledge_graph(self, building_results: List[ExtractionResult]) -> nx.DiGraph:
        """Build a knowledge graph specifically for building domain"""
        # Filter to ensure only building results
        building_only = [r for r in building_results if r.domain == "building"]
        return self.build_knowledge_graph(building_only)
    
    def save_extraction_results(self, results: List[ExtractionResult], output_file: str):
        """Save extraction results to JSON file"""
        output_path = self.workspace_path / "output" / "extractions" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "source_text": result.source_text,
                "domain": result.domain,
                "extracted_frames": result.extracted_frames,
                "confidence_scores": result.confidence_scores,
                "relationships": result.relationships,
                "lexical_units": result.lexical_units,
                "timestamp": result.timestamp
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Extraction results saved to: {output_path}")
    
    def save_robotics_results(self, robotics_results: List[ExtractionResult]):
        """Save robotics-specific extraction results"""
        # Filter to ensure only robotics results
        robotics_only = [r for r in robotics_results if r.domain == "robotics"]
        self.save_extraction_results(robotics_only, "robot_test_result.json")
    
    def save_building_results(self, building_results: List[ExtractionResult]):
        """Save building-specific extraction results"""
        # Filter to ensure only building results
        building_only = [r for r in building_results if r.domain == "building"]
        self.save_extraction_results(building_only, "building_test_result.json")
    
    def save_knowledge_graph(self, graph: nx.DiGraph, filename: str = "knowledge_graph.json"):
        """Save knowledge graph to JSON file"""
        output_path = self.workspace_path / "output" / "extractions" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert graph to serializable format
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, node_data in graph.nodes(data=True):
            graph_data["nodes"].append({
                "id": node_id,
                "frame_type": node_data.get("frame_type"),
                "domain": node_data.get("domain"),
                "confidence": node_data.get("confidence", 0.0),
                "source_text": node_data.get("source_text", ""),
                "data": node_data.get("data", {})
            })
        
        # Add edges
        for source, target, edge_data in graph.edges(data=True):
            graph_data["edges"].append({
                "source": source,
                "target": target,
                "relation": edge_data.get("relation", "")
            })
        
        graph_data["metadata"] = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Knowledge graph saved to: {output_path}")

def main():
    """Example usage of the Semantic Extraction System with separate domain processing"""
    
    # Initialize the system
    extractor = SemanticExtractionSystem()
    
    # Example robot descriptions (multiple for testing)
    robot_descriptions = [
        """
        The mobile manipulation robot is equipped with a 6-DOF robotic arm for precise object handling.
        It features an RGB-D camera for object detection and a LiDAR sensor for navigation.
        The robot can autonomously navigate through building environments, inspect electrical outlets,
        and perform maintenance tasks. It has a gripper with force feedback for safe object manipulation.
        """,
        """
        The autonomous inspection robot has a camera sensor and ultrasonic sensors for navigation.
        It can detect defects in building structures and measure dimensional accuracy.
        The robot is equipped with a mobile base and can operate in indoor environments.
        """
    ]
    
    # Example building descriptions (multiple for testing)
    building_descriptions = [
        """
        The smart office building has an advanced HVAC system with automated temperature control.
        It includes motion sensors for occupancy detection, LED lighting with dimming capabilities,
        and a fire suppression system with smoke detectors. The building's electrical system
        features smart outlets that can be monitored remotely for energy consumption.
        """,
        """
        The automated residential building features a climate control system and electrical panels.
        The building has integrated security systems and energy management capabilities.
        Smart sensors monitor the building's environmental conditions and optimize energy usage.
        """
    ]
    
    # Process domains separately
    print("🤖 Processing robotics descriptions...")
    robotics_results = extractor.process_robotics_descriptions(robot_descriptions)
    
    print("\n🏗️ Processing building descriptions...")
    building_results = extractor.process_building_descriptions(building_descriptions)
    
    # Build separate knowledge graphs
    print("\n📊 Building separate knowledge graphs...")
    robotics_graph = extractor.build_robotics_knowledge_graph(robotics_results)
    building_graph = extractor.build_building_knowledge_graph(building_results)
    
    # Combine graphs for overall knowledge representation (no cross-domain interactions)
    combined_graph = nx.compose(robotics_graph, building_graph)
    
    # Save domain-specific results
    print("\n💾 Saving results...")
    extractor.save_robotics_results(robotics_results)
    extractor.save_building_results(building_results)
    extractor.save_knowledge_graph(combined_graph, "knowledge_graph.json")
    
    print("\n🎯 Semantic extraction completed!")
    print(f"🤖 Robotics: {len(robotics_results)} extractions, {robotics_graph.number_of_nodes()} nodes, {robotics_graph.number_of_edges()} edges")
    print(f"🏗️ Building: {len(building_results)} extractions, {building_graph.number_of_nodes()} nodes, {building_graph.number_of_edges()} edges")
    print(f"📊 Combined knowledge graph: {combined_graph.number_of_nodes()} total nodes, {combined_graph.number_of_edges()} total edges")
    
    print("\n📁 Output files generated:")
    print("   - output/extractions/robot_test_result.json")
    print("   - output/extractions/building_test_result.json") 
    print("   - output/extractions/knowledge_graph.json")

if __name__ == "__main__":
    main()
