#!/usr/bin/env python3
"""
Semantic Knowledge System for Robotics and Buildings

A GraphRAG and LLM-powered system that uses frame semantics for deep knowledge
representation and reasoning about robots and buildings. This system creates
a semantic knowledge graph from the frames and enables intelligent querying
using frame-aware retrieval and generation.

Key Features:
- Frame-based knowledge graph construction
- Semantic similarity and relationship mapping
- LLM-powered frame-aware reasoning
- GraphRAG for contextual knowledge retrieval
- Cross-domain reasoning (robotics ↔ buildings)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import networkx as nx
from dataclasses import dataclass
import hashlib

@dataclass
class FrameNode:
    """Represents a semantic frame node in the knowledge graph"""
    frame_name: str
    frame_type: str  # 'robotics' or 'building'
    description: str
    lexical_units: List[str]
    frame_elements: Dict[str, Any]
    frame_relations: Dict[str, List[str]]
    examples: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class ConceptNode:
    """Represents a specific concept/instance node"""
    concept_id: str
    concept_type: str  # 'component', 'system', 'process', 'actor', etc.
    frame_source: str
    properties: Dict[str, Any]
    relationships: List[Tuple[str, str]]  # (relation_type, target_concept_id)

class SemanticKnowledgeSystem:
    """Main system for frame-based semantic knowledge representation"""
    
    def __init__(self):
        self.knowledge_graph = nx.MultiDiGraph()
        self.frames: Dict[str, FrameNode] = {}
        self.concepts: Dict[str, ConceptNode] = {}
        self.frame_hierarchies = {
            'robotics': {},
            'buildings': {}
        }
        
    def load_frames(self, frames_directory: str):
        """Load all frames from both robotics and building domains"""
        
        # Load robotics frames
        robotics_frames_dir = os.path.join(frames_directory, "robotics")
        if os.path.exists(robotics_frames_dir):
            self._load_domain_frames(robotics_frames_dir, "robotics")
        
        # Load building frames  
        building_frames_dir = os.path.join(frames_directory, "building")
        if os.path.exists(building_frames_dir):
            self._load_domain_frames(building_frames_dir, "building")
            
        # Also check the output/frames directory
        output_frames_dir = os.path.join("output", "frames")
        if os.path.exists(output_frames_dir):
            self._load_domain_frames(output_frames_dir, "building")
            
    def _load_domain_frames(self, directory: str, domain: str):
        """Load frames from a specific domain directory"""
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                        
                    frame_node = FrameNode(
                        frame_name=frame_data.get('frame', filename.replace('.json', '')),
                        frame_type=domain,
                        description=frame_data.get('description', ''),
                        lexical_units=frame_data.get('lexical_units', []),
                        frame_elements=frame_data.get('frame_elements', {}),
                        frame_relations=frame_data.get('frame_relations', {}),
                        examples=frame_data.get('example_sentences', []),
                        metadata=frame_data.get('metadata', {})
                    )
                    
                    self.frames[frame_node.frame_name] = frame_node
                    self.knowledge_graph.add_node(
                        frame_node.frame_name,
                        node_type='frame',
                        domain=domain,
                        data=frame_node
                    )
                    
                except Exception as e:
                    print(f"Warning: Could not load frame {filepath}: {e}")
    
    def build_semantic_graph(self):
        """Build the complete semantic knowledge graph"""
        
        # Add frame relationships
        self._add_frame_relationships()
        
        # Extract and add concepts from frames
        self._extract_concepts_from_frames()
        
        # Add semantic similarity edges
        self._add_semantic_similarities()
        
        # Add cross-domain mappings
        self._add_cross_domain_mappings()
        
        # Generate frame hierarchies
        self._generate_frame_hierarchies()
        
    def _add_frame_relationships(self):
        """Add explicit frame relationship edges"""
        for frame_name, frame_node in self.frames.items():
            relations = frame_node.frame_relations
            
            # Add inheritance relationships
            for parent in relations.get('inherits_from', []):
                if parent in self.frames:
                    self.knowledge_graph.add_edge(
                        frame_name, parent, 
                        relation_type='inherits_from',
                        weight=1.0
                    )
                    
            # Add subframe relationships
            for subframe in relations.get('has_subframes', []):
                if subframe in self.frames:
                    self.knowledge_graph.add_edge(
                        frame_name, subframe,
                        relation_type='has_subframe',
                        weight=0.8
                    )
                    
            # Add usage relationships
            for used_frame in relations.get('uses', []):
                if used_frame in self.frames:
                    self.knowledge_graph.add_edge(
                        frame_name, used_frame,
                        relation_type='uses',
                        weight=0.6
                    )
    
    def _extract_concepts_from_frames(self):
        """Extract specific concepts and instances from frame data"""
        
        for frame_name, frame_node in self.frames.items():
            
            # Extract concepts from lexical units
            for lexical_unit in frame_node.lexical_units:
                concept_id = f"{frame_name}:{lexical_unit}"
                
                concept = ConceptNode(
                    concept_id=concept_id,
                    concept_type=self._infer_concept_type(frame_name),
                    frame_source=frame_name,
                    properties={
                        'lexical_unit': lexical_unit,
                        'domain': frame_node.frame_type,
                        'frame_description': frame_node.description
                    },
                    relationships=[]
                )
                
                self.concepts[concept_id] = concept
                self.knowledge_graph.add_node(
                    concept_id,
                    node_type='concept',
                    domain=frame_node.frame_type,
                    data=concept
                )
                
                # Link concept to its frame
                self.knowledge_graph.add_edge(
                    concept_id, frame_name,
                    relation_type='evokes_frame',
                    weight=1.0
                )
                
            # Extract concepts from examples
            self._extract_concepts_from_examples(frame_node)
    
    def _extract_concepts_from_examples(self, frame_node: FrameNode):
        """Extract specific instances from frame examples"""
        
        examples = []
        
        # Get examples from different sources
        if isinstance(frame_node.examples, list):
            examples.extend(frame_node.examples)
            
        # Check for training examples
        training_file = f"output/training_examples/{frame_node.frame_name}_training_examples.json"
        if os.path.exists(training_file):
            try:
                with open(training_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                    examples.extend(training_data.get('training_examples', []))
            except:
                pass
                
        # Check for FrameNet examples
        framenet_examples = getattr(frame_node, 'framenet_connections', {}).get('framenet_examples', [])
        examples.extend(framenet_examples)
        
        # Process examples to extract concepts
        for example in examples:
            if isinstance(example, dict):
                self._process_example_for_concepts(example, frame_node)
    
    def _process_example_for_concepts(self, example: Dict[str, Any], frame_node: FrameNode):
        """Process a single example to extract semantic concepts"""
        
        # Extract from frame elements if available
        frame_elements = example.get('frame_elements', {})
        
        for fe_name, fe_value in frame_elements.items():
            if isinstance(fe_value, str) and fe_value.strip():
                concept_id = f"{frame_node.frame_name}:{fe_name}:{self._hash_string(fe_value)}"
                
                concept = ConceptNode(
                    concept_id=concept_id,
                    concept_type=fe_name.lower(),
                    frame_source=frame_node.frame_name,
                    properties={
                        'frame_element': fe_name,
                        'value': fe_value,
                        'domain': frame_node.frame_type,
                        'example_context': example.get('sentence', ''),
                        'semantic_type': self._get_semantic_type(frame_node, fe_name)
                    },
                    relationships=[]
                )
                
                self.concepts[concept_id] = concept
                self.knowledge_graph.add_node(
                    concept_id,
                    node_type='instance',
                    domain=frame_node.frame_type,
                    data=concept
                )
                
                # Link to frame
                self.knowledge_graph.add_edge(
                    concept_id, frame_node.frame_name,
                    relation_type='instance_of_frame',
                    weight=0.9
                )
    
    def _add_semantic_similarities(self):
        """Add semantic similarity edges based on shared properties"""
        
        concept_list = list(self.concepts.values())
        
        for i, concept1 in enumerate(concept_list):
            for concept2 in concept_list[i+1:]:
                similarity = self._calculate_semantic_similarity(concept1, concept2)
                
                if similarity > 0.3:  # Threshold for meaningful similarity
                    self.knowledge_graph.add_edge(
                        concept1.concept_id, concept2.concept_id,
                        relation_type='semantically_similar',
                        weight=similarity
                    )
    
    def _add_cross_domain_mappings(self):
        """Add mappings between robotics and building concepts"""
        
        # Define cross-domain mappings based on functional similarity
        cross_mappings = {
            'sensor': ['sensor', 'detector', 'monitor'],
            'actuator': ['actuator', 'controller', 'valve'],
            'component': ['component', 'element', 'part'],
            'system': ['system', 'subsystem', 'assembly'],
            'process': ['process', 'procedure', 'operation'],
            'function': ['function', 'capability', 'purpose'],
            'maintenance': ['maintenance', 'service', 'repair'],
            'control': ['control', 'regulation', 'management']
        }
        
        for concept_id1, concept1 in self.concepts.items():
            for concept_id2, concept2 in self.concepts.items():
                if (concept1.frame_source != concept2.frame_source and
                    concept1.properties.get('domain') != concept2.properties.get('domain')):
                    
                    # Check for functional mapping
                    mapping_strength = self._check_cross_domain_mapping(
                        concept1, concept2, cross_mappings
                    )
                    
                    if mapping_strength > 0.4:
                        self.knowledge_graph.add_edge(
                            concept_id1, concept_id2,
                            relation_type='cross_domain_equivalent',
                            weight=mapping_strength
                        )
    
    def _generate_frame_hierarchies(self):
        """Generate hierarchical representations of frame structures"""
        
        for domain in ['robotics', 'building']:
            domain_frames = {name: frame for name, frame in self.frames.items() 
                           if frame.frame_type == domain}
            
            hierarchy = {}
            
            # Build hierarchy based on inheritance relationships
            for frame_name, frame_node in domain_frames.items():
                parents = frame_node.frame_relations.get('inherits_from', [])
                children = frame_node.frame_relations.get('is_inherited_by', [])
                
                hierarchy[frame_name] = {
                    'parents': parents,
                    'children': children,
                    'level': self._calculate_hierarchy_level(frame_name, domain_frames),
                    'concepts_count': len([c for c in self.concepts.values() 
                                         if c.frame_source == frame_name])
                }
            
            self.frame_hierarchies[domain] = hierarchy
    
    def query_semantic_knowledge(self, query: str, domain: str = None, 
                               context_frames: List[str] = None) -> Dict[str, Any]:
        """Query the semantic knowledge system with natural language"""
        
        # Parse query to identify relevant frames and concepts
        relevant_frames = self._identify_relevant_frames(query, domain)
        relevant_concepts = self._identify_relevant_concepts(query, domain)
        
        # Get contextual subgraph
        subgraph = self._extract_contextual_subgraph(
            relevant_frames + relevant_concepts, context_frames
        )
        
        # Prepare knowledge context for LLM
        knowledge_context = self._prepare_knowledge_context(subgraph, query)
        
        return {
            'query': query,
            'relevant_frames': relevant_frames,
            'relevant_concepts': relevant_concepts,
            'knowledge_context': knowledge_context,
            'subgraph_stats': {
                'nodes': len(subgraph.nodes()),
                'edges': len(subgraph.edges()),
                'domains': list(set([subgraph.nodes[n].get('domain', 'unknown') 
                                   for n in subgraph.nodes()]))
            }
        }
    
    def get_frame_aware_context(self, entity: str, reasoning_type: str = 'comprehensive') -> Dict[str, Any]:
        """Get comprehensive frame-aware context for an entity"""
        
        # Find all relevant frames and concepts
        matches = self._find_entity_matches(entity)
        
        if not matches:
            return {'entity': entity, 'context': 'No semantic frame matches found'}
        
        # Build comprehensive context
        context = {
            'entity': entity,
            'direct_matches': matches,
            'semantic_network': {},
            'cross_domain_connections': {},
            'reasoning_paths': []
        }
        
        # Get semantic network around matches
        for match in matches:
            neighbors = list(self.knowledge_graph.neighbors(match))
            context['semantic_network'][match] = {
                'neighbors': neighbors,
                'edge_types': [self.knowledge_graph[match][neighbor][0].get('relation_type', 'unknown') 
                              for neighbor in neighbors]
            }
        
        # Find cross-domain connections
        context['cross_domain_connections'] = self._find_cross_domain_paths(matches)
        
        # Generate reasoning paths
        context['reasoning_paths'] = self._generate_reasoning_paths(matches, reasoning_type)
        
        return context
    
    def generate_semantic_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the semantic knowledge system"""
        
        summary = {
            'system_overview': {
                'total_frames': len(self.frames),
                'total_concepts': len(self.concepts),
                'graph_nodes': len(self.knowledge_graph.nodes()),
                'graph_edges': len(self.knowledge_graph.edges()),
                'domains': list(self.frame_hierarchies.keys())
            },
            'domain_breakdown': {},
            'frame_statistics': {},
            'concept_statistics': {},
            'relationship_analysis': {},
            'semantic_coverage': {}
        }
        
        # Domain breakdown
        for domain in ['robotics', 'building']:
            domain_frames = [f for f in self.frames.values() if f.frame_type == domain]
            domain_concepts = [c for c in self.concepts.values() 
                             if c.properties.get('domain') == domain]
            
            summary['domain_breakdown'][domain] = {
                'frames': len(domain_frames),
                'concepts': len(domain_concepts),
                'avg_lexical_units': sum(len(f.lexical_units) for f in domain_frames) / len(domain_frames) if domain_frames else 0,
                'frame_hierarchy': self.frame_hierarchies.get(domain, {})
            }
        
        # Frame statistics
        for frame_name, frame_node in self.frames.items():
            summary['frame_statistics'][frame_name] = {
                'lexical_units_count': len(frame_node.lexical_units),
                'frame_elements_count': len(frame_node.frame_elements.get('core', {})) + len(frame_node.frame_elements.get('peripheral', {})),
                'examples_count': len(frame_node.examples),
                'domain': frame_node.frame_type
            }
        
        # Relationship analysis
        edge_types = {}
        for edge in self.knowledge_graph.edges(data=True):
            relation_type = edge[2].get('relation_type', 'unknown')
            edge_types[relation_type] = edge_types.get(relation_type, 0) + 1
        
        summary['relationship_analysis'] = edge_types
        
        return summary
    
    # Helper methods
    def _infer_concept_type(self, frame_name: str) -> str:
        """Infer concept type from frame name"""
        if 'component' in frame_name.lower():
            return 'component'
        elif 'system' in frame_name.lower():
            return 'system'
        elif 'process' in frame_name.lower():
            return 'process'
        elif 'function' in frame_name.lower():
            return 'function'
        elif 'actor' in frame_name.lower():
            return 'actor'
        else:
            return 'entity'
    
    def _hash_string(self, text: str) -> str:
        """Generate short hash for string"""
        return hashlib.md5(text.encode()).hexdigest()[:8]
    
    def _get_semantic_type(self, frame_node: FrameNode, fe_name: str) -> str:
        """Get semantic type for frame element"""
        core_elements = frame_node.frame_elements.get('core', {})
        peripheral_elements = frame_node.frame_elements.get('peripheral', {})
        
        if fe_name in core_elements:
            return core_elements[fe_name].get('semantic_type', 'unknown')
        elif fe_name in peripheral_elements:
            return peripheral_elements[fe_name].get('semantic_type', 'unknown')
        else:
            return 'unknown'
    
    def _calculate_semantic_similarity(self, concept1: ConceptNode, concept2: ConceptNode) -> float:
        """Calculate semantic similarity between two concepts"""
        similarity = 0.0
        
        # Type similarity
        if concept1.concept_type == concept2.concept_type:
            similarity += 0.4
        
        # Frame source similarity
        if concept1.frame_source == concept2.frame_source:
            similarity += 0.3
        
        # Property similarity
        prop1_values = set(str(v) for v in concept1.properties.values())
        prop2_values = set(str(v) for v in concept2.properties.values())
        
        if prop1_values and prop2_values:
            overlap = len(prop1_values.intersection(prop2_values))
            total = len(prop1_values.union(prop2_values))
            similarity += 0.3 * (overlap / total) if total > 0 else 0
        
        return min(similarity, 1.0)
    
    def _check_cross_domain_mapping(self, concept1: ConceptNode, concept2: ConceptNode, 
                                  mappings: Dict[str, List[str]]) -> float:
        """Check if two concepts have cross-domain mapping"""
        
        # Get key terms from both concepts
        terms1 = self._extract_key_terms(concept1)
        terms2 = self._extract_key_terms(concept2)
        
        max_strength = 0.0
        
        for term1 in terms1:
            for term2 in terms2:
                for mapping_key, mapping_values in mappings.items():
                    if term1 in mapping_values and term2 in mapping_values:
                        strength = 0.8 if term1 == term2 else 0.6
                        max_strength = max(max_strength, strength)
        
        return max_strength
    
    def _extract_key_terms(self, concept: ConceptNode) -> List[str]:
        """Extract key terms from a concept for mapping"""
        terms = []
        
        # Add concept type
        terms.append(concept.concept_type)
        
        # Add lexical unit if available
        if 'lexical_unit' in concept.properties:
            terms.append(concept.properties['lexical_unit'])
        
        # Add frame element if available
        if 'frame_element' in concept.properties:
            terms.append(concept.properties['frame_element'].lower())
        
        # Add value words if available
        if 'value' in concept.properties:
            value_words = concept.properties['value'].lower().split()
            terms.extend([word for word in value_words if len(word) > 3])
        
        return terms
    
    def _calculate_hierarchy_level(self, frame_name: str, domain_frames: Dict[str, FrameNode]) -> int:
        """Calculate hierarchy level of a frame"""
        if frame_name not in domain_frames:
            return 0
        
        parents = domain_frames[frame_name].frame_relations.get('inherits_from', [])
        if not parents:
            return 0
        
        max_parent_level = 0
        for parent in parents:
            if parent in domain_frames:
                max_parent_level = max(max_parent_level, 
                                     self._calculate_hierarchy_level(parent, domain_frames))
        
        return max_parent_level + 1
    
    def _identify_relevant_frames(self, query: str, domain: str = None) -> List[str]:
        """Identify frames relevant to the query"""
        relevant = []
        query_lower = query.lower()
        
        for frame_name, frame_node in self.frames.items():
            if domain and frame_node.frame_type != domain:
                continue
            
            # Check frame name
            if frame_name.lower() in query_lower:
                relevant.append(frame_name)
                continue
            
            # Check lexical units
            for lu in frame_node.lexical_units:
                if lu.lower() in query_lower:
                    relevant.append(frame_name)
                    break
            
            # Check description
            if any(word in frame_node.description.lower() for word in query_lower.split()):
                if frame_name not in relevant:
                    relevant.append(frame_name)
        
        return relevant
    
    def _identify_relevant_concepts(self, query: str, domain: str = None) -> List[str]:
        """Identify concepts relevant to the query"""
        relevant = []
        query_lower = query.lower()
        
        for concept_id, concept in self.concepts.items():
            if domain and concept.properties.get('domain') != domain:
                continue
            
            # Check concept properties
            for prop_value in concept.properties.values():
                if isinstance(prop_value, str) and prop_value.lower() in query_lower:
                    relevant.append(concept_id)
                    break
        
        return relevant
    
    def _extract_contextual_subgraph(self, seed_nodes: List[str], 
                                   context_frames: List[str] = None) -> nx.MultiDiGraph:
        """Extract a contextual subgraph around seed nodes"""
        subgraph_nodes = set(seed_nodes)
        
        # Add immediate neighbors
        for node in seed_nodes:
            if node in self.knowledge_graph:
                subgraph_nodes.update(self.knowledge_graph.neighbors(node))
        
        # Add context frames if specified
        if context_frames:
            subgraph_nodes.update(context_frames)
        
        return self.knowledge_graph.subgraph(subgraph_nodes)
    
    def _prepare_knowledge_context(self, subgraph: nx.MultiDiGraph, query: str) -> Dict[str, Any]:
        """Prepare knowledge context for LLM reasoning"""
        
        context = {
            'query': query,
            'frames': {},
            'concepts': {},
            'relationships': [],
            'semantic_patterns': []
        }
        
        # Extract frame information
        for node in subgraph.nodes(data=True):
            if node[1].get('node_type') == 'frame':
                frame_data = node[1]['data']
                context['frames'][node[0]] = {
                    'description': frame_data.description,
                    'lexical_units': frame_data.lexical_units[:10],  # Limit for context
                    'domain': frame_data.frame_type
                }
        
        # Extract concept information
        for node in subgraph.nodes(data=True):
            if node[1].get('node_type') in ['concept', 'instance']:
                concept_data = node[1]['data']
                context['concepts'][node[0]] = {
                    'type': concept_data.concept_type,
                    'frame_source': concept_data.frame_source,
                    'key_properties': {k: v for k, v in list(concept_data.properties.items())[:5]}
                }
        
        # Extract relationships
        for edge in subgraph.edges(data=True):
            context['relationships'].append({
                'from': edge[0],
                'to': edge[1],
                'relation': edge[2].get('relation_type', 'unknown'),
                'weight': edge[2].get('weight', 0.5)
            })
        
        return context
    
    def _find_entity_matches(self, entity: str) -> List[str]:
        """Find all frame/concept matches for an entity"""
        matches = []
        entity_lower = entity.lower()
        
        # Check frames
        for frame_name in self.frames:
            if entity_lower in frame_name.lower():
                matches.append(frame_name)
        
        # Check concepts
        for concept_id, concept in self.concepts.items():
            for prop_value in concept.properties.values():
                if isinstance(prop_value, str) and entity_lower in prop_value.lower():
                    matches.append(concept_id)
                    break
        
        return matches
    
    def _find_cross_domain_paths(self, matches: List[str]) -> Dict[str, Any]:
        """Find paths between different domains"""
        cross_paths = {}
        
        for match1 in matches:
            for match2 in matches:
                if match1 != match2:
                    try:
                        if (match1 in self.knowledge_graph and 
                            match2 in self.knowledge_graph):
                            path = nx.shortest_path(self.knowledge_graph, match1, match2)
                            if len(path) <= 4:  # Reasonable path length
                                cross_paths[f"{match1}->{match2}"] = path
                    except nx.NetworkXNoPath:
                        continue
        
        return cross_paths
    
    def _generate_reasoning_paths(self, matches: List[str], reasoning_type: str) -> List[Dict[str, Any]]:
        """Generate semantic reasoning paths"""
        paths = []
        
        for match in matches:
            if match in self.knowledge_graph:
                neighbors = list(self.knowledge_graph.neighbors(match))
                
                for neighbor in neighbors[:3]:  # Limit for performance
                    edge_data = self.knowledge_graph[match][neighbor][0]
                    
                    paths.append({
                        'from': match,
                        'to': neighbor,
                        'relation': edge_data.get('relation_type', 'unknown'),
                        'reasoning': self._generate_reasoning_explanation(match, neighbor, edge_data)
                    })
        
        return paths
    
    def _generate_reasoning_explanation(self, from_node: str, to_node: str, edge_data: Dict[str, Any]) -> str:
        """Generate natural language explanation for reasoning path"""
        relation = edge_data.get('relation_type', 'unknown')
        
        explanations = {
            'inherits_from': f"{from_node} is a specialized type of {to_node}",
            'has_subframe': f"{from_node} contains the conceptual structure of {to_node}",
            'uses': f"{from_node} utilizes or requires {to_node}",
            'evokes_frame': f"{from_node} activates the semantic frame {to_node}",
            'semantically_similar': f"{from_node} has similar semantic properties to {to_node}",
            'cross_domain_equivalent': f"{from_node} serves a similar function to {to_node} in a different domain"
        }
        
        return explanations.get(relation, f"{from_node} is related to {to_node} via {relation}")

def main():
    """Main function to demonstrate the semantic knowledge system"""
    
    print("🧠 Building Semantic Knowledge System for Robotics and Buildings")
    print("=" * 70)
    
    # Initialize system
    system = SemanticKnowledgeSystem()
    
    # Load frames
    print("📚 Loading semantic frames...")
    system.load_frames(".")
    
    # Build knowledge graph
    print("🔗 Building semantic knowledge graph...")
    system.build_semantic_graph()
    
    # Generate summary
    print("📊 Generating system summary...")
    summary = system.generate_semantic_summary()
    
    print(f"\n✅ Semantic Knowledge System Ready!")
    print(f"📈 System Statistics:")
    print(f"   • Total Frames: {summary['system_overview']['total_frames']}")
    print(f"   • Total Concepts: {summary['system_overview']['total_concepts']}")
    print(f"   • Graph Nodes: {summary['system_overview']['graph_nodes']}")
    print(f"   • Graph Edges: {summary['system_overview']['graph_edges']}")
    print(f"   • Domains: {', '.join(summary['system_overview']['domains'])}")
    
    # Example queries
    print(f"\n🔍 Example Semantic Queries:")
    
    example_queries = [
        "What sensors are used in buildings?",
        "How do robot actuators compare to building actuators?", 
        "What maintenance processes apply to HVAC systems?",
        "What are the structural components in buildings?"
    ]
    
    for query in example_queries:
        print(f"\n📝 Query: {query}")
        result = system.query_semantic_knowledge(query)
        print(f"   Relevant Frames: {len(result['relevant_frames'])}")
        print(f"   Relevant Concepts: {len(result['relevant_concepts'])}")
        print(f"   Knowledge Subgraph: {result['subgraph_stats']['nodes']} nodes, {result['subgraph_stats']['edges']} edges")
    
    return system

if __name__ == "__main__":
    system = main()
