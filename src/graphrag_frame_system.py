#!/usr/bin/env python3
"""
GraphRAG Semantic Frame System

A sophisticated GraphRAG implementation that uses semantic frames as the knowledge
representation layer. This system enables contextual retrieval and generation
based on frame semantics, supporting complex reasoning about robotics and buildings.

Key Features:
- Frame-aware knowledge graph construction
- Semantic chunking based on frame elements
- Multi-hop reasoning across frame relationships  
- Cross-domain knowledge bridging
- LLM-powered frame-aware generation
"""

import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import networkx as nx

@dataclass
class FrameEntity:
    """Represents an entity within a semantic frame context"""
    entity_id: str
    entity_type: str  # 'frame', 'concept', 'instance', 'relationship'
    source_frame: str
    content: str
    properties: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    frame_elements: Optional[Dict[str, str]] = None

@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of knowledge"""
    chunk_id: str
    frame_context: str
    content: str
    entities: List[FrameEntity]
    semantic_type: str  # 'frame_definition', 'example', 'relationship', 'cross_domain'
    relevance_score: float = 0.0

class GraphRAGFrameSystem:
    """GraphRAG system built on semantic frame foundations"""
    
    def __init__(self):
        self.knowledge_graph = nx.MultiDiGraph()
        self.semantic_chunks: Dict[str, SemanticChunk] = {}
        self.frame_entities: Dict[str, FrameEntity] = {}
        self.frame_hierarchies = {}
        self.domain_mappings = {}
        
        # GraphRAG specific components
        self.chunk_index = {}  # For fast chunk retrieval
        self.entity_relationships = {}
        self.semantic_clusters = {}
        
    def ingest_frame_knowledge(self, frames_directory: str):
        """Ingest and process semantic frames for GraphRAG"""
        
        print("🔍 Ingesting semantic frame knowledge...")
        
        # Load building frames
        building_frames_dir = os.path.join("output", "frames")
        if os.path.exists(building_frames_dir):
            self._process_domain_frames(building_frames_dir, "building")
        
        # Load training examples
        training_dir = os.path.join("output", "training_examples")
        if os.path.exists(training_dir):
            self._process_training_examples(training_dir)
        
        # Build semantic graph
        self._build_semantic_graph()
        
        # Create semantic chunks
        self._create_semantic_chunks()
        
        # Build retrieval indices
        self._build_retrieval_indices()
        
    def _process_domain_frames(self, directory: str, domain: str):
        """Process frames from a domain directory"""
        
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        frame_data = json.load(f)
                    
                    self._extract_frame_entities(frame_data, domain)
                    
                except Exception as e:
                    print(f"Warning: Could not process {filepath}: {e}")
    
    def _extract_frame_entities(self, frame_data: Dict[str, Any], domain: str):
        """Extract entities from frame data"""
        
        frame_name = frame_data.get('frame', 'unknown')
        
        # Create frame entity
        frame_entity = FrameEntity(
            entity_id=f"frame:{frame_name}",
            entity_type='frame',
            source_frame=frame_name,
            content=frame_data.get('description', ''),
            properties={
                'domain': domain,
                'lexical_units': frame_data.get('lexical_units', []),
                'frame_elements': frame_data.get('frame_elements', {}),
                'relations': frame_data.get('frame_relations', {}),
                'metadata': frame_data.get('metadata', {})
            }
        )
        
        self.frame_entities[frame_entity.entity_id] = frame_entity
        
        # Extract lexical unit entities
        for lexical_unit in frame_data.get('lexical_units', []):
            lu_entity = FrameEntity(
                entity_id=f"lexical_unit:{frame_name}:{lexical_unit}",
                entity_type='concept',
                source_frame=frame_name,
                content=f"Lexical unit '{lexical_unit}' evokes the {frame_name} frame",
                properties={
                    'domain': domain,
                    'lexical_unit': lexical_unit,
                    'frame_evoking': True
                }
            )
            
            self.frame_entities[lu_entity.entity_id] = lu_entity
        
        # Extract frame element entities
        frame_elements = frame_data.get('frame_elements', {})
        for category in ['core', 'peripheral']:
            for fe_name, fe_data in frame_elements.get(category, {}).items():
                fe_entity = FrameEntity(
                    entity_id=f"frame_element:{frame_name}:{fe_name}",
                    entity_type='frame_element',
                    source_frame=frame_name,
                    content=fe_data.get('description', ''),
                    properties={
                        'domain': domain,
                        'element_name': fe_name,
                        'semantic_type': fe_data.get('semantic_type', 'unknown'),
                        'category': category,
                        'lexicon_reference': fe_data.get('lexicon_reference', '')
                    }
                )
                
                self.frame_entities[fe_entity.entity_id] = fe_entity
        
        # Extract example entities
        for i, example in enumerate(frame_data.get('example_sentences', [])):
            if isinstance(example, str):
                example_entity = FrameEntity(
                    entity_id=f"example:{frame_name}:{i}",
                    entity_type='example',
                    source_frame=frame_name,
                    content=example,
                    properties={
                        'domain': domain,
                        'example_type': 'basic'
                    }
                )
                
                self.frame_entities[example_entity.entity_id] = example_entity
        
        # Extract FrameNet examples if available
        framenet_examples = frame_data.get('framenet_connections', {}).get('framenet_examples', [])
        for i, fn_example in enumerate(framenet_examples):
            if isinstance(fn_example, dict):
                fn_entity = FrameEntity(
                    entity_id=f"framenet_example:{frame_name}:{i}",
                    entity_type='framenet_example',
                    source_frame=frame_name,
                    content=fn_example.get('sentence', ''),
                    properties={
                        'domain': domain,
                        'target': fn_example.get('target', ''),
                        'framenet_frame': fn_example.get('frame', ''),
                        'frame_elements': fn_example.get('frame_elements', [])
                    },
                    frame_elements={fe.get('fe_name', ''): fe.get('text', '') 
                                   for fe in fn_example.get('frame_elements', [])}
                )
                
                self.frame_entities[fn_entity.entity_id] = fn_entity
    
    def _process_training_examples(self, training_dir: str):
        """Process training examples for additional semantic content"""
        
        for filename in os.listdir(training_dir):
            if filename.endswith('_training_examples.json'):
                filepath = os.path.join(training_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        training_data = json.load(f)
                    
                    self._extract_training_entities(training_data)
                    
                except Exception as e:
                    print(f"Warning: Could not process {filepath}: {e}")
    
    def _extract_training_entities(self, training_data: Dict[str, Any]):
        """Extract entities from training examples"""
        
        frame_name = training_data.get('frame', 'unknown')
        
        for example in training_data.get('training_examples', []):
            example_id = example.get('id', 'unknown')
            
            # Create training example entity
            training_entity = FrameEntity(
                entity_id=f"training:{frame_name}:{example_id}",
                entity_type='training_example',
                source_frame=frame_name,
                content=example.get('sentence', ''),
                properties={
                    'example_id': example_id,
                    'target': example.get('target', ''),
                    'annotation_notes': example.get('annotation_notes', ''),
                    'training_purpose': 'semantic_annotation'
                },
                frame_elements=example.get('frame_elements', {})
            )
            
            self.frame_entities[training_entity.entity_id] = training_entity
    
    def _build_semantic_graph(self):
        """Build the semantic knowledge graph"""
        
        print("🔗 Building semantic graph...")
        
        # Add all entities as nodes
        for entity_id, entity in self.frame_entities.items():
            self.knowledge_graph.add_node(
                entity_id,
                entity_type=entity.entity_type,
                source_frame=entity.source_frame,
                content=entity.content,
                data=entity
            )
        
        # Add frame relationships
        self._add_frame_relationships()
        
        # Add semantic relationships
        self._add_semantic_relationships()
        
        # Add cross-domain mappings
        self._add_cross_domain_mappings()
    
    def _add_frame_relationships(self):
        """Add explicit frame-based relationships"""
        
        # Group entities by frame
        frame_groups = {}
        for entity_id, entity in self.frame_entities.items():
            frame = entity.source_frame
            if frame not in frame_groups:
                frame_groups[frame] = []
            frame_groups[frame].append(entity_id)
        
        # Add intra-frame relationships
        for frame, entities in frame_groups.items():
            frame_entity_id = f"frame:{frame}"
            
            for entity_id in entities:
                if entity_id != frame_entity_id:
                    self.knowledge_graph.add_edge(
                        entity_id, frame_entity_id,
                        relation_type='belongs_to_frame',
                        weight=1.0
                    )
        
        # Add inter-frame relationships based on frame relations
        for entity_id, entity in self.frame_entities.items():
            if entity.entity_type == 'frame':
                relations = entity.properties.get('relations', {})
                
                # Inheritance relationships
                for parent_frame in relations.get('inherits_from', []):
                    parent_entity_id = f"frame:{parent_frame}"
                    if parent_entity_id in self.frame_entities:
                        self.knowledge_graph.add_edge(
                            entity_id, parent_entity_id,
                            relation_type='inherits_from',
                            weight=0.9
                        )
                
                # Usage relationships
                for used_frame in relations.get('uses', []):
                    used_entity_id = f"frame:{used_frame}"
                    if used_entity_id in self.frame_entities:
                        self.knowledge_graph.add_edge(
                            entity_id, used_entity_id,
                            relation_type='uses',
                            weight=0.7
                        )
    
    def _add_semantic_relationships(self):
        """Add semantic similarity relationships"""
        
        # Find semantically similar entities
        entity_list = list(self.frame_entities.values())
        
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                similarity = self._calculate_semantic_similarity(entity1, entity2)
                
                if similarity > 0.4:  # Threshold for meaningful similarity
                    self.knowledge_graph.add_edge(
                        entity1.entity_id, entity2.entity_id,
                        relation_type='semantically_similar',
                        weight=similarity
                    )
    
    def _add_cross_domain_mappings(self):
        """Add cross-domain conceptual mappings"""
        
        # Define functional equivalences between domains
        equivalences = {
            'sensor': ['sensor', 'detector', 'monitor', 'gauge'],
            'actuator': ['actuator', 'controller', 'valve', 'motor'],
            'component': ['component', 'element', 'part', 'unit'],
            'system': ['system', 'subsystem', 'assembly', 'infrastructure'],
            'maintenance': ['maintenance', 'service', 'inspection', 'repair'],
            'control': ['control', 'regulation', 'management', 'automation']
        }
        
        # Find cross-domain mappings
        for concept, terms in equivalences.items():
            entities_by_term = {term: [] for term in terms}
            
            # Group entities by matching terms
            for entity_id, entity in self.frame_entities.items():
                content_lower = entity.content.lower()
                
                for term in terms:
                    if term in content_lower or term in str(entity.properties).lower():
                        entities_by_term[term].append(entity_id)
            
            # Create cross-domain connections
            for term1, entities1 in entities_by_term.items():
                for term2, entities2 in entities_by_term.items():
                    if term1 != term2:
                        for entity1_id in entities1[:3]:  # Limit connections
                            for entity2_id in entities2[:3]:
                                entity1 = self.frame_entities[entity1_id]
                                entity2 = self.frame_entities[entity2_id]
                                
                                # Only connect across domains
                                domain1 = entity1.properties.get('domain', '')
                                domain2 = entity2.properties.get('domain', '')
                                
                                if domain1 != domain2 and domain1 and domain2:
                                    self.knowledge_graph.add_edge(
                                        entity1_id, entity2_id,
                                        relation_type='cross_domain_equivalent',
                                        weight=0.6,
                                        concept_category=concept
                                    )
    
    def _create_semantic_chunks(self):
        """Create semantically coherent chunks for retrieval"""
        
        print("📦 Creating semantic chunks...")
        
        # Create frame-based chunks
        self._create_frame_chunks()
        
        # Create example-based chunks  
        self._create_example_chunks()
        
        # Create relationship-based chunks
        self._create_relationship_chunks()
        
        # Create cross-domain chunks
        self._create_cross_domain_chunks()
    
    def _create_frame_chunks(self):
        """Create chunks representing complete frame knowledge"""
        
        frame_entities = {eid: e for eid, e in self.frame_entities.items() 
                         if e.entity_type == 'frame'}
        
        for frame_entity_id, frame_entity in frame_entities.items():
            # Gather all entities belonging to this frame
            frame_name = frame_entity.source_frame
            related_entities = [e for e in self.frame_entities.values() 
                              if e.source_frame == frame_name]
            
            # Create comprehensive frame content
            content_parts = [
                f"Frame: {frame_name}",
                f"Description: {frame_entity.content}",
                f"Domain: {frame_entity.properties.get('domain', 'unknown')}"
            ]
            
            # Add lexical units
            lexical_units = frame_entity.properties.get('lexical_units', [])
            if lexical_units:
                content_parts.append(f"Lexical Units: {', '.join(lexical_units[:10])}")
            
            # Add frame elements
            frame_elements = frame_entity.properties.get('frame_elements', {})
            if frame_elements:
                core_elements = list(frame_elements.get('core', {}).keys())
                peripheral_elements = list(frame_elements.get('peripheral', {}).keys())
                content_parts.append(f"Core Elements: {', '.join(core_elements)}")
                content_parts.append(f"Peripheral Elements: {', '.join(peripheral_elements)}")
            
            chunk = SemanticChunk(
                chunk_id=f"frame_chunk:{frame_name}",
                frame_context=frame_name,
                content="\n".join(content_parts),
                entities=related_entities,
                semantic_type='frame_definition',
                relevance_score=1.0
            )
            
            self.semantic_chunks[chunk.chunk_id] = chunk
    
    def _create_example_chunks(self):
        """Create chunks from examples with frame element annotations"""
        
        example_entities = [e for e in self.frame_entities.values() 
                          if e.entity_type in ['training_example', 'framenet_example']]
        
        for example_entity in example_entities:
            if example_entity.frame_elements:
                # Create rich example chunk with frame element context
                content_parts = [
                    f"Example from {example_entity.source_frame} frame:",
                    f"Sentence: {example_entity.content}"
                ]
                
                # Add frame element annotations
                if example_entity.frame_elements:
                    content_parts.append("Frame Element Annotations:")
                    for fe_name, fe_value in example_entity.frame_elements.items():
                        content_parts.append(f"  {fe_name}: {fe_value}")
                
                # Add target information
                target = example_entity.properties.get('target', '')
                if target:
                    content_parts.append(f"Target: {target}")
                
                chunk = SemanticChunk(
                    chunk_id=f"example_chunk:{example_entity.entity_id}",
                    frame_context=example_entity.source_frame,
                    content="\n".join(content_parts),
                    entities=[example_entity],
                    semantic_type='example',
                    relevance_score=0.8
                )
                
                self.semantic_chunks[chunk.chunk_id] = chunk
    
    def _create_relationship_chunks(self):
        """Create chunks representing semantic relationships"""
        
        # Group entities by relationship types
        for edge in self.knowledge_graph.edges(data=True):
            source_id, target_id, edge_data = edge
            relation_type = edge_data.get('relation_type', 'unknown')
            
            source_entity = self.frame_entities.get(source_id)
            target_entity = self.frame_entities.get(target_id)
            
            if source_entity and target_entity:
                content = f"""Semantic Relationship: {relation_type}
Source: {source_entity.content[:100]}...
Target: {target_entity.content[:100]}...
Relationship Strength: {edge_data.get('weight', 0.5)}
Source Frame: {source_entity.source_frame}
Target Frame: {target_entity.source_frame}"""
                
                chunk = SemanticChunk(
                    chunk_id=f"relation_chunk:{source_id}_{target_id}_{relation_type}",
                    frame_context=f"{source_entity.source_frame}-{target_entity.source_frame}",
                    content=content,
                    entities=[source_entity, target_entity],
                    semantic_type='relationship',
                    relevance_score=edge_data.get('weight', 0.5)
                )
                
                self.semantic_chunks[chunk.chunk_id] = chunk
    
    def _create_cross_domain_chunks(self):
        """Create chunks highlighting cross-domain connections"""
        
        # Find cross-domain relationships
        cross_domain_edges = [(s, t, d) for s, t, d in self.knowledge_graph.edges(data=True)
                             if d.get('relation_type') == 'cross_domain_equivalent']
        
        # Group by concept category
        category_groups = {}
        for source_id, target_id, edge_data in cross_domain_edges:
            category = edge_data.get('concept_category', 'general')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append((source_id, target_id, edge_data))
        
        # Create chunks for each category
        for category, edges in category_groups.items():
            if len(edges) >= 2:  # Only create chunk if multiple connections
                content_parts = [f"Cross-Domain Mappings: {category.title()}"]
                
                entities = []
                for source_id, target_id, edge_data in edges[:5]:  # Limit size
                    source_entity = self.frame_entities.get(source_id)
                    target_entity = self.frame_entities.get(target_id)
                    
                    if source_entity and target_entity:
                        entities.extend([source_entity, target_entity])
                        
                        source_domain = source_entity.properties.get('domain', 'unknown')
                        target_domain = target_entity.properties.get('domain', 'unknown')
                        
                        content_parts.append(
                            f"{source_domain} '{source_entity.content[:50]}...' ↔ "
                            f"{target_domain} '{target_entity.content[:50]}...'"
                        )
                
                chunk = SemanticChunk(
                    chunk_id=f"cross_domain_chunk:{category}",
                    frame_context=f"cross_domain_{category}",
                    content="\n".join(content_parts),
                    entities=entities,
                    semantic_type='cross_domain',
                    relevance_score=0.7
                )
                
                self.semantic_chunks[chunk.chunk_id] = chunk
    
    def _build_retrieval_indices(self):
        """Build indices for efficient retrieval"""
        
        print("🔍 Building retrieval indices...")
        
        # Build chunk index by frame
        frame_index = {}
        for chunk_id, chunk in self.semantic_chunks.items():
            frame = chunk.frame_context
            if frame not in frame_index:
                frame_index[frame] = []
            frame_index[frame].append(chunk_id)
        
        self.chunk_index['by_frame'] = frame_index
        
        # Build chunk index by semantic type
        type_index = {}
        for chunk_id, chunk in self.semantic_chunks.items():
            semantic_type = chunk.semantic_type
            if semantic_type not in type_index:
                type_index[semantic_type] = []
            type_index[semantic_type].append(chunk_id)
        
        self.chunk_index['by_type'] = type_index
        
        # Build entity relationship index
        for entity_id, entity in self.frame_entities.items():
            neighbors = list(self.knowledge_graph.neighbors(entity_id))
            self.entity_relationships[entity_id] = neighbors
    
    def retrieve_context(self, query: str, max_chunks: int = 10, 
                        domain_filter: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve relevant context chunks for a query using GraphRAG"""
        
        # Parse query to identify key concepts and frames
        query_analysis = self._analyze_query(query, domain_filter)
        
        # Retrieve relevant chunks
        relevant_chunks = self._retrieve_relevant_chunks(
            query_analysis, max_chunks, domain_filter
        )
        
        # Rank and score chunks
        scored_chunks = self._score_chunks(relevant_chunks, query_analysis)
        
        # Build comprehensive context
        context = self._build_retrieval_context(scored_chunks, query_analysis)
        
        return context
    
    def _analyze_query(self, query: str, domain_filter: Optional[str] = None) -> Dict[str, Any]:
        """Analyze query to identify relevant semantic concepts"""
        
        query_lower = query.lower()
        
        analysis = {
            'query': query,
            'domain_filter': domain_filter,
            'mentioned_frames': [],
            'mentioned_concepts': [],
            'mentioned_entities': [],
            'semantic_intent': self._infer_semantic_intent(query),
            'complexity_level': 'basic'
        }
        
        # Find mentioned frames
        for frame_name in self.frame_entities:
            if frame_name.startswith('frame:'):
                clean_name = frame_name.replace('frame:', '').lower()
                if clean_name in query_lower:
                    analysis['mentioned_frames'].append(frame_name)
        
        # Find mentioned concepts
        concept_terms = ['sensor', 'actuator', 'component', 'system', 'process', 
                        'function', 'maintenance', 'control', 'building', 'robot']
        
        for term in concept_terms:
            if term in query_lower:
                analysis['mentioned_concepts'].append(term)
        
        # Determine complexity
        complexity_indicators = ['compare', 'relationship', 'how', 'why', 'cross-domain']
        if any(indicator in query_lower for indicator in complexity_indicators):
            analysis['complexity_level'] = 'complex'
        
        return analysis
    
    def _infer_semantic_intent(self, query: str) -> str:
        """Infer the semantic intent of the query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return 'definition'
        elif any(word in query_lower for word in ['how', 'process', 'procedure']):
            return 'process'
        elif any(word in query_lower for word in ['compare', 'difference', 'similar']):
            return 'comparison'
        elif any(word in query_lower for word in ['example', 'instance', 'case']):
            return 'examples'
        elif any(word in query_lower for word in ['relationship', 'connection', 'related']):
            return 'relationships'
        else:
            return 'general'
    
    def _retrieve_relevant_chunks(self, query_analysis: Dict[str, Any], 
                                max_chunks: int, domain_filter: Optional[str]) -> List[str]:
        """Retrieve chunks relevant to the query"""
        
        relevant_chunk_ids = set()
        
        # Get chunks from mentioned frames
        for frame_name in query_analysis['mentioned_frames']:
            clean_frame = frame_name.replace('frame:', '')
            frame_chunks = self.chunk_index['by_frame'].get(clean_frame, [])
            relevant_chunk_ids.update(frame_chunks)
        
        # Get chunks by semantic intent
        intent = query_analysis['semantic_intent']
        intent_mapping = {
            'definition': ['frame_definition'],
            'examples': ['example'],
            'relationships': ['relationship'],
            'comparison': ['cross_domain'],
            'process': ['frame_definition', 'example']
        }
        
        for semantic_type in intent_mapping.get(intent, ['frame_definition', 'example']):
            type_chunks = self.chunk_index['by_type'].get(semantic_type, [])
            relevant_chunk_ids.update(type_chunks)
        
        # Add concept-based chunks
        for concept in query_analysis['mentioned_concepts']:
            for chunk_id, chunk in self.semantic_chunks.items():
                if concept in chunk.content.lower():
                    relevant_chunk_ids.add(chunk_id)
        
        # Apply domain filter
        if domain_filter:
            filtered_chunks = []
            for chunk_id in relevant_chunk_ids:
                chunk = self.semantic_chunks[chunk_id]
                chunk_domains = [e.properties.get('domain', '') for e in chunk.entities]
                if domain_filter in chunk_domains:
                    filtered_chunks.append(chunk_id)
            relevant_chunk_ids = set(filtered_chunks)
        
        return list(relevant_chunk_ids)[:max_chunks]
    
    def _score_chunks(self, chunk_ids: List[str], query_analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Score chunks based on relevance to query"""
        
        scored_chunks = []
        
        for chunk_id in chunk_ids:
            chunk = self.semantic_chunks[chunk_id]
            score = chunk.relevance_score
            
            # Boost score based on semantic intent match
            intent = query_analysis['semantic_intent']
            if intent == 'definition' and chunk.semantic_type == 'frame_definition':
                score += 0.3
            elif intent == 'examples' and chunk.semantic_type == 'example':
                score += 0.3
            elif intent == 'comparison' and chunk.semantic_type == 'cross_domain':
                score += 0.4
            
            # Boost score for mentioned concepts
            for concept in query_analysis['mentioned_concepts']:
                if concept in chunk.content.lower():
                    score += 0.2
            
            # Boost score for frame matches
            for frame in query_analysis['mentioned_frames']:
                clean_frame = frame.replace('frame:', '')
                if clean_frame in chunk.frame_context:
                    score += 0.3
            
            scored_chunks.append((chunk_id, score))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks
    
    def _build_retrieval_context(self, scored_chunks: List[Tuple[str, float]], 
                               query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive context for LLM reasoning"""
        
        context = {
            'query': query_analysis['query'],
            'semantic_intent': query_analysis['semantic_intent'],
            'domain_filter': query_analysis['domain_filter'],
            'retrieved_chunks': [],
            'frame_context': {},
            'cross_domain_connections': [],
            'reasoning_guidance': self._generate_reasoning_guidance(query_analysis)
        }
        
        # Add top chunks with metadata
        for chunk_id, score in scored_chunks[:10]:
            chunk = self.semantic_chunks[chunk_id]
            
            context['retrieved_chunks'].append({
                'chunk_id': chunk_id,
                'content': chunk.content,
                'semantic_type': chunk.semantic_type,
                'frame_context': chunk.frame_context,
                'relevance_score': score,
                'entity_count': len(chunk.entities)
            })
        
        # Add frame hierarchies for context
        involved_frames = list(set([chunk.frame_context for _, chunk in 
                                  [(cid, self.semantic_chunks[cid]) for cid, _ in scored_chunks[:5]]]))
        
        for frame in involved_frames:
            if frame in self.frame_hierarchies:
                context['frame_context'][frame] = self.frame_hierarchies[frame]
        
        # Add cross-domain connections if relevant
        if query_analysis['semantic_intent'] in ['comparison', 'relationships']:
            cross_chunks = [chunk for chunk in context['retrieved_chunks'] 
                          if chunk['semantic_type'] == 'cross_domain']
            context['cross_domain_connections'] = cross_chunks
        
        return context
    
    def _generate_reasoning_guidance(self, query_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate guidance for LLM reasoning based on query intent"""
        
        intent = query_analysis['semantic_intent']
        
        guidance = {
            'definition': "Focus on frame definitions, core elements, and lexical units. Provide clear conceptual boundaries.",
            'examples': "Use concrete examples with frame element annotations. Show how concepts manifest in real scenarios.",
            'comparison': "Highlight similarities and differences across domains. Use cross-domain mappings and equivalences.",
            'relationships': "Explain semantic relationships, inheritance hierarchies, and frame interactions.",
            'process': "Describe procedures, workflows, and temporal sequences. Include actor roles and responsibilities.",
            'general': "Provide comprehensive overview drawing from frame definitions, examples, and relationships."
        }
        
        return {
            'primary_guidance': guidance.get(intent, guidance['general']),
            'frame_awareness': "Always consider frame-semantic context and relationships",
            'cross_domain': "Look for connections between robotics and building domains when relevant",
            'semantic_precision': "Use precise frame-semantic terminology and concepts"
        }
    
    def _calculate_semantic_similarity(self, entity1: FrameEntity, entity2: FrameEntity) -> float:
        """Calculate semantic similarity between entities"""
        
        similarity = 0.0
        
        # Entity type similarity
        if entity1.entity_type == entity2.entity_type:
            similarity += 0.3
        
        # Frame similarity
        if entity1.source_frame == entity2.source_frame:
            similarity += 0.4
        
        # Content similarity (simple overlap)
        words1 = set(entity1.content.lower().split())
        words2 = set(entity2.content.lower().split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            similarity += 0.3 * (overlap / total) if total > 0 else 0
        
        return min(similarity, 1.0)

def main():
    """Demonstrate the GraphRAG semantic frame system"""
    
    print("🚀 GraphRAG Semantic Frame System")
    print("=" * 50)
    
    # Initialize system
    system = GraphRAGFrameSystem()
    
    # Ingest frame knowledge
    system.ingest_frame_knowledge(".")
    
    print(f"\n✅ System Ready!")
    print(f"📈 Knowledge Graph Statistics:")
    print(f"   • Frame Entities: {len(system.frame_entities)}")
    print(f"   • Semantic Chunks: {len(system.semantic_chunks)}")
    print(f"   • Graph Nodes: {len(system.knowledge_graph.nodes())}")
    print(f"   • Graph Edges: {len(system.knowledge_graph.edges())}")
    
    # Example queries
    example_queries = [
        "What are building sensors and how do they work?",
        "Compare robot actuators to building actuators",
        "What maintenance processes apply to HVAC systems?",
        "How are building components structured?"
    ]
    
    print(f"\n🔍 Example GraphRAG Retrievals:")
    
    for query in example_queries:
        print(f"\n📝 Query: {query}")
        context = system.retrieve_context(query, max_chunks=5)
        
        print(f"   Intent: {context['semantic_intent']}")
        print(f"   Retrieved Chunks: {len(context['retrieved_chunks'])}")
        print(f"   Frame Contexts: {len(context['frame_context'])}")
        
        # Show top chunk
        if context['retrieved_chunks']:
            top_chunk = context['retrieved_chunks'][0]
            print(f"   Top Chunk: {top_chunk['semantic_type']} (score: {top_chunk['relevance_score']:.2f})")
    
    return system

if __name__ == "__main__":
    system = main()
