#!/usr/bin/env python3
"""
LLM-Powered Frame-Semantic Reasoning System

An advanced reasoning system that combines semantic frame knowledge with LLM
capabilities for intelligent query processing about robotics and buildings.
Uses frame semantics as the knowledge representation layer and GraphRAG
for contextual retrieval.

Key Features:
- Frame-aware prompt engineering
- Semantic reasoning with frame context
- Cross-domain knowledge bridging
- Multi-hop reasoning across frame relationships
- Professional domain expertise integration
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Import our frame systems
from semantic_knowledge_system import SemanticKnowledgeSystem
from graphrag_frame_system import GraphRAGFrameSystem

@dataclass
class ReasoningContext:
    """Context for frame-semantic reasoning"""
    query: str
    domain: Optional[str]
    frames_involved: List[str]
    concepts_identified: List[str]
    semantic_intent: str
    reasoning_type: str
    knowledge_chunks: List[Dict[str, Any]]
    cross_domain_connections: List[Dict[str, Any]]

class FrameSemanticReasoner:
    """LLM-powered reasoning system using frame semantics"""
    
    def __init__(self):
        self.knowledge_system = SemanticKnowledgeSystem()
        self.graphrag_system = GraphRAGFrameSystem()
        self.reasoning_templates = {}
        self.domain_expertise = {}
        
        # Initialize systems
        self._initialize_systems()
        self._load_reasoning_templates()
        self._load_domain_expertise()
    
    def _initialize_systems(self):
        """Initialize the underlying knowledge systems"""
        print("🧠 Initializing Frame-Semantic Reasoning System...")
        
        # Load semantic knowledge
        self.knowledge_system.load_frames(".")
        self.knowledge_system.build_semantic_graph()
        
        # Load GraphRAG knowledge
        self.graphrag_system.ingest_frame_knowledge(".")
        
        print("✅ Frame-semantic knowledge systems ready!")
    
    def _load_reasoning_templates(self):
        """Load prompt templates for different reasoning tasks"""
        
        self.reasoning_templates = {
            'definition': {
                'system_prompt': """You are a frame-semantic expert specializing in robotics and building domains. 
Your role is to provide precise, frame-aware definitions using semantic frame theory.

FRAME-SEMANTIC GUIDELINES:
- Always consider the semantic frame that concepts evoke
- Identify core and peripheral frame elements
- Use professional domain terminology
- Reference frame relationships and hierarchies
- Provide examples with frame element annotations

KNOWLEDGE INTEGRATION:
- Use ISO standards terminology for buildings (IFC, BIM, IDM)
- Apply robotics frame semantics for robot concepts
- Consider cross-domain equivalences when relevant
- Maintain semantic precision and professional accuracy""",
                
                'user_template': """Based on the frame-semantic knowledge provided below, give a comprehensive definition of: {query}

SEMANTIC FRAME CONTEXT:
{frame_context}

RELEVANT KNOWLEDGE CHUNKS:
{knowledge_chunks}

REASONING GUIDANCE:
{reasoning_guidance}

Please provide:
1. Frame-semantic definition with core elements
2. Professional domain context
3. Key frame elements and their roles
4. Examples with frame annotations
5. Relationships to other frames/concepts"""
            },
            
            'comparison': {
                'system_prompt': """You are a cross-domain frame-semantic analyst specializing in robotics and buildings.
Your expertise lies in identifying semantic similarities and differences across domains using frame theory.

COMPARISON METHODOLOGY:
- Identify equivalent frames across domains
- Compare frame elements and their semantic roles
- Highlight functional similarities and differences  
- Use cross-domain mappings and equivalences
- Consider professional context and standards

FRAME-AWARE ANALYSIS:
- Map corresponding semantic roles across domains
- Identify shared frame structures
- Explain domain-specific adaptations
- Reference professional standards and practices""",
                
                'user_template': """Compare the following concepts across robotics and building domains: {query}

CROSS-DOMAIN KNOWLEDGE:
{cross_domain_connections}

FRAME CONTEXTS:
{frame_context}

SEMANTIC MAPPINGS:
{knowledge_chunks}

Provide a frame-semantic comparison including:
1. Equivalent frames and concepts across domains
2. Shared semantic roles and frame elements
3. Domain-specific variations and adaptations
4. Professional context and standards alignment
5. Examples showing cross-domain equivalences"""
            },
            
            'process': {
                'system_prompt': """You are a process semantics expert for robotics and building operations.
You understand procedural frames, actor roles, and temporal sequences in technical domains.

PROCESS FRAME ANALYSIS:
- Identify Actor, Process, Goal, and Method frame elements
- Map temporal sequences and dependencies
- Consider professional roles and responsibilities
- Reference standards and best practices
- Include safety and compliance considerations

DOMAIN EXPERTISE:
- Building processes: Construction, maintenance, operations, inspections
- Robot processes: Control, navigation, manipulation, sensing
- Cross-domain: Automation, monitoring, maintenance, safety""",
                
                'user_template': """Explain the process or procedure for: {query}

PROCESS FRAME CONTEXT:
{frame_context}

PROCEDURAL KNOWLEDGE:
{knowledge_chunks}

ACTOR AND ROLE INFORMATION:
{reasoning_guidance}

Provide a frame-semantic process description including:
1. Process frame structure and core elements
2. Actor roles and responsibilities
3. Temporal sequence and dependencies
4. Standards and compliance requirements
5. Examples with frame element annotations"""
            },
            
            'examples': {
                'system_prompt': """You are a frame-semantic example generator for robotics and building domains.
You excel at creating concrete instances that demonstrate frame structures and semantic relationships.

EXAMPLE GENERATION PRINCIPLES:
- Use real-world scenarios from both domains
- Include frame element annotations
- Show semantic relationships clearly
- Reference professional standards and practices
- Provide diverse examples across contexts

ANNOTATION METHODOLOGY:
- Mark frame-evoking expressions
- Identify core and peripheral elements
- Show inter-frame relationships
- Include professional terminology""",
                
                'user_template': """Provide concrete examples for: {query}

FRAME-SEMANTIC EXAMPLES:
{knowledge_chunks}

DOMAIN CONTEXTS:
{frame_context}

EXAMPLE GUIDANCE:
{reasoning_guidance}

Generate frame-semantic examples including:
1. Real-world scenarios from robotics and/or buildings
2. Frame element annotations for key examples
3. Cross-domain examples if applicable
4. Professional context and standards
5. Diverse example types and complexity levels"""
            },
            
            'relationships': {
                'system_prompt': """You are a semantic relationship analyst for frame-based knowledge systems.
You specialize in mapping and explaining relationships between frames, concepts, and domains.

RELATIONSHIP ANALYSIS:
- Frame inheritance and specialization
- Cross-frame dependencies and usage
- Semantic similarity and equivalence
- Cross-domain mappings and bridges
- Professional domain connections

ANALYTICAL APPROACH:
- Use frame relationship types (inherits_from, uses, subframe_of)
- Identify semantic roles and mappings
- Explain professional domain connections
- Consider standards alignment and practices""",
                
                'user_template': """Analyze the relationships for: {query}

RELATIONSHIP NETWORK:
{knowledge_chunks}

FRAME HIERARCHIES:
{frame_context}

CROSS-DOMAIN CONNECTIONS:
{cross_domain_connections}

Provide relationship analysis including:
1. Frame inheritance and specialization hierarchies
2. Cross-frame dependencies and usage patterns
3. Semantic equivalences and similarities
4. Cross-domain mappings and bridges
5. Professional domain relationship contexts"""
            },
            
            'comprehensive': {
                'system_prompt': """You are a comprehensive frame-semantic expert covering robotics and building domains.
You provide thorough analysis combining definitions, examples, relationships, and cross-domain insights.

COMPREHENSIVE ANALYSIS APPROACH:
- Integrate frame definitions with examples
- Show relationships and hierarchies
- Include cross-domain perspectives
- Reference professional standards
- Provide actionable insights

KNOWLEDGE SYNTHESIS:
- Combine multiple frame perspectives
- Integrate examples with theory
- Show practical applications
- Consider professional context""",
                
                'user_template': """Provide comprehensive frame-semantic analysis for: {query}

COMPLETE KNOWLEDGE BASE:
{knowledge_chunks}

FRAME NETWORK:
{frame_context}

CROSS-DOMAIN INSIGHTS:
{cross_domain_connections}

REASONING APPROACH:
{reasoning_guidance}

Provide comprehensive analysis including:
1. Frame-semantic definitions and core concepts
2. Concrete examples with annotations
3. Relationship networks and hierarchies
4. Cross-domain connections and mappings
5. Professional applications and standards
6. Actionable insights and recommendations"""
            }
        }
    
    def _load_domain_expertise(self):
        """Load domain-specific expertise for enhanced reasoning"""
        
        self.domain_expertise = {
            'building': {
                'standards': ['ISO 16739-1 (IFC)', 'ISO 19650 (BIM)', 'ISO 29481 (IDM)'],
                'key_concepts': ['asset', 'component', 'system', 'function', 'process'],
                'professional_roles': ['architect', 'engineer', 'contractor', 'facility_manager'],
                'expertise_areas': ['construction', 'MEP', 'structural', 'sustainability', 'operations'],
                'frame_priorities': ['Building', 'Building_System', 'Building_Component', 'Building_Process']
            },
            
            'robotics': {
                'standards': ['IEEE Standards', 'ISO 8373', 'ROS Standards'],
                'key_concepts': ['sensor', 'actuator', 'controller', 'navigation', 'manipulation'],
                'professional_roles': ['robotics_engineer', 'control_engineer', 'system_integrator'],
                'expertise_areas': ['perception', 'control', 'planning', 'HMI', 'safety'],
                'frame_priorities': ['Robotics_System', 'Sensor', 'Actuator', 'Control', 'Navigation']
            },
            
            'cross_domain': {
                'shared_concepts': ['sensor', 'actuator', 'system', 'maintenance', 'automation'],
                'equivalence_mappings': {
                    'sensor': ['building_sensor', 'robot_sensor'],
                    'actuator': ['building_actuator', 'robot_actuator'],
                    'system': ['building_system', 'robotic_system'],
                    'maintenance': ['building_maintenance', 'robot_maintenance']
                },
                'integration_areas': ['smart_buildings', 'automated_systems', 'IoT', 'digital_twins']
            }
        }
    
    def reason_about_query(self, query: str, domain: Optional[str] = None, 
                          reasoning_type: str = 'comprehensive') -> Dict[str, Any]:
        """Main reasoning function using frame semantics and LLM"""
        
        print(f"🤔 Reasoning about: {query}")
        
        # Step 1: Analyze query and retrieve context
        reasoning_context = self._build_reasoning_context(query, domain, reasoning_type)
        
        # Step 2: Generate frame-aware prompt
        prompt_data = self._generate_frame_aware_prompt(reasoning_context)
        
        # Step 3: Simulate LLM reasoning (in real implementation, this would call an LLM)
        reasoning_result = self._simulate_frame_semantic_reasoning(prompt_data, reasoning_context)
        
        # Step 4: Post-process and enhance result
        enhanced_result = self._enhance_reasoning_result(reasoning_result, reasoning_context)
        
        return enhanced_result
    
    def _build_reasoning_context(self, query: str, domain: Optional[str], 
                               reasoning_type: str) -> ReasoningContext:
        """Build comprehensive reasoning context"""
        
        # Get GraphRAG context
        graphrag_context = self.graphrag_system.retrieve_context(
            query, max_chunks=10, domain_filter=domain
        )
        
        # Get semantic knowledge context  
        semantic_context = self.knowledge_system.query_semantic_knowledge(
            query, domain=domain
        )
        
        # Identify frames and concepts
        frames_involved = list(set(
            [chunk['frame_context'] for chunk in graphrag_context['retrieved_chunks']] +
            semantic_context['relevant_frames']
        ))
        
        concepts_identified = list(set(
            graphrag_context.get('mentioned_concepts', []) +
            [concept.split(':')[-1] for concept in semantic_context['relevant_concepts']]
        ))
        
        return ReasoningContext(
            query=query,
            domain=domain,
            frames_involved=frames_involved,
            concepts_identified=concepts_identified,
            semantic_intent=graphrag_context['semantic_intent'],
            reasoning_type=reasoning_type,
            knowledge_chunks=graphrag_context['retrieved_chunks'],
            cross_domain_connections=graphrag_context.get('cross_domain_connections', [])
        )
    
    def _generate_frame_aware_prompt(self, context: ReasoningContext) -> Dict[str, str]:
        """Generate LLM prompt with frame-semantic awareness"""
        
        # Select appropriate template
        intent = context.semantic_intent
        if intent not in self.reasoning_templates:
            intent = 'comprehensive'
        
        template = self.reasoning_templates[intent]
        
        # Prepare context components
        frame_context = self._format_frame_context(context)
        knowledge_chunks = self._format_knowledge_chunks(context)
        reasoning_guidance = self._format_reasoning_guidance(context)
        cross_domain_context = self._format_cross_domain_context(context)
        
        # Generate prompt
        system_prompt = template['system_prompt']
        user_prompt = template['user_template'].format(
            query=context.query,
            frame_context=frame_context,
            knowledge_chunks=knowledge_chunks,
            reasoning_guidance=reasoning_guidance,
            cross_domain_connections=cross_domain_context
        )
        
        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'context_metadata': {
                'frames_involved': context.frames_involved,
                'concepts_identified': context.concepts_identified,
                'semantic_intent': context.semantic_intent,
                'reasoning_type': context.reasoning_type
            }
        }
    
    def _format_frame_context(self, context: ReasoningContext) -> str:
        """Format frame context for LLM prompt"""
        
        frame_info = []
        
        for frame_name in context.frames_involved[:5]:  # Limit for prompt size
            # Get frame information from knowledge system
            if frame_name in self.knowledge_system.frames:
                frame = self.knowledge_system.frames[frame_name]
                
                frame_info.append(f"""
Frame: {frame_name}
Description: {frame.description}
Domain: {frame.frame_type}
Key Lexical Units: {', '.join(frame.lexical_units[:8])}
Core Elements: {', '.join(frame.frame_elements.get('core', {}).keys())}
""")
        
        return "\n".join(frame_info) if frame_info else "No specific frame context available."
    
    def _format_knowledge_chunks(self, context: ReasoningContext) -> str:
        """Format knowledge chunks for LLM prompt"""
        
        chunks_text = []
        
        for i, chunk in enumerate(context.knowledge_chunks[:8]):  # Limit for prompt size
            chunks_text.append(f"""
Chunk {i+1} ({chunk['semantic_type']}):
{chunk['content']}
Relevance Score: {chunk['relevance_score']:.2f}
Frame Context: {chunk['frame_context']}
""")
        
        return "\n".join(chunks_text) if chunks_text else "No specific knowledge chunks available."
    
    def _format_reasoning_guidance(self, context: ReasoningContext) -> str:
        """Format reasoning guidance for LLM prompt"""
        
        guidance = []
        
        # Add domain-specific guidance
        if context.domain in self.domain_expertise:
            domain_info = self.domain_expertise[context.domain]
            guidance.append(f"Domain Standards: {', '.join(domain_info['standards'])}")
            guidance.append(f"Key Concepts: {', '.join(domain_info['key_concepts'])}")
            guidance.append(f"Professional Roles: {', '.join(domain_info['professional_roles'])}")
        
        # Add cross-domain guidance if multiple domains involved
        if len(set([chunk.get('domain', 'unknown') for chunk in context.knowledge_chunks])) > 1:
            cross_info = self.domain_expertise['cross_domain']
            guidance.append(f"Cross-Domain Concepts: {', '.join(cross_info['shared_concepts'])}")
            guidance.append("Consider cross-domain equivalences and mappings")
        
        # Add reasoning type guidance
        reasoning_guidance = {
            'definition': "Focus on precise frame-semantic definitions with core elements",
            'comparison': "Highlight cross-domain similarities and differences",
            'process': "Emphasize temporal sequences and actor roles",
            'examples': "Provide concrete instances with frame annotations",
            'relationships': "Map semantic relationships and hierarchies",
            'comprehensive': "Integrate multiple perspectives with actionable insights"
        }
        
        guidance.append(reasoning_guidance.get(context.reasoning_type, "Provide comprehensive analysis"))
        
        return "\n".join(guidance)
    
    def _format_cross_domain_context(self, context: ReasoningContext) -> str:
        """Format cross-domain connections for LLM prompt"""
        
        if not context.cross_domain_connections:
            return "No cross-domain connections identified."
        
        connections = []
        for connection in context.cross_domain_connections[:3]:  # Limit for prompt size
            connections.append(f"""
Cross-Domain Connection:
{connection['content']}
Semantic Type: {connection['semantic_type']}
""")
        
        return "\n".join(connections)
    
    def _simulate_frame_semantic_reasoning(self, prompt_data: Dict[str, str], 
                                         context: ReasoningContext) -> Dict[str, Any]:
        """Simulate LLM reasoning (replace with actual LLM call in production)"""
        
        # This is a simulation - in real implementation, this would call an LLM
        # with the generated prompts
        
        reasoning_result = {
            'query': context.query,
            'semantic_intent': context.semantic_intent,
            'reasoning_type': context.reasoning_type,
            'analysis': self._generate_simulated_analysis(context),
            'frame_semantic_insights': self._generate_frame_insights(context),
            'cross_domain_analysis': self._generate_cross_domain_analysis(context),
            'professional_context': self._generate_professional_context(context),
            'actionable_insights': self._generate_actionable_insights(context),
            'confidence_score': 0.85,  # Simulated confidence
            'sources_used': len(context.knowledge_chunks)
        }
        
        return reasoning_result
    
    def _generate_simulated_analysis(self, context: ReasoningContext) -> str:
        """Generate simulated analysis based on context"""
        
        # This would be replaced by actual LLM generation
        frames = context.frames_involved[:3]
        concepts = context.concepts_identified[:5]
        
        analysis_parts = [
            f"Frame-semantic analysis of '{context.query}':",
            f"Primary frames involved: {', '.join(frames)}",
            f"Key concepts identified: {', '.join(concepts)}",
            f"Semantic intent: {context.semantic_intent}",
            ""
        ]
        
        if context.domain:
            domain_info = self.domain_expertise.get(context.domain, {})
            analysis_parts.append(f"Domain expertise applied: {context.domain}")
            analysis_parts.append(f"Relevant standards: {', '.join(domain_info.get('standards', []))}")
        
        return "\n".join(analysis_parts)
    
    def _generate_frame_insights(self, context: ReasoningContext) -> List[str]:
        """Generate frame-semantic insights"""
        
        insights = []
        
        for frame in context.frames_involved[:3]:
            if frame in self.knowledge_system.frames:
                frame_obj = self.knowledge_system.frames[frame]
                insights.append(
                    f"Frame '{frame}' provides {frame_obj.frame_type} domain perspective with "
                    f"{len(frame_obj.lexical_units)} lexical units"
                )
        
        return insights
    
    def _generate_cross_domain_analysis(self, context: ReasoningContext) -> str:
        """Generate cross-domain analysis"""
        
        if not context.cross_domain_connections:
            return "No significant cross-domain connections identified for this query."
        
        return f"Cross-domain analysis reveals {len(context.cross_domain_connections)} connections between robotics and building domains, highlighting shared conceptual structures and functional equivalences."
    
    def _generate_professional_context(self, context: ReasoningContext) -> Dict[str, Any]:
        """Generate professional context information"""
        
        professional_context = {}
        
        if context.domain in self.domain_expertise:
            domain_info = self.domain_expertise[context.domain]
            professional_context[context.domain] = {
                'standards': domain_info['standards'],
                'roles': domain_info['professional_roles'],
                'expertise_areas': domain_info['expertise_areas']
            }
        
        return professional_context
    
    def _generate_actionable_insights(self, context: ReasoningContext) -> List[str]:
        """Generate actionable insights based on reasoning"""
        
        insights = [
            "Frame-semantic analysis provides structured understanding of domain concepts",
            "Professional standards alignment ensures industry-relevant context",
            "Cross-domain mappings enable knowledge transfer between robotics and buildings"
        ]
        
        if context.reasoning_type == 'comparison':
            insights.append("Cross-domain comparison reveals functional equivalences and adaptation strategies")
        elif context.reasoning_type == 'process':
            insights.append("Process frame analysis identifies actor roles and temporal dependencies")
        
        return insights
    
    def _enhance_reasoning_result(self, result: Dict[str, Any], 
                                context: ReasoningContext) -> Dict[str, Any]:
        """Enhance reasoning result with additional frame-semantic metadata"""
        
        enhanced_result = result.copy()
        
        enhanced_result.update({
            'frame_metadata': {
                'frames_analyzed': context.frames_involved,
                'concepts_identified': context.concepts_identified,
                'knowledge_chunks_used': len(context.knowledge_chunks),
                'cross_domain_connections': len(context.cross_domain_connections)
            },
            'semantic_structure': {
                'primary_frames': context.frames_involved[:3],
                'semantic_relationships': self._extract_semantic_relationships(context),
                'domain_coverage': self._analyze_domain_coverage(context)
            },
            'quality_metrics': {
                'frame_coverage': len(context.frames_involved) / max(1, len(context.concepts_identified)),
                'knowledge_depth': len(context.knowledge_chunks),
                'cross_domain_breadth': len(context.cross_domain_connections)
            }
        })
        
        return enhanced_result
    
    def _extract_semantic_relationships(self, context: ReasoningContext) -> List[str]:
        """Extract key semantic relationships from context"""
        
        relationships = []
        
        # Extract from knowledge chunks
        for chunk in context.knowledge_chunks:
            if chunk['semantic_type'] == 'relationship':
                relationships.append(f"Relationship in {chunk['frame_context']}")
        
        return relationships[:5]  # Limit for clarity
    
    def _analyze_domain_coverage(self, context: ReasoningContext) -> Dict[str, int]:
        """Analyze domain coverage in the reasoning context"""
        
        domain_coverage = {}
        
        for chunk in context.knowledge_chunks:
            # Extract domain information from chunk entities
            # This would be more sophisticated in real implementation
            if 'building' in chunk['frame_context'].lower():
                domain_coverage['building'] = domain_coverage.get('building', 0) + 1
            elif 'robot' in chunk['frame_context'].lower():
                domain_coverage['robotics'] = domain_coverage.get('robotics', 0) + 1
        
        return domain_coverage

def main():
    """Demonstrate the frame-semantic reasoning system"""
    
    print("🧠 Frame-Semantic LLM Reasoning System")
    print("=" * 50)
    
    # Initialize reasoner
    reasoner = FrameSemanticReasoner()
    
    print(f"\n✅ Frame-Semantic Reasoner Ready!")
    
    # Example reasoning queries
    example_queries = [
        ("What is a building sensor?", "building", "definition"),
        ("Compare robot actuators to building actuators", None, "comparison"),
        ("How is building maintenance performed?", "building", "process"),
        ("Give examples of building components", "building", "examples"),
        ("What are the relationships between building systems?", "building", "relationships")
    ]
    
    print(f"\n🔍 Example Frame-Semantic Reasoning:")
    
    for query, domain, reasoning_type in example_queries:
        print(f"\n📝 Query: {query}")
        print(f"   Domain: {domain or 'cross-domain'}")
        print(f"   Reasoning Type: {reasoning_type}")
        
        result = reasoner.reason_about_query(query, domain, reasoning_type)
        
        print(f"   Frames Analyzed: {len(result['frame_metadata']['frames_analyzed'])}")
        print(f"   Knowledge Chunks: {result['frame_metadata']['knowledge_chunks_used']}")
        print(f"   Confidence: {result['confidence_score']:.2f}")
        print(f"   Key Insight: {result['actionable_insights'][0]}")
    
    return reasoner

if __name__ == "__main__":
    reasoner = main()
