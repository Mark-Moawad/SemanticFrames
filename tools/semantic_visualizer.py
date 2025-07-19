#!/usr/bin/env python3
"""
Interactive Semantic Extraction Visualizer

A comprehensive web-based visualization system for exploring semantic frame extraction
results from construction robotics scenarios. Features interactive knowledge graphs,
frame structure visualization, and side-by-side text-to-frame comparisons.
"""

import json
import re
import sys
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Add the src directory to the Python path for importing the real extraction system
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load Cytoscape stylesheets
cyto.load_extra_layouts()

class SemanticVisualizationSystem:
    """
    Interactive visualization system for semantic frame extraction results
    """
    
    def __init__(self, extraction_results_file: str = None):
        """
        Initialize the visualization system
        
        Args:
            extraction_results_file: Path to JSON file with extraction results
        """
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                            suppress_callback_exceptions=True)
        
        # Add custom CSS for lexical unit hover effects
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .lexical-unit-annotation:hover {
                        transform: scale(1.05) !important;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
                        z-index: 10 !important;
                    }
                    
                    .lexical-unit-annotation {
                        display: inline-block;
                        transition: all 0.3s ease;
                    }
                    
                    /* Legend badge styles matching knowledge graph colors */
                    .legend-robot { background-color: #2c3e50 !important; color: white !important; }
                    .legend-building { background-color: #2ecc71 !important; color: white !important; }
                    .legend-component-robotics { background-color: #e74c3c !important; color: white !important; }
                    .legend-component-building { background-color: #ff8c00 !important; color: white !important; }
                    .legend-capability { background-color: #3498db !important; color: white !important; }
                    .legend-system { background-color: #ffc107 !important; color: black !important; }
                    .legend-action { background-color: #27ae60 !important; color: white !important; }
                    .legend-process { background-color: #6c757d !important; color: white !important; }
                    
                    .legend-badge {
                        padding: 4px 8px !important;
                        border-radius: 12px !important;
                        font-size: 12px !important;
                        font-weight: bold !important;
                        margin-right: 8px !important;
                        display: inline-block !important;
                        border: none !important;
                    }
                        transition: all 0.3s ease;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        self.extraction_results = []
        self.knowledge_graph = nx.DiGraph()
        
        # Initialize the real semantic extraction system for lexical units
        try:
            from semantic_extraction_system import SemanticExtractionSystem
            self.extraction_system = SemanticExtractionSystem()
            print("✅ Real lexical unit extraction system loaded")
        except ImportError as e:
            print(f"⚠️ Warning: Could not import real extraction system, using fallback patterns: {e}")
            self.extraction_system = None
        
        if extraction_results_file:
            self.load_extraction_results(extraction_results_file)
        
        self.setup_layout()
        self.setup_callbacks()
    
    def load_extraction_results(self, file_path: str):
        """Load extraction results from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.extraction_results = json.load(f)
            print(f"✅ Loaded {len(self.extraction_results)} extraction results")
            self._build_knowledge_graph()
        except Exception as e:
            print(f"❌ Error loading extraction results: {e}")
    
    def _build_knowledge_graph(self):
        """Build NetworkX graph from extraction results"""
        self.knowledge_graph = nx.DiGraph()
        
        for i, result in enumerate(self.extraction_results):
            domain = result.get('domain', 'unknown')
            
            # Add nodes for each frame
            for frame_name, frame_data in result.get('extracted_frames', {}).items():
                node_id = f"{domain}_{frame_name}_{i}"
                self.knowledge_graph.add_node(
                    node_id,
                    label=f"{frame_name}",
                    domain=domain,
                    frame_type=frame_name,
                    data=frame_data,
                    source_text=result.get('source_text', '')[:200] + "...",
                    confidence=result.get('confidence_scores', {}).get(frame_name, 0.5),
                    scenario_id=i
                )
            
            # Add edges for relationships
            for source, relation, target in result.get('relationships', []):
                source_id = f"{domain}_{source}_{i}"
                target_id = f"{domain}_{target}_{i}"
                if self.knowledge_graph.has_node(source_id) and self.knowledge_graph.has_node(target_id):
                    self.knowledge_graph.add_edge(source_id, target_id, relation=relation)
    
    def _identify_lexical_units(self, text: str, extracted_frames: Dict, lexical_units_data: Dict = None) -> List[Tuple[str, int, int, List[str], List[str]]]:
        """
        Identify lexical units (words/phrases) that evoke semantic frames
        
        Args:
            text: Source text to analyze
            extracted_frames: Dictionary of extracted semantic frames
            lexical_units_data: Pre-extracted lexical units with frame evocations from JSON data
            
        Returns:
            List of tuples: (lexical_unit, start_pos, end_pos, explicit_frames, implicit_frames)
        """
        lexical_annotations = []
        
        # First, try to use pre-extracted lexical units data if available
        if lexical_units_data:
            text_lower = text.lower()
            for lexical_unit, unit_data in lexical_units_data.items():
                # Find all occurrences of this lexical unit in the text
                unit_pattern = re.compile(re.escape(lexical_unit), re.IGNORECASE)
                for match in unit_pattern.finditer(text):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Get explicit frame evocation
                    explicit_frames = [unit_data['frame']] if unit_data.get('evocation') == 'explicit' else []
                    
                    # Get implicit frame evocations
                    implicit_frames = []
                    if 'implicit_frames' in unit_data:
                        for implicit_frame in unit_data['implicit_frames']:
                            if implicit_frame.get('evocation') == 'explicit':
                                explicit_frames.append(implicit_frame['frame'])
                            elif implicit_frame.get('evocation') == 'implicit':
                                implicit_frames.append(implicit_frame['frame'])
                    
                    # If primary frame is implicit, add to implicit list
                    if unit_data.get('evocation') == 'implicit':
                        implicit_frames.append(unit_data['frame'])
                    
                    lexical_annotations.append((lexical_unit, start_pos, end_pos, explicit_frames, implicit_frames))
        
        elif self.extraction_system:
            # Fallback: Use the real extraction system
            # Determine domain from extracted frames
            has_robot = 'Robot' in extracted_frames
            has_building = 'Building' in extracted_frames
            
            if has_robot and has_building:
                domain = "general"
            elif has_robot:
                domain = "robotics"
            elif has_building:
                domain = "building"
            else:
                domain = "general"
            
            # Get lexical units from the real extraction system
            lexical_units = self.extraction_system._extract_lexical_units(text, extracted_frames, domain)
            
            # Convert to the format expected by the visualizer
            text_lower = text.lower()
            for word, mapping in lexical_units.items():
                frame = mapping['frame']
                evocation = mapping['evocation']
                
                # Find all occurrences of this word in the text
                word_pattern = re.compile(re.escape(word), re.IGNORECASE)
                for match in word_pattern.finditer(text):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    if evocation == "explicit":
                        explicit_frames = [frame]
                        implicit_frames = []
                    else:
                        explicit_frames = []
                        implicit_frames = [frame]
                    
                    lexical_annotations.append((word, start_pos, end_pos, explicit_frames, implicit_frames))
        else:
            # Fallback to hardcoded patterns if real system not available
            lexical_annotations = self._identify_lexical_units_fallback(text, extracted_frames)
        
        # Sort by position and remove overlaps
        lexical_annotations.sort(key=lambda x: x[1])
        return self._remove_overlapping_annotations(lexical_annotations)
    
    def _identify_lexical_units_fallback(self, text: str, extracted_frames: Dict) -> List[Tuple[str, int, int, List[str], List[str]]]:
        """
        Fallback lexical unit identification using hardcoded patterns (original implementation)
        """
        lexical_annotations = []
        
        # Determine domain from extracted frames to use appropriate patterns
        has_robot = 'Robot' in extracted_frames
        has_building = 'Building' in extracted_frames
        has_capability = 'Capability' in extracted_frames
        has_action = 'Action' in extracted_frames
        has_system = 'System' in extracted_frames
        has_process = 'Process' in extracted_frames
        
        # Define domain-specific frame-evoking patterns
        frame_patterns = {}
        
        # ROBOTICS DOMAIN PATTERNS (only if robotics frames are present)
        if has_robot or has_capability or has_action:
            robotics_patterns = {
                # Robot frame patterns - explicit Robot, implicit Capability + Action cascade
                r'\b(?:mobile\s+)?robot\b': (['Robot'], ['Capability', 'Action']),
                r'\brobot(?:ic)?\s+(?:arm|manipulator|actuator)\b': (['Robot', 'Component'], ['Capability', 'Action']),
                r'\bautonomo\w+\s+(?:vehicle|system|robot)\b': (['Robot'], ['Capability', 'Action']),
                r'\bmobile\s+platform\b': (['Robot', 'Component'], ['Capability', 'Action']),
                
                # Robotics Component patterns - explicit Component, implicit Capability + Action cascade
                r'\b(?:camera|sensor|lidar|ultrasonic|infrared)\b': (['Component'], ['Capability', 'Action']),
                r'\b(?:actuator|motor|servo|joint|wheel|gripper|end.effector)\b': (['Component'], ['Capability', 'Action']),
                r'\b(?:manipulator\s+arm|robotic\s+arm|mechanical\s+arm)\b': (['Component'], ['Capability', 'Action']),
                r'\b(?:processor|computer|controller|cpu|gpu)\b': (['Component'], ['Capability']),
                r'\b(?:battery|power\s+supply|energy\s+storage)\b': (['Component'], []),
                r'\b(?:transmission|gearbox|drivetrain|mechanism)\b': (['Component'], ['Capability', 'Action']),
                
                # Robotics Capability patterns - explicit Capability, strong Action implication
                r'\b(?:navigation|path\s+planning|localization|mapping)\b': (['Capability'], ['Action']),
                r'\b(?:manipulation|grasping|picking|placing)\b': (['Capability'], ['Action']),
                r'\b(?:vision|perception|sensing|detection)\b': (['Capability'], ['Action']),
                r'\b(?:communication|networking|data\s+transmission)\b': (['Capability'], []),
                r'\b(?:mobility|movement|locomotion|maneuvering)\b': (['Capability'], ['Action']),
                
                # Robotics Action patterns - explicit Action, may require Component/Capability
                r'\b(?:move|moving|navigate|navigating|travel|traveling)\b': (['Action'], ['Capability']),
                r'\b(?:pick|picking|grasp|grasping|grab|grabbing)\b': (['Action'], ['Capability', 'Component']),
                r'\b(?:place|placing|put|putting|position|positioning)\b': (['Action'], ['Capability']),
                r'\b(?:scan|scanning|detect|detecting|sense|sensing)\b': (['Action'], ['Capability', 'Component']),
                r'\b(?:avoid|avoiding|dodge|dodging|evade|evading)\b': (['Action'], ['Capability']),
                r'\b(?:lift|lifting|raise|raising|lower|lowering)\b': (['Action'], ['Capability', 'Component']),
            }
            frame_patterns.update(robotics_patterns)
        
        # BUILDING DOMAIN PATTERNS (only if building frames are present)
        if has_building or has_system or has_process:
            building_patterns = {
                # Building frame patterns - explicit Building, may imply System/Process
                r'\b(?:building|structure|facility|construction|edifice)\b': (['Building'], ['System']),
                r'\b(?:commercial\s+building|office\s+building|residential\s+building)\b': (['Building'], ['System']),
                r'\b(?:smart\s+building|intelligent\s+building)\b': (['Building'], ['System', 'Process']),
                
                # Building Component patterns - explicit Component, belongs to System
                r'\b(?:hvac|heating|ventilation|air\s+conditioning)\b': (['Component'], ['System']),
                r'\b(?:electrical\s+(?:panel|system|equipment))\b': (['Component'], ['System']),
                r'\b(?:plumbing|pipe|duct|valve)\b': (['Component'], ['System']),
                r'\b(?:fire\s+(?:safety|alarm|sprinkler))\b': (['Component'], ['System']),
                r'\b(?:elevator|lift|escalator)\b': (['Component'], ['System']),
                r'\b(?:beam|column|wall|floor|ceiling|foundation)\b': (['Component'], []),
                r'\b(?:door|window|roof|insulation)\b': (['Component'], []),
                
                # Building System patterns - explicit System, may imply Process
                r'\b(?:climate\s+control|environmental\s+control)\b': (['System'], ['Process']),
                r'\b(?:power\s+distribution|electrical\s+supply)\b': (['System'], ['Process']),
                r'\b(?:safety\s+(?:management|system))\b': (['System'], ['Process']),
                r'\b(?:vertical\s+transportation|elevator\s+system)\b': (['System'], ['Process']),
                r'\b(?:building\s+automation|smart\s+controls)\b': (['System'], ['Process']),
                r'\b(?:energy\s+management|power\s+management)\b': (['System'], ['Process']),
                
                # Building Process patterns - explicit Process, requires System
                r'\b(?:temperature\s+regulation|climate\s+control)\b': (['Process'], ['System']),
                r'\b(?:emergency\s+response|safety\s+management)\b': (['Process'], ['System']),
                r'\b(?:energy\s+optimization|power\s+management)\b': (['Process'], ['System']),
                r'\b(?:occupant\s+transportation|people\s+movement)\b': (['Process'], ['System']),
                r'\b(?:building\s+automation|automated\s+control)\b': (['Process'], ['System']),
                
                # Space and room patterns
                r'\b(?:room|office|corridor|hallway|entrance|lobby)\b': (['Building'], []),
                r'\b(?:floor|level|storey|story)\b': (['Building'], []),
            }
            frame_patterns.update(building_patterns)
        
        # GENERAL CONSTRUCTION PATTERNS (apply to both domains)
        general_patterns = {
            r'\b(?:concrete|steel|rebar|lumber|drywall)\b': (['Component'], []),
            r'\b(?:construction\s+equipment|heavy\s+machinery)\b': (['Component'], []),
        }
        frame_patterns.update(general_patterns)
        
        # Find all pattern matches in the text
        for pattern, (explicit_frames, implicit_frames) in frame_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                lexical_unit = match.group()
                start_pos = match.start()
                end_pos = match.end()
                
                # Filter frames based on what was actually extracted
                relevant_explicit = [f for f in explicit_frames if f in extracted_frames]
                relevant_implicit = [f for f in implicit_frames if f in extracted_frames]
                
                if relevant_explicit or relevant_implicit:  # Only add if there are relevant frames
                    lexical_annotations.append((lexical_unit, start_pos, end_pos, relevant_explicit, relevant_implicit))
        
        return lexical_annotations
    
    def _remove_overlapping_annotations(self, lexical_annotations: List[Tuple[str, int, int, List[str], List[str]]]) -> List[Tuple[str, int, int, List[str], List[str]]]:
        """Remove overlapping annotations, keeping the first one"""
        filtered_annotations = []
        last_end = 0
        for annotation in lexical_annotations:
            if annotation[1] >= last_end:  # No overlap
                filtered_annotations.append(annotation)
                last_end = annotation[2]
        return filtered_annotations
    
    def _create_annotated_text(self, text: str, extracted_frames: Dict, lexical_units_data: Dict = None) -> html.Div:
        """
        Create annotated text with highlighted lexical units and hover tooltips showing explicit/implicit frames
        """
        annotations = self._identify_lexical_units(text, extracted_frames, lexical_units_data)
        
        if not annotations:
            # No annotations found, return plain text
            return html.Div(text, style={'white-space': 'pre-wrap'})
        
        # Build annotated HTML elements
        elements = []
        last_pos = 0
        
        # Detect domain from extracted frames to distinguish component colors
        domain = "robotics"  # default
        if "Building" in extracted_frames:
            domain = "building"
        
        # Color mapping for different frame types matching knowledge graph colors
        frame_colors = {
            'Robot': '#2c3e50',
            'Component': '#e74c3c' if domain == "robotics" else '#ff8c00',  # Red for robotics, Orange for building
            'Capability': '#3498db',
            'Action': '#27ae60',
            'Building': '#2ecc71',  # Fixed to match knowledge graph
            'System': '#ffc107',
            'Process': '#6c757d'
        }
        
        for lexical_unit, start_pos, end_pos, explicit_frames, implicit_frames in annotations:
            # Add any text before this annotation
            if start_pos > last_pos:
                elements.append(text[last_pos:start_pos])
            
            # Create the primary frame color (use first explicit frame, or first implicit if no explicit)
            primary_frame = explicit_frames[0] if explicit_frames else implicit_frames[0]
            primary_color = frame_colors.get(primary_frame, '#7f8c8d')
            
            # Create enhanced tooltip text showing explicit and implicit evocations
            tooltip_parts = []
            if explicit_frames:
                tooltip_parts.append(f"Explicitly evokes: {', '.join(explicit_frames)}")
            if implicit_frames:
                tooltip_parts.append(f"Implicitly evokes: {', '.join(implicit_frames)}")
            tooltip_text = " | ".join(tooltip_parts)
            
            # Determine border style - solid for explicit, dashed for mixed explicit/implicit
            border_style = 'solid' if explicit_frames and not implicit_frames else 'dashed'
            text_decoration = 'underline' if explicit_frames else 'underline'
            
            # Create the annotated span with enhanced hover tooltip
            annotated_span = html.Span(
                lexical_unit,
                id=f"lexical-{start_pos}-{end_pos}",
                style={
                    'color': primary_color,
                    'font-weight': 'bold',
                    'text-decoration': text_decoration,
                    'text-decoration-color': primary_color,
                    'text-decoration-thickness': '2px',
                    'text-decoration-style': border_style,
                    'cursor': 'pointer',
                    'border-radius': '4px',
                    'padding': '2px 4px',
                    'background-color': f"{primary_color}20",  # Light background
                    'position': 'relative',
                    'transition': 'all 0.3s ease',
                    'margin': '0 1px',
                    # Add border to distinguish explicit vs implicit
                    'border': f'1px {border_style} {primary_color}40'
                },
                title=tooltip_text,  # Enhanced browser tooltip
                className="lexical-unit-annotation"
            )
            
            elements.append(annotated_span)
            last_pos = end_pos
        
        # Add any remaining text
        if last_pos < len(text):
            elements.append(text[last_pos:])
        
        return html.Div(
            elements,
            style={
                'white-space': 'pre-wrap',
                'line-height': '1.8',
                'font-family': 'monospace',
                'font-size': '14px',
                'padding': '15px',
                'background-color': '#f8f9fa',
                'border-radius': '8px',
                'border': '1px solid #dee2e6'
            },
            id='annotated-text-container'
        )
    
    def setup_layout(self):
        """Setup the Dash app layout"""
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🤖🏗️ Semantic Frame Extraction Visualizer", 
                           className="text-center mb-4"),
                    html.P("Interactive visualization of construction robotics semantic knowledge extraction",
                           className="text-center text-muted mb-4")
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📊 Visualization Controls", className="card-title"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Scenario:"),
                                    dcc.Dropdown(
                                        id='scenario-dropdown',
                                        options=[],
                                        value=None,
                                        placeholder="Choose a scenario to analyze"
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label("Domain Filter:"),
                                    dcc.Dropdown(
                                        id='domain-filter',
                                        options=[
                                            {'label': 'All Domains', 'value': 'all'},
                                            {'label': 'Robotics', 'value': 'robotics'},
                                            {'label': 'Building', 'value': 'building'},
                                            {'label': 'General', 'value': 'general'}
                                        ],
                                        value='all'
                                    )
                                ], width=6)
                            ]),
                            
                            html.Hr(),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🔄 Refresh Data", id="refresh-btn", 
                                             color="primary", size="sm"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Button("📊 Show Statistics", id="stats-btn", 
                                             color="info", size="sm"),
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Main Content Tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="🎯 Frame Analysis", tab_id="frame-analysis"),
                        dbc.Tab(label="🌐 Knowledge Graph", tab_id="knowledge-graph"),
                        dbc.Tab(label="📝 Text-to-Frame", tab_id="text-frame"),
                        dbc.Tab(label="📈 Statistics", tab_id="statistics")
                    ], id="main-tabs", active_tab="frame-analysis")
                ])
            ]),
            
            # Content Area
            html.Div(id="tab-content", className="mt-4"),
            
            # Footer
            html.Hr(),
            html.P("Semantic Frame Extraction Visualizer for Construction Robotics", 
                   className="text-center text-muted small")
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup Dash callbacks for interactivity"""
        
        @self.app.callback(
            Output('scenario-dropdown', 'options'),
            Input('refresh-btn', 'n_clicks')
        )
        def update_scenario_options(n_clicks):
            """Update scenario dropdown options"""
            if not self.extraction_results:
                return []
            
            options = []
            for i, result in enumerate(self.extraction_results):
                # Try to extract title from source text or use index
                source_text = result.get('source_text', '')
                title = source_text[:50] + "..." if len(source_text) > 50 else f"Scenario {i+1}"
                
                options.append({
                    'label': f"{i+1}. {title} ({result.get('domain', 'unknown')})",
                    'value': i
                })
            
            return options
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('scenario-dropdown', 'value'),
             Input('domain-filter', 'value')]
        )
        def update_tab_content(active_tab, selected_scenario, domain_filter):
            """Update content based on active tab"""
            
            if active_tab == "frame-analysis":
                return self._create_frame_analysis_content(selected_scenario, domain_filter)
            elif active_tab == "knowledge-graph":
                return self._create_knowledge_graph_content(domain_filter)
            elif active_tab == "text-frame":
                return self._create_text_frame_content(selected_scenario, domain_filter)
            elif active_tab == "statistics":
                return self._create_statistics_content(domain_filter)
            else:
                return html.Div("Select a tab to view content")
        
        # Hierarchical tree interaction callback
        @self.app.callback(
            Output('node-detail-info', 'children'),
            [Input('hierarchical-tree', 'tapNodeData'),
             Input('hierarchical-tree', 'mouseoverNodeData')],
            prevent_initial_call=True
        )
        def display_node_info(tap_data, hover_data):
            """Display detailed information when node is tapped or hovered"""
            node_data = tap_data or hover_data
            
            if not node_data:
                return html.Div()
            
            node_type = node_data.get('type', 'unknown')
            node_label = node_data.get('label', 'Unknown')
            node_details = node_data.get('details', '{}')
            confidence = node_data.get('confidence', 0.0)
            domain = node_data.get('domain', 'unknown')
            
            # Create confidence badge
            if confidence > 0.7:
                badge_color = "success"
            elif confidence > 0.5:
                badge_color = "warning"
            else:
                badge_color = "danger"
            
            # Icon mapping for different node types
            icons = {
                'robot': '🤖',
                'building': '🏗️',
                'capability': '⚡',
                'component': '🔧',
                'action': '🎯'
            }
            
            icon = icons.get(node_type, '📦')
            
            return dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        f"{icon} {node_label}",
                        dbc.Badge(f"Confidence: {confidence:.2f}", color=badge_color, className="ms-2"),
                        dbc.Badge(f"Domain: {domain.title()}", color="secondary", className="ms-2")
                    ])
                ]),
                dbc.CardBody([
                    html.H6(f"Type: {node_type.title()}", className="text-muted mb-3"),
                    html.H6("📋 Detailed Information:", className="mb-2"),
                    html.Pre(
                        node_details,
                        style={
                            'max-height': '400px',
                            'overflow-y': 'auto',
                            'background-color': '#f8f9fa',
                            'padding': '15px',
                            'border-radius': '8px',
                            'font-size': '12px',
                            'border': '1px solid #dee2e6'
                        }
                    )
                ])
            ], className="mb-3", color="light")
        
        # Enhanced lexical unit hover callback (for future interactivity)
        @self.app.callback(
            Output('lexical-hover-info', 'children'),
            [Input('annotated-text-container', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_lexical_unit_interaction(n_clicks):
            """Handle interactions with lexical units in annotated text"""
            # This can be expanded for more sophisticated interactions
            return html.Div()
    
    def _create_frame_analysis_content(self, selected_scenario, domain_filter=None):
        """Create frame analysis content"""
        if selected_scenario is None or not self.extraction_results:
            return dbc.Alert("Please select a scenario to analyze", color="warning")
        
        if selected_scenario >= len(self.extraction_results):
            return dbc.Alert("Invalid scenario selected", color="danger")
        
        result = self.extraction_results[selected_scenario]
        
        # Apply domain filter if specified
        if domain_filter and domain_filter != 'all' and result.get('domain') != domain_filter:
            return dbc.Alert(f"Selected scenario is not from {domain_filter} domain", color="warning")
        
        # Create frame structure visualization
        frame_cards = []
        for frame_name, frame_data in result.get('extracted_frames', {}).items():
            confidence = result.get('confidence_scores', {}).get(frame_name, 0.0)
            
            # Create confidence badge
            if confidence > 0.7:
                badge_color = "success"
            elif confidence > 0.5:
                badge_color = "warning"
            else:
                badge_color = "danger"
            
            frame_card = dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        f"🎯 {frame_name}",
                        dbc.Badge(f"{confidence:.2f}", color=badge_color, className="ms-2")
                    ])
                ]),
                dbc.CardBody([
                    html.Pre(json.dumps(frame_data, indent=2, ensure_ascii=False),
                            style={'max-height': '300px', 'overflow-y': 'auto',
                                   'background-color': '#f8f9fa', 'padding': '10px',
                                   'border-radius': '5px', 'font-size': '12px'})
                ])
            ], className="mb-3")
            
            frame_cards.append(frame_card)
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4(f"📊 Frame Analysis - Scenario {selected_scenario + 1}"),
                    html.P(f"Domain: {result.get('domain', 'unknown').title()}", 
                           className="text-muted"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col(frame_cards)
            ])
        ])
    
    def _create_knowledge_graph_content(self, domain_filter):
        """Create interactive hierarchical tree content"""
        if not self.extraction_results:
            return dbc.Alert("No extraction data available", color="warning")
        
        # Build hierarchical tree structure from extraction results
        tree_elements = self._build_hierarchical_tree(domain_filter)
        
        if not tree_elements:
            return dbc.Alert("No data matches the selected domain filter", color="warning")
        
        # Count nodes and edges
        nodes = [elem for elem in tree_elements if 'source' not in elem.get('data', {})]
        edges = [elem for elem in tree_elements if 'source' in elem.get('data', {})]
        
        # Enhanced stylesheet for hierarchical tree with VERY LARGE fonts
        stylesheet = [
            # Root node (Robot/Building)
            {
                'selector': '.root',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '180px',
                    'background-color': '#2c3e50',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': '#2c3e50',
                    'width': '200px',
                    'height': '120px',
                    'font-size': '24px',  # Much larger!
                    'font-weight': 'bold',
                    'shape': 'round-rectangle',
                    'cursor': 'pointer',
                    'border-width': 4,
                    'border-color': '#34495e'
                }
            },
            # Capability nodes
            {
                'selector': '.capability',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '160px',
                    'background-color': '#3498db',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': '#3498db',
                    'width': '180px',
                    'height': '100px',
                    'font-size': '20px',  # Much larger!
                    'font-weight': 'bold',
                    'shape': 'round-rectangle',
                    'cursor': 'pointer',
                    'border-width': 3,
                    'border-color': '#2980b9'
                }
            },
            # Robotics Component nodes
            {
                'selector': '.component.domain-robotics',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '140px',
                    'background-color': '#e74c3c',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': '#e74c3c',
                    'width': '160px',
                    'height': '90px',
                    'font-size': '18px',
                    'font-weight': 'bold',
                    'shape': 'ellipse',
                    'cursor': 'pointer',
                    'border-width': 3,
                    'border-color': '#c0392b'
                }
            },
            # Building Component nodes
            {
                'selector': '.component.domain-building',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '140px',
                    'background-color': '#ff8c00',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': '#ff8c00',
                    'width': '160px',
                    'height': '90px',
                    'font-size': '18px',
                    'font-weight': 'bold',
                    'shape': 'round-rectangle',
                    'cursor': 'pointer',
                    'border-width': 3,
                    'border-color': '#e67e00'
                }
            },
            # Action nodes
            {
                'selector': '.action',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '120px',
                    'background-color': '#27ae60',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': '#27ae60',
                    'width': '140px',
                    'height': '80px',
                    'font-size': '16px',  # Much larger!
                    'font-weight': 'bold',
                    'shape': 'diamond',
                    'cursor': 'pointer',
                    'border-width': 3,
                    'border-color': '#229954'
                }
            },
            # System nodes (Building domain)
            {
                'selector': '.system',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '160px',
                    'background-color': '#ffc107',
                    'color': 'black',
                    'text-outline-width': 2,
                    'text-outline-color': '#ffc107',
                    'width': '180px',
                    'height': '100px',
                    'font-size': '20px',
                    'font-weight': 'bold',
                    'shape': 'round-rectangle',
                    'cursor': 'pointer',
                    'border-width': 3,
                    'border-color': '#e0a800'
                }
            },
            # Process nodes (Building domain)
            {
                'selector': '.process',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '120px',
                    'background-color': '#6c757d',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': '#6c757d',
                    'width': '140px',
                    'height': '80px',
                    'font-size': '16px',
                    'font-weight': 'bold',
                    'shape': 'hexagon',
                    'cursor': 'pointer',
                    'border-width': 3,
                    'border-color': '#545b62'
                }
            },
            # Building domain styling
            {
                'selector': '.domain-building.root',
                'style': {
                    'background-color': '#2ecc71',
                    'text-outline-color': '#2ecc71',
                    'border-color': '#27ae60'
                }
            },
            # Hover effects
            {
                'selector': 'node:hover',
                'style': {
                    'border-width': 6,
                    'border-opacity': 1,
                    'font-size': '+6px',  # Even bigger on hover!
                    'z-index': 999
                }
            },
            # Edges (parent-child relationships)
            {
                'selector': 'edge',
                'style': {
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'width': 4,
                    'line-color': '#7f8c8d',
                    'target-arrow-color': '#7f8c8d',
                    'content': 'data(relation)',
                    'font-size': '16px',  # Much larger edge labels!
                    'font-weight': 'bold',
                    'color': '#2c3e50',
                    'text-background-color': 'white',
                    'text-background-opacity': 0.95,
                    'text-background-padding': '6px',
                    'text-border-width': 2,
                    'text-border-color': '#bdc3c7'
                }
            }
        ]
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("� Hierarchical Semantic Tree"),
                    html.P(f"Showing {len(nodes)} nodes, {len(edges)} relationships"),
                    dbc.Alert([
                        html.Strong("💡 Interaction Guide: "),
                        "Click nodes to expand/collapse children • Hover for details • Drag to rearrange"
                    ], color="info", className="mb-3"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    cyto.Cytoscape(
                        id='hierarchical-tree',
                        elements=tree_elements,
                        style={'width': '100%', 'height': '900px'},  # Taller for bigger nodes
                        layout={
                            'name': 'dagre',
                            'rankDir': 'TB',  # Top to Bottom
                            'spacingFactor': 3.0,  # Much more spacing for larger nodes
                            'nodeSep': 150,  # More horizontal separation
                            'rankSep': 180,  # More vertical separation
                            'animate': True,
                            'animationDuration': 500
                        },
                        stylesheet=stylesheet,
                        responsive=True
                    )
                ], width=8),
                
                # Node details panel
                dbc.Col([
                    html.Div(
                        id='node-detail-info',
                        children=[
                            dbc.Alert([
                                html.I(className="fas fa-info-circle me-2"),
                                "Hover over or click nodes to see detailed information"
                            ], color="light", className="text-center")
                        ]
                    )
                ], width=4)
            ])
        ])
    
    def _build_hierarchical_tree(self, domain_filter):
        """Build hierarchical tree elements from extraction results"""
        elements = []
        created_node_ids = set()  # Track created nodes to avoid ID errors
        
        for idx, result in enumerate(self.extraction_results):
            domain = result.get('domain', 'unknown')
            
            # Apply domain filter
            if domain_filter != 'all' and domain != domain_filter:
                continue
            
            extracted_frames = result.get('extracted_frames', {})
            confidence_scores = result.get('confidence_scores', {})
            
            # Create root node (Robot or Building)
            root_id = None
            root_frame = None
            root_type = None
            
            if 'Robot' in extracted_frames:
                root_frame = extracted_frames['Robot']
                root_type = 'Robot'
                root_id = f"robot_{idx}"
            elif 'Building' in extracted_frames:
                root_frame = extracted_frames['Building']
                root_type = 'Building'
                root_id = f"building_{idx}"
            
            if root_frame and root_id:
                # Create a clean label for the root
                agent_info = root_frame.get('Agent', {})
                robot_type = agent_info.get('robot_type', root_type)
                clean_label = robot_type.replace('_', ' ').title()
                
                elements.append({
                    'data': {
                        'id': root_id,
                        'label': clean_label,
                        'type': root_type.lower(),
                        'domain': domain,
                        'confidence': confidence_scores.get(root_type, 0.8),
                        'details': json.dumps(root_frame, indent=2),
                        'expanded': True
                    },
                    'classes': f'root domain-{domain}'
                })
                created_node_ids.add(root_id)
            
            # Create component nodes FIRST (Robot → Component)
            component_ids = {}  # Track component IDs for capability connections
            if 'Component' in extracted_frames:
                components = extracted_frames['Component']
                for comp_name, comp_data in components.items():
                    if comp_name == 'frame':  # Skip frame identifier
                        continue
                    
                    comp_id = f"comp_{comp_name}_{idx}"
                    component_ids[comp_name] = comp_id
                    
                    # Create clean component label
                    clean_comp_label = comp_name.replace('_', ' ').title()
                    
                    elements.append({
                        'data': {
                            'id': comp_id,
                            'label': clean_comp_label,
                            'type': 'component',
                            'domain': domain,
                            'confidence': confidence_scores.get('Component', 0.8),
                            'details': json.dumps(comp_data, indent=2),
                            'expanded': False
                        },
                        'classes': f'component domain-{domain}'
                    })
                    created_node_ids.add(comp_id)
                    
                    # Add edge from root to component - only if root exists
                    if root_id and root_id in created_node_ids:
                        elements.append({
                            'data': {
                                'source': root_id,
                                'target': comp_id,
                                'relation': 'has'
                            }
                        })
            
            # Create system nodes for BUILDING DOMAIN (Building → System → Component)
            system_ids = {}  # Track system IDs for process connections
            if 'System' in extracted_frames:
                systems = extracted_frames['System']
                for sys_name, sys_data in systems.items():
                    if sys_name == 'frame':  # Skip frame identifier
                        continue
                    
                    sys_id = f"sys_{sys_name}_{idx}"
                    system_ids[sys_name] = sys_id
                    
                    # Create clean system label
                    clean_sys_label = sys_name.replace('_system', '').replace('_', ' ').title()
                    
                    elements.append({
                        'data': {
                            'id': sys_id,
                            'label': clean_sys_label,
                            'type': 'system',
                            'domain': domain,
                            'confidence': confidence_scores.get('System', 0.8),
                            'details': json.dumps(sys_data, indent=2),
                            'expanded': False
                        },
                        'classes': f'system domain-{domain}'
                    })
                    created_node_ids.add(sys_id)
                    
                    # Connect systems to components they contain
                    # Look for components contained within this system
                    if 'Component' in extracted_frames:
                        components = extracted_frames['Component']
                        for comp_name, comp_data in components.items():
                            if comp_name == 'frame':
                                continue
                            parent_system = comp_data.get('system') or comp_data.get('parent_system')
                            if parent_system == sys_name and comp_name in component_ids:
                                comp_id = component_ids[comp_name]
                                # Add edge from system to component (system contains component)
                                elements.append({
                                    'data': {
                                        'source': sys_id,
                                        'target': comp_id,
                                        'relation': 'contains'
                                    }
                                })
            
            # Create capability nodes SECOND (Component → Capability)
            capability_ids = {}  # Track capability IDs for action connections
            if 'Capability' in extracted_frames:
                capabilities = extracted_frames['Capability']
                for cap_name, cap_data in capabilities.items():
                    if cap_name == 'frame':  # Skip frame identifier
                        continue
                    
                    cap_id = f"cap_{cap_name}_{idx}"
                    capability_ids[cap_name] = cap_id
                    
                    # Create clean capability label
                    clean_cap_label = cap_name.replace('_capability', '').replace('_', ' ').title()
                    
                    elements.append({
                        'data': {
                            'id': cap_id,
                            'label': clean_cap_label,
                            'type': 'capability',
                            'domain': domain,
                            'confidence': confidence_scores.get('Capability', 0.8),
                            'details': json.dumps(cap_data, indent=2),
                            'expanded': False
                        },
                        'classes': f'capability domain-{domain}'
                    })
                    created_node_ids.add(cap_id)
                    
                    # Connect capabilities to components that enable them
                    # Look for components that enable this capability
                    for comp_name, comp_data in components.items() if 'Component' in extracted_frames else []:
                        if comp_name == 'frame':
                            continue
                        enabled_capability = comp_data.get('enables_capability')
                        if enabled_capability == cap_name and comp_name in component_ids:
                            comp_id = component_ids[comp_name]
                            # Add edge from component to capability
                            elements.append({
                                'data': {
                                    'source': comp_id,
                                    'target': cap_id,
                                    'relation': 'enables'
                                }
                            })
            
            # Create process nodes for BUILDING DOMAIN (System → Process)
            process_ids = {}  # Track process IDs
            if 'Process' in extracted_frames:
                processes = extracted_frames['Process']
                for proc_name, proc_data in processes.items():
                    if proc_name == 'frame':  # Skip frame identifier
                        continue
                    
                    proc_id = f"proc_{proc_name}_{idx}"
                    process_ids[proc_name] = proc_id
                    
                    # Create clean process label
                    clean_proc_label = proc_name.replace('_', ' ').title()
                    
                    elements.append({
                        'data': {
                            'id': proc_id,
                            'label': clean_proc_label,
                            'type': 'process',
                            'domain': domain,
                            'confidence': confidence_scores.get('Process', 0.8),
                            'details': json.dumps(proc_data, indent=2),
                            'expanded': False
                        },
                        'classes': f'process domain-{domain}'
                    })
                    created_node_ids.add(proc_id)
                    
                    # Connect processes to systems that enable them
                    required_systems = proc_data.get('required_systems', [])
                    if not required_systems:
                        # Also check singular form as fallback
                        single_sys = proc_data.get('required_system')
                        if single_sys:
                            required_systems = [single_sys]
                    
                    for req_system in required_systems:
                        # Try to match system by name
                        for sys_name in system_ids:
                            if req_system == sys_name or req_system in sys_name or sys_name in req_system:
                                sys_id = system_ids[sys_name]
                                # Add edge from system to process
                                elements.append({
                                    'data': {
                                        'source': sys_id,
                                        'target': proc_id,
                                        'relation': 'enables'
                                    }
                                })
                                break
            
            # Create action nodes LAST (Capability → Action)
            action_ids = {}  # Track action IDs
            if 'Action' in extracted_frames:
                actions = extracted_frames['Action']
                for action_name, action_data in actions.items():
                    if action_name == 'frame':  # Skip frame identifier
                        continue
                    
                    action_id = f"act_{action_name}_{idx}"
                    action_ids[action_name] = action_id
                    
                    # Create clean action label
                    clean_action_label = action_name.replace('_', ' ').title()
                    
                    elements.append({
                        'data': {
                            'id': action_id,
                            'label': clean_action_label,
                            'type': 'action',
                            'domain': domain,
                            'confidence': confidence_scores.get('Action', 0.8),
                            'details': json.dumps(action_data, indent=2),
                            'expanded': False
                        },
                        'classes': f'action domain-{domain}'
                    })
                    created_node_ids.add(action_id)
                    
                    # Connect actions to capabilities that perform them
                    required_capabilities = action_data.get('required_capabilities', [])
                    if not required_capabilities:
                        # Also check singular form as fallback
                        single_cap = action_data.get('required_capability')
                        if single_cap:
                            required_capabilities = [single_cap]
                    
                    for req_capability in required_capabilities:
                        # Try to match capability by name
                        for cap_name in capability_ids:
                            if req_capability == cap_name or req_capability in cap_name or cap_name in req_capability:
                                cap_id = capability_ids[cap_name]
                                # Add edge from capability to action
                                elements.append({
                                    'data': {
                                        'source': cap_id,
                                        'target': action_id,
                                        'relation': 'performs'
                                    }
                                })
                                break
        
        print(f"🔍 Debug: Created {len([e for e in elements if 'source' not in e.get('data', {})])} nodes")
        print(f"🔍 Debug: Created {len([e for e in elements if 'source' in e.get('data', {})])} edges")
        
        return elements
    
    def _create_text_frame_content(self, selected_scenario, domain_filter=None):
        """Create text-to-frame comparison content with annotated lexical units"""
        if selected_scenario is None or not self.extraction_results:
            return dbc.Alert("Please select a scenario to analyze", color="warning")
        
        result = self.extraction_results[selected_scenario]
        
        # Apply domain filter if specified
        if domain_filter and domain_filter != 'all' and result.get('domain') != domain_filter:
            return dbc.Alert(f"Selected scenario is not from {domain_filter} domain", color="warning")
        source_text = result.get('source_text', 'No source text available')
        extracted_frames = result.get('extracted_frames', {})
        lexical_units_data = result.get('lexical_units', {})
        
        # Create annotated text with lexical unit highlighting
        annotated_text = self._create_annotated_text(source_text, extracted_frames, lexical_units_data)
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4(f"📝 Text-to-Frame Extraction - Scenario {selected_scenario + 1}"),
                    html.Hr()
                ])
            ]),
            
            # Legend for lexical unit colors and explicit/implicit evocations
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.Strong("🎨 Lexical Unit Legend: "),
                        html.Br(),
                        html.Strong("Robotics Domain: "),
                        html.Span("Robot", className="legend-badge legend-robot"),
                        html.Span("Component", className="legend-badge legend-component-robotics", title="Robotics components (sensors, actuators)"), 
                        html.Span("Capability", className="legend-badge legend-capability"),
                        html.Span("Action", className="legend-badge legend-action"),
                        html.Br(),
                        html.Strong("Building Domain: "),
                        html.Span("Building", className="legend-badge legend-building"),
                        html.Span("Component", className="legend-badge legend-component-building", title="Building components (HVAC, electrical)"),
                        html.Span("System", className="legend-badge legend-system"),
                        html.Span("Process", className="legend-badge legend-process"),
                        html.Br(),
                        html.Small([
                            "💡 ", html.Strong("Solid underline"), " = Explicit frame evocation • ",
                            html.Strong("Dashed underline"), " = Mixed explicit + implicit evocation • ",
                            "Hover for details"
                        ], className="text-muted")
                    ], color="light", className="mb-3")
                ])
            ]),
            
            dbc.Row([
                # Annotated Original Text
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("📄 Original Text with Lexical Unit Annotations"),
                            html.Small("Underlined words/phrases evoke semantic frames", className="text-muted")
                        ]),
                        dbc.CardBody([
                            annotated_text
                        ], style={'max-height': '500px', 'overflow-y': 'auto'})
                    ])
                ], width=6),
                
                # Extracted Frames
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🎯 Extracted Semantic Frames")),
                        dbc.CardBody([
                            html.Pre(json.dumps(extracted_frames, indent=2, ensure_ascii=False),
                                    style={'max-height': '500px', 'overflow-y': 'auto',
                                           'font-size': '12px'})
                        ])
                    ])
                ], width=6)
            ]),
            
            # Optional hover info area (for future enhancement)
            html.Div(id='lexical-hover-info', className="mt-3"),
            
            html.Hr(),
            
            # Confidence and Relationships
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("📈 Confidence Scores")),
                        dbc.CardBody([
                            html.Div([
                                dbc.Badge(f"{frame}: {score:.2f}", 
                                         color="success" if score > 0.7 else "warning" if score > 0.5 else "danger",
                                         className="me-2 mb-2")
                                for frame, score in result.get('confidence_scores', {}).items()
                            ])
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("🔗 Relationships")),
                        dbc.CardBody([
                            html.Ul([
                                html.Li(f"{rel[0]} → {rel[1]} → {rel[2]}")
                                for rel in result.get('relationships', [])[:10]  # Show first 10
                            ]) if result.get('relationships') else html.P("No relationships found")
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_statistics_content(self, domain_filter=None):
        """Create statistics dashboard content"""
        if not self.extraction_results:
            return dbc.Alert("No extraction results available for statistics", color="warning")
        
        # Filter results by domain if specified
        filtered_results = self.extraction_results
        if domain_filter and domain_filter != 'all':
            filtered_results = [r for r in self.extraction_results if r.get('domain') == domain_filter]
            if not filtered_results:
                return dbc.Alert(f"No extraction results found for {domain_filter} domain", color="warning")
        
        # Calculate statistics
        domain_counts = {}
        frame_type_counts = {}
        confidence_scores = []
        
        for result in filtered_results:
            domain = result.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            for frame_name in result.get('extracted_frames', {}).keys():
                frame_type_counts[frame_name] = frame_type_counts.get(frame_name, 0) + 1
            
            confidence_scores.extend(result.get('confidence_scores', {}).values())
        
        # Create charts
        domain_fig = px.pie(
            values=list(domain_counts.values()),
            names=list(domain_counts.keys()),
            title="Domain Distribution"
        )
        
        frame_fig = px.bar(
            x=list(frame_type_counts.keys()),
            y=list(frame_type_counts.values()),
            title="Frame Type Frequency"
        )
        
        confidence_fig = px.histogram(
            x=confidence_scores,
            nbins=20,
            title="Confidence Score Distribution"
        )
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("📈 Extraction Statistics"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=domain_fig)
                ], width=4),
                dbc.Col([
                    dcc.Graph(figure=frame_fig)
                ], width=4),
                dbc.Col([
                    dcc.Graph(figure=confidence_fig)
                ], width=4)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📊 Summary Statistics"),
                            html.P(f"Total Scenarios: {len(filtered_results)}"),
                            html.P(f"Average Confidence: {sum(confidence_scores)/len(confidence_scores):.3f}" if confidence_scores else "N/A"),
                            html.P(f"Knowledge Graph Nodes: {self.knowledge_graph.number_of_nodes()}"),
                            html.P(f"Knowledge Graph Edges: {self.knowledge_graph.number_of_edges()}")
                        ])
                    ])
                ])
            ])
        ])
    
    def run(self, debug=True, port=8050):
        """Run the visualization app"""
        print(f"🚀 Starting Semantic Visualization System on http://localhost:{port}")
        self.app.run(debug=debug, port=port)

def main():
    """Main function to run the visualizer"""
    
    # Check if extraction results exist (prioritize domain-specific files)
    extraction_files = [
        "output/extractions/test_extraction_results.json",  # Keep for backward compatibility
        "output/extractions/robot_test_result.json",         # New robotics-specific file
        "output/extractions/building_test_result.json",      # New building-specific file
        "output/extractions/combined_test_results.json",     # Legacy
        "output/extractions/construction_robotics_extraction.json"  # Legacy
    ]
    
    extraction_file = None
    for file_path in extraction_files:
        if Path(file_path).exists():
            extraction_file = file_path
            break
    
    if not extraction_file:
        print("⚠️  No extraction results found!")
        print("💡 Run one of these to generate extraction data:")
        print("   python src/semantic_extraction_system.py")
        print("   python tests/test_extraction_system.py")
        
        # Create demo visualizer with no data
        visualizer = SemanticVisualizationSystem()
    else:
        print(f"✅ Loading extraction results from {extraction_file}")
        visualizer = SemanticVisualizationSystem(extraction_file)
    
    # Run the app
    visualizer.run(debug=True, port=8050)

if __name__ == "__main__":
    main()
