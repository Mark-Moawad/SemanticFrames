#!/usr/bin/env python3
"""
Web-based Semantic Frame Annotation Tool

A simple Flask web interface for annotating robotics sentences with semantic frames.
Much easier to use than the command-line tool.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
from datetime import datetime
from typing import Dict, List, Any

# Configure Flask to look for templates in the correct directory
app = Flask(__name__, template_folder='../templates')

class WebAnnotationTool:
    def __init__(self):
        # Robotics frames (current default)
        self.robotics_frames = ["Robot", "Component", "Capability", "Action"]
        self.robotics_frame_elements = {
            "Robot": ["Agent", "Function", "Domain"],
            "Component": ["Component", "Whole", "Function"],
            "Capability": ["Capability", "Agent", "Enabler"],
            "Action": ["Action", "Agent", "Instrument", "Target"]
        }
        
        # Building frames (IFC-compliant hierarchy)
        self.building_frames = ["Building", "System", "Component", "Process"]
        self.building_frame_elements = {
            "Building": ["Asset", "Function", "Location"],
            "System": ["System", "Building", "Function"],
            "Component": ["Component", "Whole", "Function"],
            "Process": ["Process", "Actor", "Target"]
        }
        
        # Default to robotics (can be made configurable)
        self.frames = self.robotics_frames
        self.frame_elements = self.robotics_frame_elements
        
        self.evocation_types = ["explicit", "implicit"]
        
        # Load test sentences
        self.load_test_sentences()
        self.load_existing_annotations()
        
    def load_test_sentences(self):
        """Load test sentences from JSON file"""
        sentences_file = "data/test_dataset/robotics_test_sentences.json"
        if os.path.exists(sentences_file):
            with open(sentences_file, 'r') as f:
                self.sentences = json.load(f)
        else:
            self.sentences = []
            
    def load_existing_annotations(self):
        """Load existing annotations if they exist"""
        annotations_file = "data/test_dataset/robotics_annotations.json"
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = []
    
    def save_annotation(self, annotation: Dict[str, Any]):
        """Save a single annotation"""
        # Check if annotation for this sentence already exists
        sentence_text = annotation["source_text"]
        existing_index = None
        
        for i, existing in enumerate(self.annotations):
            if existing["source_text"] == sentence_text:
                existing_index = i
                break
        
        # Update or append
        if existing_index is not None:
            self.annotations[existing_index] = annotation
        else:
            self.annotations.append(annotation)
        
        # Save to file
        annotations_file = "data/test_dataset/robotics_annotations.json"
        os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)

# Global tool instance
tool = WebAnnotationTool()

@app.route('/')
def index():
    """Main annotation interface"""
    total_sentences = len(tool.sentences)
    annotated_count = len(tool.annotations)
    
    return render_template('annotation_interface.html',
                         sentences=json.dumps(tool.sentences),
                         frames=json.dumps(tool.frames),
                         frame_elements=json.dumps(tool.frame_elements),
                         evocation_types=json.dumps(tool.evocation_types),
                         total_sentences=total_sentences,
                         annotated_count=annotated_count)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    """Save annotation via AJAX"""
    try:
        data = request.json
        
        # Create annotation structure
        annotation = {
            "source_text": data["sentence"],
            "domain": "robotics",
            "lexical_units": data["lexical_units"],
            "timestamp": datetime.now().isoformat()
        }
        
        tool.save_annotation(annotation)
        
        return jsonify({"success": True, "message": "Annotation saved successfully!"})
    
    except Exception as e:
        return jsonify({"success": False, "message": f"Error saving annotation: {str(e)}"})

@app.route('/get_annotations')
def get_annotations():
    """Return current annotations as JSON"""
    return jsonify(tool.annotations)

@app.route('/get_annotation/<int:sentence_index>')
def get_annotation_for_sentence(sentence_index):
    """Get annotation for a specific sentence by index"""
    try:
        if sentence_index < 0 or sentence_index >= len(tool.sentences):
            return jsonify({"success": False, "message": "Invalid sentence index"})
        
        sentence = tool.sentences[sentence_index]
        
        # Find annotation for this sentence
        for annotation in tool.annotations:
            if annotation.get("source_text") == sentence:
                return jsonify({"success": True, "annotation": annotation})
        
        return jsonify({"success": False, "message": "No annotation found for this sentence"})
    
    except Exception as e:
        return jsonify({"success": False, "message": f"Error loading annotation: {str(e)}"})

@app.route('/get_progress')
def get_progress():
    """Get current annotation progress"""
    return jsonify({
        "total_sentences": len(tool.sentences),
        "annotated_count": len(tool.annotations),
        "progress_percentage": (len(tool.annotations) / len(tool.sentences) * 100) if len(tool.sentences) > 0 else 0
    })

@app.route('/export')
def export_annotations():
    """Export annotations in benchmark format"""
    return jsonify({
        "total_sentences": len(tool.sentences),
        "annotated_sentences": len(tool.annotations),
        "annotations": tool.annotations
    })

if __name__ == '__main__':
    # Create templates directory
    templates_dir = "templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    print("🌐 Starting Web Annotation Tool...")
    print("📝 Navigate to: http://localhost:5000")
    print("💾 Annotations will be saved to: data/test_dataset/robotics_annotations.json")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
