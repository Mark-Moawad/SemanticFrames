#!/usr/bin/env python3
"""
Semantic Frame Annotation Tool

A simple interactive tool for annotating robotics sentences with:
- Frame evocations (Robot, Component, Capability, Action)  
- Lexical units and their frame elements
- Explicit/implicit frame relationships

This creates ground truth data for benchmarking the LLM + GraphRAG pipeline.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

class FrameAnnotationTool:
    def __init__(self):
        self.frames = ["Robot", "Component", "Capability", "Action"]
        self.robot_elements = ["Agent", "Function", "Domain"]
        self.component_elements = ["Component", "Whole", "Function"]
        self.capability_elements = ["Capability", "Agent", "Enabler"]
        self.action_elements = ["Action", "Agent", "Instrument", "Target"]
        
        self.frame_elements = {
            "Robot": self.robot_elements,
            "Component": self.component_elements, 
            "Capability": self.capability_elements,
            "Action": self.action_elements
        }
        
        self.evocation_types = ["explicit", "implicit"]
        
        # Load robotics lexicon for reference
        self.load_robotics_lexicon()
        
    def load_robotics_lexicon(self):
        """Load robotics lexicon for term lookup"""
        lexicon_path = "data/resources/robotics_lexicon.json"
        if os.path.exists(lexicon_path):
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                self.robotics_lexicon = json.load(f)
            print("✅ Loaded robotics lexicon for reference")
        else:
            self.robotics_lexicon = {}
            print("⚠️ Robotics lexicon not found")
    
    def find_lexicon_match(self, term: str) -> List[str]:
        """Find matching terms in robotics lexicon"""
        matches = []
        term_lower = term.lower()
        
        for section_id, section in self.robotics_lexicon.items():
            if "terms" in section:
                for lexicon_term in section["terms"]:
                    if term_lower == lexicon_term.lower():
                        matches.append(f"{section_id}: {lexicon_term}")
        return matches
    
    def display_sentence(self, sentence: str, index: int, total: int):
        """Display current sentence for annotation"""
        print("\n" + "="*80)
        print(f"📝 ANNOTATION [{index}/{total}]")
        print("="*80)
        print(f"SENTENCE: {sentence}")
        print("="*80)
        
    def get_lexical_units(self, sentence: str) -> List[str]:
        """Extract potential lexical units from sentence"""
        print("\n🔍 STEP 1: Identify Lexical Units")
        print("-" * 40)
        print("Enter lexical units found in the sentence.")
        print("Examples: 'mobile robot', 'manipulator arm', 'camera sensor'")
        print("Type 'done' when finished.")
        
        units = []
        while True:
            unit = input(f"\nLexical unit #{len(units)+1}: ").strip()
            if unit.lower() == 'done':
                break
            if unit:
                # Check robotics lexicon
                matches = self.find_lexicon_match(unit)
                if matches:
                    print(f"  📚 Found in lexicon: {matches[0]}")
                units.append(unit)
        
        return units
    
    def annotate_lexical_unit(self, unit: str) -> Dict[str, Any]:
        """Annotate a single lexical unit"""
        print(f"\n📋 Annotating: '{unit}'")
        print("-" * 40)
        
        # Primary frame
        print("Available frames:", ", ".join(self.frames))
        while True:
            primary_frame = input("Primary frame: ").strip()
            if primary_frame in self.frames:
                break
            print(f"❌ Invalid frame. Choose from: {', '.join(self.frames)}")
        
        # Frame element
        elements = self.frame_elements[primary_frame]
        print(f"Available elements for {primary_frame}:", ", ".join(elements))
        while True:
            element = input("Frame element: ").strip()
            if element in elements:
                break
            print(f"❌ Invalid element. Choose from: {', '.join(elements)}")
        
        # Evocation type
        print("Evocation types:", ", ".join(self.evocation_types))
        while True:
            evocation = input("Evocation (explicit/implicit): ").strip().lower()
            if evocation in self.evocation_types:
                break
            print("❌ Invalid evocation. Use 'explicit' or 'implicit'")
        
        # Implicit frames (optional)
        implicit_frames = []
        print("\n🔗 Add implicit frame evocations? (y/n):")
        if input().lower().startswith('y'):
            print("Enter implicit frames (type 'done' when finished):")
            while True:
                frame = input("Implicit frame: ").strip()
                if frame.lower() == 'done':
                    break
                if frame in self.frames:
                    elem_options = self.frame_elements[frame]
                    print(f"Elements for {frame}:", ", ".join(elem_options))
                    element = input("Element: ").strip()
                    if element in elem_options:
                        evoc = input("Evocation (explicit/implicit): ").strip().lower()
                        if evoc in self.evocation_types:
                            implicit_frames.append({
                                "frame": frame,
                                "element": element,
                                "evocation": evoc
                            })
        
        annotation = {
            "frame": primary_frame,
            "element": element,
            "evocation": evocation
        }
        
        if implicit_frames:
            annotation["implicit_frames"] = implicit_frames
            
        return annotation
    
    def annotate_sentence(self, sentence: str, index: int, total: int) -> Dict[str, Any]:
        """Annotate a complete sentence"""
        self.display_sentence(sentence, index, total)
        
        # Get lexical units
        lexical_units = self.get_lexical_units(sentence)
        
        if not lexical_units:
            print("⚠️ No lexical units identified. Skipping sentence.")
            return None
        
        # Annotate each lexical unit
        annotations = {}
        for unit in lexical_units:
            annotations[unit] = self.annotate_lexical_unit(unit)
        
        # Ask for domain confirmation
        print(f"\n🏷️ Domain: robotics (confirm? y/n):")
        domain = "robotics" if input().lower().startswith('y') else input("Enter domain: ").strip()
        
        result = {
            "source_text": sentence,
            "domain": domain,
            "lexical_units": annotations,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def run_annotation_session(self, sentences: List[str], output_file: str):
        """Run complete annotation session"""
        print("\n🎯 SEMANTIC FRAME ANNOTATION TOOL")
        print("="*50)
        print("Instructions:")
        print("- Identify lexical units in each sentence")
        print("- Assign primary frame and element")
        print("- Mark evocation as explicit/implicit")
        print("- Add implicit frame evocations if present")
        print("="*50)
        
        annotations = []
        
        for i, sentence in enumerate(sentences, 1):
            try:
                annotation = self.annotate_sentence(sentence, i, len(sentences))
                if annotation:
                    annotations.append(annotation)
                    print(f"✅ Annotated sentence {i}/{len(sentences)}")
                
                # Save progress
                if i % 5 == 0:
                    self.save_annotations(annotations, output_file)
                    print(f"💾 Progress saved after {i} sentences")
                    
            except KeyboardInterrupt:
                print("\n⏸️ Annotation interrupted. Saving progress...")
                break
        
        # Final save
        self.save_annotations(annotations, output_file)
        print(f"\n🎉 Annotation complete! Saved {len(annotations)} annotations to {output_file}")
        
        return annotations
    
    def save_annotations(self, annotations: List[Dict], output_file: str):
        """Save annotations to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)


def create_robotics_test_sentences() -> List[str]:
    """Create 20 simple robotics test sentences"""
    return [
        # Basic robot descriptions
        "The mobile robot has a manipulator arm and camera sensor",
        "The industrial robot performs welding operations",
        "The service robot navigates through the warehouse",
        "The humanoid robot walks on two legs",
        "The autonomous robot avoids obstacles using lidar",
        
        # Component descriptions  
        "The robot arm has six degrees of freedom",
        "The gripper grasps objects with precision",
        "The vision system detects defective parts",
        "The mobile base moves on four wheels",
        "The end effector holds the workpiece",
        
        # Action descriptions
        "The robot picks up the box",
        "The manipulator places components on the assembly line", 
        "The mobile platform transports materials",
        "The robotic system assembles electronic devices",
        "The automation cell packages products",
        
        # Capability descriptions
        "The robot has navigation capabilities",
        "The system provides manipulation functions",
        "The sensor enables object recognition",
        "The controller manages robot movements",
        "The software plans optimal paths"
    ]


def main():
    """Main annotation workflow"""
    print("🤖 ROBOTICS FRAME ANNOTATION TOOL")
    print("="*50)
    
    # Create test sentences
    sentences = create_robotics_test_sentences()
    print(f"📝 Created {len(sentences)} test sentences")
    
    # Initialize annotation tool
    tool = FrameAnnotationTool()
    
    # Output file
    output_file = "data/test_dataset/robotics_annotations.json"
    
    print(f"\n🎯 Starting annotation session...")
    print(f"📁 Results will be saved to: {output_file}")
    print("\nReady to start? (press Enter)")
    input()
    
    # Run annotation
    annotations = tool.run_annotation_session(sentences, output_file)
    
    print(f"\n✅ ANNOTATION COMPLETE!")
    print(f"📊 Total sentences annotated: {len(annotations)}")
    print(f"📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
