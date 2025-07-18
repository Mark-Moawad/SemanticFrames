# Semantic Robot Frames Documentation

This document describes the semantic frame decomposition of the robot blueprint into four core entities following SRDL (Semantic Robot Description Language) principles, ISO 8373 robotics standards, and FrameNet's linguistic framework.

## Overview

The robot blueprint has been decomposed into four interconnected semantic frames, each augmented with **lexical units** and **frame elements** following Charles Fillmore's FrameNet approach:

1. **Robot** - Root frame representing the complete robotic system
2. **Component** - Physical parts and subsystems that constitute the robot
3. **Capability** - Functional abilities that emerge from component integration
4. **Action** - Specific behaviors and tasks the robot can perform

## FrameNet Integration

Each semantic frame now includes:

### Lexical Units
Words and terms from the robotics lexicon (ISO 8373) that can evoke each frame:
- **Robot**: "robot", "industrial robot", "service robot", "autonomous system", "cobot"
- **Component**: "actuator", "sensor", "end-effector", "gripper", "controller", "servo"
- **Capability**: "manipulation", "locomotion", "navigation", "perception", "autonomy"
- **Action**: "pick", "place", "grasp", "navigate", "assemble", "inspect", "monitor"

### Frame Elements
Semantic roles that entities play within each frame, categorized as:

#### Core Elements
Essential roles that define the frame's meaning:
- **Robot**: Agent, Function, Domain
- **Component**: Component, Whole, Function  
- **Capability**: Agent, Capability, Domain
- **Action**: Agent, Action, Object

#### Peripheral Elements
Additional roles that provide context:
- **Robot**: Designer, Operator, Configuration, Specifications
- **Component**: Specifications, Material, Configuration, Interface
- **Capability**: Enabler, Performance, Degree, Method
- **Action**: Instrument, Method, Parameters, Result

#### Extra-thematic Elements
Background information and contextual details:
- Common across frames: Time, Place, Purpose, Condition

## Annotated Examples

The file `robotics_frame_examples.jsonl` contains training examples showing how lexical units evoke frames and how frame elements are identified in robot descriptions:

```json
{
  "instruction": "The word 'robot' evokes the 'Robot' frame. Identify the frame elements in this sentence.",
  "input": "The industrial robot in the automotive assembly line performs welding operations with high precision.",
  "output": {
    "target": "robot",
    "frame": "Robot", 
    "frame_elements": [
      {"fe_name": "Agent", "text": "The industrial robot"},
      {"fe_name": "Domain", "text": "automotive assembly line"},
      {"fe_name": "Function", "text": "performs welding operations"},
      {"fe_name": "Specifications", "text": "with high precision"}
    ]
  }
}
```

## Frame Hierarchy

```
Robot (Root Frame)
├── has_component → Component
├── exhibits_capability → Capability
└── performs_action → Action

Component
├── part_of → Robot
├── enables → Capability
└── used_by → Action

Capability
├── exhibited_by → Robot
├── enabled_by → Component
└── supports → Action

Action
├── performed_by → Robot
├── requires → Capability
└── uses → Component
```

## Frame Descriptions

### Robot Frame
- **Purpose**: Central entity representing the complete robotic system
- **Key Properties**: Identifier, type classification, coordinate systems, operating modes
- **Relationships**: Aggregates components, exhibits capabilities, performs actions
- **Constraints**: Component-capability consistency, operational safety compliance

### Component Frame  
- **Purpose**: Physical elements that constitute the robot hardware
- **Key Properties**: Component types (based on ISO 8373), specifications, operating conditions
- **Relationships**: Belongs to robot, enables capabilities, used by actions
- **Constraints**: Physical compatibility, operational limits, safety boundaries

### Capability Frame
- **Purpose**: Functional abilities that emerge from component integration
- **Key Properties**: Capability types, performance metrics, environmental constraints
- **Relationships**: Exhibited by robot, enabled by components, supports actions
- **Constraints**: Component availability, prerequisite satisfaction, environmental compatibility

### Action Frame
- **Purpose**: Specific executable behaviors and tasks
- **Key Properties**: Action types, parameters, preconditions, postconditions
- **Relationships**: Performed by robot, requires capabilities, uses components
- **Constraints**: Capability availability, precondition satisfaction, resource availability

## Design Principles

### SRDL Compliance
- **Hierarchical Structure**: Clear parent-child relationships between entities
- **Semantic Consistency**: Consistent property naming and typing across frames
- **Modularity**: Each frame can be independently extended or modified
- **Interoperability**: Standard interfaces for cross-frame relationships

### ISO 8373 Integration
- **Terminology Alignment**: Component types and robot classifications follow ISO standards
- **Safety Considerations**: Safety requirements embedded in constraints and properties
- **Performance Metrics**: Standardized measurement criteria for capabilities and actions

## Usage Guidelines

### Frame Instantiation
1. Create Robot instance as root entity
2. Define Component instances for all physical parts
3. Specify Capability instances based on component integration
4. Define Action instances that utilize capabilities

### Relationship Management
- Use explicit relationship properties to link frame instances
- Validate constraints before establishing relationships
- Maintain referential integrity across frame connections

### Extensibility
- Add new component types to Component frame taxonomy
- Extend capability categories for specialized robot functions
- Define custom action types for domain-specific behaviors

## Benefits

### For LLM/RAG Systems
- **Structured Knowledge**: Clear semantic boundaries enable precise querying
- **Contextual Understanding**: Relationships provide rich context for reasoning
- **Scalable Architecture**: Frame-based design supports incremental knowledge growth

### For Robot Development
- **Modular Design**: Component-based architecture supports reusable designs
- **Capability Mapping**: Clear mapping from hardware to functional abilities
- **Action Planning**: Structured action definitions enable automated planning

### For System Integration
- **Standard Interfaces**: Consistent frame structure enables tool interoperability
- **Validation Framework**: Built-in constraints support system verification
- **Documentation**: Self-documenting structure reduces integration complexity

## File Structure

```
output/frames/
├── Robot.json                    # Root robot semantic frame with lexical units & frame elements
├── Component.json                # Physical component semantic frame with robotics terminology
├── Capability.json               # Functional capability semantic frame with performance metrics
├── Action.json                   # Executable action semantic frame with behavioral terms
├── robotics_frame_examples.jsonl # Annotated training examples following FrameNet format
└── README.md                     # This documentation file
```

Each frame file now contains:
- **Frame definition**: Core entity description
- **Lexical units**: Terms from robotics lexicon that evoke the frame
- **Frame elements**: Semantic roles (core, peripheral, extra-thematic)
- **Properties schema**: Typed property specifications  
- **Relationships**: Cross-frame connection definitions
- **Constraints**: Validation rules and requirements

## Applications

### Text Processing & Understanding
The augmented frames enable semantic parsing of robot descriptions:
1. **Lexical Unit Recognition**: Identify robotics terms that trigger frame activation
2. **Frame Element Extraction**: Parse sentences to identify semantic roles
3. **Knowledge Integration**: Connect parsed information to structured robot models
4. **Cross-frame Reasoning**: Understand relationships between components, capabilities, and actions

### Training Data Generation  
The robotics frame examples provide templates for:
- Training semantic parsers for robotics domain
- Generating synthetic robot descriptions with proper frame structure
- Evaluating frame-based understanding systems
- Creating robotics-specific NLP models

This FrameNet-inspired approach bridges the gap between natural language robot descriptions and structured semantic knowledge, enabling more intelligent robot understanding and reasoning systems.
