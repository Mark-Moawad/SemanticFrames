# Building Semantic Frames

## 📋 Overview

This directory contains semantic frames for the building/construction domain, structured parallel to the robotics frames but incorporating ISO building standards and professional building terminology.

## 🏗️ Frame Hierarchy

```
Building (Root Frame)
├── Building_System (Infrastructure level)
│   └── Building_Component (Physical elements)
├── Building_Function (Capabilities/Services)
│   └── Building_Process (Operations/Procedures)
└── Actor (People/Organizations)
```

## 📁 Frame Files

### Core Frames
- **`Building.json`** - Root frame for building entities (based on ISO 16739-1 IFC)
- **`Building_System.json`** - Building infrastructure systems (HVAC, Electrical, etc.)
- **`Building_Component.json`** - Individual physical elements (walls, doors, sensors)
- **`Building_Function.json`** - High-level building capabilities (shelter, climate control)
- **`Building_Process.json`** - Building operations and procedures (maintenance, inspection)
- **`Actor.json`** - People and organizations in building processes (ISO 19650 roles)

## 🔗 Integration Sources

### ISO Standards Integration
- **ISO 16739-1:2018** - IFC Data Schema (Industry Foundation Classes)
- **ISO 19650-1/2:2018** - BIM Management using Information Management
- **ISO 29481-1:2016** - Information Delivery Manual (IDM) methodology

### Lexicon Integration
- **Source**: `building_lexicon.json` (135 professional terms)
- **Standards**: 10 cleaned ISO building standards files
- **Languages**: English + German→English translations

### FrameNet Integration
- **Primary Frames**: Buildings, Building_subparts, Infrastructure
- **Related Frames**: Architecture, Locale_by_use, Containers
- **Dataset**: `framenet_1.7_training_data.jsonl`

## 🎯 Frame Element Structure

Each frame follows this consistent structure:

```json
{
  "frame": "Frame_Name",
  "description": "Frame definition with ISO standard reference",
  "lexical_units": ["vocabulary", "terms", "that", "evoke", "frame"],
  "frame_elements": {
    "core": {
      "Element": {
        "description": "Core semantic role",
        "semantic_type": "FrameNet_type",
        "lexicon_reference": "building_lexicon term mapping"
      }
    },
    "peripheral": {
      "Optional_Element": {
        "description": "Non-core semantic role"
      }
    }
  },
  "frame_relations": {},
  "iso_standards_alignment": {},
  "lexicon_integration": {},
  "framenet_connections": {},
  "example_sentences": []
}
```

## 🔄 Frame Relations

### Inheritance Hierarchy
- `Building` ← `Building_System` ← `Building_Component`
- `Building` ← `Building_Function` ← `Building_Process`

### Semantic Relations
- **Building has_system Building_System**
- **Building_System consists_of Building_Component**
- **Building has_function Building_Function**
- **Building_Function enables Building_Process**
- **Actor performs Building_Process**

## 🏢 Domain-Specific Features

### IFC Element Mapping
Building components map directly to IFC entity types:
- `IfcWall`, `IfcDoor`, `IfcWindow` (Building envelope)
- `IfcBeam`, `IfcColumn`, `IfcSlab` (Structural elements)
- `IfcPipe`, `IfcDuct`, `IfcSensor` (Building services)

### ISO 19650 Actor Roles
Actor frame incorporates BIM management roles:
- Appointing Party / Appointed Party
- Lead Appointed Party / Task Team
- Facility Manager / Building Operator

### Building Process Categories
- **Operational**: Daily operation, energy management
- **Maintenance**: Preventive, corrective, cleaning
- **Lifecycle**: Commissioning, renovation, decommissioning
- **Management**: Asset management, compliance

## 📊 FrameNet Alignment

### Buildings Frame
- **Frame Elements**: Building, Name, Descriptor, Relative_location
- **Lexical Units**: bar, building, facility, structure, establishment

### Building_subparts Frame  
- **Frame Elements**: Building_part, Whole, Orientation
- **Lexical Units**: belfry, room, floor, wall, door, window

### Infrastructure Frame
- **Frame Elements**: Infrastructure, Activity, Resource, Possessor
- **Lexical Units**: infrastructure, system, network, facility

## 🎯 Usage Examples

### Building Frame
```
"The [Asset office building] serves a [Function commercial] purpose."
```

### Building_System Frame
```
"The [System HVAC system] provides [Function climate control]."
```

### Building_Process Frame
```
"The [Agent facility manager] performs [Process maintenance] on the [System electrical system]."
```

## 🔧 Implementation Notes

### Lexicon References
Frame elements include `lexicon_reference` fields that link to specific terms in `building_lexicon.json`:
```json
"lexicon_reference": "actor - person, organization or organizational unit involved in construction process"
```

### ISO Standards Alignment
Each frame includes `iso_standards_alignment` sections mapping frame elements to specific ISO standard concepts.

### FrameNet Connections
Frames reference corresponding FrameNet frames and provide `framenet_connections` for linguistic grounding.

## 📈 Comparison with Robotics Frames

| **Aspect** | **Robotics** | **Buildings** |
|------------|--------------|---------------|
| **Root Entity** | Robot | Building |
| **Physical Parts** | Component | Building_System → Building_Component |
| **Capabilities** | Capability | Building_Function |
| **Operations** | Action | Building_Process |
| **People** | (implicit) | Actor (explicit) |
| **Standards** | ISO 8373 | ISO 16739-1, 19650, 29481 |
| **Domain Focus** | Automation | Construction/Facility Management |

The building frames provide equivalent semantic richness to robotics frames while incorporating building-specific concepts, standards, and terminology appropriate for construction and facility management domains.
