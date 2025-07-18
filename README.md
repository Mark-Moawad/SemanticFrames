# SemanticFrames Project

A comprehensive framework for semantic robot description and understanding using FrameNet-inspired linguistic structures combined with ISO 8373 robotics standards.

## 🎯 Project Overview

This project bridges natural language processing and robotics by creating semantic frames that enable intelligent understanding of robot descriptions. It combines:

- **ISO 8373 Robotics Standards**: Comprehensive robotics terminology and definitions
- **FrameNet Linguistic Framework**: Semantic role labeling and frame-based understanding
- **SRDL Principles**: Semantic Robot Description Language for structured knowledge representation

## 📁 Project Structure

```
SemanticFrames/
├── data/                           # Core data and resources
│   ├── dataset/                    # Training datasets (FrameNet, etc.)
│   ├── FNdata-1.7/                # FrameNet 1.7 data
│   ├── resources/                  # Robotics lexicon and reference materials
│   └── ...
├── output/                         # Generated outputs and results
│   ├── blueprints/                 # Robot blueprint JSON files
│   └── frames/                     # Semantic frame definitions
├── scripts/                        # Processing and utility scripts
│   ├── processing/                 # Data processing scripts
│   ├── visualization/              # Visualization scripts
│   └── build_robotics_lexicon.py   # Main lexicon builder
├── lib/                           # Shared libraries and modules
└── venv/                          # Python virtual environment
```

## 🚀 Key Components

### 1. Robotics Lexicon (`data/resources/robotics_lexicon.json`)
- **128 sections** from ISO 8373 standard
- Structured terminology with definitions and cross-references
- Enables precise robotics knowledge representation

### 2. Semantic Frames (`output/frames/`)
- **Robot Frame**: Root entity for complete robotic systems
- **Component Frame**: Physical parts and subsystems
- **Capability Frame**: Functional abilities and skills
- **Action Frame**: Executable behaviors and tasks

### 3. FrameNet Integration
- **Lexical Units**: Terms that evoke each semantic frame
- **Frame Elements**: Semantic roles (Agent, Object, Instrument, etc.)
- **Annotated Examples**: Training data for semantic parsing

## 🛠️ Getting Started

### Prerequisites
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Build Robotics Lexicon
```bash
python scripts/build_robotics_lexicon.py
```

### Generate Semantic Frames
The semantic frames are pre-built in `output/frames/` with:
- Frame definitions following FrameNet structure
- Lexical units from robotics domain
- Semantic role annotations
- Cross-frame relationships

### Usage Examples
```python
# Load robotics lexicon
import json
with open('data/resources/robotics_lexicon.json', 'r') as f:
    lexicon = json.load(f)

# Load semantic frames
with open('output/frames/Robot.json', 'r') as f:
    robot_frame = json.load(f)
```

## 📊 Applications

### 1. Natural Language Robot Understanding
- Parse robot descriptions into structured semantic representations
- Extract entities, relationships, and capabilities from text
- Enable intelligent query answering about robot systems

### 2. Semantic Role Labeling
- Train NLP models for robotics domain
- Identify semantic roles in robot documentation
- Generate structured knowledge from unstructured text

### 3. Knowledge Graph Construction
- Build comprehensive robot knowledge graphs
- Link components, capabilities, and actions
- Support automated reasoning and planning

## 🔬 Research Applications

### Semantic Parsing
- **Input**: "The industrial robot in the automotive assembly line performs welding operations with high precision"
- **Output**: Robot frame with Agent="industrial robot", Domain="automotive assembly line", Function="welding operations"

### Frame-based Understanding
- Automatic identification of lexical units that evoke frames
- Extraction of frame elements and semantic roles
- Cross-frame relationship mapping

## 📈 Future Extensions

- [ ] Multi-modal frame integration (vision + language)
- [ ] Dynamic frame learning from robot documentation
- [ ] Integration with robot planning and control systems
- [ ] Expansion to other engineering domains

## 📚 References

- **ISO 8373:2021**: Robotics — Vocabulary
- **FrameNet**: Frame semantics and the structure of language
- **SRDL**: Semantic Robot Description Language

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code structure and documentation style
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Building bridges between natural language and robotics through semantic understanding.*