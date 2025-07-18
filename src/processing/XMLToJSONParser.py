import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

def create_training_dataset(framenet_path):
    """
    Parses the FrameNet 1.7 XML data to create a JSON Lines dataset
    for fine-tuning a Large Language Model.

    Args:
        framenet_path (str): The path to the root of the FrameNet 1.7
                             data directory (e.g., 'fndata-1.7/').
    """
    lu_dir = os.path.join(framenet_path, 'lu')
    data_path = 'data/'
    output_path = os.path.join(data_path, 'dataset')
    output_file = 'framenet_1.7_training_data.jsonl'
    training_data = []
    
    # The namespace is required to find elements in the XML files
    ns = {'fn': 'http://framenet.icsi.berkeley.edu'}

    print(f"Looking for LU files in: {lu_dir}")
    if not os.path.isdir(lu_dir):
        print(f"Error: Directory not found at {lu_dir}")
        print("Please ensure the framenet_path is correct.")
        return

    lu_files = [f for f in os.listdir(lu_dir) if f.endswith('.xml')]
    print(f"Found {len(lu_files)} LU files to process.")

    # Use tqdm for a progress bar
    for filename in tqdm(lu_files, desc="Processing LU files"):
        try:
            lu_path = os.path.join(lu_dir, filename)
            tree = ET.parse(lu_path)
            root = tree.getroot()

            frame_name = root.get('frame')
            lu_name = root.get('name')

            # Find all sentences with manual annotations
            for sentence in root.findall('.//fn:sentence', ns):
                annotation_sets = sentence.findall('.//fn:annotationSet[@status="MANUAL"]', ns)
                
                for a_set in annotation_sets:
                    sentence_text_element = sentence.find('fn:text', ns)
                    if sentence_text_element is None or sentence_text_element.text is None:
                        continue
                    sentence_text = sentence_text_element.text

                    # Extract the target word
                    target_element = a_set.find('.//fn:layer[@name="Target"]/fn:label', ns)
                    if target_element is None:
                        continue
                        
                    target_start_str = target_element.get('start')
                    target_end_str = target_element.get('end')

                    if target_start_str is None or target_end_str is None:
                        continue
                    
                    target_start = int(target_start_str)
                    target_end = int(target_end_str)
                    target_text = sentence_text[target_start : target_end + 1]

                    # Extract all frame elements
                    frame_elements = []
                    fe_labels = a_set.findall('.//fn:layer[@name="FE"]/fn:label', ns)
                    for label in fe_labels:
                        fe_name = label.get('name')
                        fe_start_str = label.get('start')
                        fe_end_str = label.get('end')

                        # This handles cases of "null instantiation" where an FE is
                        # present conceptually but not in the text.
                        if fe_start_str is not None and fe_end_str is not None:
                            fe_start = int(fe_start_str)
                            fe_end = int(fe_end_str)
                            fe_text = sentence_text[fe_start : fe_end + 1]
                            frame_elements.append({
                                "fe_name": fe_name,
                                "text": fe_text
                            })
                    
                    # Only include examples with at least one frame element
                    if not frame_elements:
                        continue

                    # Assemble the JSON object for this training example
                    instruction = (f"The word '{target_text}' evokes the '{frame_name}' frame. "
                                   f"Identify the arguments of this frame in the sentence.")
                    
                    training_example = {
                        "instruction": instruction,
                        "input": sentence_text,
                        "output": {
                            "target": target_text,
                            "frame": frame_name,
                            "frame_elements": frame_elements
                        }
                    }
                    training_data.append(training_example)

        except ET.ParseError as e:
            print(f"Could not parse {filename}: {e}")
        except Exception as e:
            print(f"An error occurred with file {filename}: {e}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    full_output_path = os.path.join(output_path, output_file)

    # Write the collected data to a .jsonl file
    with open(full_output_path, 'w', encoding='utf-8') as f:
        for entry in training_data:
            json.dump(entry, f)
            f.write('\n')
            
    print(f"\nProcessing complete.")
    print(f"Successfully created {len(training_data)} training examples.")
    print(f"Dataset saved to: {full_output_path}")


if __name__ == '__main__':
    # IMPORTANT: Update this path to the location of your FrameNet 1.7 data
    path_to_framenet = 'data/FNdata-1.7/' 
    create_training_dataset(path_to_framenet)
