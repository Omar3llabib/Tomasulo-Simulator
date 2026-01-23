import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.index import TomasuloEngine

def test_label_parsing():
    engine = TomasuloEngine()
    
    # Program with label on separate line
    lines = [
        "ADD F1, F0, 3",
        "L1:",
        "SUB F1, F1, 1",
        "BNE F1, F2, L1"
    ]
    
    print("Setting instructions with separate label line...")
    engine.reset()
    engine.set_instructions(lines)
    
    print(f"Instructions loaded: {len(engine.instruction_list)}")
    print(f"Labels found: {engine.labels}")
    
    if "L1" in engine.labels:
        print(f"SUCCESS: Found label L1 at index {engine.labels['L1']}")
    else:
        print("FAILURE: Label L1 not found!")

if __name__ == "__main__":
    test_label_parsing()
