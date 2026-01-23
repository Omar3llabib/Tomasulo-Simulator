import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.index import TomasuloEngine

def test_repro():
    engine = TomasuloEngine()
    
    # User provided code
    lines = [
        "L1:",
        "ADD F1, F2, F3",
        "BNE R1, R2, L1"
    ]
    
    print("Loading user instructions...")
    engine.reset()
    engine.set_instructions(lines)
    
    print(f"Instructions loaded: {len(engine.instruction_list)}")
    for i, inst in enumerate(engine.instruction_list):
        print(f"{i}: {inst}")
    print(f"Labels: {engine.labels}")
    
    iterations = 2
    engine.target_iterations = iterations
    print(f"Setting target iterations to {iterations}")
    
    # Run simulation
    cycles = 0
    max_cycles = 20
    while cycles < max_cycles:
        state = engine.step_forward()
        print(f"Cycle {state.cycle}: Instructions Issued: {state.stats['instructions_issued']}")
        
        # Stop if done
        if engine.instruction_counter >= len(engine.instruction_list) and not any(rs.busy for rs in engine.reservation_stations) and engine.rob[engine.rob_head].busy == False:
             break
        cycles += 1
    
    print(f"Simulation finished in {cycles} cycles")
    print(f"Trace length: {len(engine.trace)}")
    
    # Expected trace length:
    # Iter 1: ADD, BNE
    # Iter 2: ADD, BNE
    # Total = 4
    for i, t in enumerate(engine.trace):
         inst = engine.instruction_list[t.instruction_index]
         print(f"{i}: {inst.type} Iter:{t.iteration} Index:{t.instruction_index}")

if __name__ == "__main__":
    test_repro()
