import sys
import os

# Add the parent directory to sys.path so we can import api.index
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.index import TomasuloEngine, InstructionType

def test_dynamic_loop():
    engine = TomasuloEngine()
    
    # Program:
    # ADDI F1, F0, 3
    # L1: SUBI F1, F1, 1
    # BNE F1, F2, L1
    
    lines = [
        "ADD F1, F0, 3",
        "L1: SUB F1, F1, 1",
        "BNE F1, F2, L1"
    ]
    
    print("Loading instructions...")
    engine.load_instructions([{
        "type": "ADD", "dest": "F1", "src1": "F0", "src2": "3"
    }, {
        "type": "SUB", "dest": "F1", "src1": "F1", "src2": "1", "label": "L1"
    }, {
        "type": "BRANCH", "dest": "", "src1": "F1", "src2": "F2", "label_target": "L1"
    }])
    
    # Alternative: use set_instructions to test parsing and init
    engine.reset()
    engine.set_instructions(lines)
    
    print(f"Instructions loaded: {len(engine.instruction_list)}")
    # Expected: 3 instructions static
    
    iterations = 3
    engine.target_iterations = iterations
    print(f"Setting target iterations to {iterations}")
    
    # Run simulation
    cycles = 0
    max_cycles = 50
    while cycles < max_cycles:
        state = engine.step_forward()
        print(f"Cycle {state.cycle}: Issued {state.stats['instructions_issued']}, Executed {state.stats['instructions_executed']}")
        
        # Check if done (no more instructions to issue and ROB empty)
        if engine.instruction_counter >= len(engine.instruction_list) and not engine.rob[engine.rob_head].busy:
             # Wait, instruction_counter might be jumping around.
             # Stop condition: instruction_counter pointed past end AND rob empty?
             # My logic falls through when loop finishes.
             if engine.instruction_counter >= len(engine.instruction_list) and not any(rs.busy for rs in engine.reservation_stations):
                 break
        
        cycles += 1
    
    print(f"Simulation finished in {cycles} cycles")
    
    # Verification
    trace_len = len(engine.trace)
    print(f"Trace length: {trace_len}")
    
    # Expected: 1 (ADDI) + 3 * 2 (SUBI, BNE) = 7
    expected_len = 1 + iterations * 2
    
    if trace_len == expected_len:
        print("SUCCESS: Trace length matches expected loop iterations.")
    else:
        print(f"FAILURE: Expected trace length {expected_len}, got {trace_len}")
    
    # Print Trace
    for i, entry in enumerate(engine.trace):
        inst = engine.instruction_list[entry.instruction_index]
        print(f"{i}: {inst.type} Iter:{entry.iteration}")

if __name__ == "__main__":
    test_dynamic_loop()
