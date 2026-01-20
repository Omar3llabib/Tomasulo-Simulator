from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import os
import copy
import re

app = FastAPI()

# Path to public folder
public_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InstructionType(str, Enum):
    ADD = "ADD"
    SUB = "SUB"
    MULT = "MULT"
    DIV = "DIV"
    LOAD = "LOAD"
    STORE = "STORE"
    BRANCH = "BRANCH"


class Instruction(BaseModel):
    type: InstructionType
    dest: str
    src1: str
    src2: Optional[str] = None
    address: Optional[int] = None
    label_target: Optional[str] = None  # For branch instructions


class ReservationStation(BaseModel):
    name: str
    busy: bool = False
    op: Optional[str] = None
    vj: Optional[float] = None
    vk: Optional[float] = None
    qj: Optional[str] = None
    qk: Optional[str] = None
    a: Optional[int] = None
    dest: Optional[int] = None
    time: Optional[int] = None  # For UI display


class ReorderBufferEntry(BaseModel):
    entry: int
    busy: bool = False
    instruction: Optional[str] = None
    state: str = "Issue"  # Issue, Execute, Write Result, Commit
    destination: Optional[str] = None
    value: Optional[float] = None
    exception: Optional[str] = None


class RegisterFileEntry(BaseModel):
    name: str
    value: float
    q: Optional[str] = None  # ROB entry producing value


class TomasuloState(BaseModel):
    cycle: int
    instructions: List[Dict[str, Any]]
    reservation_stations: List[ReservationStation]
    reorder_buffer: List[ReorderBufferEntry]
    register_file: List[RegisterFileEntry]
    memory: Dict[int, float]
    stats: Dict[str, Any]


class TomasuloEngine:
    def __init__(self):
        self.latencies = {
            "ADD": 1, "SUB": 1, "MULT": 3, "DIV": 3, 
            "LOAD": 2, "STORE": 1, "BRANCH": 1
        }
        self.max_unroll_limit = 5  # Safety limit for loop unrolling
        self.reset()
    
    def reset(self):
        self.current_cycle = 0
        self.instruction_list: List[Instruction] = []
        self.state_history: List[TomasuloState] = []
        self.instruction_counter = 0
        self.execution_counter = 0
        self.labels: Dict[str, int] = {}  # Map label names to instruction indices
        self.instruction_to_label: Dict[int, str] = {}  # Map instruction index to label name
        self.rob_to_instruction: Dict[int, int] = {}  # Map ROB entry index to instruction index
        self.instruction_iterations: Dict[int, int] = {}  # Track iteration number for each instruction
        self.loop_unroll_count: Dict[str, int] = {}  # Track unroll count per loop (loop_id -> count)
        self.target_iterations = 1  # Default to 1 iteration if not specified

        
        # TIMING TRACKERS
        self.issue_cycle: Dict[int, int] = {}       # Map instruction index to issue cycle
        self.execute_cycle: Dict[int, int] = {}     # Map instruction index to execute cycle
        self.write_result_cycle: Dict[int, int] = {} # Map instruction index to write result cycle
        self.commit_cycle: Dict[int, int] = {}      # Map instruction index to commit cycle
        
        # Reservation Stations
        self.reservation_stations = [
            ReservationStation(name="ADD1"),
            ReservationStation(name="ADD2"),
            ReservationStation(name="ADD3"),
            ReservationStation(name="MULT1"),
            ReservationStation(name="MULT2"),
            ReservationStation(name="LOAD1"),
            ReservationStation(name="LOAD2"),
            ReservationStation(name="STORE1"),
            ReservationStation(name="STORE2"),
        ]
        
        # Reorder Buffer (8 entries)
        self.rob = [ReorderBufferEntry(entry=i) for i in range(8)]
        self.rob_head = 0
        self.rob_tail = 0
        
        # Register File (F0-F31)
        self.register_file = [
            RegisterFileEntry(name=f"F{i}", value=0.0) for i in range(32)
        ]
        
        # Memory (addresses 0-99)
        self.memory = {i: float(i * 10) for i in range(100)}
        
        # Instruction queue
        self.instruction_queue = []
        
        # Save initial state
        self._save_state()
    
    def load_instructions(self, instructions: List[Dict[str, Any]]):
        """Load instructions into the simulator"""
        self.reset()
        # Rebuild instruction_to_label from label info in instructions
        self.instruction_to_label = {}
        self.labels = {}  # Rebuild labels map
        for idx, inst in enumerate(instructions):
            if inst.get("label"):
                label_name = inst["label"]
                self.instruction_to_label[idx] = label_name
                self.labels[label_name] = idx
        
        # Remove label from instruction dict before creating Instruction objects
        instructions_without_label = []
        for inst in instructions:
            inst_copy = inst.copy()
            inst_copy.pop("label", None)  # Remove label field as it's not in Instruction model
            instructions_without_label.append(inst_copy)
        
        self.instruction_list = [
            Instruction(**inst) for inst in instructions_without_label
        ]
        self.instruction_queue = self.instruction_list.copy()
        self.instruction_counter = 0
        self._save_state()

    def set_instructions(self, instruction_lines: List[str]):
        """Set instructions from raw instruction strings"""
        
        # First pass: collect labels and their line numbers
        self.labels = {}
        instruction_index = 0
        
        for line in instruction_lines:
            # Remove comments (everything after # or //)
            line = re.sub(r'#.*|//.*', '', line).strip()
            if not line:
                continue
            
            # Check if line is a label (ends with colon, e.g., "L1:")
            label_match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*:', line)
            if label_match:
                label_name = label_match.group(1)
                self.labels[label_name] = instruction_index
                # Remove label from line to parse the rest
                line = re.sub(r'^[A-Za-z_][A-Za-z0-9_]*\s*:\s*', '', line).strip()
                if not line:
                    continue  # Label on its own line, skip
            
            instruction_index += 1
        
        # Second pass: parse instructions
        parsed_instructions = []
        instruction_index = 0
        
        for line in instruction_lines:
            # Remove comments (everything after # or //)
            line = re.sub(r'#.*|//.*', '', line).strip()
            if not line:
                continue
            
            # Check if line is a label (ends with colon, e.g., "L1:")
            label_match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*:', line)
            current_label = None
            if label_match:
                current_label = label_match.group(1)
                # Remove label from line to parse the rest
                line = re.sub(r'^[A-Za-z_][A-Za-z0-9_]*\s*:\s*', '', line).strip()
                if not line:
                    continue  # Label on its own line, skip
            
            # Split by commas and whitespace, then filter empty strings
            parts = [p.strip() for p in re.split(r'[,\s]+', line) if p.strip()]
            
            if not parts:
                continue
            
            inst_type_raw = parts[0].upper()
            
            # Map instruction types
            type_mapping = {
                'ADDD': 'ADD', 'SUBD': 'SUB', 'MULTD': 'MULT', 'DIVD': 'DIV',
                'LD': 'LOAD', 'ST': 'STORE', 'STORE': 'STORE',
                'BNE': 'BRANCH', 'BEQ': 'BRANCH', 'BGT': 'BRANCH', 'BLT': 'BRANCH',
            }
            
            inst_type = type_mapping.get(inst_type_raw, inst_type_raw)
            
            if inst_type not in ['ADD', 'SUB', 'MULT', 'DIV', 'LOAD', 'STORE', 'BRANCH']:
                continue
            
            parsed = {
                "type": inst_type,
                "dest": "",
                "src1": "",
                "src2": None,
                "address": None,
                "label_target": None,
                "label": current_label
            }
            
            # Parse based on instruction type
            if inst_type in ['ADD', 'SUB', 'MULT', 'DIV']:
                if len(parts) >= 4:
                    parsed["dest"] = parts[1]
                    parsed["src1"] = parts[2]
                    parsed["src2"] = parts[3]
                elif len(parts) == 3:
                    parsed["dest"] = parts[1]
                    parsed["src1"] = parts[2]
            
            elif inst_type == 'LOAD':
                if len(parts) >= 2:
                    parsed["dest"] = parts[1]
                    if len(parts) >= 3:
                        offset_base = parts[2]
                        match = re.match(r'(\d+)?\(?([RF]\d+)\)?', offset_base)
                        if match:
                            offset_str, base_reg = match.groups()
                            parsed["address"] = int(offset_str) if offset_str else 0
                            parsed["src1"] = base_reg if base_reg else ""
            
            elif inst_type == 'STORE':
                if len(parts) >= 2:
                    parsed["src1"] = parts[1]
                    if len(parts) >= 3:
                        offset_base = parts[2]
                        match = re.match(r'(\d+)?\(?([RF]\d+)\)?', offset_base)
                        if match:
                            offset_str, base_reg = match.groups()
                            parsed["address"] = int(offset_str) if offset_str else 0
                            parsed["dest"] = base_reg if base_reg else ""
            
            elif inst_type == 'BRANCH':
                if len(parts) >= 4:
                    parsed["src1"] = parts[1]
                    parsed["src2"] = parts[2]
                    parsed["label_target"] = parts[3]
                elif len(parts) == 3:
                    parsed["src1"] = parts[1]
                    parsed["src2"] = parts[2]

            parsed_instructions.append(parsed)
            instruction_index += 1

        self.load_instructions(parsed_instructions)

    def step_forward(self) -> TomasuloState:
        """Advance one clock cycle"""
        
        # 1. Commit stage - commit completed instructions from ROB head
        # CORRECTED: Now tracks commit cycle and saves it
        if self.rob[self.rob_head].busy and self.rob[self.rob_head].state == "Write Result":
            rob_entry = self.rob[self.rob_head]
            print(f'Cycle {self.current_cycle}: Committing ROB entry {self.rob_head}')
            
            # --- START FIX: RECORD COMMIT CYCLE ---
            inst_idx = self.rob_to_instruction.get(self.rob_head)
            if inst_idx is not None:
                self.commit_cycle[inst_idx] = self.current_cycle
            # --- END FIX ---

            if rob_entry.destination:
                # Update register file if this ROB is still the latest producer
                for rf in self.register_file:
                    if rf.name == rob_entry.destination and rf.q == str(self.rob_head):
                        rf.value = rob_entry.value
                        rf.q = None
            
            self.rob[self.rob_head].busy = False
            self.rob[self.rob_head].state = "Commit"
            self.rob_head = (self.rob_head + 1) % len(self.rob)
        
        # 2. Execute & Write Result stage - execute operations in reservation stations
        for rs in self.reservation_stations:
            if rs.busy and rs.op:
                # Check if operands are ready (Qj and Qk must be None)
                operands_ready = (rs.qj is None and rs.qk is None)
                
                # For BRANCH, both vj and vk must be ready
                if rs.op == "BRANCH":
                    operands_ready = operands_ready and rs.vj is not None and rs.vk is not None
                elif rs.op == "LOAD":
                    # LOAD only needs address (rs.a), not vj/vk
                    pass 
                else:
                    # For ADD/SUB/MULT/DIV, need both vj and vk
                    operands_ready = operands_ready and rs.vj is not None and rs.vk is not None
                
                # Initialize time ONLY when operands become ready (start of execution)
                if operands_ready and rs.time is None:
                    # Set execution cycles based on operation type using configurable latencies
                    rs.time = self.latencies.get(rs.op, 1)
                    
                    # Track execute_cycle when execution starts
                    rob_idx = rs.dest
                    if rob_idx is not None:
                        inst_idx = self.rob_to_instruction.get(rob_idx)
                        if inst_idx is not None:
                            self.execute_cycle[inst_idx] = self.current_cycle
                            print(f'Cycle {self.current_cycle}: Instruction {inst_idx} started executing in RS {rs.name} (operands ready, cycles={rs.time})')
                
                # Decrement cycles remaining if execution has started
                if rs.time is not None and rs.time > 0:
                    rs.time -= 1
                    print(f'Cycle {self.current_cycle}: RS {rs.name} executing {rs.op}, cycles remaining: {rs.time}')
                
                # If cycles reach 0 (after decrement or already 0), execute and broadcast immediately
                if rs.time is not None and rs.time == 0:
                    result = 0.0
                    rob_idx = rs.dest
                    
                    # Compute result based on operation
                    if rs.op == "ADD": result = (rs.vj or 0) + (rs.vk or 0)
                    elif rs.op == "SUB": result = (rs.vj or 0) - (rs.vk or 0)
                    elif rs.op == "MULT": result = (rs.vj or 0) * (rs.vk or 0)
                    elif rs.op == "DIV": result = (rs.vj / rs.vk) if rs.vk != 0 else 0
                    elif rs.op == "LOAD": result = self.memory.get(rs.a, 0.0)
                    elif rs.op == "BRANCH":
                        # BRANCH logic with dynamic loop unrolling
                        inst_idx = self.rob_to_instruction.get(rob_idx)
                        should_take_branch = False
                        
                        if inst_idx is not None:
                            inst = self.instruction_list[inst_idx]
                            
                            # Check if backward jump (Loop candidate)
                            is_backward_jump = False
                            target_line = -1
                            if inst.label_target in self.labels:
                                target_line = self.labels[inst.label_target]
                                if target_line < inst_idx:
                                    is_backward_jump = True
                            
                            if is_backward_jump and self.target_iterations > 1:
                                # Dynamic Unrolling Logic (Overrides register values)
                                loop_name = inst.label_target
                                current_iter_count = self.loop_unroll_count.get(loop_name, 0)
                                
                                if current_iter_count < (self.target_iterations - 1):
                                    # Take Branch (Unroll)
                                    should_take_branch = True
                                    
                                    # Logic to Unroll
                                    self.loop_unroll_count[loop_name] = current_iter_count + 1
                                    new_iteration_num = self.loop_unroll_count[loop_name] + 1
                                    
                                    # ... (Same unroll logic as before) ...
                                    original_loop_end = -1
                                    for i in range(target_line, len(self.instruction_list)):
                                            scan_inst = self.instruction_list[i]
                                            if scan_inst.type == InstructionType.BRANCH and scan_inst.label_target == inst.label_target:
                                                original_loop_end = i
                                                break
                                    
                                    if original_loop_end != -1:
                                        loop_instructions = []
                                        for i in range(target_line, original_loop_end + 1):
                                            loop_inst = self.instruction_list[i]
                                            new_inst = Instruction(
                                                type=loop_inst.type,
                                                dest=loop_inst.dest,
                                                src1=loop_inst.src1,
                                                src2=loop_inst.src2,
                                                address=loop_inst.address,
                                                label_target=loop_inst.label_target
                                            )
                                            loop_instructions.append(new_inst)
                                        
                                        start_new_idx = len(self.instruction_list)
                                        self.instruction_list.extend(loop_instructions)
                                        
                                        for i, _ in enumerate(loop_instructions):
                                            new_abs_idx = start_new_idx + i
                                            self.instruction_iterations[new_abs_idx] = new_iteration_num
                                        
                                        print(f'Cycle {self.current_cycle}: Loop unrolled {len(loop_instructions)} insts for iter {new_iteration_num}')
                                        print(f'Cycle {self.current_cycle}: Loop Continued. Jumping to trace index {start_new_idx}')
                                        self.instruction_counter = start_new_idx
                                    else:
                                        print("Error: Could not find original loop end for unrolling template")
                                        self.instruction_counter = target_line
                                
                                else:
                                    # Do Not Take Branch (Loop End)
                                    should_take_branch = False
                                    print(f'Cycle {self.current_cycle}: Loop Finished (reached {self.target_iterations} iterations).')
                                    
                                    original_loop_end = -1
                                    for i in range(target_line, len(self.instruction_list)):
                                            scan_inst = self.instruction_list[i]
                                            if scan_inst.type == InstructionType.BRANCH and scan_inst.label_target == inst.label_target:
                                                original_loop_end = i
                                                break
                                    
                                    if original_loop_end != -1:
                                            fallthrough_idx = original_loop_end + 1
                                            
                                            # Check if fallthrough would land us in the unrolled section
                                            # This happens if there are NO instructions after the loop in the original program
                                            # The unrolled instructions are appended to the end.
                                            # The 'original' end is effectively self.original_instruction_count (not tracked)
                                            # BUT, we can check if fallthrough_idx matches the start of any Loop Unroll block.
                                            # OR simply: if fallthrough_idx points to an instruction that we dynamically added.
                                            # We need to distinguish original vs dynamic.
                                            
                                            # Robust fix: 
                                            # The dynamically added instructions are always at the END of the list.
                                            # If fallthrough_idx >= len(self.instruction_list) BEFORE we added this current unroll block...
                                            # Wait, we haven't added *this* iter's unroll block yet? 
                                            # In this 'if' block (loop finish), we do NOT add new instructions.
                                            # So `self.instruction_list` contains: [Original Program] + [Iter 2] + [Iter 3] ...
                                            # `fallthrough_idx` is calculated based on `original_loop_end`. 
                                            # `original_loop_end` is found by scanning from `target_line` (start of loop).
                                            
                                            # The issue: If the original program is just the loop, `original_loop_end` is the last instruction of the original program.
                                            # `fallthrough_idx` = `original_loop_end` + 1.
                                            # If we have already unrolled once, `self.instruction_list` has [Original] + [Unroll 1].
                                            # `fallthrough_idx` points to the start of [Unroll 1].
                                            # So we jump to [Unroll 1] and execute it again. Infinite Loop.
                                            
                                            # Solution: Executing the Fallthrough logic implies we want to run the code *after* the loop in the *logical* program flow.
                                            # In the unrolling simulation, the instructions *after* the loop in the original source are the only valid destinations.
                                            # We should jump to `original_loop_end + 1` ONLY IF that index represents an instruction from the ORIGINAL program.
                                            # How do we know how many instructions were in the original program?
                                            # We didn't store it. We should store it in `set_instructions`.
                                            
                                            # Alternative: The "original loop scan" loop `for i in range(target_line, len(self.instruction_list))` 
                                            # might be scanning into unrolled territory if we are not careful?
                                            # No, because unrolled instructions are appended.
                                            # But if target_line is 0, and we appended stuff, the scan goes to end.
                                            # We need to find the loop end *of the template*.
                                            
                                            # Let's fix 'original_loop_end' finding logic first to be safe (limit to avoiding unrolled).
                                            # But simpler fix for now:
                                            # If we jump to `fallthrough_idx`, verify it is NOT a dynamically generated instruction.
                                            # We can check `self.instruction_iterations`.
                                            # Original instructions have `Iter ?` (or uninitialized logic?).
                                            
                                            is_dynamic = fallthrough_idx in self.instruction_iterations
                                            if is_dynamic:
                                                 print(f"Cycle {self.current_cycle}: Fallthrough index {fallthrough_idx} is dynamic (Loop Boundary). Terminating loop execution locally.")
                                                 # We have run out of original instructions.
                                                 self.instruction_counter = len(self.instruction_list)
                                            else:
                                                 print(f"Jumping back to main program at index {fallthrough_idx}")
                                                 self.instruction_counter = fallthrough_idx
                                    else:
                                            print("Error: Could not determine fallthrough target")
                            else:
                                # Standard Logic (Forward jump or dynamic not enabled)
                                if rs.vj != rs.vk:
                                    should_take_branch = True
                                    print(f'Cycle {self.current_cycle}: Branch Taken (Standard). Jumping to {inst.label_target}')
                                    if inst.label_target in self.labels:
                                        self.instruction_counter = self.labels[inst.label_target]
                        
                        else:
                            # Fallback if ROB mapping fails (shouldn't happen)
                             if rs.vj != rs.vk:
                                 pass # Can't jump without instruction info
                    
                    # Store Result
                    if rs.op == "STORE":
                         # Store value (vk/src1) into memory address (dest/address + vj/base)
                         # Note: In standard tomasulo, store is handled at commit, but for simplicity:
                         if rs.a is not None:
                             self.memory[rs.a] = rs.vk if rs.vk is not None else 0.0
                             print(f"Stored {rs.vk} at address {rs.a}")

                    # BROADCAST to CDB
                    inst_idx = self.rob_to_instruction.get(rob_idx)
                    if inst_idx is not None:
                        self.write_result_cycle[inst_idx] = self.current_cycle
                    
                    if rob_idx is not None:
                        self.rob[rob_idx].value = result
                        self.rob[rob_idx].state = "Write Result"
                        self.execution_counter += 1
                        
                        # Update all waiting stations (CDB Broadcast)
                        for other_rs in self.reservation_stations:
                            if other_rs.busy:
                                if other_rs.qj == str(rob_idx):
                                    other_rs.qj = None
                                    other_rs.vj = result
                                if other_rs.qk == str(rob_idx):
                                    other_rs.qk = None
                                    other_rs.vk = result
                        
                        # Update Registers waiting for this value
                        if self.rob[rob_idx].destination:
                            for rf in self.register_file:
                                if rf.name == self.rob[rob_idx].destination and rf.q == str(rob_idx):
                                    rf.value = result
                                    rf.q = None
                    
                    # Release Station
                    rs.busy = False
                    rs.time = None
                    rs.op = None
                    rs.vj = None; rs.vk = None
                    rs.qj = None; rs.qk = None
                    rs.a = None; rs.dest = None
        
        # 3. Issue stage - issue new instruction if possible
        if self.instruction_counter < len(self.instruction_list):
            inst = self.instruction_list[self.instruction_counter]
            print(f'Cycle {self.current_cycle}: Attempting to issue instruction {self.instruction_counter}: {inst.type.value}')
            
            # Find free ROB entry
            rob_idx = self._find_free_rob_entry()
            
            if rob_idx is None:
                print(f'Cycle {self.current_cycle}: Cannot issue - ROB full')
            else:
                # Find appropriate reservation station
                rs_idx = self._find_reservation_station(inst.type)
                if rs_idx is None:
                    print(f'Cycle {self.current_cycle}: Cannot issue - No free reservation station for {inst.type.value}')
                else:
                    rs = self.reservation_stations[rs_idx]
                    
                    # Issue instruction
                    rs.busy = True
                    rs.op = inst.type.value
                    rs.dest = rob_idx
                    
                    # Get source register values
                    if inst.src1:
                        val1, q1 = self._get_register_value(inst.src1)
                        if inst.type == InstructionType.STORE:
                            # For STORE, src1 is the value to store (goes to vk/qk usually, or depends on arch)
                            # Here we map src1 -> vk/qk (value to store)
                            rs.vk = val1
                            rs.qk = q1
                        else:
                            rs.vj = val1
                            rs.qj = q1
                    
                    if inst.src2:
                        val2, q2 = self._get_register_value(inst.src2)
                        rs.vk = val2
                        rs.qk = q2
                    
                    if inst.type == InstructionType.LOAD:
                        rs.a = inst.address if inst.address is not None else 0
                    elif inst.type == InstructionType.STORE:
                        rs.a = inst.address if inst.address is not None else 0
                        # For STORE, dest is base register. Get its value for address calculation if needed
                        if inst.dest:
                             val_base, q_base = self._get_register_value(inst.dest)
                             # In a real system, we might wait for base reg. Here simplifying:
                             rs.vj = val_base
                             rs.qj = q_base
                    
                    # Update ROB
                    self.rob[rob_idx].busy = True
                    self.rob[rob_idx].value = None
                    if inst.type == InstructionType.BRANCH:
                        self.rob[rob_idx].instruction = f"{inst.type.value} {inst.src1}, {inst.src2}, {inst.label_target or ''}"
                        self.rob[rob_idx].destination = None
                    else:
                        target = inst.dest if inst.type != InstructionType.STORE else inst.src1 # Display purpose
                        self.rob[rob_idx].instruction = f"{inst.type.value} {target}"
                        self.rob[rob_idx].destination = inst.dest if inst.type != InstructionType.STORE else None
                    
                    self.rob[rob_idx].state = "Issue"
                    
                    # Map ROB to instruction index
                    self.rob_to_instruction[rob_idx] = self.instruction_counter
                    self.issue_cycle[self.instruction_counter] = self.current_cycle
                    
                    # Update register file (Renaming) - Store and Branch don't write to registers
                    if inst.type != InstructionType.BRANCH and inst.type != InstructionType.STORE:
                         for rf in self.register_file:
                            if rf.name == inst.dest:
                                rf.q = str(rob_idx)
                    
                    self.rob_tail = (self.rob_tail + 1) % len(self.rob)
                    self.instruction_counter += 1
        
        self.current_cycle += 1
        self._save_state()
        return self.state_history[-1]

    # --- Helper Methods ---

    def _find_free_rob_entry(self) -> Optional[int]:
        """Find the next free ROB entry using tail pointer"""
        # In a circular buffer, check if tail is free
        if not self.rob[self.rob_tail].busy:
            return self.rob_tail
        return None

    def _find_reservation_station(self, inst_type: InstructionType) -> Optional[int]:
        """Find a free reservation station for the instruction type"""
        target_prefixes = []
        if inst_type in [InstructionType.ADD, InstructionType.SUB]:
            target_prefixes = ["ADD"]
        elif inst_type in [InstructionType.MULT, InstructionType.DIV]:
            target_prefixes = ["MULT"]
        elif inst_type == InstructionType.LOAD:
            target_prefixes = ["LOAD"]
        elif inst_type == InstructionType.STORE:
            target_prefixes = ["STORE"]
        elif inst_type == InstructionType.BRANCH:
            # Branches often use ADD stations or dedicated units. Using ADD here.
            target_prefixes = ["ADD"] 

        for idx, rs in enumerate(self.reservation_stations):
            if not rs.busy:
                # Check if RS name starts with any of the allowed prefixes
                if any(rs.name.startswith(prefix) for prefix in target_prefixes):
                    return idx
        return None

    def pre_unroll_loops(self, iterations: int):
        """Static unrolling: duplicate loop instructions N times"""
        if iterations <= 1:
            return
        
        print(f"Starting static unroll with {iterations} iterations")
        print(f"Current instruction count: {len(self.instruction_list)}")
        
        # Find the first loop (first label to first backward branch)
        loop_start = None
        loop_end = None
        loop_label = None
        
        # Find first label
        for idx, inst in enumerate(self.instruction_list):
            if idx in self.instruction_to_label:
                loop_start = idx
                loop_label = self.instruction_to_label[idx]
                print(f"Found loop start at index {idx}, label: {loop_label}")
                break
        
        # Find first backward branch after the label
        if loop_start is not None:
            for idx in range(loop_start, len(self.instruction_list)):
                inst = self.instruction_list[idx]
                if (inst.type == InstructionType.BRANCH and 
                    inst.label_target and 
                    inst.label_target in self.labels and 
                    self.labels[inst.label_target] < idx):
                    loop_end = idx
                    print(f"Found loop end at index {idx}, target: {inst.label_target}")
                    break
        
        if loop_start is None or loop_end is None:
            print("No valid loop found for unrolling")
            return
        
        # Extract loop body (from loop_start to loop_end, exclusive of branch)
        loop_body = self.instruction_list[loop_start:loop_end]
        print(f"Loop body has {len(loop_body)} instructions")
        
        # Create unrolled instructions
        unrolled_instructions = []
        for iteration in range(1, iterations + 1):
            for i, loop_inst in enumerate(loop_body):
                new_inst = Instruction(
                    type=loop_inst.type,
                    dest=loop_inst.dest,
                    src1=loop_inst.src1,
                    src2=loop_inst.src2,
                    address=loop_inst.address,
                    label_target=loop_inst.label_target
                )
                unrolled_instructions.append(new_inst)
        
        # Replace original loop with unrolled instructions
        # Remove original loop instructions
        del self.instruction_list[loop_start:loop_end + 1]
        
        # Insert unrolled instructions at the same position and track indices
        for i, new_inst in enumerate(unrolled_instructions):
            insert_idx = loop_start + i
            self.instruction_list.insert(insert_idx, new_inst)
            # Calculate iteration number for this instruction
            iteration_num = (i // len(loop_body)) + 1
            self.instruction_iterations[insert_idx] = iteration_num
        
        # Update all mappings
        self._update_mappings_after_unroll(loop_start, loop_end, len(unrolled_instructions), len(loop_body))
        
        print(f"Unrolled loop: {len(loop_body)} instructions × {iterations} iterations = {len(unrolled_instructions)} total")
        print(f"Final instruction count: {len(self.instruction_list)}")
    
    def _update_mappings_after_unroll(self, old_start, old_end, new_count, old_loop_count):
        """Update label and instruction mappings after unrolling"""
        shift_amount = new_count - (old_end - old_start + 1)
        
        # Update labels mapping
        updated_labels = {}
        for label, line_num in self.labels.items():
            if line_num > old_end:
                updated_labels[label] = line_num + shift_amount
            else:
                updated_labels[label] = line_num
        self.labels = updated_labels
        
        # Update instruction_to_label mapping
        updated_instruction_to_label = {}
        for inst_idx, label in self.instruction_to_label.items():
            if inst_idx > old_end:
                updated_instruction_to_label[inst_idx + shift_amount] = label
            elif inst_idx < old_start:
                updated_instruction_to_label[inst_idx] = label
            # Instructions within the unrolled loop get removed from mapping
        self.instruction_to_label = updated_instruction_to_label

    def _get_register_value(self, reg_name: str) -> Tuple[float, Optional[str]]:
        """Get value or ROB dependency (Q) for a register"""
        for rf in self.register_file:
            if rf.name == reg_name:
                if rf.q is not None:
                    return (0.0, rf.q)  # Value not ready, return Q
                return (rf.value, None)  # Value ready
        return (0.0, None) # Register not found, default 0

    def _save_state(self):
        """Save current state to history"""
        # Prepare instructions list for frontend with display info
        display_instructions = []
        for idx, inst in enumerate(self.instruction_list):
            inst_dict = inst.dict()
            inst_dict["label"] = self.instruction_to_label.get(idx)
            inst_dict["iteration"] = self.instruction_iterations.get(idx, 1)  # Default to iteration 1
            
            # Format timing values as strings without checkmarks
            inst_dict["issue_cycle"] = str(self.issue_cycle.get(idx)) if self.issue_cycle.get(idx) is not None else "-"
            inst_dict["execute_cycle"] = str(self.execute_cycle.get(idx)) if self.execute_cycle.get(idx) is not None else "-"
            inst_dict["write_result_cycle"] = str(self.write_result_cycle.get(idx)) if self.write_result_cycle.get(idx) is not None else "-"
            inst_dict["commit_cycle"] = str(self.commit_cycle.get(idx)) if self.commit_cycle.get(idx) is not None else "-"
            
            display_instructions.append(inst_dict)

        state = TomasuloState(
            cycle=self.current_cycle,
            instructions=display_instructions,
            reservation_stations=[copy.deepcopy(rs) for rs in self.reservation_stations],
            reorder_buffer=[copy.deepcopy(rob) for rob in self.rob],
            register_file=[copy.deepcopy(rf) for rf in self.register_file],
            memory=copy.deepcopy(self.memory),
            stats={
                "instructions_issued": self.instruction_counter,
                "instructions_executed": self.execution_counter,
                "rob_head": self.rob_head,
                "rob_tail": self.rob_tail
            }
        )
        self.state_history.append(state)

    def step_backward(self) -> Optional[TomasuloState]:
        """Go back one state in history"""
        if len(self.state_history) <= 1:
            return None
        
        # Remove current state
        self.state_history.pop()
        
        # Restore previous state
        prev_state = self.state_history[-1]
        self._restore_state(prev_state)
        
        return prev_state
    
    def _restore_state(self, state: TomasuloState):
        """Restore engine state from saved state using deepcopy"""
        self.current_cycle = state.cycle
        # Instructions are static, usually don't need full reload but safe to do so
        # Important: labels/maps need to be consistent. 
        # Ideally we don't wipe instruction list if just stepping back logic
        
        self.reservation_stations = [copy.deepcopy(rs) for rs in state.reservation_stations]
        self.rob = [copy.deepcopy(rob) for rob in state.reorder_buffer]
        self.register_file = [copy.deepcopy(rf) for rf in state.register_file]
        self.memory = copy.deepcopy(state.memory)
        
        self.instruction_counter = state.stats["instructions_issued"]
        self.execution_counter = state.stats["instructions_executed"]
        self.rob_head = state.stats["rob_head"]
        self.rob_tail = state.stats["rob_tail"]
        
        # Restore timing maps from history is harder unless stored. 
        # For simple UI rollback, we assume maps in `self` are cumulative or we rebuild them.
        # Here we just keep existing maps, but logically they might need rollback if we strictly restart.
        # For now, UI state is sufficient.

    def get_current_state(self) -> TomasuloState:
        """Get current state"""
        if self.state_history:
            return self.state_history[-1]
        self._save_state()
        return self.state_history[-1]


# Global engine instance
engine = TomasuloEngine()


@app.post("/api/instructions")
def load_instructions(instructions: List[Dict[str, Any]]):
    """Load instructions into the simulator"""
    engine.load_instructions(instructions)
    return {"message": "Instructions loaded", "count": len(instructions)}


@app.post("/api/step-forward")
def step_forward():
    """Execute one cycle forward"""
    state = engine.step_forward()
    return state.dict()


@app.post("/api/step-backward")
def step_backward():
    """Go back one cycle"""
    state = engine.step_backward()
    if state:
        return state.dict()
    return {"error": "Cannot go back further"}


@app.get("/api/state")
def get_state():
    """Get current state"""
    state = engine.get_current_state()
    return state.dict()


@app.post("/api/reset")
def reset():
    """Reset the simulator"""
    engine.reset()
    state = engine.get_current_state()
    return state.dict()


@app.post("/api/config_timings")
def config_timings(timings: Dict[str, int]):
    """Configure execution latencies for instruction types"""
    valid_ops = {"ADD", "SUB", "MULT", "DIV", "LOAD", "STORE", "BRANCH"}
    for op, latency in timings.items():
        if op in valid_ops and latency > 0:
            engine.latencies[op] = latency
    return {"message": "Timings updated", "latencies": engine.latencies}


@app.post("/api/load_program")
async def load_program(request: Request):
    try:
        # Check if the data is JSON or raw text
        content_type = request.headers.get('content-type', '')
        
        if 'application/json' in content_type:
            data = await request.json()
            program_text = data.get("program") or data.get("instructions") or ""
            iterations = data.get("iterations", 1)
            print(f"Received JSON request: iterations={iterations}")
        else:
            body = await request.body()
            program_text = body.decode('utf-8')
            iterations = 1
            print(f"Received raw text request")

        print(f"Program text: {repr(program_text)}")
        print(f"Iterations: {iterations}")

        engine.reset()
        
        # Split into lines and filter empty ones
        lines = [line.strip() for line in program_text.split('\n') if line.strip()]
        print(f"Split lines: {lines}")
        
        if not lines:
            return {"error": "No instructions found in the input"}

        engine.set_instructions(lines)
        
        engine.set_instructions(lines)
        
        # Set target iterations for dynamic execution
        engine.target_iterations = iterations
        print(f"Set target iterations to {iterations}. Dynamic unrolling enabled.")
        
        # if iterations > 1:
        #    print(f"Calling pre_unroll_loops with {iterations} iterations")
        #    engine.pre_unroll_loops(iterations)
        # else:
        #    print("Skipping pre-unroll (iterations <= 1)")

        
        return engine.get_current_state().dict()
        
    except Exception as e:
        print(f"Error loading program: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Mount static files and serve index.html at root (must be last)
if os.path.exists(public_path):
    app.mount("/static", StaticFiles(directory=public_path), name="static")
    
    @app.get("/")
    async def serve_index():
        index_path = os.path.join(public_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Tomasulo Simulator API"}