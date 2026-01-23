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


class TraceEntry(BaseModel):
    trace_index: int
    instruction_index: int
    iteration: int
    issue_cycle: Optional[int] = None
    execute_cycle: Optional[int] = None
    write_result_cycle: Optional[int] = None
    commit_cycle: Optional[int] = None



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
        self.total_issued_count = 0
        self.execution_counter = 0
        self.labels: Dict[str, int] = {}  # Map label names to instruction indices
        self.instruction_to_label: Dict[int, str] = {}  # Map instruction index to label name
        self.rob_to_instruction: Dict[int, int] = {}  # Map ROB entry index to instruction index
        self.instruction_iterations: Dict[int, int] = {}  # Track iteration number for each instruction
        self.loop_unroll_count: Dict[str, int] = {}  # Track unroll count per loop (loop_id -> count)
        self.target_iterations = 1  # Default to 1 iteration if not specified

        # TRACE STORAGE
        self.trace: List[TraceEntry] = []
        
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
        pending_label = None
        
        for line in instruction_lines:
            # Remove comments (everything after # or //)
            line = re.sub(r'#.*|//.*', '', line).strip()
            if not line:
                continue
            
            # Check if line is a label (ends with colon, e.g., "L1:")
            label_match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*:', line)
            if label_match:
                pending_label = label_match.group(1)
                # Remove label from line to parse the rest
                line = re.sub(r'^[A-Za-z_][A-Za-z0-9_]*\s*:\s*', '', line).strip()
                if not line:
                    continue  # Label on its own line, skip to next line but keep pending_label
            
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
                "label": pending_label
            }
            pending_label = None
            
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
        if self.rob[self.rob_head].busy and self.rob[self.rob_head].state == "Write Result":
            rob_entry = self.rob[self.rob_head]
            print(f'Cycle {self.current_cycle}: Committing ROB entry {self.rob_head}')
            
            # Use trace_idx from rob_to_instruction
            trace_idx = self.rob_to_instruction.get(self.rob_head)
            if trace_idx is not None and trace_idx < len(self.trace):
                self.trace[trace_idx].commit_cycle = self.current_cycle

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
                        trace_idx = self.rob_to_instruction.get(rob_idx)
                        if trace_idx is not None and trace_idx < len(self.trace):
                            self.trace[trace_idx].execute_cycle = self.current_cycle
                            print(f'Cycle {self.current_cycle}: Trace {trace_idx} started executing in RS {rs.name} (operands ready, cycles={rs.time})')
                
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
                        # Branch logic handles control flow during Execute/Writeback if handling misprediction
                        # But for this simple simulator with dynamic Issue, control flow happens at Issue (Prediction).
                        # Here we just check correctness or update history.
                        # Since we simulate "perfect" or "immediate" flow in Issue for now (or static unroll style previously),
                        # we technically don't need complex branch recovery for this specific request unless requested.
                        # The user wants "process will put the instructions in issue Q".
                        pass # Branch execution mainly signals completion here.
                    
                    # Store Result
                    if rs.op == "STORE":
                         # Store value (vk/src1) into memory address (dest/address + vj/base)
                         # Note: In standard tomasulo, store is handled at commit, but for simplicity:
                         if rs.a is not None:
                             self.memory[rs.a] = rs.vk if rs.vk is not None else 0.0
                             print(f"Stored {rs.vk} at address {rs.a}")

                    # BROADCAST to CDB
                    trace_idx = self.rob_to_instruction.get(rob_idx)
                    if trace_idx is not None and trace_idx < len(self.trace):
                        self.trace[trace_idx].write_result_cycle = self.current_cycle
                    
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
        # Check against instruction_list length
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
                    
                    # Create Trace Entry
                    trace_idx = len(self.trace)
                    
                    # Determine iteration
                    # Simple heuristic: if we visited this PC before, increment iteration?
                    # Or usage self.loop_unroll_count logic?
                    # Let's use the loop_unroll_count for BRANCH mapping or just global count?
                    # Better: self.instruction_iterations maps static PC -> current iteration
                    # Initialize if missing
                    if self.instruction_counter not in self.instruction_iterations:
                        self.instruction_iterations[self.instruction_counter] = 1
                    
                    current_iter = self.instruction_iterations[self.instruction_counter]
                    
                    new_trace = TraceEntry(
                        trace_index=trace_idx,
                        instruction_index=self.instruction_counter,
                        iteration=current_iter,
                        issue_cycle=self.current_cycle
                    )
                    self.trace.append(new_trace)
                    
                    # Issue instruction (Use inst data)
                    rs.busy = True
                    rs.op = inst.type.value
                    rs.dest = rob_idx
                    
                    # Get source register values
                    if inst.src1:
                        val1, q1 = self._get_register_value(inst.src1)
                        if inst.type == InstructionType.STORE:
                            # For STORE, src1 is the value to store
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
                        if inst.dest:
                             val_base, q_base = self._get_register_value(inst.dest)
                             rs.vj = val_base
                             rs.qj = q_base
                    
                    # Update ROB
                    self.rob[rob_idx].busy = True
                    self.rob[rob_idx].value = None
                    if inst.type == InstructionType.BRANCH:
                        self.rob[rob_idx].instruction = f"{inst.type.value} {inst.src1}, {inst.src2}, {inst.label_target or ''}"
                        self.rob[rob_idx].destination = None
                    else:
                        target = inst.dest if inst.type != InstructionType.STORE else inst.src1
                        self.rob[rob_idx].instruction = f"{inst.type.value} {target}"
                        self.rob[rob_idx].destination = inst.dest if inst.type != InstructionType.STORE else None
                    
                    self.rob[rob_idx].state = "Issue"
                    
                    # Map ROB -> Trace Index
                    self.rob_to_instruction[rob_idx] = trace_idx
                    
                    # Update register file (Renaming)
                    if inst.type != InstructionType.BRANCH and inst.type != InstructionType.STORE:
                         for rf in self.register_file:
                            if rf.name == inst.dest:
                                rf.q = str(rob_idx)
                    
                    self.rob_tail = (self.rob_tail + 1) % len(self.rob)
                    self.total_issued_count += 1
                    
                    # --- CONTROL FLOW UPDATE (PC) ---
                    # Default: Advance PC
                    next_pc = self.instruction_counter + 1
                    
                    if inst.type == InstructionType.BRANCH and inst.label_target:
                        # Dynamic Loop Handling
                        loop_label = inst.label_target
                        
                        # Initialize loop counter for this label if needed
                        if loop_label not in self.loop_unroll_count:
                             self.loop_unroll_count[loop_label] = 0
                             
                        # Check if we should branch (Loop logic)
                        # The logic: if current iterations < target_iterations -> Take branch
                        # We need to know which loop this is.
                        # Assuming label_target points to start of loop.
                        
                        current_loop_iter = self.loop_unroll_count[loop_label]
                        if current_loop_iter < (self.target_iterations - 1):
                            # Take Branch
                            if loop_label in self.labels:
                                next_pc = self.labels[loop_label]
                                self.loop_unroll_count[loop_label] += 1
                                # Also update iteration count for instructions in the loop?
                                # That happens when we visit them.
                                # But we need to make sure when we jump back, we increment the 'iteration' counter for those instructions?
                                # A simple global map `instruction_iterations` might fail if next_pc jumps back.
                                # Correct way: When jumping back, we expect to see instructions again.
                                # We can just increment iteration count for the target instruction logic?
                                # Actually, `instruction_iterations` map should probably be updated here?
                                # No, let's update `instruction_iterations` as we encounter them. 
                                # But how do we know it's a new iteration for *that* instruction?
                                # If we jump back to L1, the instruction at L1 is visited again.
                                # trace doesn't store this state. 
                                # We can infer iteration from loop_unroll_count + 1?
                                pass
                            else:
                                print(f"Warning: Label {loop_label} not found")
                        else:
                            # Parse through loop end, reset specific loop counter?
                            # Or just fall through.
                            print(f"Loop {loop_label} finished {self.target_iterations} iterations. Falling through.")
                            # Optional: Reset counter if we re-enter loop later from outside? 
                            # For single pass simulator, fine to leave as is.
                        
                    self.instruction_counter = next_pc
                    
                    # Update iteration count for the NEXT instruction
                    if next_pc < len(self.instruction_list):
                        # Capture current PC from trace entry or just calculate it?
                        # We updated self.instruction_counter. The executed instruction was at 'inst.instruction_index' (trace_entry.instruction_index).
                        # But we constructed 'new_trace' earlier using 'self.instruction_counter' (before update).
                        # Let's use 'new_trace.instruction_index' as 'current_pc'.
                        current_pc = new_trace.instruction_index
                        
                        if next_pc <= current_pc: 
                             # Backward jump (or stay same) - New iteration for target
                             current_val = self.instruction_iterations.get(next_pc, 0)
                             self.instruction_iterations[next_pc] = current_val + 1
                        else:
                             # Forward flow - Propagate iteration count
                             self.instruction_iterations[next_pc] = current_iter

        
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
        # Prepare instructions list for frontend with display info from TRACE
        display_instructions = []
        for idx, trace_entry in enumerate(self.trace):
            inst = self.instruction_list[trace_entry.instruction_index]
            inst_dict = inst.dict()
            
            # Label should only show if it matches the static one? 
            # Or should we show labels for repeated loops? 
            # Static labels are mapped to instruction_indices.
            # If trace_entry.instruction_index has a label, show it?
            # Yes, show labels to understand where "L1" is in the trace.
            inst_dict["label"] = self.instruction_to_label.get(trace_entry.instruction_index)
            # Only show label on the first iteration? Or all? User preference.
            # For clarity, let's show it.
            
            inst_dict["iteration"] = trace_entry.iteration
            
            # Format timing values as strings without checkmarks
            inst_dict["issue_cycle"] = str(trace_entry.issue_cycle) if trace_entry.issue_cycle is not None else "-"
            inst_dict["execute_cycle"] = str(trace_entry.execute_cycle) if trace_entry.execute_cycle is not None else "-"
            inst_dict["write_result_cycle"] = str(trace_entry.write_result_cycle) if trace_entry.write_result_cycle is not None else "-"
            inst_dict["commit_cycle"] = str(trace_entry.commit_cycle) if trace_entry.commit_cycle is not None else "-"
            
            display_instructions.append(inst_dict)

        state = TomasuloState(
            cycle=self.current_cycle,
            instructions=display_instructions,
            reservation_stations=[copy.deepcopy(rs) for rs in self.reservation_stations],
            reorder_buffer=[copy.deepcopy(rob) for rob in self.rob],
            register_file=[copy.deepcopy(rf) for rf in self.register_file],
            memory=copy.deepcopy(self.memory),
            stats={
                "instructions_issued": self.total_issued_count,
                "instructions_executed": self.execution_counter,
                "pc": self.instruction_counter,
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
        
        self.memory = copy.deepcopy(state.memory)
        
        self.total_issued_count = state.stats["instructions_issued"]
        self.instruction_counter = state.stats.get("pc", 0) # Restore PC
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