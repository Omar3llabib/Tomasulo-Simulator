from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Tuple, Union

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
    INTEGER = "INTEGER"  # For addi, sltu operations


class Instruction(BaseModel):
    type: InstructionType
    dest: str
    src1: str
    src2: Optional[Union[str, int]] = None
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
    cdb_available_cycle: Optional[int] = None  # Track when CDB value becomes available


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


class ExecutionUnitEntry(BaseModel):
    fu_type: str
    op: str
    remaining: int
    rob_idx: int
    trace_idx: int
    vj: Optional[float] = None
    vk: Optional[float] = None
    a: Optional[int] = None


class RegisterFileEntry(BaseModel):
    name: str
    value: float
    q: Optional[str] = None  # ROB entry producing value


class TomasuloState(BaseModel):
    cycle: int
    instructions: List[Dict[str, Any]]
    reservation_stations: List[ReservationStation]
    execution_units: List[ExecutionUnitEntry]
    reorder_buffer: List[ReorderBufferEntry]
    register_file: List[RegisterFileEntry]
    memory: Dict[int, float]
    stats: Dict[str, Any]


class TomasuloEngine:
    def __init__(self):
        self.latencies = {
            "ADD": 10, "SUB": 10, "MULT": 15, "DIV": 40, 
            "LOAD": 1, "STORE": 1, "BRANCH": 1, "INTEGER": 1
        }
        
        self.max_unroll_limit = 5  # Safety limit for loop unrolling
        
        # Hardware configuration
        self.issue_width = 1
        self.cdb_write_limit = 1
        self.fu_pipelining = False  # Non-pipelined for Exercise 3.15
        
        self.reset()
    
    def set_latencies(self, load_store: int = 2, add_sub: int = 2, mult: int = 10, div: int = 40):
        """Set functional unit latencies"""
        self.latencies.update({
            "LOAD": load_store,
            "STORE": load_store,
            "ADD": add_sub,
            "SUB": add_sub,
            "MULT": mult,
            "DIV": div
        })
        print(f"Updated latencies: LOAD/STORE={load_store}, ADD/SUB={add_sub}, MULT={mult}, DIV={div}")
    
    def configure_hardware(self, issue_width: int = 1, cdb_write_limit: int = 1, fu_pipelining: bool = True):
        """Configure hardware settings"""
        self.issue_width = issue_width
        self.cdb_write_limit = cdb_write_limit
        self.fu_pipelining = fu_pipelining
        print(f"Hardware config: issue_width={issue_width}, cdb_write_limit={cdb_write_limit}, fu_pipelining={fu_pipelining}")
    
    def configure_reservation_stations(self, int_count: int = 3, load_count: int = 2, 
                                     store_count: int = 2, fp_add_count: int = 2, fp_mult_count: int = 2):
        """Configure reservation station counts and rebuild RS array"""
        new_reservation_stations = []
        
        # Create integer stations
        for i in range(int_count):
            new_reservation_stations.append(ReservationStation(name=f"INT{i+1}"))
        
        # Create load stations
        for i in range(load_count):
            new_reservation_stations.append(ReservationStation(name=f"LOAD{i+1}"))
        
        # Create store stations
        for i in range(store_count):
            new_reservation_stations.append(ReservationStation(name=f"STORE{i+1}"))
        
        # Create FP Add stations
        for i in range(fp_add_count):
            new_reservation_stations.append(ReservationStation(name=f"FP_ADD{i+1}"))
        
        # Create FP Mult stations
        for i in range(fp_mult_count):
            new_reservation_stations.append(ReservationStation(name=f"FP_MULT{i+1}"))
        
        self.reservation_stations = new_reservation_stations
        print(f"RS config: INT={int_count}, LOAD={load_count}, STORE={store_count}, FP_ADD={fp_add_count}, FP_MULT={fp_mult_count}")
        print(f"Total reservation stations: {len(self.reservation_stations)}")
    
    def reset(self):
        self.current_cycle = 1  # Start at Cycle 1 per textbook standards
        
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
        self.pre_unrolled = False

        # Loop control timing (dynamic loops)
        self.branch_pending = False
        self.loop_ready_cycle = 1

        # TRACE STORAGE
        self.trace: List[TraceEntry] = []
        
        # Track non-pipelined FU occupancy (fu_type -> busy until cycle)
        self.fu_busy_until: Dict[str, int] = {}
        
        # Execution units (in-flight execution)
        self.execution_units: List[ExecutionUnitEntry] = []
        
        # Reservation Stations
        self.reservation_stations = [
            ReservationStation(name="INT1"),
            ReservationStation(name="INT2"),
            ReservationStation(name="INT3"),
            ReservationStation(name="FP_ADD1"),
            ReservationStation(name="FP_ADD2"),
            ReservationStation(name="FP_MULT1"),
            ReservationStation(name="FP_MULT2"),
            ReservationStation(name="LOAD1"),
            ReservationStation(name="LOAD2"),
            ReservationStation(name="STORE1"),
            ReservationStation(name="STORE2"),
        ]
        
        # Reorder Buffer (40 entries to accommodate 27 unrolled instructions)
        self.rob = [ReorderBufferEntry(entry=i) for i in range(40)]
        self.rob_head = 0
        self.rob_tail = 0
        
        # Register File (F0-F31, X0-X31)
        self.register_file = [
            *(RegisterFileEntry(name=f"F{i}", value=0.0) for i in range(32)),
            *(RegisterFileEntry(name=f"X{i}", value=0.0) for i in range(32))
        ]
        
        # Memory (addresses 0-99)
        self.memory = {i: float(i * 10) for i in range(100)}
        
        # Instruction queue
        self.instruction_queue = []
        
        # Save initial state
        self._save_state()
    
    def load_instructions(self, instructions: List[Dict[str, Any]], reset_state: bool = True):
        """Load instructions into the simulator"""
        if reset_state:
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
                'ADDD': 'ADD', 'FADD.D': 'ADD', 'FADDD': 'ADD',
                'SUBD': 'SUB', 'FSUB.D': 'SUB', 'FSUBD': 'SUB',
                'MULTD': 'MULT', 'FMUL.D': 'MULT', 'FMULD': 'MULT',
                'DIVD': 'DIV', 'FDIV.D': 'DIV', 'FDIVD': 'DIV',
                'LD': 'LOAD', 'FLD': 'LOAD', 'FLD.D': 'LOAD',
                'ST': 'STORE', 'SD': 'STORE', 'FSD': 'STORE', 'FSD.D': 'STORE',
                'STORE': 'STORE',
                'BNE': 'BRANCH', 'BEQ': 'BRANCH', 'BGT': 'BRANCH', 'BLT': 'BRANCH', 'BNEZ': 'BRANCH',
                'ADDI': 'INTEGER', 'SLTU': 'INTEGER',  # Map integer instructions to INTEGER type
            }
            inst_type = type_mapping.get(inst_type_raw, inst_type_raw)
            
            if inst_type not in ['ADD', 'SUB', 'MULT', 'DIV', 'LOAD', 'STORE', 'BRANCH', 'INTEGER']:
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
                    parsed["dest"] = parts[1].upper()
                    parsed["src1"] = parts[2].upper()
                    parsed["src2"] = parts[3].upper()
                elif len(parts) == 3:
                    parsed["dest"] = parts[1].upper()
                    parsed["src1"] = parts[2].upper()
            
            elif inst_type == 'INTEGER':
                # Handle addi/sltu integer operations
                if len(parts) >= 3:
                    parsed["dest"] = parts[1].upper()
                    parsed["src1"] = parts[2].upper()
                    if len(parts) >= 4:
                        src2_token = parts[3].lstrip('#')
                        if re.fullmatch(r"-?\d+", src2_token):
                            parsed["src2"] = int(src2_token)
                        else:
                            parsed["src2"] = src2_token.upper()
                    elif inst_type_raw == 'ADDI':
                        # Default missing immediate to 0 for timing-only runs
                        parsed["src2"] = 0
                    parsed["label_target"] = inst_type_raw  # Track integer op type
            
            elif inst_type == 'LOAD':
                if len(parts) >= 2:
                    parsed["dest"] = parts[1].upper()
                    if len(parts) >= 3:
                        offset_base = parts[2]
                        match = re.match(r'(\d+)?\(?([RFXx]\d+)\)?', offset_base)
                        if match:
                            offset_str, base_reg = match.groups()
                            parsed["address"] = int(offset_str) if offset_str else 0
                            parsed["src1"] = base_reg.upper() if base_reg else ""
            
            elif inst_type == 'STORE':
                if len(parts) >= 2:
                    parsed["src1"] = parts[1].upper()
                    if len(parts) >= 3:
                        offset_base = parts[2]
                        match = re.match(r'(\d+)?\(?([RFXx]\d+)\)?', offset_base)
                        if match:
                            offset_str, base_reg = match.groups()
                            parsed["address"] = int(offset_str) if offset_str else 0
                            parsed["dest"] = base_reg.upper() if base_reg else ""
            
            elif inst_type == 'BRANCH':
                if len(parts) >= 4:
                    parsed["src1"] = parts[1].upper()
                    parsed["src2"] = parts[2].upper()
                    parsed["label_target"] = parts[3]
                elif len(parts) == 3:
                    # Handle BNEZ: BNEZ reg, label (src1=reg, src2=None, label_target=label)
                    if inst_type_raw == 'BNEZ':
                        parsed["src1"] = parts[1].upper()
                        parsed["src2"] = None
                        parsed["label_target"] = parts[2]
                    else:
                        parsed["src1"] = parts[1].upper()
                        parsed["src2"] = parts[2].upper()
                elif len(parts) == 2:
                    # Handle BNEZ with exactly 2 parts: BNEZ reg label
                    parsed["src1"] = parts[1].upper()
                    parsed["src2"] = None
                    parsed["label_target"] = parts[2] if len(parts) > 2 else None

            parsed_instructions.append(parsed)
            instruction_index += 1

        self.load_instructions(parsed_instructions, reset_state=False)

    def pre_unroll_loops(self, iterations: int):
        """Statically unroll loops for the specified number of iterations"""
        if iterations <= 1:
            return
        
        print(f"Pre-unrolling loops for {iterations} iterations")
        
        # Find all loops (branches that jump backward)
        loops = []
        for i, inst in enumerate(self.instruction_list):
            if inst.type == InstructionType.BRANCH and inst.label_target:
                if inst.label_target in self.labels:
                    target_idx = self.labels[inst.label_target]
                    # Check if this is a backward branch (loop)
                    if target_idx <= i:
                        loops.append({
                            'start_idx': target_idx,
                            'end_idx': i,
                            'label': inst.label_target,
                            'branch_inst': inst
                        })
        
        if not loops:
            print("No loops found to unroll - loading single iteration")
            return  # Continue with single iteration, don't do anything
        
        print(f"Found {len(loops)} loop(s) to unroll")
        
        # Sort loops by start index (outermost first)
        loops.sort(key=lambda x: x['start_idx'])
        
        # Create new instruction list with unrolled loops
        new_instructions = []
        new_labels = {}
        new_instruction_iterations: Dict[int, int] = {}
        
        i = 0
        while i < len(self.instruction_list):
            # Check if this instruction is the start of a loop
            loop_start = None
            for loop in loops:
                if i == loop['start_idx']:
                    loop_start = loop
                    break
            
            if loop_start:
                # This is a loop start - unroll it
                print(f"Unrolling loop '{loop_start['label']}' from index {loop_start['start_idx']} to {loop_start['end_idx']}")
                
                # Create unique label for this loop instance
                base_label = loop_start['label']
                
                # Add the loop body for each iteration
                for iter_num in range(iterations):
                    # Add labels for this iteration
                    iter_label = f"{base_label}_iter{iter_num}"
                    
                    # Process each instruction in the loop body
                    for loop_idx in range(loop_start['start_idx'], loop_start['end_idx'] + 1):
                        loop_inst = self.instruction_list[loop_idx]
                        
                        # Create a copy of the instruction
                        new_inst = Instruction(
                            type=loop_inst.type,
                            dest=loop_inst.dest,
                            src1=loop_inst.src1,
                            src2=loop_inst.src2,
                            address=loop_inst.address,
                            label_target=loop_inst.label_target
                        )
                        
                        # Update label targets for this iteration
                        if new_inst.label_target:
                            # If the branch targets the loop start, update to this iteration
                            if new_inst.label_target == base_label:
                                if iter_num < iterations - 1:  # Not the last iteration
                                    new_inst.label_target = iter_label
                                else:
                                    # Last iteration - remove the branch or make it fall through
                                    new_inst.label_target = None
                        
                        # Add instruction with iteration-specific label if this is the loop start
                        if loop_idx == loop_start['start_idx']:
                            # Add the instruction with the iteration label
                            new_instructions.append(new_inst)
                            new_instruction_iterations[len(new_instructions) - 1] = iter_num + 1
                            new_labels[iter_label] = len(new_instructions) - 1
                        else:
                            new_instructions.append(new_inst)
                            new_instruction_iterations[len(new_instructions) - 1] = iter_num + 1
                
                # Skip the original loop instructions
                i = loop_start['end_idx'] + 1
            else:
                # Regular instruction - just copy it
                inst = self.instruction_list[i]
                new_inst = Instruction(
                    type=inst.type,
                    dest=inst.dest,
                    src1=inst.src1,
                    src2=inst.src2,
                    address=inst.address,
                    label_target=inst.label_target
                )
                
                # Update any label targets that were affected by unrolling
                if new_inst.label_target and new_inst.label_target in self.labels:
                    original_target = self.labels[new_inst.label_target]
                    # Check if this target was affected by loop unrolling
                    for loop in loops:
                        if original_target >= loop['start_idx'] and original_target <= loop['end_idx']:
                            # This target was inside an unrolled loop - adjust it
                            new_inst.label_target = f"{new_inst.label_target}_iter{iterations-1}"
                            break
                
                new_instructions.append(new_inst)
                new_instruction_iterations[len(new_instructions) - 1] = 1
                i += 1
        
        # Update the engine state with the unrolled instructions
        self.instruction_list = new_instructions
        self.labels = new_labels
        self.instruction_iterations = new_instruction_iterations
        self.pre_unrolled = True
        
        # Rebuild instruction_to_label mapping
        self.instruction_to_label = {}
        for label, idx in new_labels.items():
            self.instruction_to_label[idx] = label
        
        print(f"Unrolled program has {len(new_instructions)} instructions")
        print(f"New labels: {list(new_labels.keys())}")
        
        # Save the unrolled state
        self._save_state()

    def _save_state(self):
        """Save current state to history"""
        display_instructions = []

        for idx, inst in enumerate(self.instruction_list):
            inst_dict = inst.dict()
            inst_dict["label"] = self.instruction_to_label.get(idx)

            trace_entry = None
            for te in self.trace:
                if te.instruction_index == idx:
                    trace_entry = te
                    break

            if trace_entry:
                inst_dict["iteration"] = trace_entry.iteration
                inst_dict["instruction"] = inst.type.value
                inst_dict["issue_at"] = trace_entry.issue_cycle
                inst_dict["exec_start"] = trace_entry.execute_cycle
                inst_dict["write_cdb"] = trace_entry.write_result_cycle
            else:
                inst_dict["iteration"] = 0
                inst_dict["instruction"] = inst.type.value
                inst_dict["issue_at"] = None
                inst_dict["exec_start"] = None
                inst_dict["write_cdb"] = None

            display_instructions.append(inst_dict)

        state = TomasuloState(
            cycle=self.current_cycle,
            instructions=display_instructions,
            reservation_stations=[copy.deepcopy(rs) for rs in self.reservation_stations],
            execution_units=copy.deepcopy(self.execution_units),
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

    def step_forward(self) -> TomasuloState:
        """Advance one clock cycle - Process stages in order: Commit -> Write Result -> Execute -> Issue"""

        write_count = 0
        issued_this_cycle = 0

        if self.rob[self.rob_head].busy and self.rob[self.rob_head].state == "Write Result":
            rob_entry = self.rob[self.rob_head]
            print(f'Cycle {self.current_cycle}: Committing ROB entry {self.rob_head}')

            trace_idx = self.rob_to_instruction.get(self.rob_head)
            if trace_idx is not None and trace_idx < len(self.trace):
                self.trace[trace_idx].commit_cycle = self.current_cycle

            if rob_entry.destination:
                for rf in self.register_file:
                    if rf.name == rob_entry.destination and rf.q == str(self.rob_head):
                        rf.value = rob_entry.value
                        rf.q = None

            self.rob[self.rob_head].busy = False
            self.rob[self.rob_head].state = "Commit"
            self.rob_head = (self.rob_head + 1) % len(self.rob)

        ready_to_write = []
        for unit in self.execution_units:
            if unit.remaining == 0:
                ready_to_write.append(unit)

        ready_to_write.sort(key=lambda unit: unit.trace_idx)
        for unit in ready_to_write:
            uses_cdb = unit.op not in {"STORE", "BRANCH"}
            if uses_cdb and write_count >= self.cdb_write_limit:
                continue

            result = 0.0
            rob_idx = unit.rob_idx
            trace_idx = unit.trace_idx

            if unit.op == "ADD":
                result = (unit.vj or 0) + (unit.vk or 0)
            elif unit.op == "SUB":
                result = (unit.vj or 0) - (unit.vk or 0)
            elif unit.op == "MULT":
                result = (unit.vj or 0) * (unit.vk or 0)
            elif unit.op == "DIV":
                result = (unit.vj / unit.vk) if unit.vk != 0 else 0
            elif unit.op == "ADDI":
                result = (unit.vj or 0) + (unit.vk or 0)
            elif unit.op == "SLTU":
                result = 1.0 if (unit.vj or 0) < (unit.vk or 0) else 0.0
            elif unit.op == "LOAD":
                result = self.memory.get(unit.a, 0.0)
            elif unit.op == "BRANCH":
                result = 0.0
                if self.branch_pending:
                    self.branch_pending = False
                    self.loop_ready_cycle = self.current_cycle

            if unit.op == "STORE":
                if unit.a is not None:
                    self.memory[unit.a] = unit.vk if unit.vk is not None else 0.0
                    print(f"Stored {unit.vk} at address {unit.a}")

            if uses_cdb and trace_idx is not None and trace_idx < len(self.trace):
                self.trace[trace_idx].write_result_cycle = self.current_cycle

            self.rob[rob_idx].value = result
            self.rob[rob_idx].state = "Write Result"
            self.execution_counter += 1

            for other_rs in self.reservation_stations:
                if other_rs.busy:
                    if other_rs.qj == str(rob_idx):
                        other_rs.qj = None
                        other_rs.vj = result
                        next_cycle = self.current_cycle + 1
                        if other_rs.cdb_available_cycle is None or next_cycle > other_rs.cdb_available_cycle:
                            other_rs.cdb_available_cycle = next_cycle
                    if other_rs.qk == str(rob_idx):
                        other_rs.qk = None
                        other_rs.vk = result
                        next_cycle = self.current_cycle + 1
                        if other_rs.cdb_available_cycle is None or next_cycle > other_rs.cdb_available_cycle:
                            other_rs.cdb_available_cycle = next_cycle

            if self.rob[rob_idx].destination:
                for rf in self.register_file:
                    if rf.name == self.rob[rob_idx].destination and rf.q == str(rob_idx):
                        rf.value = self.rob[rob_idx].value
                        rf.q = None

            self.execution_units = [u for u in self.execution_units if u != unit]
            if uses_cdb:
                write_count += 1

        fu_in_use = set()
        for unit in self.execution_units:
            if unit.remaining > 0:
                unit.remaining -= 1

        for rs in self.reservation_stations:
            if rs.busy and rs.op:
                operands_ready = (rs.qj is None and rs.qk is None)
                cdb_ready = True
                if rs.cdb_available_cycle is not None and self.current_cycle < rs.cdb_available_cycle:
                    cdb_ready = False

                if rs.op == "BRANCH":
                    if rs.qk is None and rs.vk is None:
                        operands_ready = operands_ready and rs.vj is not None
                    else:
                        operands_ready = operands_ready and rs.vj is not None and rs.vk is not None
                elif rs.op == "LOAD":
                    pass
                else:
                    operands_ready = operands_ready and rs.vj is not None and rs.vk is not None

                fu_type = None
                if rs.op in ["ADD", "SUB"]:
                    fu_type = "ADD_SUB"
                elif rs.op in ["MULT", "DIV"]:
                    fu_type = "MULT_DIV"
                elif rs.op == "LOAD":
                    fu_type = "LOAD"
                elif rs.op == "STORE":
                    fu_type = "STORE"
                elif rs.op == "BRANCH":
                    fu_type = "INT"
                elif rs.op in ["INTEGER", "ADDI", "SLTU"]:
                    fu_type = "INT"

                busy_until = self.fu_busy_until.get(fu_type, 0)
                fu_available = self.fu_pipelining or (fu_type not in fu_in_use and self.current_cycle > busy_until)

                if operands_ready and cdb_ready and rs.time is None and fu_available:
                    trace_idx = self.rob_to_instruction.get(rs.dest)
                    if trace_idx is not None and trace_idx < len(self.trace):
                        if self.trace[trace_idx].iteration >= 2 and self.current_cycle <= self.loop_ready_cycle:
                            continue
                    exec_cycles = self.latencies.get(rs.op, self.latencies.get("INTEGER", 1))
                    remaining_cycles = max(exec_cycles - 1, 0)
                    rob_idx = rs.dest
                    trace_idx = self.rob_to_instruction.get(rob_idx)

                    if rob_idx is not None and trace_idx is not None:
                        self.execution_units.append(
                            ExecutionUnitEntry(
                                fu_type=fu_type,
                                op=rs.op,
                                remaining=remaining_cycles,
                                rob_idx=rob_idx,
                                trace_idx=trace_idx,
                                vj=rs.vj,
                                vk=rs.vk,
                                a=rs.a
                            )
                        )

                    if not self.fu_pipelining:
                        fu_in_use.add(fu_type)
                        self.fu_busy_until[fu_type] = self.current_cycle + exec_cycles - 1

                    if trace_idx is not None and trace_idx < len(self.trace):
                        self.trace[trace_idx].execute_cycle = self.current_cycle
                        print(f'Cycle {self.current_cycle}: Trace {trace_idx} started executing in RS {rs.name} (operands ready, cycles={exec_cycles})')

                    rs.busy = False
                    rs.time = None
                    rs.op = None
                    rs.vj = None
                    rs.vk = None
                    rs.qj = None
                    rs.qk = None
                    rs.a = None
                    rs.dest = None
                    rs.cdb_available_cycle = None

        while (issued_this_cycle < self.issue_width and
               self.instruction_counter < len(self.instruction_list)):
            inst = self.instruction_list[self.instruction_counter]
            print(f'Cycle {self.current_cycle}: Attempting to issue instruction {self.instruction_counter}: {inst.type.value}')

            rob_idx = self._find_free_rob_entry()
            if rob_idx is None:
                print(f'Cycle {self.current_cycle}: Cannot issue - ROB full')
                break

            rs_idx = self._find_reservation_station(inst.type)
            if rs_idx is None:
                print(f'Cycle {self.current_cycle}: Cannot issue - No free reservation station for {inst.type.value}')
                break

            rs = self.reservation_stations[rs_idx]

            if self.instruction_counter not in self.instruction_iterations:
                self.instruction_iterations[self.instruction_counter] = 1

            current_iter = self.instruction_iterations[self.instruction_counter]
            if current_iter >= 2 and (
                self.branch_pending
                or self.current_cycle < self.loop_ready_cycle
            ):
                print(
                    f"Cycle {self.current_cycle}: Delaying issue for iteration {current_iter} until branch resolves"
                )
                break

            trace_idx = len(self.trace)
            new_trace = TraceEntry(
                trace_index=trace_idx,
                instruction_index=self.instruction_counter,
                iteration=current_iter,
                issue_cycle=self.current_cycle
            )
            self.trace.append(new_trace)

            rs.busy = True
            if inst.type == InstructionType.INTEGER and inst.label_target:
                rs.op = inst.label_target
            else:
                rs.op = inst.type.value
            rs.dest = rob_idx

            if inst.src1:
                val1, q1 = self._get_register_value(inst.src1)
                if inst.type == InstructionType.STORE:
                    rs.vk = val1
                    rs.qk = q1
                else:
                    rs.vj = val1
                    rs.qj = q1

            if inst.src2 is not None:
                if inst.type == InstructionType.INTEGER and isinstance(inst.src2, int):
                    rs.vk = float(inst.src2)
                    rs.qk = None
                else:
                    val2, q2 = self._get_register_value(inst.src2)
                    rs.vk = val2
                    rs.qk = q2
            elif inst.type == InstructionType.INTEGER and rs.op == "ADDI":
                rs.vk = 0.0
                rs.qk = None

            if inst.type == InstructionType.LOAD:
                rs.a = inst.address if inst.address is not None else 0
            elif inst.type == InstructionType.STORE:
                rs.a = inst.address if inst.address is not None else 0
                if inst.dest:
                    val_base, q_base = self._get_register_value(inst.dest)
                    rs.vj = val_base
                    rs.qj = q_base

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
            self.rob_to_instruction[rob_idx] = trace_idx

            if inst.type != InstructionType.BRANCH and inst.type != InstructionType.STORE:
                for rf in self.register_file:
                    if rf.name == inst.dest:
                        rf.q = str(rob_idx)

            self.rob_tail = (self.rob_tail + 1) % len(self.rob)
            self.total_issued_count += 1
            issued_this_cycle += 1

            next_pc = self.instruction_counter + 1
            if inst.type == InstructionType.BRANCH and inst.label_target and not self.pre_unrolled:
                loop_label = inst.label_target
                if loop_label not in self.loop_unroll_count:
                    self.loop_unroll_count[loop_label] = 0

                current_loop_iter = self.loop_unroll_count[loop_label]
                if current_loop_iter < (self.target_iterations - 1):
                    if loop_label in self.labels:
                        next_pc = self.labels[loop_label]
                        self.loop_unroll_count[loop_label] += 1
                        self.branch_pending = True
                    else:
                        print(f"Warning: Label {loop_label} not found")
                else:
                    print(f"Loop {loop_label} finished {self.target_iterations} iterations. Falling through.")

            self.instruction_counter = next_pc

            if not self.pre_unrolled and next_pc < len(self.instruction_list):
                current_pc = new_trace.instruction_index
                if next_pc <= current_pc:
                    current_val = self.instruction_iterations.get(next_pc, 0)
                    self.instruction_iterations[next_pc] = current_val + 1
                else:
                    self.instruction_iterations[next_pc] = current_iter

        self.current_cycle += 1
        self._save_state()
        return self.state_history[-1]

    def step_backward(self) -> Optional[TomasuloState]:
        """Go back one state in history"""
        if len(self.state_history) <= 1:
            return None

        self.state_history.pop()
        prev_state = self.state_history[-1]
        self._restore_state(prev_state)
        return prev_state

    def _restore_state(self, state: TomasuloState):
        """Restore engine state from saved state using deepcopy"""
        self.current_cycle = state.cycle
        self.reservation_stations = [copy.deepcopy(rs) for rs in state.reservation_stations]
        self.rob = [copy.deepcopy(rob) for rob in state.reorder_buffer]
        self.register_file = [copy.deepcopy(rf) for rf in state.register_file]
        self.execution_units = [copy.deepcopy(unit) for unit in state.execution_units]
        self.memory = copy.deepcopy(state.memory)

        self.total_issued_count = state.stats["instructions_issued"]
        self.instruction_counter = state.stats.get("pc", 0)
        self.execution_counter = state.stats["instructions_executed"]
        self.rob_head = state.stats["rob_head"]
        self.rob_tail = state.stats["rob_tail"]

    def get_current_state(self) -> TomasuloState:
        """Get current state"""
        if self.state_history:
            return self.state_history[-1]
        self._save_state()
        return self.state_history[-1]

    def _find_free_rob_entry(self) -> Optional[int]:
        """Find the next free ROB entry using tail pointer"""
        if not self.rob[self.rob_tail].busy:
            return self.rob_tail
        return None

    def _find_reservation_station(self, inst_type: InstructionType) -> Optional[int]:
        """Find a free reservation station for the instruction type"""
        target_prefixes = []

        if inst_type == InstructionType.LOAD:
            target_prefixes = ["LOAD"]
        elif inst_type == InstructionType.STORE:
            target_prefixes = ["STORE"]
        elif inst_type in [InstructionType.MULT, InstructionType.DIV]:
            target_prefixes = ["FP_MULT"]
        elif inst_type in [InstructionType.ADD, InstructionType.SUB]:
            target_prefixes = ["FP_ADD"]
        elif inst_type == InstructionType.INTEGER:
            target_prefixes = ["INT"]
        elif inst_type == InstructionType.BRANCH:
            target_prefixes = ["INT"]

        for idx, rs in enumerate(self.reservation_stations):
            if not rs.busy and any(rs.name.startswith(prefix) for prefix in target_prefixes):
                return idx
        return None

    def _get_register_value(self, reg_name: str) -> Tuple[float, Optional[str]]:
        """Get value or ROB dependency (Q) for a register"""
        for rf in self.register_file:
            if rf.name == reg_name:
                if rf.q is not None:
                    return (0.0, rf.q)
                return (rf.value, None)
        return (0.0, None)


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
    try:
        state = engine.step_forward()
        return state.dict()
    except Exception as e:
        print(f"Error in step_forward: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

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
            latencies = data.get("latencies", {})
            hardware_config = data.get("hardware_config", {})
            rs_config = data.get("rs_config", {})
            print(f"Received JSON request: iterations={iterations}, latencies={latencies}")
            print(f"Hardware config: {hardware_config}, RS config: {rs_config}")
        else:
            body = await request.body()
            program_text = body.decode('utf-8')
            iterations = 1
            latencies = {}
            hardware_config = {}
            rs_config = {}
            print(f"Received raw text request")

        print(f"Program text: {repr(program_text)}")
        print(f"Iterations: {iterations}")

        engine.reset()
        
        # Configure hardware settings
        if hardware_config:
            engine.configure_hardware(
                issue_width=hardware_config.get("issue_width", 1),
                cdb_write_limit=hardware_config.get("cdb_write_limit", 1),
                fu_pipelining=hardware_config.get("fu_pipelining", True)
            )
        
        # Configure reservation stations
        if rs_config:
            engine.configure_reservation_stations(
                int_count=rs_config.get("int", 3),
                load_count=rs_config.get("load", 2),
                store_count=rs_config.get("store", 2),
                fp_add_count=rs_config.get("fp_add", 2),
                fp_mult_count=rs_config.get("fp_mult", 2)
            )
        
        # Set custom latencies if provided
        if latencies:
            engine.set_latencies(
                load_store=latencies.get("load_store", 2),
                add_sub=latencies.get("add_sub", 2),
                mult=latencies.get("mult", 10),
                div=latencies.get("div", 40)
            )
        
        # Split into lines and filter empty ones
        lines = [line.strip() for line in program_text.split('\n') if line.strip()]
        print(f"Split lines: {lines}")
        
        if not lines:
            return {"error": "No instructions found in the input"}

        engine.set_instructions(lines)
        
        # Pre-unroll loops if iterations > 1
        if iterations > 1:
            print(f"Calling pre_unroll_loops with {iterations} iterations")
            engine.pre_unroll_loops(iterations)
        else:
            print("Skipping pre-unroll (iterations <= 1)")
        engine.target_iterations = iterations
        
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