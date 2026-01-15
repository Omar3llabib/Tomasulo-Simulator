from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import os

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


class Instruction(BaseModel):
    type: InstructionType
    dest: str
    src1: str
    src2: Optional[str] = None
    address: Optional[int] = None


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
        self.reset()
    
    def reset(self):
        self.current_cycle = 0
        self.instruction_list: List[Instruction] = []
        self.state_history: List[TomasuloState] = []
        self.instruction_counter = 0
        self.execution_counter = 0
        
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
        
        # Register File (F0-F15)
        self.register_file = [
            RegisterFileEntry(name=f"F{i}", value=0.0) for i in range(16)
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
        self.instruction_list = [
            Instruction(**inst) for inst in instructions
        ]
        self.instruction_queue = self.instruction_list.copy()
        self.instruction_counter = 0
        self._save_state()
    
    def _save_state(self):
        """Save current state to history"""
        state = TomasuloState(
            cycle=self.current_cycle,
            instructions=[
                {
                    "type": inst.type.value,
                    "dest": inst.dest,
                    "src1": inst.src1,
                    "src2": inst.src2,
                    "address": inst.address,
                }
                for inst in self.instruction_list
            ],
            reservation_stations=[rs.copy() for rs in self.reservation_stations],
            reorder_buffer=[rob.copy() for rob in self.rob],
            register_file=[rf.copy() for rf in self.register_file],
            memory=self.memory.copy(),
            stats={
                "instructions_issued": self.instruction_counter,
                "instructions_executed": self.execution_counter,
                "cycles": self.current_cycle,
                "rob_head": self.rob_head,
                "rob_tail": self.rob_tail,
            },
        )
        self.state_history.append(state)
    
    def _find_free_rob_entry(self) -> Optional[int]:
        """Find free ROB entry"""
        for i in range(len(self.rob)):
            idx = (self.rob_tail + i) % len(self.rob)
            if not self.rob[idx].busy:
                return idx
        return None
    
    def _find_reservation_station(self, op_type: InstructionType) -> Optional[int]:
        """Find available reservation station for operation"""
        station_map = {
            InstructionType.ADD: ["ADD1", "ADD2", "ADD3"],
            InstructionType.SUB: ["ADD1", "ADD2", "ADD3"],
            InstructionType.MULT: ["MULT1", "MULT2"],
            InstructionType.DIV: ["MULT1", "MULT2"],
            InstructionType.LOAD: ["LOAD1", "LOAD2"],
            InstructionType.STORE: ["STORE1", "STORE2"],
        }
        
        target_stations = station_map.get(op_type, [])
        for rs in self.reservation_stations:
            if rs.name in target_stations and not rs.busy:
                return self.reservation_stations.index(rs)
        return None
    
    def _get_register_value(self, reg_name: str) -> Tuple[Optional[float], Optional[str]]:
        """Get register value and source (ROB entry if waiting)"""
        for rf in self.register_file:
            if rf.name == reg_name:
                if rf.q is None:
                    return rf.value, None
                else:
                    # Check if ROB entry has result
                    rob_idx = int(rf.q)
                    if self.rob[rob_idx].value is not None:
                        return self.rob[rob_idx].value, None
                    return None, rf.q
        return None, None
    
    def step_forward(self) -> TomasuloState:
        """Execute one cycle forward"""
        # 1. Commit stage - commit completed instructions from ROB head
        if self.rob[self.rob_head].busy and self.rob[self.rob_head].state == "Write Result":
            rob_entry = self.rob[self.rob_head]
            if rob_entry.destination:
                # Update register file
                for rf in self.register_file:
                    if rf.name == rob_entry.destination and rf.q == str(self.rob_head):
                        rf.value = rob_entry.value
                        rf.q = None
            self.rob[self.rob_head].busy = False
            self.rob[self.rob_head].state = "Commit"
            self.rob_head = (self.rob_head + 1) % len(self.rob)
        
        # 2. Write Result stage - write results from execution units
        for i, rob_entry in enumerate(self.rob):
            if rob_entry.busy and rob_entry.state == "Execute":
                # Check if execution is complete (simplified - assume 1 cycle for now)
                rob_entry.state = "Write Result"
                # Broadcast result to waiting reservation stations
                for rs in self.reservation_stations:
                    if rs.qj == str(i):
                        rs.qj = None
                        rs.vj = rob_entry.value
                    if rs.qk == str(i):
                        rs.qk = None
                        rs.vk = rob_entry.value
        
        # 3. Execute stage - execute operations in reservation stations
        for rs in self.reservation_stations:
            if rs.busy and rs.op and rs.vj is not None:
                # Check if operands are ready
                if rs.qj is None and (rs.vk is not None or rs.qk is None):
                    if rs.op in ["ADD", "SUB"]:
                        if rs.vk is not None:
                            result = rs.vj + rs.vk if rs.op == "ADD" else rs.vj - rs.vk
                            if rs.dest is not None:
                                self.rob[rs.dest].value = result
                                self.rob[rs.dest].state = "Execute"
                                self.execution_counter += 1
                            rs.busy = False
                    elif rs.op in ["MULT", "DIV"]:
                        if rs.vk is not None:
                            result = rs.vj * rs.vk if rs.op == "MULT" else rs.vj / rs.vk
                            if rs.dest is not None:
                                self.rob[rs.dest].value = result
                                self.rob[rs.dest].state = "Execute"
                                self.execution_counter += 1
                            rs.busy = False
                    elif rs.op == "LOAD":
                        if rs.a is not None:
                            result = self.memory.get(rs.a, 0.0)
                            if rs.dest is not None:
                                self.rob[rs.dest].value = result
                                self.rob[rs.dest].state = "Execute"
                                self.execution_counter += 1
                            rs.busy = False
        
        # 4. Issue stage - issue new instruction if possible
        if self.instruction_counter < len(self.instruction_list):
            inst = self.instruction_list[self.instruction_counter]
            
            # Find free ROB entry
            rob_idx = self._find_free_rob_entry()
            if rob_idx is None:
                pass  # ROB full, cannot issue
            else:
                # Find appropriate reservation station
                rs_idx = self._find_reservation_station(inst.type)
                if rs_idx is not None:
                    rs = self.reservation_stations[rs_idx]
                    
                    # Issue instruction
                    rs.busy = True
                    rs.op = inst.type.value
                    rs.dest = rob_idx
                    
                    # Get source register values
                    if inst.src1:
                        val1, q1 = self._get_register_value(inst.src1)
                        rs.vj = val1
                        rs.qj = q1
                    
                    if inst.src2:
                        val2, q2 = self._get_register_value(inst.src2)
                        rs.vk = val2
                        rs.qk = q2
                    elif inst.type == InstructionType.LOAD:
                        rs.a = inst.address if inst.address is not None else 0
                    
                    # Update ROB
                    self.rob[rob_idx].busy = True
                    self.rob[rob_idx].instruction = f"{inst.type.value} {inst.dest}"
                    self.rob[rob_idx].destination = inst.dest
                    self.rob[rob_idx].state = "Issue"
                    
                    # Update register file
                    for rf in self.register_file:
                        if rf.name == inst.dest:
                            rf.q = str(rob_idx)
                    
                    self.rob_tail = (self.rob_tail + 1) % len(self.rob)
                    self.instruction_counter += 1
        
        self.current_cycle += 1
        self._save_state()
        return self.state_history[-1]
    
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
        """Restore engine state from saved state"""
        self.current_cycle = state.cycle
        self.instruction_list = [
            Instruction(**inst) for inst in state.instructions
        ]
        self.reservation_stations = state.reservation_stations
        self.rob = state.reorder_buffer
        self.register_file = state.register_file
        self.memory = state.memory.copy()
        self.instruction_counter = state.stats["instructions_issued"]
        self.execution_counter = state.stats["instructions_executed"]
        self.rob_head = state.stats["rob_head"]
        self.rob_tail = state.stats["rob_tail"]
    
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


# Mount static files and serve index.html at root (must be last)
if os.path.exists(public_path):
    # Serve index.html at root
    @app.get("/")
    async def serve_index():
        index_path = os.path.join(public_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Tomasulo Simulator API"}
    
    # Mount static files directory for any other assets
    app.mount("/static", StaticFiles(directory=public_path), name="static")
