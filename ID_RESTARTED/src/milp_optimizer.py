"""
Mixed-Integer Linear Programming (MILP) Scheduling & Classroom Optimization Engine
----------------------------------------------------------------------------------
Implements the formal MILP model solving Equations (1) to (5):

  Minimize Z = sum_{t,r} E_r * y_{r,t} + omega * sum_{t,r} (C_r * y_{r,t} - sum_i A_{i,t} * x_{i,r,t})

Subject to:
  1. Room Capacity: sum_i A_{i,t} * x_{i,r,t} <= C_r * y_{r,t}
  2. Single Allocation: sum_r x_{i,r,t} = 1
  3. Curriculum Consistency: x_{i,r,t} + x_{j,r,t} <= 1 + I(S_i = S_j)
  4. Room Activation: x_{i,r,t} <= y_{r,t}
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pulp

@dataclass
class ClassroomSpec:
    room_id: str
    capacity: int
    power_kw: float  # kWh hourly electrical rating (HVAC + Lighting)

@dataclass
class SectionSpec:
    section_id: str
    subject: str
    curriculum: str
    timeslot: str
    registered: int
    attendance: int

class MILPClassroomOptimizer:
    def __init__(self, omega: float = 0.05):
        """
        :param omega: Penalty parameter weighting seat under-utilization vs energy cost.
        """
        self.omega = omega
        self.rooms: Dict[str, ClassroomSpec] = {}
        self.sections: Dict[str, SectionSpec] = {}

    def add_classroom(self, room_id: str, capacity: int, power_kw: float = 3.5):
        self.rooms[room_id] = ClassroomSpec(room_id, capacity, power_kw)

    def add_section(self, section: SectionSpec):
        self.sections[section.section_id] = section

    def solve_optimization(self) -> Dict:
        """
        Formulates and solves the MILP model using PuLP (CBC Solver).
        """
        prob = pulp.LpProblem("Dynamic_Classroom_Lecture_Merging", pulp.LpMinimize)

        # Decision Variables
        # x[i, r, t] = 1 if section i is allocated to room r at timeslot t
        # y[r, t] = 1 if room r is active at timeslot t
        x = {}
        y = {}

        timeslots = list(set(s.timeslot for s in self.sections.values()))
        section_ids = list(self.sections.keys())
        room_ids = list(self.rooms.keys())

        for r in room_ids:
            for t in timeslots:
                y[r, t] = pulp.LpVariable(f"y_{r}_{t}", cat=pulp.LpBinary)
                for i in section_ids:
                    x[i, r, t] = pulp.LpVariable(f"x_{i}_{r}_{t}", cat=pulp.LpBinary)

        # Objective Function Equation (1): Minimize Energy Consumption + Under-utilization Penalty
        objective_terms = []
        for t in timeslots:
            for r in room_ids:
                e_r = self.rooms[r].power_kw
                c_r = self.rooms[r].capacity
                
                # Energy term
                objective_terms.append(e_r * y[r, t])
                
                # Seat waste penalty term: omega * (C_r * y_{r,t} - sum_i A_{i,t} * x_{i,r,t})
                objective_terms.append(self.omega * c_r * y[r, t])
                for i in section_ids:
                    if self.sections[i].timeslot == t:
                        a_it = self.sections[i].attendance
                        objective_terms.append(-self.omega * a_it * x[i, r, t])

        prob += pulp.lpSum(objective_terms), "Total_Energy_and_Wastage_Objective"

        # Constraint 1: Room Capacity Equation (2)
        for t in timeslots:
            for r in room_ids:
                c_r = self.rooms[r].capacity
                attendances = [self.sections[i].attendance * x[i, r, t] for i in section_ids if self.sections[i].timeslot == t]
                prob += pulp.lpSum(attendances) <= c_r * y[r, t], f"Capacity_Constraint_{r}_{t}"

        # Constraint 2: Single Room Allocation per Active Section Equation (3)
        for i in section_ids:
            t = self.sections[i].timeslot
            prob += pulp.lpSum([x[i, r, t] for r in room_ids]) == 1, f"Single_Allocation_Section_{i}"

        # Constraint 3: Curriculum Consistency Equation (4)
        for t in timeslots:
            active_sections = [i for i in section_ids if self.sections[i].timeslot == t]
            for idx1 in range(len(active_sections)):
                for idx2 in range(idx1 + 1, len(active_sections)):
                    i1 = active_sections[idx1]
                    i2 = active_sections[idx2]
                    same_curriculum = 1 if self.sections[i1].curriculum == self.sections[i2].curriculum else 0
                    for r in room_ids:
                        prob += x[i1, r, t] + x[i2, r, t] <= 1 + same_curriculum, f"Curriculum_Match_{i1}_{i2}_{r}_{t}"

        # Constraint 4: Activation Relation Equation (5)
        for i in section_ids:
            t = self.sections[i].timeslot
            for r in room_ids:
                prob += x[i, r, t] <= y[r, t], f"Activation_Link_{i}_{r}_{t}"

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)

        results = {
            "status": pulp.LpStatus[status],
            "objective_value": pulp.value(prob.objective),
            "allocations": [],
            "vacated_rooms": [],
            "active_rooms": []
        }

        if status == pulp.LpStatusOptimal:
            for t in timeslots:
                for r in room_ids:
                    is_active = pulp.value(y[r, t]) > 0.5
                    assigned_secs = [i for i in section_ids if self.sections[i].timeslot == t and pulp.value(x[i, r, t]) > 0.5]
                    if is_active:
                        results["active_rooms"].append((r, t, assigned_secs))
                        results["allocations"].append({
                            "room_id": r,
                            "timeslot": t,
                            "assigned_sections": assigned_secs,
                            "total_attendance": sum(self.sections[i].attendance for i in assigned_secs),
                            "capacity": self.rooms[r].capacity
                        })
                    else:
                        results["vacated_rooms"].append((r, t))

        return results

def run_milp_demo():
    optimizer = MILPClassroomOptimizer(omega=0.05)

    # Add Classrooms
    optimizer.add_classroom("Room_A101", capacity=60, power_kw=3.5)
    optimizer.add_classroom("Room_A102", capacity=60, power_kw=3.5)
    optimizer.add_classroom("Room_B201", capacity=120, power_kw=6.0)

    # Add Lecture Sections (Simulated 10:00-11:00 AM)
    optimizer.add_section(SectionSpec("SEC_A1", "Data Structures", "CS_SEM3", "10:00-11:00", registered=60, attendance=15))
    optimizer.add_section(SectionSpec("SEC_A2", "Data Structures", "CS_SEM3", "10:00-11:00", registered=60, attendance=18))
    optimizer.add_section(SectionSpec("SEC_B1", "Algorithms", "CS_SEM5", "10:00-11:00", registered=60, attendance=50))

    res = optimizer.solve_optimization()

    print("\n" + "="*70)
    print("=== MILP MATHEMATICAL CLASSROOM OPTIMIZATION SOLVER RESULT ===")
    print("="*70)
    print(f"Solver Status: {res['status']}")
    print(f"Objective Score Z: {res['objective_value']:.4f}")
    print("\nOptimized Room Allocations:")
    for alloc in res["allocations"]:
        print(f" - [{alloc['room_id']}] Timeslot: {alloc['timeslot']} | Sections: {alloc['assigned_sections']} | "
              f"Combined Students: {alloc['total_attendance']}/{alloc['capacity']}")

    print("\nVacated Classrooms (Power Shutdown Triggered):")
    for room, t in res["vacated_rooms"]:
        print(f" - [{room}] at Timeslot {t} -> VACATED & OFF")

    print("="*70 + "\n")
    return res

if __name__ == "__main__":
    run_milp_demo()
