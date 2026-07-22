"""
ASHRAE Thermodynamic & Energy Savings Calculator
-------------------------------------------------
Calculates thermal cooling load (Q_saved), electrical energy saved (Delta E),
and utility cost savings (Delta S) for vacated classrooms according to ASHRAE guidelines.

Formulas:
  Q_saved = Q_ventilation + Q_lighting + Q_occupant
  Q_ventilation = m_dot * C_pa * (T_OA - T_set)
  Q_lighting = P_lighting * eta_thermal
  Q_occupant = N_students * q_sensible
  Delta E = ( (Q_saved / COP) + P_lighting ) * Delta_t
  Delta S = Delta E * Rate_utility
"""

from dataclasses import dataclass

@dataclass
class ThermalParams:
    c_pa: float = 1.005          # Specific heat capacity of air (kJ/kg K)
    t_oa: float = 34.0           # Outdoor ambient temperature (°C)
    t_set: float = 24.0          # Indoor setpoint temperature (°C)
    eta_thermal: float = 0.33     # Lighting thermal dissipation fraction
    q_sensible: float = 0.075    # Average occupant sensible heat dissipation (kW per student)
    cop: float = 3.2             # HVAC Coefficient of Performance
    utility_rate_usd: float = 0.15 # Commercial utility cost per kWh ($/kWh)

def calculate_classroom_energy_savings(
    vacated_rooms_count: int,
    avg_room_lighting_kw: float = 0.8,
    avg_fresh_air_m_dot: float = 0.35, # kg/s per room
    avg_occupants_saved: int = 25,
    duration_hours: float = 1.0,
    params: ThermalParams = ThermalParams()
) -> dict:
    """
    Computes thermodynamic load reduction and financial savings when rooms are vacated.
    """
    # 1. Thermal Load Components per room (kW)
    q_ventilation = avg_fresh_air_m_dot * params.c_pa * (params.t_oa - params.t_set)
    q_lighting = avg_room_lighting_kw * params.eta_thermal
    q_occupant = avg_occupants_saved * params.q_sensible

    q_saved_per_room_kw = q_ventilation + q_lighting + q_occupant
    total_q_saved_kw = q_saved_per_room_kw * vacated_rooms_count

    # 2. Total Saved Electrical Energy (kWh)
    # Delta E = ( (Q_saved / COP) + P_lighting ) * Delta_t
    elec_kw_per_room = (q_saved_per_room_kw / params.cop) + avg_room_lighting_kw
    delta_e_kwh = elec_kw_per_room * vacated_rooms_count * duration_hours

    # 3. Utility Cost Savings ($)
    delta_s_usd = delta_e_kwh * params.utility_rate_usd

    results = {
        "vacated_rooms": vacated_rooms_count,
        "duration_hours": duration_hours,
        "q_ventilation_kw_per_room": round(q_ventilation, 3),
        "q_lighting_kw_per_room": round(q_lighting, 3),
        "q_occupant_kw_per_room": round(q_occupant, 3),
        "total_thermal_load_saved_kw": round(total_q_saved_kw, 3),
        "electrical_energy_saved_kwh": round(delta_e_kwh, 3),
        "cost_savings_usd": round(delta_s_usd, 3),
        "cost_savings_inr": round(delta_s_usd * 83.5, 2)
    }

    return results

if __name__ == "__main__":
    res = calculate_classroom_energy_savings(vacated_rooms_count=2, duration_hours=1.0)
    print("\n" + "="*70)
    print("=== ASHRAE THERMODYNAMIC ENERGY SAVINGS EVALUATION RESULT ===")
    print("="*70)
    for k, v in res.items():
        print(f" {k:<32}: {v}")
    print("="*70 + "\n")
