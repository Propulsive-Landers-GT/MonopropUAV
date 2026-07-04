"""
Simulates the angular movement of a Valworx 60° V-port ball valve driven by
Reefs Triple7 servos (2:1 servo-to-valve ratio)

Outputs:
  - Matplotlib plot of servo angle, valve open %, and thrust vs. time
  - CSV log in ./logs/<YYYY-MM-DD_HH-MM>.csv with raw simulation data
"""

import csv
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Servo travel: 0° – 180°  (fully closed → fully open)
# Valve travel: 0° – 90°   (2:1 ratio: valve_angle = servo_angle / 2)
SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 180.0
VALVE_MAX_DEG = 90.0        # 100% open


def valve_open_fraction(servo_angle_deg: float) -> float:
    valve_angle = servo_angle_deg / 2.0           # 2:1 mechanical ratio
    fraction = valve_angle / VALVE_MAX_DEG        # normalise to 0–1
    return max(0.0, min(1.0, fraction))

# ── N2O liquid density table (isobaric_liquid_nitrous_oxide.xlsx, 6.5 MPa) ───
_ISO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isobaric_liquid_nitrous_oxide.xlsx")
_iso_T:   np.ndarray = np.array([])
_iso_rho: np.ndarray = np.array([])


def _load_iso_table() -> None:
    """Load and cache the isobaric N2O density table on first call."""
    global _iso_T, _iso_rho
    if _iso_T.size:
        return
    df = pd.read_excel(_ISO_PATH)
    _iso_T   = df["Temperature (K)"].to_numpy(dtype=float)
    _iso_rho = df["Density (kg/m3)"].to_numpy(dtype=float)


def get_n2o_density(T_K: float) -> float:
    """Liquid N2O density (kg/m³) at 6.5 MPa, interpolated from NIST isobaric table.

    Valid range: ~288.7 K – 304.5 K.  Values outside the table are extrapolated
    linearly by numpy.interp (clamps to endpoint values).
    """
    _load_iso_table()
    return float(np.interp(T_K, _iso_T, _iso_rho))


# ── angleSolver2.m physics ───────────────────────────────────────────────────
# Tank / injector constants (match angleSolver2.m)
_P1_PA    = 6.5e6       # tank pressure (Pa)
_A_INJ    = 3.9528e-5   # injector geometric area (m²)
_CD_INJ   = 0.32        # injector discharge coefficient
_CD_VALVE = 0.245       # valve discharge coefficient

# Cubic polynomial: valve mechanical angle (deg) = f(effective valve area, m²)
# theta = -1e14·A³ + 1e10·A² + 959373·A - 1.4629   (from angleSolver2.m line 43)
_ANGLE_POLY = (-1e14, 1e10, 959373.0, -1.4629)

# A_ev at which the polynomial reaches its maximum (d theta/d A_ev = 0).
# Beyond this point the cubic turns over, so the binary search stays below it.
# Rearranged to standard form: 3e14·A² − 2e10·A − 959373 = 0
# Positive root ≈ 9.895e-5 m²
_A_EV_PEAK = (2e10 + math.sqrt((2e10)**2 + 4 * 3e14 * 959373)) / (2 * 3e14)

_G0 = 9.80665  # standard gravity (m/s²)


def _angle_from_A_ev(A_ev: float) -> float:
    """Valve mechanical angle (deg) from effective valve area (m²) via the cubic fit."""
    a, b, c, d = _ANGLE_POLY
    return a * A_ev**3 + b * A_ev**2 + c * A_ev + d


def _A_ev_from_valve_angle(valve_angle_deg: float, tol: float = 1e-14, max_iter: int = 64) -> float:
    """Binary search for A_ev given a target valve angle.

    The polynomial is monotonically increasing on [0, _A_EV_PEAK], so the
    search is well-conditioned within that range.
    """
    if valve_angle_deg <= _ANGLE_POLY[3]:   # below the zero-flow intercept
        return 0.0

    lo, hi = 0.0, _A_EV_PEAK
    if _angle_from_A_ev(hi) < valve_angle_deg:
        return hi   # angle exceeds the polynomial maximum; clamp to peak

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if abs(_angle_from_A_ev(mid) - valve_angle_deg) < tol:
            return mid
        if _angle_from_A_ev(mid) < valve_angle_deg:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def _mdot_from_A_ev(A_ev: float, Pc_bar: float, rho_n2o: float) -> float:
    """Closed-form mass flow rate from effective valve area.

    Derived by treating the MTV valve and injector as two orifices in series
    sharing the same total pressure drop (P_tank − P_chamber):

        m_dot = √(2ρ·ΔP_total) · (Cv·Av·Ci·Ai) / √((Ci·Ai)² + (Cv·Av)²)

    where Cv·Av = A_ev·Cd_valve and Ci·Ai = Cd_inj·A_inj.
    """
    Pc_pa    = Pc_bar * 1e5
    headroom = _P1_PA - Pc_pa
    if headroom <= 0.0 or A_ev <= 0.0:
        return 0.0
    Cv_Av = A_ev * _CD_VALVE
    Ci_Ai = _CD_INJ * _A_INJ
    return math.sqrt(2.0 * rho_n2o * headroom) * Cv_Av * Ci_Ai / math.sqrt(Ci_Ai**2 + Cv_Av**2)


def mdot_from_valve_angle(valve_angle_deg: float, Pc_bar: float, rho_n2o: float) -> float:
    """Mass flow rate (kg/s) from the current valve mechanical angle and chamber pressure.

    Two-step inversion of the angleSolver2 model:
      1. Binary search in A_ev space (monotonic cubic polynomial).
      2. Closed-form two-orifice-in-series formula to convert A_ev → m_dot.
    """
    A_ev = _A_ev_from_valve_angle(valve_angle_deg)
    return _mdot_from_A_ev(A_ev, Pc_bar, rho_n2o)


def thrust_from_mdot(m_dot: float, isp_s: float) -> float:
    """Thrust (N) from mass flow rate (kg/s): F = m_dot · Isp · g0."""
    return m_dot * isp_s * _G0


# ─────────────────────────────────────────────────────────────────────────────


def _target_at_time(t: float, servo_schedule: dict) -> float:
    """Return the commanded servo angle at time t.

    Takes the most-recent entry in servo_schedule whose key ≤ t.
    If t precedes all defined times, returns 0° (fully closed).
    """
    past_times = sorted(k for k in servo_schedule if k <= t)
    if not past_times:
        return 0.0
    return float(servo_schedule[past_times[-1]])


def simulate(
    tick_s: float,
    servo_schedule: dict,
    total_time_s: float,
    p_gain: float = 10.0,
    unloaded_speed_deg_s: float = 343.8,
    stall_torque: float = 70.0,
    valve_torque: float = 30.0,
    chamber_pressure_bar: float = 20.0,
    rho_n2o: float = 802.0,
    isp_s: float = 220.0,
) -> tuple[list, list, list, list]:
    """Run the valve servo simulation.

    A proportional controller drives the servo toward its commanded angle.
    The P controller computes a velocity command (p_gain × position error),
    which is then clamped to a load-adjusted speed limit derived from the
    servo's torque-speed curve. This matches the MTV dynamics in device_sim.rs.

    Thrust at each step is computed by inverting the angleSolver2 physics model:
    valve angle → effective area (binary search) → mass flow rate → thrust.

    Args:
        tick_s:                 Integration time step (s).
        servo_schedule:         Dict mapping {time_s: servo_angle_deg}.
                                Between defined times the target is unchanged.
        total_time_s:           Simulation end time (s).
        p_gain:                 Proportional gain (deg/s per deg of error).
        unloaded_speed_deg_s:   No-load servo speed (deg/s).
        stall_torque:           Servo stall torque (N·m or any consistent unit).
        valve_torque:           Resistive torque from the valve load (same unit).
        chamber_pressure_bar:   Fixed chamber back-pressure (bar) for thrust model.
        rho_n2o:                N2O liquid density (kg/m³) at tank conditions.
                                Use get_n2o_density(T_tank_K) to derive from the
                                isobaric NIST table (isobaric_liquid_nitrous_oxide.xlsx).
        isp_s:                  Specific impulse (s) for thrust conversion.

    Returns:
        Four lists: (times_s, servo_angles_deg, open_percents, thrusts_N)
    """
    load_factor = min(valve_torque / stall_torque, 1.0)
    speed_limit = unloaded_speed_deg_s * (1.0 - load_factor)

    times:         list[float] = []
    servo_angles:  list[float] = []
    open_percents: list[float] = []
    thrusts:       list[float] = []

    servo_pos = 0.0
    servo_vel = 0.0

    n_steps = int(total_time_s / tick_s) + 1

    for i in range(n_steps):
        t = i * tick_s

        target  = _target_at_time(t, servo_schedule)
        error   = target - servo_pos
        command = error * p_gain

        servo_vel = max(-speed_limit, min(speed_limit, command))
        servo_pos += servo_vel * tick_s
        servo_pos = max(SERVO_MIN_DEG, min(SERVO_MAX_DEG, servo_pos))

        x = valve_open_fraction(servo_pos)

        valve_angle = servo_pos / 2.0
        m_dot  = mdot_from_valve_angle(valve_angle, chamber_pressure_bar, rho_n2o)
        thrust = thrust_from_mdot(m_dot, isp_s)

        times.append(t)
        servo_angles.append(servo_pos)
        open_percents.append(x * 100.0)
        thrusts.append(thrust)

    return times, servo_angles, open_percents, thrusts


def save_log(
    times: list,
    servo_angles: list,
    open_percents: list,
    thrusts: list,
) -> str:
    """Write raw simulation data to ./logs/<YYYY-MM-DD_HH-MM>.csv."""
    os.makedirs("logs", exist_ok=True)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M") + ".csv"
    filepath = os.path.join("logs", filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "servo_angle_deg", "valve_open_pct", "thrust_N"])
        for t, s, o, f_ in zip(times, servo_angles, open_percents, thrusts):
            writer.writerow([round(t, 4), round(s, 4), round(o, 4), round(f_, 4)])

    print(f"Log saved → {filepath}")
    return filepath


def plot_results(
    times: list,
    servo_angles: list,
    open_percents: list,
    thrusts: list,
) -> None:
    """Display a three-panel plot of servo angle, valve open %, and thrust."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(times, servo_angles, color="darkorange", linewidth=1.5)
    axes[0].set_ylabel("Servo Angle (°)")
    axes[0].set_ylim(-5, 185)
    axes[0].set_title("MTV Valve Simulation")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(times, open_percents, color="seagreen", linewidth=1.5)
    axes[1].set_ylabel("Valve Open (%)")
    axes[1].set_ylim(-2, 105)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(times, thrusts, color="crimson", linewidth=1.8)
    axes[2].set_ylabel("Thrust (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def main():

    # How often a data point is collected / integration step (seconds)
    tick_s: float = 0.05

    # P controller gain (deg/s per deg of position error)
    p_gain: float = 10.0

    # Reefs Triple7 spec: ~343.8 deg/s no-load (0.523 s / 180°)
    unloaded_speed_deg_s: float = 343.8

    # Servo stall torque and valve load torque (N·m)
    stall_torque: float = 70.0
    valve_torque: float = 30.0

    # Fixed chamber back-pressure for the thrust model (bar)
    chamber_pressure_bar: float = 20.0

    # N2O liquid density (kg/m³) at tank conditions.
    # T1_K = 294.26 K matches angleSolver2.m; change if your tank temperature differs.
    T_tank_K: float = 294.26
    rho_n2o: float = get_n2o_density(T_tank_K)
    print(f"N2O density at {T_tank_K} K (6.5 MPa): {rho_n2o:.2f} kg/m³")

    # Specific impulse (s) — used to convert mass flow rate to thrust
    isp_s: float = 220.0

    # Servo schedule: {time_in_seconds: target_servo_angle_in_degrees}
    # The P controller drives the servo toward the commanded angle.
    # Between two defined times the target remains at the earlier value.
    servo_schedule: dict = {
        0.0:   0.0,   # t=0 s  → servo at   0° (valve fully closed)
        2.0:  60.0,   # t=2 s  → servo to  60° (valve at ~33% open)
        5.0: 120.0,   # t=5 s  → servo to 120° (valve fully open)
        8.0:   0.0,   # t=8 s  → close valve
    }

    total_time_s = max(servo_schedule.keys()) + 3.0   # 3 s buffer after last event

    times, servo_angles, open_percents, thrusts = simulate(
        tick_s=tick_s,
        servo_schedule=servo_schedule,
        total_time_s=total_time_s,
        p_gain=p_gain,
        unloaded_speed_deg_s=unloaded_speed_deg_s,
        stall_torque=stall_torque,
        valve_torque=valve_torque,
        chamber_pressure_bar=chamber_pressure_bar,
        rho_n2o=rho_n2o,
        isp_s=isp_s,
    )

    save_log(times, servo_angles, open_percents, thrusts)
    plot_results(times, servo_angles, open_percents, thrusts)


if __name__ == "__main__":
    main()
