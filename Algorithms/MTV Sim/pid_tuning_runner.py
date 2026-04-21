"""
pid_tuning_runner.py
--------------------
Entry point for running and tuning the MTV PID controller simulation.

All constants live here so they can be adjusted during experimental tuning
without touching the engine module.

Typical usage
-------------
    python "pid_tuning_runner.py" <mode>
"""

import importlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# ---------------------------------------------------------------------------
# Load the engine module by path (its filename contains spaces).
# ---------------------------------------------------------------------------
_ENGINE_PATH = Path(__file__).parent / "Dynamic Mdot PID.py"
_spec   = importlib.util.spec_from_file_location("dynamic_mdot_pid", _ENGINE_PATH)
_engine = importlib.util.module_from_spec(_spec)
sys.modules["dynamic_mdot_pid"] = _engine
_spec.loader.exec_module(_engine)

SimConfig                     = _engine.SimConfig
MTVSimulation                 = _engine.MTVSimulation
FeedSideModel                 = _engine.FeedSideModel
get_or_build_cea_grid         = _engine.get_or_build_cea_grid
throat_area_from_design_point = _engine.throat_area_from_design_point
print_time_marching_summary   = _engine.print_time_marching_summary
run_steady_valve_sweep        = _engine.run_steady_valve_sweep
plot_steady_valve_sweep       = _engine.plot_steady_valve_sweep

# ---------------------------------------------------------------------------
# Feed system
# ---------------------------------------------------------------------------
PT_BAR         = 55.0       # Tank pressure [bar]
TEMPERATURE_C  = 6.0        # Propellant temperature [°C]
CDA_PFS_M2     = 3.1e-5     # Upstream PFS effective CdA [m²]
CDA_MTV_MAX_M2 = 9e-4       # MTV fully-open CdA [m²]

ANGLE_DATA_PCT = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
CV_DATA        = np.array([0.000, 0.000, 0.120, 0.236, 0.539, 0.643, 1.081, 1.587, 2.615, 3.664, 5.525], dtype=float)

# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------
CD_INJ   = 0.30
A_INJ_M2 = 5.90701952e-5

# ---------------------------------------------------------------------------
# Grain / motor
# ---------------------------------------------------------------------------
L_GRAIN_M        = 0.31
D_PORT_INITIAL_M = 0.05
D_PORT_MAX_M     = 0.12
RHO_FUEL         = 900.0

# ---------------------------------------------------------------------------
# CEA / design point
# ---------------------------------------------------------------------------
DESIGN_THRUST_N = 1000.0
DESIGN_PC_BAR   = 40.0
DESIGN_OF       = 7.0
EPSILON         = 5.4830    # Nozzle expansion ratio

CEA_PC_MIN_BAR  = 5.0
CEA_PC_MAX_BAR  = 65.0
CEA_PC_STEP_BAR = 5.0
CEA_OF_MIN      = 0.5
CEA_OF_MAX      = 12.0
CEA_OF_STEP     = 0.50

# ---------------------------------------------------------------------------
# Time-march
# ---------------------------------------------------------------------------
BURN_TIME_S = 15.0
DT_S        = 0.05
TAU_PC_S    = 0.20          # Chamber pressure lag time constant [s]
TAU_MDOT_S  = 0.50          # Oxidiser mass-flow lag time constant [s]

# ---------------------------------------------------------------------------
# PID gains  ← primary tuning targets
# ---------------------------------------------------------------------------
PID_KP = 0.8
PID_KI = 0.3
PID_KD = 0.0

PID_TRIM_MIN_PCT         = -20.0
PID_TRIM_MAX_PCT         =  20.0
PID_VALVE_MIN_PCT        =  15.0
PID_VALVE_MAX_PCT        = 100.0
PID_INITIAL_VALVE_PCT    =  40.0
PID_INTEGRAL_MIN         = -50.0
PID_INTEGRAL_MAX         =  50.0
PID_RATE_LIMIT_PCT_PER_S = 120.0

# ---------------------------------------------------------------------------
# Servo actuator
# ---------------------------------------------------------------------------
SERVO_KP             = 20.0     # [(%/s) per % position error]
SERVO_MAX_RATE_PCT_S = 120.0    # Hard slew-rate ceiling [%/s]

# ---------------------------------------------------------------------------
# Feedforward sweep range
# ---------------------------------------------------------------------------
FF_ANGLE_START_PCT = 25.0
FF_ANGLE_END_PCT   = 100.0

# ---------------------------------------------------------------------------
# Solver bounds
# ---------------------------------------------------------------------------
M_DOT_MIN = 0.15
M_DOT_MAX = 0.60

# ---------------------------------------------------------------------------
# Thrust schedule — edit to match your test profile
# ---------------------------------------------------------------------------
def thrust_schedule(t_s: float) -> float:
    if t_s < 2.0:
        return 1000.0
    elif t_s < 5.0:
        return 900.0
    elif t_s < 8.0:
        return 1100.0
    else:
        return 950.0


# ===========================================================================
# Internal helpers
# ===========================================================================

def _make_config(**overrides) -> SimConfig:
    """Build a SimConfig from the local constants, with optional overrides."""
    defaults = dict(
        pt_bar=PT_BAR,
        temperature_c=TEMPERATURE_C,
        cda_pfs_m2=CDA_PFS_M2,
        cda_mtv_max_m2=CDA_MTV_MAX_M2,
        d_port_initial_m=D_PORT_INITIAL_M,
        d_port_max_m=D_PORT_MAX_M,
        burn_time_s=BURN_TIME_S,
        dt_s=DT_S,
        tau_pc_s=TAU_PC_S,
        tau_mdot_s=TAU_MDOT_S,
        servo_kp=SERVO_KP,
        servo_max_rate_pct_per_s=SERVO_MAX_RATE_PCT_S,
        pid_kp=PID_KP,
        pid_ki=PID_KI,
        pid_kd=PID_KD,
        pid_trim_min_pct=PID_TRIM_MIN_PCT,
        pid_trim_max_pct=PID_TRIM_MAX_PCT,
        pid_valve_min_pct=PID_VALVE_MIN_PCT,
        pid_valve_max_pct=PID_VALVE_MAX_PCT,
        pid_initial_valve_pct=PID_INITIAL_VALVE_PCT,
        pid_integral_min=PID_INTEGRAL_MIN,
        pid_integral_max=PID_INTEGRAL_MAX,
        pid_rate_limit_pct_per_s=PID_RATE_LIMIT_PCT_PER_S,
        thrust_schedule=thrust_schedule,
    )
    defaults.update(overrides)
    return SimConfig(**defaults)


def _compute_iae(history: dict) -> float:
    """Integral Absolute Error of thrust vs the commanded thrust schedule."""
    t  = np.array(history["time_s"],   dtype=float)
    F  = np.array(history["thrust_N"], dtype=float)
    Ft = np.array([thrust_schedule(ti) for ti in t], dtype=float)
    valid = np.isfinite(F) & np.isfinite(Ft)
    if np.count_nonzero(valid) < 2:
        return np.inf
    return float(np.trapezoid(np.abs(F[valid] - Ft[valid]), t[valid]))


# ===========================================================================
# Shared setup — build CEA grid and throat geometry once for all runs
# ===========================================================================

def build_shared_setup():
    cea_grid = get_or_build_cea_grid(
        pc_min_bar=CEA_PC_MIN_BAR,
        pc_max_bar=CEA_PC_MAX_BAR,
        pc_step_bar=CEA_PC_STEP_BAR,
        of_min=CEA_OF_MIN,
        of_max=CEA_OF_MAX,
        of_step=CEA_OF_STEP,
    )

    throat_area_m2, design_perf = throat_area_from_design_point(
        design_thrust_n=DESIGN_THRUST_N,
        design_pc_bar=DESIGN_PC_BAR,
        design_of=DESIGN_OF,
        cea_grid=cea_grid,
    )

    throat_diameter_m = math.sqrt(4.0 * throat_area_m2 / np.pi)
    exit_area_m2      = EPSILON * throat_area_m2
    exit_diameter_m   = math.sqrt(4.0 * exit_area_m2 / np.pi)

    print("\n==============================================================")
    print("DESIGN REFERENCE")
    print(f"Design thrust:           {DESIGN_THRUST_N:.3f} N")
    print(f"Design chamber pressure: {DESIGN_PC_BAR:.3f} bar")
    print(f"Design O/F:              {DESIGN_OF:.3f}")
    print(f"Design C*:               {design_perf['cstar']:.3f} m/s")
    print(f"Design Cf:               {design_perf['cf']:.5f}")
    print(f"Throat area:             {throat_area_m2:.6e} m^2")
    print(f"Exit area:               {exit_area_m2:.6e} m^2")
    print(f"Throat diameter:         {1000.0 * throat_diameter_m:.4f} mm")
    print(f"Exit diameter:           {1000.0 * exit_diameter_m:.4f} mm")
    print("==============================================================\n")

    return cea_grid, throat_area_m2


# ===========================================================================
# Section 1 — Single nominal run
# ===========================================================================

def run_single(cea_grid, throat_area_m2):
    sim     = MTVSimulation(_make_config(), cea_grid, throat_area_m2)
    history = sim.run()
    iae     = _compute_iae(history)
    print_time_marching_summary(history)
    print(f"Thrust IAE: {iae:.2f} N·s")
    sim.plot(history)
    return history


# ===========================================================================
# Section 2 — PID optimiser
# ===========================================================================

def run_optimize(cea_grid, throat_area_m2):
    """
    Optimise the parameters listed in OPTIMIZE_BOUNDS to minimise thrust IAE.

    Progress is printed each evaluation and a convergence plot is shown at
    the end alongside the best-run time history.
    """
    param_names  = list(OPTIMIZE_BOUNDS.keys())
    bounds_list  = [OPTIMIZE_BOUNDS[p] for p in param_names]
    x0           = [_make_config().__dict__[p] for p in param_names]

    # Progress tracking
    evals    = []   # (iteration, param_values, iae)
    best_iae = [np.inf]
    call_n   = [0]

    def objective(x):
        overrides = {name: float(np.clip(val, lo, hi))
                     for name, val, (lo, hi) in zip(param_names, x, bounds_list)}
        try:
            sim     = MTVSimulation(_make_config(**overrides), cea_grid, throat_area_m2)
            history = sim.run()
            iae     = _compute_iae(history)
        except Exception as exc:
            iae = 1e9
            print(f"  [eval {call_n[0]:4d}] FAILED ({exc})")

        call_n[0] += 1
        is_best = iae < best_iae[0]
        if is_best:
            best_iae[0] = iae

        param_str = "  ".join(f"{n}={v:.4f}" for n, v in zip(param_names, x))
        marker    = " *" if is_best else ""
        print(f"  eval {call_n[0]:4d} | {param_str} | IAE = {iae:10.2f} N·s{marker}")

        evals.append((call_n[0], list(x), iae))
        return iae

    print("=" * 60)
    print(f"Optimising: {param_names}")
    print(f"Method:     {OPTIMIZE_METHOD}")
    print(f"Bounds:     {dict(zip(param_names, bounds_list))}")
    print(f"x0:         {dict(zip(param_names, x0))}")
    print("=" * 60)

    if OPTIMIZE_METHOD == "nelder-mead":
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter":  OPTIMIZE_MAX_ITER,
                "xatol":    1e-3,
                "fatol":    0.5,
                "adaptive": True,
                "disp":     True,
            },
        )

    elif OPTIMIZE_METHOD == "differential-evolution":
        result = differential_evolution(
            objective,
            bounds_list,
            maxiter=OPTIMIZE_MAX_ITER,
            tol=0.01,
            seed=42,
            disp=True,
            polish=True,    # Nelder-Mead polish pass after DE
        )

    else:
        raise ValueError(f"Unknown OPTIMIZE_METHOD: {OPTIMIZE_METHOD!r}. "
                         "Choose 'nelder-mead' or 'differential-evolution'.")

    # ---- results ----
    best_params = {name: float(np.clip(val, lo, hi))
                   for name, val, (lo, hi) in zip(param_names, result.x, bounds_list)}

    print("\n" + "=" * 60)
    print("OPTIMISATION COMPLETE")
    print(f"  Converged:  {result.success}")
    print(f"  Message:    {result.message}")
    print(f"  Evaluations:{call_n[0]}")
    print(f"  Best IAE:   {result.fun:.2f} N·s")
    print("  Best parameters:")
    for name, val in best_params.items():
        print(f"    {name:30s} = {val:.6f}")
    print("=" * 60)

    _plot_convergence(evals, param_names)

    print("\nRunning final simulation with optimised parameters...")
    sim     = MTVSimulation(_make_config(**best_params), cea_grid, throat_area_m2)
    history = sim.run()
    print_time_marching_summary(history)
    sim.plot(history)

    return result, best_params


def _plot_convergence(evals: list, param_names: list):
    """Plot IAE vs evaluation count and parameter trajectories."""
    iters  = [e[0]  for e in evals]
    iae_vals = [e[2] for e in evals]
    params = np.array([e[1] for e in evals])

    # Running best
    best_so_far = np.minimum.accumulate(iae_vals)

    n_params = len(param_names)
    fig, axs = plt.subplots(n_params + 1, 1, figsize=(12, 3 * (n_params + 1)), sharex=True)
    fig.suptitle("Optimiser convergence", fontsize=13)

    axs[0].semilogy(iters, iae_vals,     alpha=0.4, color="steelblue", label="IAE per eval")
    axs[0].semilogy(iters, best_so_far,  color="crimson",  linewidth=2, label="Best so far")
    axs[0].set_ylabel("IAE [N·s]")
    axs[0].legend()
    axs[0].grid(True)

    for i, name in enumerate(param_names):
        axs[i + 1].plot(iters, params[:, i], color="darkorange")
        axs[i + 1].set_ylabel(name)
        axs[i + 1].grid(True)

    axs[-1].set_xlabel("Evaluation #")
    plt.tight_layout()
    plt.show()

# ===========================================================================
# Main
# ===========================================================================

# ---------------------------------------------------------------------------
# Optimizer settings
# ---------------------------------------------------------------------------
# Which PID parameters to optimise and their [min, max] search bounds.
# Remove a key to hold that parameter fixed at the value defined above.
OPTIMIZE_BOUNDS = {
    "pid_kp": (0.0, 5.0),
    "pid_ki": (0.0, 2.0),
    "pid_kd": (0.0, 0.5),
}

# "nelder-mead" — fast, local (good starting point required).
# "differential-evolution" — slow, global (no starting point needed).
OPTIMIZE_METHOD  = "nelder-mead"
OPTIMIZE_MAX_ITER = 300        # Maximum optimizer iterations / generations


if __name__ == "__main__":
    cea_grid, throat_area_m2 = build_shared_setup()
    mode = sys.argv[1]
    if mode == "Single":
        run_single(cea_grid, throat_area_m2)

    if mode == "Optimize":
        run_optimize(cea_grid, throat_area_m2)