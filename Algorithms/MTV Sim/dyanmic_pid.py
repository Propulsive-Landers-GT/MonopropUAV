from pathlib import Path
from datetime import datetime
import subprocess
import os
import re
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List

import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================
# USER INPUTS / CONSTANTS
# ============================================================

FLUID = "NITROUSOXIDE"

# -----------------------------
# FEED SYSTEM INPUTS
# -----------------------------
PT_BAR_DEFAULT = 55.0
TEMPERATURE_C_DEFAULT = 6.0

# Upstream PFS effective CdA
CDA_PFS_M2_DEFAULT = 3.1e-5
CDA_MTV_MAX_M2_DEFAULT = 9e-4   # keep your current value; verify physically

# Valve angle (%) vs Cv table
ANGLE_DATA_PCT_DEFAULT = np.array(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    dtype=float,
)

CV_DATA_DEFAULT = np.array(
    [0.000, 0.000, 0.120, 0.236, 0.539, 0.643, 1.081, 1.587, 2.615, 3.664, 5.525],
    dtype=float,
)

# -----------------------------
# INJECTOR INPUTS
# -----------------------------
CD_INJ = 0.30
A_INJ_M2 = 5.90701952e-5
CDA_INJ_M2 = CD_INJ * A_INJ_M2 * 1.25

# -----------------------------
# HYBRID GRAIN / MOTOR INPUTS
# -----------------------------
L_GRAIN_M = 0.31
D_PORT_INITIAL_M = 0.05
A_REG = 0.0000722
N_REG = 0.67
RHO_FUEL = 900.0
EPSILON = 5.4830

# -----------------------------
# THROAT SIZING DESIGN POINT
# -----------------------------
DESIGN_THRUST_N = 1000.0
DESIGN_PC_BAR = 40.0
DESIGN_OF = 7.0

# -----------------------------
# CEA SETTINGS
# -----------------------------
CEA_PC_MIN_BAR = 5.0
CEA_PC_MAX_BAR = 65.0
CEA_PC_STEP_BAR = 5.0

CEA_OF_MIN = 0.5
CEA_OF_MAX = 12.0
CEA_OF_STEP = 0.50

PE_BAR = 1.01325

ETA_CSTAR = 0.95
ETA_CF = 0.95
G0 = 9.80665

DEBUG = True

# -----------------------------
# TIME-MARCH SETTINGS
# -----------------------------
DT_S = 0.05
BURN_TIME_S = 15.0
D_PORT_MAX_M = 0.12
STOP_ON_SOLVER_FAILURE = True
TAU_PC_S = 0.20
TAU_MDOT_S = 0.5

# Progress printing
SHOW_PROGRESS = True
PROGRESS_PRINT_EVERY_N_STEPS = 250
PROGRESS_PRINT_MIN_INTERVAL_S = 0.5

# -----------------------------
# CONTROL SETTINGS
# -----------------------------
USE_PID = True
USE_FEEDFORWARD = True

def thrust_target_schedule_n(t_s: float) -> float:
    """
    Commanded thrust versus time.
    Replace this with your own schedule later.
    """
    if t_s < 2.0:
        return 1000.0
    elif t_s < 5.0:
        return 900.0
    elif t_s < 8.0:
        return 1100.0
    else:
        return 950.0

FF_SWEEP_ANGLE_START_PCT = 25.0
FF_SWEEP_ANGLE_END_PCT = 100.0
FF_SWEEP_ANGLE_STEP_PCT = 1.0
FF_REBUILD_INTERVAL_S = 0.25

PID_KP = 0.8
PID_KI = 0.3
PID_KD = 0.00

PID_TRIM_MIN_PCT = -20.0
PID_TRIM_MAX_PCT = 20.0

PID_VALVE_MIN_PCT = 15.0
PID_VALVE_MAX_PCT = 100.0

PID_INITIAL_VALVE_PCT = 40.0
PID_INTEGRAL_MIN = -50.0
PID_INTEGRAL_MAX = 50.0

PID_RATE_LIMIT_PCT_PER_S = 120.0

# -----------------------------
# CEA PATHS
# -----------------------------
CEA_EXE = Path(r"C:\Users\chald\Downloads\cea-main\cea-main\build\source\cea.exe")
CEA_DATA_DIR = Path(r"C:\Users\chald\Downloads\cea-main\cea-main\data")
BASE_DIR = Path(r"C:\Users\chald\FeedSystemDesign\pythonProject1\CEA Run")

# -----------------------------
# CEA CACHE SETTINGS
# -----------------------------
CEA_CACHE_DIR = BASE_DIR / "CEA_Cache"
CEA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

USE_CEA_CACHE = True
REBUILD_CEA_CACHE = False

# Optional parallel CEA-grid build
USE_PARALLEL_CEA_GRID_BUILD = True
MAX_CEA_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# -----------------------------
# FEEDFORWARD CACHE SETTINGS
# -----------------------------
FF_CACHE_DIR = BASE_DIR / "FF_Cache"
FF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

USE_FF_CACHE = True
REBUILD_FF_CACHE = False

# -----------------------------
# SOLVER SETTINGS
# -----------------------------
M_DOT_MIN = 0.15
M_DOT_MAX = 0.6
M_DOT_RES_TOL = 1e-6
MAX_OUTER_ITER = 80

PC_FIXED_POINT_TOL_BAR = 1e-5
PC_FIXED_POINT_MAX_ITER = 120

# -----------------------------
# OPTIONAL LEGACY SCHEDULE HELPER
# -----------------------------
def valve_angle_schedule_pct(t_s: float) -> float:
    """
    Legacy open-loop schedule. Not used when feedforward/PID is active,
    but kept so you can still test open-loop behavior if needed.
    """
    if t_s < 2.0:
        return 100.0
    elif t_s < 4.0:
        return 80.0
    elif t_s < 6.0:
        return 60.0
    elif t_s < 8.0:
        return 40.0
    elif t_s < 10.0:
        return 20.0
    else:
        return 100.0


# ============================================================
# BASIC HELPERS
# ============================================================

def debug_print(msg: str) -> None:
    if DEBUG:
        print(msg)

def bar_to_pa(p_bar: float) -> float:
    return p_bar * 1e5

def pa_to_bar(p_pa: float) -> float:
    return p_pa / 1e5

def safe_sqrt(x: float) -> float:
    return math.sqrt(max(x, 0.0))

def safe_float(x, default=np.nan):
    try:
        return float(str(x).replace("D", "E"))
    except Exception:
        return default

def compute_cstar_from_isp_cf(isp_s: float, cf: float) -> float:
    if np.isnan(isp_s) or np.isnan(cf) or cf <= 0:
        return np.nan
    return isp_s * G0 / cf


# ============================================================
# NITROUS PROPERTY MODEL
# ============================================================

@dataclass
class NitrousState:
    pressure_bar: float
    temperature_C: float
    density_kg_m3: float
    saturation_pressure_bar: float
    pressure_margin_above_sat_bar: float


class NitrousProperties:
    def __init__(self, fluid: str = FLUID) -> None:
        self.fluid = fluid

    def density_from_PT(self, pressure_bar: float, temperature_C: float) -> float:
        pressure_pa = bar_to_pa(pressure_bar)
        temperature_K = temperature_C + 273.15
        return CP.PropsSI("D", "P", pressure_pa, "T", temperature_K, self.fluid)

    def saturation_pressure_bar(self, temperature_C: float) -> float:
        temperature_K = temperature_C + 273.15
        p_sat_pa = CP.PropsSI("P", "T", temperature_K, "Q", 0, self.fluid)
        return pa_to_bar(p_sat_pa)

    def state_from_PT(self, pressure_bar: float, temperature_C: float) -> NitrousState:
        rho = self.density_from_PT(pressure_bar, temperature_C)
        p_sat_bar = self.saturation_pressure_bar(temperature_C)

        return NitrousState(
            pressure_bar=pressure_bar,
            temperature_C=temperature_C,
            density_kg_m3=rho,
            saturation_pressure_bar=p_sat_bar,
            pressure_margin_above_sat_bar=pressure_bar - p_sat_bar,
        )


# ============================================================
# FEED SYSTEM MODELS
# ============================================================

class UpstreamPFSModel:
    def __init__(
        self,
        tank_pressure_bar: float = PT_BAR_DEFAULT,
        cda_pfs_m2: float = CDA_PFS_M2_DEFAULT,
        fluid: str = FLUID,
    ) -> None:
        self.tank_pressure_bar = tank_pressure_bar
        self.cda_pfs_m2 = cda_pfs_m2
        self.props = NitrousProperties(fluid=fluid)

    def calc_density_upstream(self, temperature_C: float) -> float:
        return self.props.density_from_PT(self.tank_pressure_bar, temperature_C)

    def calc_pfs_dp_bar(self, mdot_kg_s: float, temperature_C: float) -> float:
        if mdot_kg_s < 0.0:
            raise ValueError("mdot_kg_s must be non-negative.")
        rho = self.calc_density_upstream(temperature_C)
        dP_pa = (mdot_kg_s / self.cda_pfs_m2) ** 2 / (2.0 * rho)
        return pa_to_bar(dP_pa)

    def calc_p1_bar(self, mdot_kg_s: float, temperature_C: float) -> float:
        return self.tank_pressure_bar - self.calc_pfs_dp_bar(mdot_kg_s, temperature_C)


class MTVModel:
    def __init__(
        self,
        cda_mtv_max_m2: float = CDA_MTV_MAX_M2_DEFAULT,
        angle_data_pct: np.ndarray = ANGLE_DATA_PCT_DEFAULT,
        cv_data: np.ndarray = CV_DATA_DEFAULT,
        fluid: str = FLUID,
    ) -> None:
        self.cda_mtv_max_m2 = cda_mtv_max_m2
        self.angle_data_pct = np.array(angle_data_pct, dtype=float)
        self.cv_data = np.array(cv_data, dtype=float)
        self.cv_max = float(np.max(self.cv_data))
        self.props = NitrousProperties(fluid=fluid)

        if len(self.angle_data_pct) != len(self.cv_data):
            raise ValueError("angle_data_pct and cv_data must have the same length.")
        if self.cv_max <= 0.0:
            raise ValueError("cv_data must contain a positive maximum value.")

    def cv_from_angle(self, angle_pct: float) -> float:
        angle_pct = float(np.clip(angle_pct, self.angle_data_pct[0], self.angle_data_pct[-1]))
        return float(np.interp(angle_pct, self.angle_data_pct, self.cv_data))

    def cv_ratio_from_angle(self, angle_pct: float) -> float:
        return self.cv_from_angle(angle_pct) / self.cv_max

    def cda_from_angle(self, angle_pct: float) -> float:
        return self.cda_mtv_max_m2 * self.cv_ratio_from_angle(angle_pct)

    def calc_density_at_p1(self, p1_bar: float, temperature_C: float) -> float:
        return self.props.density_from_PT(p1_bar, temperature_C)

    def calc_p2_from_mdot(self, angle_pct: float, mdot_kg_s: float, p1_bar: float, temperature_C: float) -> float:
        cda_theta = self.cda_from_angle(angle_pct)

        # Treat tiny CdA as effectively closed
        if not np.isfinite(cda_theta) or cda_theta < 1e-10:
            return np.nan

        if not np.isfinite(p1_bar) or p1_bar <= 0.0:
            return np.nan

        rho1 = self.calc_density_at_p1(p1_bar, temperature_C)
        if not np.isfinite(rho1) or rho1 <= 0.0:
            return np.nan

        dP_pa = (mdot_kg_s / cda_theta) ** 2 / (2.0 * rho1)
        p2_bar = p1_bar - pa_to_bar(dP_pa)

        if not np.isfinite(p2_bar) or p2_bar <= 0.0:
            return np.nan

        return p2_bar


class FeedSideModel:
    def __init__(
        self,
        tank_pressure_bar: float = PT_BAR_DEFAULT,
        cda_pfs_m2: float = CDA_PFS_M2_DEFAULT,
        cda_mtv_max_m2: float = CDA_MTV_MAX_M2_DEFAULT,
        angle_data_pct: np.ndarray = ANGLE_DATA_PCT_DEFAULT,
        cv_data: np.ndarray = CV_DATA_DEFAULT,
        fluid: str = FLUID,
    ) -> None:
        self.upstream = UpstreamPFSModel(
            tank_pressure_bar=tank_pressure_bar,
            cda_pfs_m2=cda_pfs_m2,
            fluid=fluid,
        )
        self.mtv = MTVModel(
            cda_mtv_max_m2=cda_mtv_max_m2,
            angle_data_pct=angle_data_pct,
            cv_data=cv_data,
            fluid=fluid,
        )

    def evaluate_from_mdot(
        self,
        angle_pct: float,
        mdot_kg_s: float,
        temperature_C: float,
    ) -> Dict[str, float]:
        p1_bar = self.upstream.calc_p1_bar(mdot_kg_s, temperature_C)
        p2_bar = self.mtv.calc_p2_from_mdot(angle_pct, mdot_kg_s, p1_bar, temperature_C)

        if not np.isfinite(p1_bar) or p1_bar <= 0.0:
            raise ValueError(
                f"Nonphysical P1: {p1_bar:.6f} bar at mdot={mdot_kg_s:.6f} kg/s, angle={angle_pct:.2f}%"
            )
        if not np.isfinite(p2_bar) or p2_bar <= 0.0:
            raise ValueError(
                f"Nonphysical P2: {p2_bar:.6f} bar at mdot={mdot_kg_s:.6f} kg/s, angle={angle_pct:.2f}%"
            )

        rho_tank = self.upstream.props.density_from_PT(self.upstream.tank_pressure_bar, temperature_C)
        rho1 = self.mtv.calc_density_at_p1(p1_bar, temperature_C)
        p_sat_bar = self.upstream.props.saturation_pressure_bar(temperature_C)

        return {
            "angle_pct": angle_pct,
            "mdot_kg_s": mdot_kg_s,
            "p_tank_bar": self.upstream.tank_pressure_bar,
            "p1_bar": p1_bar,
            "p2_bar": p2_bar,
            "dP_pfs_bar": self.upstream.tank_pressure_bar - p1_bar,
            "dP_mtv_bar": p1_bar - p2_bar,
            "rho_tank_kg_m3": rho_tank,
            "rho1_kg_m3": rho1,
            "cv_theta": self.mtv.cv_from_angle(angle_pct),
            "cv_ratio_theta": self.mtv.cv_ratio_from_angle(angle_pct),
            "cda_pfs_m2": self.upstream.cda_pfs_m2,
            "cda_mtv_theta_m2": self.mtv.cda_from_angle(angle_pct),
            "cda_mtv_max_m2": self.mtv.cda_mtv_max_m2,
            "saturation_pressure_bar": p_sat_bar,
            "tank_margin_above_sat_bar": self.upstream.tank_pressure_bar - p_sat_bar,
        }


# ============================================================
# CEA HELPERS
# ============================================================

def build_cea_input_text(pc_bar: float, of_ratio: float, pressure_ratio: float) -> str:
    return f"""problem rocket equilibrium
  p(bar)={pc_bar}
  o/f={of_ratio}
  pi/p={pressure_ratio}
reactants
  fuel=Paraffin wt%=100 t(k)=298.15 h,cal/mol=-444600 C 73 H 124
  oxid=N2O wt%=100 t(k)=298.15
output siunits
end
"""

def run_cea_case(case_name: str, pc_bar: float, of_ratio: float, pressure_ratio: float, results_dir: Path) -> str:
    base = results_dir / case_name
    inp_path = base.with_suffix(".inp")
    out_path = base.with_suffix(".out")

    inp_path.write_text(build_cea_input_text(pc_bar, of_ratio, pressure_ratio), encoding="utf-8")

    env = os.environ.copy()
    env["CEA_DATA_DIR"] = str(CEA_DATA_DIR)

    subprocess.run(
        [str(CEA_EXE), str(base)],
        check=True,
        cwd=str(results_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return out_path.read_text(encoding="utf-8", errors="ignore")

def parse_property_row(out_text: str, row_name: str) -> List[float]:
    float_pattern = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
    row_name_up = row_name.upper()

    for line in out_text.splitlines():
        line_clean = line.strip().upper().replace("D", "E")
        if line_clean.startswith(row_name_up):
            nums = re.findall(float_pattern, line_clean)
            return [safe_float(x) for x in nums]
    return []

def parse_cea_output(out_text: str) -> dict:
    data = {
        "cf": np.nan,
        "isp_raw": np.nan,
        "ivac_raw": np.nan,
        "tc": np.nan,
        "gamma_c": np.nan,
        "gamma_e": np.nan,
        "mw_c": np.nan,
        "mw_e": np.nan,
        "mach_e": np.nan,
        "sonic_e": np.nan,
        "cstar": np.nan,
    }

    vals = parse_property_row(out_text, "CF")
    if len(vals) >= 1:
        data["cf"] = vals[-1]

    vals = parse_property_row(out_text, "ISP")
    if len(vals) >= 1:
        data["isp_raw"] = vals[-1]

    vals = parse_property_row(out_text, "IVAC")
    if len(vals) >= 1:
        data["ivac_raw"] = vals[-1]

    vals = parse_property_row(out_text, "T, K")
    if len(vals) >= 1:
        data["tc"] = vals[0]

    vals = parse_property_row(out_text, "GAMM")
    if len(vals) >= 1:
        data["gamma_c"] = vals[0]
    if len(vals) >= 2:
        data["gamma_e"] = vals[-1]

    vals = parse_property_row(out_text, "M, (1/N)")
    if len(vals) >= 1:
        data["mw_c"] = vals[0]
    if len(vals) >= 2:
        data["mw_e"] = vals[-1]

    if np.isnan(data["mw_c"]):
        vals = parse_property_row(out_text, "MOL WT")
        if len(vals) >= 1:
            data["mw_c"] = vals[0]
        if len(vals) >= 2:
            data["mw_e"] = vals[-1]

    vals = parse_property_row(out_text, "MACH")
    if len(vals) >= 1:
        data["mach_e"] = vals[-1]

    vals = parse_property_row(out_text, "SON VEL")
    if len(vals) >= 1:
        data["sonic_e"] = vals[-1]

    isp_s = data["isp_raw"] / G0 if not np.isnan(data["isp_raw"]) else np.nan
    data["cstar"] = compute_cstar_from_isp_cf(isp_s, data["cf"])
    return data

def fill_nan_1d(arr: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        raise RuntimeError("Array has no valid values; cannot fill NaNs.")
    return np.interp(x_vals, x_vals[valid], arr[valid])

def fill_nan_grid_along_of(grid: np.ndarray, of_values: np.ndarray, grid_name: str) -> np.ndarray:
    filled = np.array(grid, dtype=float, copy=True)

    for i in range(filled.shape[0]):
        row = filled[i, :]
        valid_count = np.count_nonzero(np.isfinite(row))

        if valid_count == 0:
            raise RuntimeError(f"{grid_name}: entire Pc row at index {i} is invalid; cannot fill.")

        if valid_count == 1:
            single_val = row[np.isfinite(row)][0]
            filled[i, :] = single_val
        elif valid_count < len(row):
            filled[i, :] = fill_nan_1d(row, of_values)

    return filled

def save_cea_grid_cache(cea_grid: dict, cache_file: Path) -> None:
    np.savez(
        cache_file,
        pc_values_bar=cea_grid["pc_values_bar"],
        of_values=cea_grid["of_values"],
        cstar_grid=cea_grid["cstar_grid"],
        cf_grid=cea_grid["cf_grid"],
        isp_grid=cea_grid["isp_grid"],
        ivac_grid=cea_grid["ivac_grid"],
        tc_grid=cea_grid["tc_grid"],
    )
    print(f"Saved CEA cache to: {cache_file}")

def load_cea_grid_cache(cache_file: Path) -> dict:
    data = np.load(cache_file)
    cea_grid = {
        "pc_values_bar": data["pc_values_bar"],
        "of_values": data["of_values"],
        "cstar_grid": data["cstar_grid"],
        "cf_grid": data["cf_grid"],
        "isp_grid": data["isp_grid"],
        "ivac_grid": data["ivac_grid"],
        "tc_grid": data["tc_grid"],
    }
    print(f"Loaded CEA cache from: {cache_file}")
    return cea_grid

def build_ff_cache_filename(
    temperature_C: float,
    d_port_m: float,
    tank_pressure_bar: float,
    cda_pfs_m2: float,
    cda_mtv_max_m2: float,
    throat_area_m2: float,
    angle_start_pct: float,
    angle_end_pct: float,
    angle_step_pct: float,
) -> Path:
    safe_name = (
        f"ff_map_"
        f"T_{temperature_C:.3f}_"
        f"dport_{d_port_m:.6f}_"
        f"pt_{tank_pressure_bar:.5f}_"
        f"cdapfs_{cda_pfs_m2:.8e}_"
        f"cdamtv_{cda_mtv_max_m2:.8e}_"
        f"at_{throat_area_m2:.8e}_"
        f"ang_{angle_start_pct:.3f}_{angle_end_pct:.3f}_{angle_step_pct:.3f}"
    ).replace(".", "p").replace("-", "m")
    return FF_CACHE_DIR / f"{safe_name}.npz"


def save_ff_map_cache(ff_map: Dict[str, np.ndarray], cache_file: Path) -> None:
    np.savez(
        cache_file,
        pc_bar=np.array(ff_map["pc_bar"], dtype=float),
        angle_pct=np.array(ff_map["angle_pct"], dtype=float),
    )
    print(f"Saved feedforward cache to: {cache_file}")


def load_ff_map_cache(cache_file: Path) -> Dict[str, np.ndarray]:
    data = np.load(cache_file)
    ff_map = {
        "pc_bar": np.array(data["pc_bar"], dtype=float),
        "angle_pct": np.array(data["angle_pct"], dtype=float),
    }
    print(f"Loaded feedforward cache from: {cache_file}")
    return ff_map

def build_cache_filename(
    pc_min_bar: float,
    pc_max_bar: float,
    pc_step_bar: float,
    of_min: float,
    of_max: float,
    of_step: float,
    pe_bar: float,
    eta_cstar: float,
    eta_cf: float,
) -> Path:
    safe_name = (
        f"cea_grid_"
        f"pc_{pc_min_bar:.3f}_{pc_max_bar:.3f}_{pc_step_bar:.3f}_"
        f"of_{of_min:.3f}_{of_max:.3f}_{of_step:.3f}_"
        f"pe_{pe_bar:.5f}_"
        f"etacstar_{eta_cstar:.5f}_"
        f"etacf_{eta_cf:.5f}"
    ).replace(".", "p")
    return CEA_CACHE_DIR / f"{safe_name}.npz"

def _run_single_cea_case(args):
    pc_bar, of_ratio, pressure_ratio, results_dir_str = args
    results_dir = Path(results_dir_str)
    case_name = f"pc_{pc_bar:.2f}_of_{of_ratio:.3f}".replace(".", "p")

    try:
        out_text = run_cea_case(case_name, pc_bar, of_ratio, pressure_ratio, results_dir)
        parsed = parse_cea_output(out_text)

        isp_s = parsed["isp_raw"] / G0 if not np.isnan(parsed["isp_raw"]) else np.nan
        ivac_s = parsed["ivac_raw"] / G0 if not np.isnan(parsed["ivac_raw"]) else np.nan

        cstar_val = parsed["cstar"] * ETA_CSTAR
        cf_val = parsed["cf"] * ETA_CF
        isp_val = isp_s * ETA_CSTAR * ETA_CF
        ivac_val = ivac_s * ETA_CSTAR * ETA_CF
        tc_val = parsed["tc"]

        if not (
            np.isfinite(cstar_val)
            and np.isfinite(cf_val)
            and np.isfinite(isp_val)
            and np.isfinite(ivac_val)
            and np.isfinite(tc_val)
        ):
            raise ValueError("Parsed CEA output contains non-finite values.")

        return {
            "ok": True,
            "pc_bar": pc_bar,
            "of_ratio": of_ratio,
            "cstar": cstar_val,
            "cf": cf_val,
            "isp": isp_val,
            "ivac": ivac_val,
            "tc": tc_val,
        }
    except Exception as e:
        return {
            "ok": False,
            "pc_bar": pc_bar,
            "of_ratio": of_ratio,
            "error": str(e),
        }

def generate_cea_grid(
    pc_min_bar: float,
    pc_max_bar: float,
    pc_step_bar: float,
    of_min: float,
    of_max: float,
    of_step: float,
    use_parallel: bool = USE_PARALLEL_CEA_GRID_BUILD,
) -> dict:
    run_name = datetime.now().strftime("cea_2d_grid_%Y-%m-%d_%H-%M-%S")
    results_dir = BASE_DIR / "CEA_Results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    pc_values = np.arange(pc_min_bar, pc_max_bar + 0.5 * pc_step_bar, pc_step_bar)
    of_values = np.arange(of_min, of_max + 0.5 * of_step, of_step)

    cstar_grid = np.full((len(pc_values), len(of_values)), np.nan, dtype=float)
    cf_grid = np.full((len(pc_values), len(of_values)), np.nan, dtype=float)
    isp_grid = np.full((len(pc_values), len(of_values)), np.nan, dtype=float)
    ivac_grid = np.full((len(pc_values), len(of_values)), np.nan, dtype=float)
    tc_grid = np.full((len(pc_values), len(of_values)), np.nan, dtype=float)

    print(f"Generating 2D CEA grid in: {results_dir}")

    jobs = []
    for i, pc_bar in enumerate(pc_values):
        pressure_ratio = pc_bar / PE_BAR
        for j, of_ratio in enumerate(of_values):
            jobs.append((i, j, pc_bar, of_ratio, pressure_ratio))

    failed_cases = []

    if use_parallel and len(jobs) > 1:
        print(f"Using parallel CEA grid build with up to {MAX_CEA_WORKERS} workers")
        with ProcessPoolExecutor(max_workers=MAX_CEA_WORKERS) as exe:
            futures = {
                exe.submit(
                    _run_single_cea_case,
                    (pc_bar, of_ratio, pressure_ratio, str(results_dir))
                ): (i, j, pc_bar, of_ratio)
                for i, j, pc_bar, of_ratio, pressure_ratio in jobs
            }

            done_count = 0
            total = len(futures)

            for fut in as_completed(futures):
                i, j, pc_bar, of_ratio = futures[fut]
                result = fut.result()
                done_count += 1

                if done_count % 25 == 0 or done_count == total:
                    print(f"CEA grid progress: {done_count}/{total}")

                if result["ok"]:
                    cstar_grid[i, j] = result["cstar"]
                    cf_grid[i, j] = result["cf"]
                    isp_grid[i, j] = result["isp"]
                    ivac_grid[i, j] = result["ivac"]
                    tc_grid[i, j] = result["tc"]
                else:
                    failed_cases.append((pc_bar, of_ratio, result["error"]))
    else:
        print("Using serial CEA grid build")
        for idx, (i, j, pc_bar, of_ratio, pressure_ratio) in enumerate(jobs, start=1):
            try:
                result = _run_single_cea_case((pc_bar, of_ratio, pressure_ratio, str(results_dir)))
                if result["ok"]:
                    cstar_grid[i, j] = result["cstar"]
                    cf_grid[i, j] = result["cf"]
                    isp_grid[i, j] = result["isp"]
                    ivac_grid[i, j] = result["ivac"]
                    tc_grid[i, j] = result["tc"]
                else:
                    failed_cases.append((pc_bar, of_ratio, result["error"]))
            except Exception as e:
                failed_cases.append((pc_bar, of_ratio, str(e)))

            if idx % 25 == 0 or idx == len(jobs):
                print(f"CEA grid progress: {idx}/{len(jobs)}")

    total_points = len(pc_values) * len(of_values)
    failed_count = len(failed_cases)
    success_count = total_points - failed_count

    print("\n================ CEA GRID SUMMARY ================")
    print(f"Total requested points: {total_points}")
    print(f"Successful points:      {success_count}")
    print(f"Failed points:          {failed_count}")

    if failed_count > 0:
        print("First few failed cases:")
        for pc_bar, of_ratio, err in failed_cases[:10]:
            print(f"  Pc={pc_bar:.2f} bar, O/F={of_ratio:.3f} -> {err}")

    cstar_grid = fill_nan_grid_along_of(cstar_grid, of_values, "cstar_grid")
    cf_grid = fill_nan_grid_along_of(cf_grid, of_values, "cf_grid")
    isp_grid = fill_nan_grid_along_of(isp_grid, of_values, "isp_grid")
    ivac_grid = fill_nan_grid_along_of(ivac_grid, of_values, "ivac_grid")
    tc_grid = fill_nan_grid_along_of(tc_grid, of_values, "tc_grid")

    if not (
        np.all(np.isfinite(cstar_grid))
        and np.all(np.isfinite(cf_grid))
        and np.all(np.isfinite(isp_grid))
        and np.all(np.isfinite(ivac_grid))
        and np.all(np.isfinite(tc_grid))
    ):
        raise RuntimeError("CEA grid still contains invalid entries after attempting to fill missing values.")

    print("CEA grid completed successfully.")
    print("=================================================\n")

    return {
        "pc_values_bar": pc_values,
        "of_values": of_values,
        "cstar_grid": cstar_grid,
        "cf_grid": cf_grid,
        "isp_grid": isp_grid,
        "ivac_grid": ivac_grid,
        "tc_grid": tc_grid,
    }

def get_or_build_cea_grid(
    pc_min_bar: float,
    pc_max_bar: float,
    pc_step_bar: float,
    of_min: float,
    of_max: float,
    of_step: float,
) -> dict:
    cache_file = build_cache_filename(
        pc_min_bar=pc_min_bar,
        pc_max_bar=pc_max_bar,
        pc_step_bar=pc_step_bar,
        of_min=of_min,
        of_max=of_max,
        of_step=of_step,
        pe_bar=PE_BAR,
        eta_cstar=ETA_CSTAR,
        eta_cf=ETA_CF,
    )

    if USE_CEA_CACHE and cache_file.exists() and not REBUILD_CEA_CACHE:
        return load_cea_grid_cache(cache_file)

    cea_grid = generate_cea_grid(
        pc_min_bar=pc_min_bar,
        pc_max_bar=pc_max_bar,
        pc_step_bar=pc_step_bar,
        of_min=of_min,
        of_max=of_max,
        of_step=of_step,
        use_parallel=USE_PARALLEL_CEA_GRID_BUILD,
    )

    if USE_CEA_CACHE:
        save_cea_grid_cache(cea_grid, cache_file)

    return cea_grid

def bilinear_interpolate(x_vals: np.ndarray, y_vals: np.ndarray, grid: np.ndarray, x: float, y: float) -> float:
    if x < x_vals[0] or x > x_vals[-1]:
        raise ValueError(f"x={x:.6f} outside interpolation bounds [{x_vals[0]:.6f}, {x_vals[-1]:.6f}]")
    if y < y_vals[0] or y > y_vals[-1]:
        raise ValueError(f"y={y:.6f} outside interpolation bounds [{y_vals[0]:.6f}, {y_vals[-1]:.6f}]")

    i1 = int(np.searchsorted(x_vals, x, side="right"))
    j1 = int(np.searchsorted(y_vals, y, side="right"))

    if i1 == 0:
        i0 = i1 = 0
    elif i1 >= len(x_vals):
        i0 = i1 = len(x_vals) - 1
    else:
        i0 = i1 - 1

    if j1 == 0:
        j0 = j1 = 0
    elif j1 >= len(y_vals):
        j0 = j1 = len(y_vals) - 1
    else:
        j0 = j1 - 1

    x0 = x_vals[i0]
    x1 = x_vals[i1]
    y0 = y_vals[j0]
    y1 = y_vals[j1]

    q00 = grid[i0, j0]
    q01 = grid[i0, j1]
    q10 = grid[i1, j0]
    q11 = grid[i1, j1]

    if i0 == i1 and j0 == j1:
        return float(q00)
    if i0 == i1:
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        return float((1.0 - ty) * q00 + ty * q01)
    if j0 == j1:
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return float((1.0 - tx) * q00 + tx * q10)

    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    return float(
        (1.0 - tx) * (1.0 - ty) * q00
        + (1.0 - tx) * ty * q01
        + tx * (1.0 - ty) * q10
        + tx * ty * q11
    )

def interpolate_cea_2d(cea_grid: dict, pc_bar: float, of_ratio: float) -> dict:
    pc_vals = cea_grid["pc_values_bar"]
    of_vals = cea_grid["of_values"]

    return {
        "cstar": bilinear_interpolate(pc_vals, of_vals, cea_grid["cstar_grid"], pc_bar, of_ratio),
        "cf": bilinear_interpolate(pc_vals, of_vals, cea_grid["cf_grid"], pc_bar, of_ratio),
        "isp": bilinear_interpolate(pc_vals, of_vals, cea_grid["isp_grid"], pc_bar, of_ratio),
        "ivac": bilinear_interpolate(pc_vals, of_vals, cea_grid["ivac_grid"], pc_bar, of_ratio),
        "tc": bilinear_interpolate(pc_vals, of_vals, cea_grid["tc_grid"], pc_bar, of_ratio),
    }


# ============================================================
# HYBRID ENGINE MODEL
# ============================================================

def throat_area_from_design_point(
    design_thrust_n: float,
    design_pc_bar: float,
    design_of: float,
    cea_grid: dict,
) -> Tuple[float, dict]:
    cea_design = interpolate_cea_2d(cea_grid, design_pc_bar, design_of)

    if cea_design["cf"] <= 0.0:
        raise ValueError("Design Cf is nonphysical.")

    at = design_thrust_n / (cea_design["cf"] * bar_to_pa(design_pc_bar))
    return at, cea_design

def regression_and_fuel_flow(d_port_m: float, m_dot_o_kg_s: float) -> Dict[str, float]:
    gox = 4.0 * m_dot_o_kg_s / (np.pi * d_port_m * d_port_m)
    r_dot = A_REG * gox ** N_REG
    m_dot_f = RHO_FUEL * np.pi * d_port_m * L_GRAIN_M * r_dot

    if not np.isfinite(m_dot_f) or m_dot_f <= 0.0:
        raise ValueError("Fuel mass flow became nonphysical.")

    of_ratio = m_dot_o_kg_s / m_dot_f

    return {
        "gox": gox,
        "r_dot": r_dot,
        "r_dot_mm": 1000.0 * r_dot,
        "m_dot_f": m_dot_f,
        "of_ratio": of_ratio,
    }


# ============================================================
# COUPLED SOLVER
# ============================================================

def mdot_residual_for_coupled_point(
    mdot_guess_kg_s: float,
    angle_pct: float,
    temperature_C: float,
    d_port_m: float,
    feed_model: FeedSideModel,
    throat_area_m2: float,
    cea_grid: dict,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    try:
        feed_result = feed_model.evaluate_from_mdot(
            angle_pct=angle_pct,
            mdot_kg_s=mdot_guess_kg_s,
            temperature_C=temperature_C,
        )

        p2_bar = feed_result["p2_bar"]
        if not np.isfinite(p2_bar) or p2_bar <= 0.0:
            raise ValueError("Nonphysical P2")

        engine_result = solve_engine_from_p2(
            p2_bar=p2_bar,
            temperature_C=temperature_C,
            d_port_m=d_port_m,
            throat_area_m2=throat_area_m2,
            cea_grid=cea_grid,
            cda_inj_m2=CDA_INJ_M2,
        )

        residual = mdot_guess_kg_s - engine_result["m_dot_o"]
        return residual, feed_result, engine_result

    except Exception as e:
        return 1e9, {"error": str(e)}, {"error": str(e)}

def solve_engine_from_p2(
    p2_bar: float,
    temperature_C: float,
    d_port_m: float,
    throat_area_m2: float,
    cea_grid: dict,
    cda_inj_m2: float = CDA_INJ_M2,
) -> Dict[str, float]:
    props = NitrousProperties(fluid=FLUID)

    if not np.isfinite(p2_bar) or p2_bar <= 0.0:
        raise ValueError(f"Injector inlet pressure P2 is invalid: {p2_bar}")

    rho2 = props.density_from_PT(p2_bar, temperature_C)

    of_min_grid = cea_grid["of_values"][0]
    of_max_grid = cea_grid["of_values"][-1]
    pc_min_grid = cea_grid["pc_values_bar"][0]
    pc_max_grid = cea_grid["pc_values_bar"][-1]

    pc_lo = pc_min_grid
    pc_hi = min(pc_max_grid, p2_bar - 1e-4)

    if pc_hi <= pc_lo:
        raise ValueError(
            f"P2={p2_bar:.6f} bar is too low to support chamber-pressure solve within CEA grid bounds."
        )

    def engine_state_at_pc(pc_bar: float) -> Dict[str, float]:
        dP_inj_bar = p2_bar - pc_bar
        dP_inj_pa = bar_to_pa(dP_inj_bar)

        if dP_inj_pa <= 0.0:
            raise ValueError("Injector pressure drop became nonphysical.")

        m_dot_o = cda_inj_m2 * safe_sqrt(2.0 * rho2 * dP_inj_pa)

        if not np.isfinite(m_dot_o) or m_dot_o <= 0.0:
            raise ValueError("Injector predicted nonphysical oxidizer flow.")

        reg = regression_and_fuel_flow(d_port_m, m_dot_o)
        of_ratio = reg["of_ratio"]

        if of_ratio < of_min_grid or of_ratio > of_max_grid:
            raise ValueError(
                f"Computed O/F={of_ratio:.6f} is outside CEA grid bounds "
                f"[{of_min_grid:.6f}, {of_max_grid:.6f}]."
            )

        cea = interpolate_cea_2d(cea_grid, pc_bar, of_ratio)

        m_dot_total = m_dot_o + reg["m_dot_f"]
        pc_required_pa = m_dot_total * cea["cstar"] / throat_area_m2
        pc_required_bar = pa_to_bar(pc_required_pa)

        thrust = cea["cf"] * bar_to_pa(pc_bar) * throat_area_m2
        isp_eff = thrust / (m_dot_total * G0)

        return {
            "converged": False,
            "p2_bar": p2_bar,
            "pc_bar": pc_bar,
            "pc_required_bar": pc_required_bar,
            "pc_error_bar": pc_required_bar - pc_bar,
            "dP_inj_bar": dP_inj_bar,
            "rho2_kg_m3": rho2,
            "m_dot_o": m_dot_o,
            "gox": reg["gox"],
            "r_dot": reg["r_dot"],
            "r_dot_mm": reg["r_dot_mm"],
            "m_dot_f": reg["m_dot_f"],
            "m_dot_total": m_dot_total,
            "of_ratio": of_ratio,
            "cstar": cea["cstar"],
            "cf": cea["cf"],
            "isp": isp_eff,
            "ivac": cea["ivac"],
            "tc": cea["tc"],
            "thrust": thrust,
        }

    state_lo = engine_state_at_pc(pc_lo)
    state_hi = engine_state_at_pc(pc_hi)

    f_lo = state_lo["pc_error_bar"]
    f_hi = state_hi["pc_error_bar"]

    if f_lo * f_hi > 0:
        raise ValueError(
            f"Could not bracket engine chamber-pressure solution at P2={p2_bar:.6f} bar. "
            f"f_lo={f_lo:.6f}, f_hi={f_hi:.6f}"
        )

    best_state = None

    for i in range(PC_FIXED_POINT_MAX_ITER):
        pc_mid = 0.5 * (pc_lo + pc_hi)
        state_mid = engine_state_at_pc(pc_mid)
        f_mid = state_mid["pc_error_bar"]
        best_state = state_mid
        best_state["iterations_used"] = i + 1

        if abs(f_mid) < PC_FIXED_POINT_TOL_BAR:
            best_state["converged"] = True
            return best_state

        if f_lo * f_mid <= 0:
            pc_hi = pc_mid
            f_hi = f_mid
        else:
            pc_lo = pc_mid
            f_lo = f_mid

    if best_state is None:
        raise RuntimeError("Engine solver failed before producing any state.")

    raise ValueError(
        f"Engine chamber-pressure bisection did not converge after {PC_FIXED_POINT_MAX_ITER} iterations. "
        f"Last Pc={best_state['pc_bar']:.6f} bar, "
        f"Pc_required={best_state['pc_required_bar']:.6f} bar, "
        f"error={best_state['pc_error_bar']:.6f} bar."
    )

def solve_coupled_operating_point(
    angle_pct: float,
    temperature_C: float,
    d_port_m: float,
    feed_model: FeedSideModel,
    throat_area_m2: float,
    cea_grid: dict,
    mdot_min: float = M_DOT_MIN,
    mdot_max: float = M_DOT_MAX,
    tol: float = M_DOT_RES_TOL,
    max_iter: int = MAX_OUTER_ITER,
) -> Dict[str, Dict[str, float]]:
    res_lo, feed_lo, eng_lo = mdot_residual_for_coupled_point(
        mdot_min, angle_pct, temperature_C, d_port_m, feed_model, throat_area_m2, cea_grid
    )
    res_hi, feed_hi, eng_hi = mdot_residual_for_coupled_point(
        mdot_max, angle_pct, temperature_C, d_port_m, feed_model, throat_area_m2, cea_grid
    )

    expand_count = 0
    local_mdot_max = mdot_max
    while res_lo * res_hi > 0 and expand_count < 4 and local_mdot_max < 1.0:
        local_mdot_max *= 1.25
        res_hi, feed_hi, eng_hi = mdot_residual_for_coupled_point(
            local_mdot_max, angle_pct, temperature_C, d_port_m, feed_model, throat_area_m2, cea_grid
        )
        expand_count += 1

    if res_lo * res_hi > 0:
        return {
            "converged": False,
            "iterations_used": 0,
            "mdot_solution_kg_s": np.nan,
            "feed": {
                "error": (
                    f"Could not bracket coupled solution.\n"
                    f"LOW END:  mdot={mdot_min:.6f}, res={res_lo:.6e}, "
                    f"feed_err={feed_lo.get('error', 'none')}, "
                    f"eng_err={eng_lo.get('error', 'none')}\n"
                    f"HIGH END: mdot={local_mdot_max:.6f}, res={res_hi:.6e}, "
                    f"feed_err={feed_hi.get('error', 'none')}, "
                    f"eng_err={eng_hi.get('error', 'none')}"
                )
            },
            "engine": {"error": "No valid bracket found for outer solve."},
        }

    best_feed = None
    best_engine = None
    best_mdot = None

    lo = mdot_min
    hi = local_mdot_max

    for i in range(max_iter):
        mdot_mid = 0.5 * (lo + hi)
        res_mid, feed_mid, eng_mid = mdot_residual_for_coupled_point(
            mdot_mid, angle_pct, temperature_C, d_port_m, feed_model, throat_area_m2, cea_grid
        )

        best_feed = feed_mid
        best_engine = eng_mid
        best_mdot = mdot_mid

        if abs(res_mid) < tol:
            return {
                "converged": True,
                "iterations_used": i + 1,
                "mdot_solution_kg_s": best_mdot,
                "feed": best_feed,
                "engine": best_engine,
            }

        if res_lo * res_mid <= 0:
            hi = mdot_mid
            res_hi = res_mid
        else:
            lo = mdot_mid
            res_lo = res_mid

    return {
        "converged": False,
        "iterations_used": max_iter,
        "mdot_solution_kg_s": best_mdot,
        "feed": best_feed,
        "engine": best_engine,
    }


# ============================================================
# PRINTING / REPORT HELPERS
# ============================================================

def print_coupled_result(result: Dict[str, Dict[str, float]], temperature_C: float) -> None:
    feed = result["feed"]
    eng = result["engine"]

    print("\n==============================================================")
    print(f"Coupled solution converged:      {result['converged']}")
    print(f"Outer iterations used:           {result['iterations_used']}")

    if "angle_pct" not in feed:
        print("Coupled solver failed before a valid feed state was found.")
        print(f"Feed error:                      {feed.get('error', 'Unknown error')}")
        print(f"Engine error:                    {eng.get('error', 'Unknown error')}")
        print("==============================================================")
        return

    print(f"Valve angle:                     {feed['angle_pct']:.2f} %")
    print(f"Temperature:                     {temperature_C:.2f} C")
    print("")
    print("---- FEED SIDE ----")
    print(f"Tank pressure Pt:                {feed['p_tank_bar']:.4f} bar")
    print(f"Pressure before MTV P1:          {feed['p1_bar']:.4f} bar")
    print(f"Injector inlet pressure P2:      {feed['p2_bar']:.4f} bar")
    print(f"Upstream PFS dP:                 {feed['dP_pfs_bar']:.4f} bar")
    print(f"MTV dP:                          {feed['dP_mtv_bar']:.4f} bar")
    print(f"Feed mass flow:                  {feed['mdot_kg_s']:.6f} kg/s")
    print(f"Tank density:                    {feed['rho_tank_kg_m3']:.3f} kg/m^3")
    print(f"MTV upstream density:            {feed['rho1_kg_m3']:.3f} kg/m^3")
    print(f"Cv(theta):                       {feed['cv_theta']:.6f}")
    print(f"Cv ratio(theta):                 {feed['cv_ratio_theta']:.6f}")
    print(f"CdA_PFS:                         {feed['cda_pfs_m2']:.6e} m^2")
    print(f"CdA_MTV(theta):                  {feed['cda_mtv_theta_m2']:.6e} m^2")
    print(f"CdA_MTV,max:                     {feed['cda_mtv_max_m2']:.6e} m^2")
    print(f"N2O saturation pressure:         {feed['saturation_pressure_bar']:.4f} bar")
    print(f"Tank margin above Psat:          {feed['tank_margin_above_sat_bar']:.4f} bar")
    print("")
    print("---- INJECTOR / ENGINE SIDE ----")
    print(f"Injector Cd:                     {CD_INJ:.4f}")
    print(f"Injector area:                   {A_INJ_M2:.8e} m^2")
    print(f"Injector CdA:                    {CDA_INJ_M2:.8e} m^2")
    print(f"Injector inlet density:          {eng['rho2_kg_m3']:.3f} kg/m^3")
    print(f"Injector dP (P2 - Pc):           {eng['dP_inj_bar']:.4f} bar")
    print(f"Chamber pressure Pc:             {eng['pc_bar']:.4f} bar")
    print(f"Oxidizer mass flow:              {eng['m_dot_o']:.6f} kg/s")
    print(f"Fuel mass flow:                  {eng['m_dot_f']:.6f} kg/s")
    print(f"Total mass flow:                 {eng['m_dot_total']:.6f} kg/s")
    print(f"O/F ratio:                       {eng['of_ratio']:.4f}")
    print(f"Gox:                             {eng['gox']:.3f} kg/m^2-s")
    print(f"Regression rate:                 {eng['r_dot_mm']:.4f} mm/s")
    print(f"C*:                              {eng['cstar']:.3f} m/s")
    print(f"Cf:                              {eng['cf']:.5f}")
    print(f"Effective Isp:                   {eng['isp']:.3f} s")
    print(f"Ivac:                            {eng['ivac']:.3f} s")
    print(f"Chamber temperature:             {eng['tc']:.3f} K")
    print(f"Predicted thrust:                {eng['thrust']:.3f} N")
    print(f"Engine inner iterations used:    {eng['iterations_used']}")
    print(f"Required chamber pressure:       {eng['pc_required_bar']:.4f} bar")
    print(f"Chamber closure error:           {eng['pc_error_bar']:.6f} bar")
    print("==============================================================")

def print_time_marching_summary(history: Dict[str, list]) -> None:
    print("\n==============================================================")
    print("TIME-MARCHING SUMMARY")
    recorded_steps = len(history["time_s"])
    print(f"Recorded steps:                  {recorded_steps}")

    if recorded_steps == 0:
        print("No time-history data recorded.")
        print("==============================================================")
        return

    final_time = history["time_s"][-1]
    final_angle = history["angle_pct"][-1]
    final_d_port = history["d_port_m"][-1]

    print(f"Final time:                      {final_time:.4f} s")
    print(f"Final valve angle:               {final_angle:.4f} %")
    print(f"Final port diameter:             {1000.0 * final_d_port:.4f} mm")

    final_ok = bool(history["solver_converged"][-1]) if len(history["solver_converged"]) > 0 else False
    if final_ok:
        print(f"Final chamber pressure:          {history['pc_bar'][-1]:.4f} bar")
        print(f"Final oxidizer flow:             {history['m_dot_o'][-1]:.6f} kg/s")
        print(f"Final fuel flow:                 {history['m_dot_f'][-1]:.6f} kg/s")
        print(f"Final O/F:                       {history['of_ratio'][-1]:.4f}")
        print(f"Final regression rate:           {history['r_dot_mm_s'][-1]:.4f} mm/s")
        print(f"Final thrust:                    {history['thrust_N'][-1]:.4f} N")

        t = np.array(history["time_s"], dtype=float)
        F = np.array(history["thrust_N"], dtype=float)
        valid = np.isfinite(t) & np.isfinite(F)
        if np.count_nonzero(valid) >= 2:
            total_impulse = np.trapezoid(F[valid], t[valid])
            print(f"Approx. total impulse:           {total_impulse:.4f} N-s")
    else:
        print("Final state is invalid / solver failed.")

    print("==============================================================")


# ============================================================
# ENGINE STATE HELPERS
# ============================================================

def compute_engine_state_from_current_pc(
    p2_bar: float,
    pc_bar: float,
    temperature_C: float,
    d_port_m: float,
    throat_area_m2: float,
    cea_grid: dict,
    cda_inj_m2: float = CDA_INJ_M2,
) -> Dict[str, float]:
    props = NitrousProperties(fluid=FLUID)

    if not np.isfinite(p2_bar) or p2_bar <= 0.0:
        raise ValueError(f"Injector inlet pressure P2 is invalid: {p2_bar}")
    if not np.isfinite(pc_bar) or pc_bar <= 0.0:
        raise ValueError(f"Chamber pressure Pc is invalid: {pc_bar}")

    dP_inj_bar = p2_bar - pc_bar
    dP_inj_pa = bar_to_pa(dP_inj_bar)

    if dP_inj_pa <= 0.0:
        return {
            "p2_bar": p2_bar,
            "pc_bar": pc_bar,
            "pc_steady_bar": np.nan,
            "dP_inj_bar": dP_inj_bar,
            "rho2_kg_m3": np.nan,
            "m_dot_o": 0.0,
            "gox": 0.0,
            "r_dot": 0.0,
            "r_dot_mm": 0.0,
            "m_dot_f": 0.0,
            "m_dot_total": 0.0,
            "of_ratio": np.nan,
            "cstar": np.nan,
            "cf": np.nan,
            "isp": np.nan,
            "ivac": np.nan,
            "tc": np.nan,
            "thrust": 0.0,
        }

    rho2 = props.density_from_PT(p2_bar, temperature_C)
    m_dot_o = cda_inj_m2 * safe_sqrt(2.0 * rho2 * dP_inj_pa)

    if not np.isfinite(m_dot_o) or m_dot_o <= 0.0:
        raise ValueError("Injector predicted nonphysical oxidizer flow.")

    reg = regression_and_fuel_flow(d_port_m, m_dot_o)
    of_ratio = reg["of_ratio"]

    of_min_grid = cea_grid["of_values"][0]
    of_max_grid = cea_grid["of_values"][-1]
    if of_ratio < of_min_grid or of_ratio > of_max_grid:
        raise ValueError(
            f"Computed O/F={of_ratio:.6f} is outside CEA grid bounds "
            f"[{of_min_grid:.6f}, {of_max_grid:.6f}]."
        )

    cea = interpolate_cea_2d(cea_grid, pc_bar, of_ratio)

    m_dot_total = m_dot_o + reg["m_dot_f"]
    pc_steady_pa = m_dot_total * cea["cstar"] / throat_area_m2
    pc_steady_bar = pa_to_bar(pc_steady_pa)

    thrust = cea["cf"] * bar_to_pa(pc_bar) * throat_area_m2
    isp_eff = thrust / (m_dot_total * G0)

    return {
        "p2_bar": p2_bar,
        "pc_bar": pc_bar,
        "pc_steady_bar": pc_steady_bar,
        "dP_inj_bar": dP_inj_bar,
        "rho2_kg_m3": rho2,
        "m_dot_o": m_dot_o,
        "gox": reg["gox"],
        "r_dot": reg["r_dot"],
        "r_dot_mm": reg["r_dot_mm"],
        "m_dot_f": reg["m_dot_f"],
        "m_dot_total": m_dot_total,
        "of_ratio": of_ratio,
        "cstar": cea["cstar"],
        "cf": cea["cf"],
        "isp": isp_eff,
        "ivac": cea["ivac"],
        "tc": cea["tc"],
        "thrust": thrust,
    }

def compute_chamber_state_from_mdot(
    m_dot_o: float,
    pc_bar: float,
    d_port_m: float,
    throat_area_m2: float,
    cea_grid: dict,
) -> Dict[str, float]:
    if not np.isfinite(m_dot_o) or m_dot_o < 0.0:
        raise ValueError(f"Oxidizer flow is invalid: {m_dot_o}")
    if not np.isfinite(pc_bar) or pc_bar <= 0.0:
        raise ValueError(f"Chamber pressure Pc is invalid: {pc_bar}")

    if m_dot_o <= 0.0:
        return {
            "m_dot_o": 0.0,
            "gox": 0.0,
            "r_dot": 0.0,
            "r_dot_mm": 0.0,
            "m_dot_f": 0.0,
            "m_dot_total": 0.0,
            "of_ratio": np.nan,
            "cstar": np.nan,
            "cf": np.nan,
            "isp": np.nan,
            "ivac": np.nan,
            "tc": np.nan,
            "thrust": 0.0,
            "pc_steady_bar": np.nan,
        }

    reg = regression_and_fuel_flow(d_port_m, m_dot_o)
    of_ratio = reg["of_ratio"]

    of_min_grid = cea_grid["of_values"][0]
    of_max_grid = cea_grid["of_values"][-1]

    if of_ratio < of_min_grid or of_ratio > of_max_grid:
        raise ValueError(
            f"Computed O/F={of_ratio:.6f} is outside CEA grid bounds "
            f"[{of_min_grid:.6f}, {of_max_grid:.6f}]."
        )

    cea = interpolate_cea_2d(cea_grid, pc_bar, of_ratio)

    m_dot_total = m_dot_o + reg["m_dot_f"]
    pc_steady_pa = m_dot_total * cea["cstar"] / throat_area_m2
    pc_steady_bar = pa_to_bar(pc_steady_pa)

    thrust = cea["cf"] * bar_to_pa(pc_bar) * throat_area_m2
    isp_eff = thrust / (m_dot_total * G0)

    return {
        "m_dot_o": m_dot_o,
        "gox": reg["gox"],
        "r_dot": reg["r_dot"],
        "r_dot_mm": reg["r_dot_mm"],
        "m_dot_f": reg["m_dot_f"],
        "m_dot_total": m_dot_total,
        "of_ratio": of_ratio,
        "cstar": cea["cstar"],
        "cf": cea["cf"],
        "isp": isp_eff,
        "ivac": cea["ivac"],
        "tc": cea["tc"],
        "thrust": thrust,
        "pc_steady_bar": pc_steady_bar,
    }


# ============================================================
# CONTROLLER HELPERS
# ============================================================

@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    setpoint: float
    output_min: float
    output_max: float
    integral_min: float = -np.inf
    integral_max: float = np.inf
    rate_limit_pct_per_s: Optional[float] = None

    integral: float = 0.0
    prev_error: Optional[float] = None
    prev_output: Optional[float] = None

    def reset(self, initial_output: float = 0.0) -> None:
        self.integral = 0.0
        self.prev_error = None
        self.prev_output = initial_output

    def update(
        self,
        measurement: float,
        dt: float,
        setpoint: Optional[float] = None,
    ) -> Dict[str, float]:
        if dt <= 0.0:
            raise ValueError("PID dt must be positive.")

        if setpoint is not None:
            self.setpoint = setpoint

        error = self.setpoint - measurement

        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt

        # Compute unsaturated output using the current (pre-update) integral so that
        # the anti-windup check can reference it before deciding whether to accumulate.
        u_unsat = self.kp * error + self.ki * self.integral + self.kd * derivative

        integral_candidate = self.integral + error * dt
        integral_candidate = float(np.clip(integral_candidate, self.integral_min, self.integral_max))

        pushing_high = (u_unsat > self.output_max) and (error > 0.0)
        pushing_low = (u_unsat < self.output_min) and (error < 0.0)

        if not (pushing_high or pushing_low):
            self.integral = integral_candidate

        u_unsat = self.kp * error + self.ki * self.integral + self.kd * derivative
        u_cmd = float(np.clip(u_unsat, self.output_min, self.output_max))

        if self.rate_limit_pct_per_s is not None and self.prev_output is not None:
            max_delta = self.rate_limit_pct_per_s * dt
            u_cmd = float(np.clip(u_cmd, self.prev_output - max_delta, self.prev_output + max_delta))
            u_cmd = float(np.clip(u_cmd, self.output_min, self.output_max))

        self.prev_error = error
        self.prev_output = u_cmd

        return {
            "output": u_cmd,
            "error": error,
            "integral": self.integral,
            "derivative": derivative,
        }

def thrust_to_pc_target_bar(
    thrust_target_n: float,
    cf_est: float,
    throat_area_m2: float,
) -> float:
    if not np.isfinite(cf_est) or cf_est <= 0.0:
        raise ValueError(f"Invalid Cf estimate: {cf_est}")

    pc_target_pa = thrust_target_n / (cf_est * throat_area_m2)
    return pa_to_bar(pc_target_pa)

def feedforward_angle_from_pc_target(
    pc_target_bar: float,
    ff_map: Dict[str, np.ndarray],
) -> float:
    pc_vals = ff_map["pc_bar"]
    angle_vals = ff_map["angle_pct"]

    pc_target_clamped = float(np.clip(pc_target_bar, pc_vals[0], pc_vals[-1]))
    return float(np.interp(pc_target_clamped, pc_vals, angle_vals))

def build_pc_to_angle_feedforward_map(
    temperature_C: float,
    d_port_m: float,
    feed_model: FeedSideModel,
    throat_area_m2: float,
    cea_grid: dict,
    angle_start_pct: float = FF_SWEEP_ANGLE_START_PCT,
    angle_end_pct: float = FF_SWEEP_ANGLE_END_PCT,
    angle_step_pct: float = FF_SWEEP_ANGLE_STEP_PCT,
) -> Dict[str, np.ndarray]:
    cache_file = build_ff_cache_filename(
        temperature_C=temperature_C,
        d_port_m=d_port_m,
        tank_pressure_bar=feed_model.upstream.tank_pressure_bar,
        cda_pfs_m2=feed_model.upstream.cda_pfs_m2,
        cda_mtv_max_m2=feed_model.mtv.cda_mtv_max_m2,
        throat_area_m2=throat_area_m2,
        angle_start_pct=angle_start_pct,
        angle_end_pct=angle_end_pct,
        angle_step_pct=angle_step_pct,
    )

    if USE_FF_CACHE and cache_file.exists() and not REBUILD_FF_CACHE:
        return load_ff_map_cache(cache_file)

    sweep = run_steady_valve_sweep(
        angle_start_pct=angle_start_pct,
        angle_end_pct=angle_end_pct,
        angle_step_pct=angle_step_pct,
        temperature_C=temperature_C,
        d_port_fixed_m=d_port_m,
        feed_model=feed_model,
        throat_area_m2=throat_area_m2,
        cea_grid=cea_grid,
        mdot_min=M_DOT_MIN,
        mdot_max=M_DOT_MAX,
    )

    angle = np.array(sweep["angle_pct"], dtype=float)
    pc = np.array(sweep["pc_bar"], dtype=float)

    valid = np.isfinite(angle) & np.isfinite(pc) & np.array(sweep["solver_converged"], dtype=bool)

    if np.count_nonzero(valid) < 2:
        raise RuntimeError("Could not build feedforward map: not enough valid steady sweep points.")

    angle = angle[valid]
    pc = pc[valid]

    idx = np.argsort(pc)
    pc_sorted = pc[idx]
    angle_sorted = angle[idx]

    pc_unique, unique_idx = np.unique(pc_sorted, return_index=True)
    angle_unique = angle_sorted[unique_idx]

    if len(pc_unique) < 2:
        raise RuntimeError("Could not build feedforward map: Pc sweep is not sufficiently distinct.")

    ff_map = {
        "pc_bar": pc_unique,
        "angle_pct": angle_unique,
    }

    if USE_FF_CACHE:
        save_ff_map_cache(ff_map, cache_file)

    return ff_map

# ============================================================
# STEADY VALVE SWEEP
# ============================================================

def run_steady_valve_sweep(
    angle_start_pct: float,
    angle_end_pct: float,
    angle_step_pct: float,
    temperature_C: float,
    d_port_fixed_m: float,
    feed_model: FeedSideModel,
    throat_area_m2: float,
    cea_grid: dict,
    mdot_min: float = M_DOT_MIN,
    mdot_max: float = M_DOT_MAX,
) -> Dict[str, list]:
    angles = np.arange(angle_start_pct, angle_end_pct + 0.5 * angle_step_pct, angle_step_pct)

    sweep = {
        "angle_pct": [],
        "d_port_m": [],
        "p_tank_bar": [],
        "p1_bar": [],
        "p2_bar": [],
        "pc_bar": [],
        "dP_pfs_bar": [],
        "dP_mtv_bar": [],
        "dP_inj_bar": [],
        "m_dot_o": [],
        "m_dot_f": [],
        "m_dot_total": [],
        "of_ratio": [],
        "gox": [],
        "r_dot_m_s": [],
        "r_dot_mm_s": [],
        "cstar": [],
        "cf": [],
        "isp_s": [],
        "ivac_s": [],
        "tc_K": [],
        "thrust_N": [],
        "solver_converged": [],
        "solver_message": [],
    }

    for angle_pct in angles:
        result = solve_coupled_operating_point(
            angle_pct=angle_pct,
            temperature_C=temperature_C,
            d_port_m=d_port_fixed_m,
            feed_model=feed_model,
            throat_area_m2=throat_area_m2,
            cea_grid=cea_grid,
            mdot_min=mdot_min,
            mdot_max=mdot_max,
            tol=M_DOT_RES_TOL,
            max_iter=MAX_OUTER_ITER,
        )

        sweep["angle_pct"].append(angle_pct)
        sweep["d_port_m"].append(d_port_fixed_m)

        if not result["converged"]:
            for key in [
                "p_tank_bar", "p1_bar", "p2_bar", "pc_bar", "dP_pfs_bar",
                "dP_mtv_bar", "dP_inj_bar", "m_dot_o", "m_dot_f",
                "m_dot_total", "of_ratio", "gox", "r_dot_m_s", "r_dot_mm_s",
                "cstar", "cf", "isp_s", "ivac_s", "tc_K", "thrust_N"
            ]:
                sweep[key].append(np.nan)

            sweep["solver_converged"].append(False)
            sweep["solver_message"].append(
                result["feed"].get("error", result["engine"].get("error", "Unknown error"))
            )
            continue

        feed = result["feed"]
        eng = result["engine"]

        sweep["p_tank_bar"].append(feed["p_tank_bar"])
        sweep["p1_bar"].append(feed["p1_bar"])
        sweep["p2_bar"].append(feed["p2_bar"])
        sweep["pc_bar"].append(eng["pc_bar"])
        sweep["dP_pfs_bar"].append(feed["dP_pfs_bar"])
        sweep["dP_mtv_bar"].append(feed["dP_mtv_bar"])
        sweep["dP_inj_bar"].append(eng["dP_inj_bar"])

        sweep["m_dot_o"].append(eng["m_dot_o"])
        sweep["m_dot_f"].append(eng["m_dot_f"])
        sweep["m_dot_total"].append(eng["m_dot_total"])
        sweep["of_ratio"].append(eng["of_ratio"])
        sweep["gox"].append(eng["gox"])
        sweep["r_dot_m_s"].append(eng["r_dot"])
        sweep["r_dot_mm_s"].append(eng["r_dot_mm"])

        sweep["cstar"].append(eng["cstar"])
        sweep["cf"].append(eng["cf"])
        sweep["isp_s"].append(eng["isp"])
        sweep["ivac_s"].append(eng["ivac"])
        sweep["tc_K"].append(eng["tc"])
        sweep["thrust_N"].append(eng["thrust"])

        sweep["solver_converged"].append(True)
        sweep["solver_message"].append("OK")

    return sweep


# ============================================================
# DYNAMIC TIME MARCHING
# ============================================================

def run_time_marching_burn(
    burn_time_s: float,
    dt_s: float,
    temperature_C: float,
    d_port_initial_m: float,
    feed_model: FeedSideModel,
    throat_area_m2: float,
    cea_grid: dict,
    tau_pc_s: float = TAU_PC_S,
    tau_mdot_s: float = TAU_MDOT_S,
    d_port_max_m: Optional[float] = None,
    use_pid: bool = USE_PID,
    use_feedforward: bool = USE_FEEDFORWARD,
) -> Dict[str, list]:
    n_steps = int(np.floor(burn_time_s / dt_s)) + 1
    d_port_m = d_port_initial_m

    pid = PIDController(
        kp=PID_KP,
        ki=PID_KI,
        kd=PID_KD,
        setpoint=DESIGN_PC_BAR,
        output_min=PID_TRIM_MIN_PCT,
        output_max=PID_TRIM_MAX_PCT,
        integral_min=PID_INTEGRAL_MIN,
        integral_max=PID_INTEGRAL_MAX,
        rate_limit_pct_per_s=PID_RATE_LIMIT_PCT_PER_S,
    )

    angle0_pct = float(np.clip(PID_INITIAL_VALVE_PCT, PID_VALVE_MIN_PCT, PID_VALVE_MAX_PCT))

    init_result = solve_coupled_operating_point(
        angle_pct=angle0_pct,
        temperature_C=temperature_C,
        d_port_m=d_port_m,
        feed_model=feed_model,
        throat_area_m2=throat_area_m2,
        cea_grid=cea_grid,
        mdot_min=M_DOT_MIN,
        mdot_max=M_DOT_MAX,
        tol=M_DOT_RES_TOL,
        max_iter=MAX_OUTER_ITER,
    )

    if not init_result["converged"]:
        raise RuntimeError(
            "Could not initialize time-marching simulation from a valid steady operating point. "
            f"Feed error: {init_result['feed'].get('error', 'none')} | "
            f"Engine error: {init_result['engine'].get('error', 'none')}"
        )

    init_feed = init_result["feed"]
    init_eng = init_result["engine"]

    pc_bar = init_eng["pc_bar"]
    m_dot_o_actual = init_eng["m_dot_o"]
    cf_est = init_eng["cf"]

    pid.reset(initial_output=0.0)

    ff_map = build_pc_to_angle_feedforward_map(
        temperature_C=temperature_C,
        d_port_m=d_port_m,
        feed_model=feed_model,
        throat_area_m2=throat_area_m2,
        cea_grid=cea_grid,
    )
    ff_last_build_t_s = 0.0

    thrust_target0_n = thrust_target_schedule_n(0.0)
    pc_target0_bar = thrust_to_pc_target_bar(thrust_target0_n, cf_est, throat_area_m2)
    pc_target0_bar = float(np.clip(pc_target0_bar, CEA_PC_MIN_BAR, CEA_PC_MAX_BAR))

    angle_ff0_pct = feedforward_angle_from_pc_target(pc_target0_bar, ff_map)

    if use_pid:
        pid_state0 = pid.update(measurement=pc_bar, dt=dt_s, setpoint=pc_target0_bar)
        angle_trim0_pct = pid_state0["output"]
        pid_error0 = pid_state0["error"]
        pid_integral0 = pid_state0["integral"]
        pid_derivative0 = pid_state0["derivative"]
    else:
        angle_trim0_pct = 0.0
        pid_error0 = np.nan
        pid_integral0 = np.nan
        pid_derivative0 = np.nan

    angle0_pct = angle_ff0_pct + angle_trim0_pct
    angle0_pct = float(np.clip(angle0_pct, PID_VALVE_MIN_PCT, PID_VALVE_MAX_PCT))

    history = {
        "time_s": [],
        "thrust_target_n": [],
        "pc_target_bar": [],
        "angle_ff_pct": [],
        "angle_trim_pct": [],
        "angle_pct": [],
        "d_port_m": [],
        "p_tank_bar": [],
        "p1_bar": [],
        "p2_bar": [],
        "pc_bar": [],
        "pc_steady_bar": [],
        "dP_pfs_bar": [],
        "dP_mtv_bar": [],
        "dP_inj_bar": [],
        "m_dot_o": [],
        "m_dot_f": [],
        "m_dot_total": [],
        "of_ratio": [],
        "gox": [],
        "r_dot_m_s": [],
        "r_dot_mm_s": [],
        "cstar": [],
        "cf": [],
        "isp_s": [],
        "ivac_s": [],
        "tc_K": [],
        "thrust_N": [],
        "pid_error_bar": [],
        "pid_integral": [],
        "pid_derivative": [],
        "solver_converged": [],
        "solver_message": [],
    }

    history["time_s"].append(0.0)
    history["thrust_target_n"].append(thrust_target0_n)
    history["pc_target_bar"].append(pc_target0_bar)
    history["angle_ff_pct"].append(angle_ff0_pct)
    history["angle_trim_pct"].append(angle_trim0_pct)
    history["angle_pct"].append(angle0_pct)
    history["d_port_m"].append(d_port_m)

    history["p_tank_bar"].append(init_feed["p_tank_bar"])
    history["p1_bar"].append(init_feed["p1_bar"])
    history["p2_bar"].append(init_feed["p2_bar"])
    history["pc_bar"].append(init_eng["pc_bar"])
    history["pc_steady_bar"].append(init_eng["pc_bar"])
    history["dP_pfs_bar"].append(init_feed["dP_pfs_bar"])
    history["dP_mtv_bar"].append(init_feed["dP_mtv_bar"])
    history["dP_inj_bar"].append(init_feed["p2_bar"] - init_eng["pc_bar"])

    history["m_dot_o"].append(init_eng["m_dot_o"])
    history["m_dot_f"].append(init_eng["m_dot_f"])
    history["m_dot_total"].append(init_eng["m_dot_total"])
    history["of_ratio"].append(init_eng["of_ratio"])
    history["gox"].append(init_eng["gox"])
    history["r_dot_m_s"].append(init_eng["r_dot"])
    history["r_dot_mm_s"].append(init_eng["r_dot_mm"])

    history["cstar"].append(init_eng["cstar"])
    history["cf"].append(init_eng["cf"])
    history["isp_s"].append(init_eng["isp"])
    history["ivac_s"].append(init_eng["ivac"])
    history["tc_K"].append(init_eng["tc"])
    history["thrust_N"].append(init_eng["thrust"])

    history["pid_error_bar"].append(pid_error0)
    history["pid_integral"].append(pid_integral0)
    history["pid_derivative"].append(pid_derivative0)

    history["solver_converged"].append(True)
    history["solver_message"].append("Initialized from steady-state solution")

    current_angle_pct = angle0_pct

    start_wall = time.time()
    last_progress_print = start_wall

    for k in range(1, n_steps):
        t_s = k * dt_s

        if SHOW_PROGRESS:
            now = time.time()
            should_print = (
                (k % PROGRESS_PRINT_EVERY_N_STEPS == 0)
                and ((now - last_progress_print) >= PROGRESS_PRINT_MIN_INTERVAL_S or k == n_steps - 1)
            )
            if should_print:
                elapsed = now - start_wall
                progress = k / n_steps
                est_total = elapsed / max(progress, 1e-9)
                remaining = max(est_total - elapsed, 0.0)
                print(
                    f"[{k:6d}/{n_steps:6d}] "
                    f"{100.0 * progress:6.2f}% | "
                    f"Sim t = {t_s:7.3f} s | "
                    f"Elapsed = {elapsed:7.1f} s | "
                    f"ETA = {remaining:7.1f} s"
                )
                last_progress_print = now

        try:
            if use_feedforward and ((t_s - ff_last_build_t_s) >= FF_REBUILD_INTERVAL_S):
                ff_map = build_pc_to_angle_feedforward_map(
                    temperature_C=temperature_C,
                    d_port_m=d_port_m,
                    feed_model=feed_model,
                    throat_area_m2=throat_area_m2,
                    cea_grid=cea_grid,
                )
                ff_last_build_t_s = t_s

            thrust_target_n = thrust_target_schedule_n(t_s)

            if not np.isfinite(cf_est) or cf_est <= 0.0:
                cf_est = max(init_eng["cf"], 1e-6)

            pc_target_bar = thrust_to_pc_target_bar(thrust_target_n, cf_est, throat_area_m2)
            pc_target_bar = float(np.clip(pc_target_bar, CEA_PC_MIN_BAR, CEA_PC_MAX_BAR))

            if use_feedforward:
                angle_ff_pct = feedforward_angle_from_pc_target(pc_target_bar, ff_map)
            else:
                angle_ff_pct = PID_INITIAL_VALVE_PCT

            if use_pid:
                pid_state = pid.update(measurement=pc_bar, dt=dt_s, setpoint=pc_target_bar)
                angle_trim_pct = pid_state["output"]
                pid_error = pid_state["error"]
                pid_integral = pid_state["integral"]
                pid_derivative = pid_state["derivative"]
            else:
                angle_trim_pct = 0.0
                pid_error = np.nan
                pid_integral = np.nan
                pid_derivative = np.nan

            angle_cmd_pct = angle_ff_pct + angle_trim_pct
            angle_cmd_pct = float(np.clip(angle_cmd_pct, PID_VALVE_MIN_PCT, PID_VALVE_MAX_PCT))
            current_angle_pct = angle_cmd_pct

            feed = feed_model.evaluate_from_mdot(
                angle_pct=current_angle_pct,
                mdot_kg_s=m_dot_o_actual,
                temperature_C=temperature_C,
            )

            props = NitrousProperties(fluid=FLUID)
            rho2 = props.density_from_PT(feed["p2_bar"], temperature_C)

            dP_inj_bar_old = feed["p2_bar"] - pc_bar
            dP_inj_pa_old = bar_to_pa(dP_inj_bar_old)

            if dP_inj_pa_old <= 0.0:
                m_dot_o_target = 0.0
            else:
                m_dot_o_target = CDA_INJ_M2 * safe_sqrt(2.0 * rho2 * dP_inj_pa_old)

            alpha_m = dt_s / tau_mdot_s if tau_mdot_s > 0.0 else 1.0
            alpha_m = min(max(alpha_m, 0.0), 1.0)
            m_dot_o_actual = m_dot_o_actual + alpha_m * (m_dot_o_target - m_dot_o_actual)

            eng = compute_chamber_state_from_mdot(
                m_dot_o=m_dot_o_actual,
                pc_bar=pc_bar,
                d_port_m=d_port_m,
                throat_area_m2=throat_area_m2,
                cea_grid=cea_grid,
            )

            if not np.isfinite(eng["pc_steady_bar"]):
                raise ValueError("pc_steady_bar became non-finite.")

            alpha_pc = dt_s / tau_pc_s if tau_pc_s > 0.0 else 1.0
            alpha_pc = min(max(alpha_pc, 0.0), 1.0)
            pc_bar = pc_bar + alpha_pc * (eng["pc_steady_bar"] - pc_bar)

            eng = compute_chamber_state_from_mdot(
                m_dot_o=m_dot_o_actual,
                pc_bar=pc_bar,
                d_port_m=d_port_m,
                throat_area_m2=throat_area_m2,
                cea_grid=cea_grid,
            )

            dP_inj_bar = feed["p2_bar"] - pc_bar
            eng["dP_inj_bar"] = dP_inj_bar

            cf_est = eng["cf"]

            history["time_s"].append(t_s)
            history["thrust_target_n"].append(thrust_target_n)
            history["pc_target_bar"].append(pc_target_bar)
            history["angle_ff_pct"].append(angle_ff_pct)
            history["angle_trim_pct"].append(angle_trim_pct)
            history["angle_pct"].append(current_angle_pct)
            history["d_port_m"].append(d_port_m)

            history["p_tank_bar"].append(feed["p_tank_bar"])
            history["p1_bar"].append(feed["p1_bar"])
            history["p2_bar"].append(feed["p2_bar"])
            history["pc_bar"].append(pc_bar)
            history["pc_steady_bar"].append(eng["pc_steady_bar"])
            history["dP_pfs_bar"].append(feed["dP_pfs_bar"])
            history["dP_mtv_bar"].append(feed["dP_mtv_bar"])
            history["dP_inj_bar"].append(eng["dP_inj_bar"])

            history["m_dot_o"].append(eng["m_dot_o"])
            history["m_dot_f"].append(eng["m_dot_f"])
            history["m_dot_total"].append(eng["m_dot_total"])
            history["of_ratio"].append(eng["of_ratio"])
            history["gox"].append(eng["gox"])
            history["r_dot_m_s"].append(eng["r_dot"])
            history["r_dot_mm_s"].append(eng["r_dot_mm"])

            history["cstar"].append(eng["cstar"])
            history["cf"].append(eng["cf"])
            history["isp_s"].append(eng["isp"])
            history["ivac_s"].append(eng["ivac"])
            history["tc_K"].append(eng["tc"])
            history["thrust_N"].append(eng["thrust"])

            history["pid_error_bar"].append(pid_error)
            history["pid_integral"].append(pid_integral)
            history["pid_derivative"].append(pid_derivative)

            history["solver_converged"].append(True)
            history["solver_message"].append("OK")

            d_port_m = d_port_m + 2.0 * eng["r_dot"] * dt_s

            if d_port_max_m is not None and d_port_m >= d_port_max_m:
                print(f"\nTime marching stopped at t = {t_s:.3f} s")
                print(f"Reason: port diameter reached limit {d_port_max_m:.6f} m")
                break

        except Exception as e:
            history["time_s"].append(t_s)
            history["thrust_target_n"].append(np.nan)
            history["pc_target_bar"].append(np.nan)
            history["angle_ff_pct"].append(np.nan)
            history["angle_trim_pct"].append(np.nan)
            history["angle_pct"].append(current_angle_pct if 'current_angle_pct' in locals() else np.nan)
            history["d_port_m"].append(d_port_m)

            for key in [
                "p_tank_bar", "p1_bar", "p2_bar", "pc_bar", "pc_steady_bar",
                "dP_pfs_bar", "dP_mtv_bar", "dP_inj_bar",
                "m_dot_o", "m_dot_f", "m_dot_total", "of_ratio", "gox",
                "r_dot_m_s", "r_dot_mm_s", "cstar", "cf", "isp_s",
                "ivac_s", "tc_K", "thrust_N",
                "pid_error_bar", "pid_integral", "pid_derivative"
            ]:
                history[key].append(np.nan)

            history["solver_converged"].append(False)
            history["solver_message"].append(str(e))
            print(f"\nTime marching stopped at t = {t_s:.3f} s")
            print(f"Reason: {e}")

            if STOP_ON_SOLVER_FAILURE:
                break

    return history


# ============================================================
# PLOTTING
# ============================================================

def plot_time_history(history: Dict[str, list]) -> None:
    t = np.array(history["time_s"], dtype=float)

    def arr(key: str) -> np.ndarray:
        return np.array(history[key], dtype=float)

    fig, axs = plt.subplots(10, 1, figsize=(14, 22), sharex=True)
    fig.suptitle("Time-Marching Engine Behavior", fontsize=14)

    axs[0].plot(t, arr("thrust_N"), label="Thrust")
    axs[0].plot(t, arr("thrust_target_n"), "--", label="Thrust target")
    axs[0].set_ylabel("Thrust [N]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, arr("pc_bar"), label="Pc")
    axs[1].plot(t, arr("pc_target_bar"), "--", label="Pc target")
    axs[1].plot(t, arr("pc_steady_bar"), ":", label="Pc_steady")
    axs[1].plot(t, arr("p2_bar"), "-.", label="P2")
    axs[1].set_ylabel("Pressure [bar]")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(t, arr("dP_inj_bar"), label="ΔP_inj")
    axs[2].set_ylabel("ΔP inj [bar]")
    axs[2].grid(True)
    axs[2].legend()

    axs[3].plot(t, arr("of_ratio"), label="O/F")
    axs[3].set_ylabel("O/F")
    axs[3].grid(True)
    axs[3].legend()

    axs[4].plot(t, arr("m_dot_o"), label="Ox")
    axs[4].plot(t, arr("m_dot_f"), label="Fuel")
    axs[4].plot(t, arr("m_dot_total"), label="Total")
    axs[4].set_ylabel("ṁ [kg/s]")
    axs[4].grid(True)
    axs[4].legend()

    axs[5].plot(t, 1000.0 * arr("d_port_m"), label="Port")
    axs[5].set_ylabel("D_port [mm]")
    axs[5].grid(True)
    axs[5].legend()

    axs[6].plot(t, arr("angle_pct"), label="Valve actual (servo out)")
    if "angle_cmd_pct" in history:
        axs[6].plot(t, np.array(history["angle_cmd_pct"], dtype=float), "--", label="Valve cmd (FF+PID)")
    axs[6].plot(t, arr("angle_ff_pct"), ":", label="Valve FF")
    axs[6].plot(t, arr("angle_trim_pct"), "-.", label="Valve PID trim")
    axs[6].set_ylabel("Valve [%]")
    axs[6].grid(True)
    axs[6].legend()

    axs[7].plot(t, arr("pid_error_bar"), label="PID error")
    axs[7].set_ylabel("Error [bar]")
    axs[7].grid(True)
    axs[7].legend()

    axs[8].plot(t, arr("pid_integral"), label="Integral")
    axs[8].plot(t, arr("pid_derivative"), label="Derivative")
    axs[8].set_ylabel("PID states")
    axs[8].grid(True)
    axs[8].legend()

    axs[9].plot(t, arr("cf"), label="Cf")
    axs[9].set_ylabel("Cf [-]")
    axs[9].set_xlabel("Time [s]")
    axs[9].grid(True)
    axs[9].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plot_steady_valve_sweep(sweep: Dict[str, list]) -> None:
    angle = np.array(sweep["angle_pct"], dtype=float)

    def arr(key: str) -> np.ndarray:
        return np.array(sweep[key], dtype=float)

    fig, axs = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    fig.suptitle("Steady Valve Sweep", fontsize=14)

    axs[0].plot(angle, arr("thrust_N"), label="Thrust")
    axs[0].set_ylabel("Thrust [N]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(angle, arr("pc_bar"), label="Pc")
    axs[1].plot(angle, arr("p2_bar"), label="P2")
    axs[1].set_ylabel("Pressure [bar]")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(angle, arr("m_dot_o"), label="Ox")
    axs[2].plot(angle, arr("m_dot_f"), label="Fuel")
    axs[2].plot(angle, arr("m_dot_total"), label="Total")
    axs[2].set_ylabel("ṁ [kg/s]")
    axs[2].grid(True)
    axs[2].legend()

    axs[3].plot(angle, arr("of_ratio"), label="O/F")
    axs[3].set_ylabel("O/F")
    axs[3].grid(True)
    axs[3].legend()

    axs[4].plot(angle, arr("dP_inj_bar"), label="ΔP_inj")
    axs[4].set_ylabel("ΔP inj [bar]")
    axs[4].grid(True)
    axs[4].legend()

    axs[5].plot(angle, arr("cf"), label="Cf")
    axs[5].set_ylabel("Cf [-]")
    axs[5].set_xlabel("Valve angle [%]")
    axs[5].grid(True)
    axs[5].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


# ============================================================
# SIMULATION CONFIG
# ============================================================

@dataclass
class SimConfig:
    """
    All tuneable parameters for one MTV simulation run.

    Build once and pass to MTVSimulation.  PID gains, servo parameters, and
    feed-system characteristics can all be varied between runs while sharing
    the expensive CEA lookup grid.
    """
    # -- Feed system --
    pt_bar: float = PT_BAR_DEFAULT
    temperature_c: float = TEMPERATURE_C_DEFAULT
    cda_pfs_m2: float = CDA_PFS_M2_DEFAULT
    cda_mtv_max_m2: float = CDA_MTV_MAX_M2_DEFAULT

    # -- Grain geometry --
    d_port_initial_m: float = D_PORT_INITIAL_M
    d_port_max_m: Optional[float] = D_PORT_MAX_M

    # -- Time-march --
    burn_time_s: float = BURN_TIME_S
    dt_s: float = DT_S
    tau_pc_s: float = TAU_PC_S
    tau_mdot_s: float = TAU_MDOT_S

    # -- Servo actuator (P-controller driving the MTV valve) --
    servo_kp: float = 20.0                   # [(%/s) per % position error]
    servo_max_rate_pct_per_s: float = 120.0  # Hard slew-rate ceiling [%/s]

    # -- PID controller --
    pid_kp: float = PID_KP
    pid_ki: float = PID_KI
    pid_kd: float = PID_KD
    pid_trim_min_pct: float = PID_TRIM_MIN_PCT
    pid_trim_max_pct: float = PID_TRIM_MAX_PCT
    pid_valve_min_pct: float = PID_VALVE_MIN_PCT
    pid_valve_max_pct: float = PID_VALVE_MAX_PCT
    pid_initial_valve_pct: float = PID_INITIAL_VALVE_PCT
    pid_integral_min: float = PID_INTEGRAL_MIN
    pid_integral_max: float = PID_INTEGRAL_MAX
    pid_rate_limit_pct_per_s: float = PID_RATE_LIMIT_PCT_PER_S

    # -- Control modes --
    use_pid: bool = USE_PID
    use_feedforward: bool = USE_FEEDFORWARD

    # -- Thrust schedule callable: (t_s: float) -> thrust_N: float --
    thrust_schedule: Optional[Callable[[float], float]] = None

    def __post_init__(self) -> None:
        if self.thrust_schedule is None:
            self.thrust_schedule = thrust_target_schedule_n


# ============================================================
# SERVO ACTUATOR MODEL
# ============================================================

@dataclass
class ServoActuator:
    """
    P-controller model of the servo driving the MTV valve.

    The physical valve position tracks the commanded angle with proportional
    rate control and a hard slew-rate ceiling, capturing the mechanical lag
    of the servo mechanism.  actual_angle_pct is the state used by the feed
    model; angle_cmd_pct is what the flight controller requests.
    """
    kp: float                    # Proportional rate gain [(%/s) per % error]
    max_rate_pct_per_s: float    # Hard slew-rate ceiling [%/s]
    angle_min_pct: float         # Lower travel stop [%]
    angle_max_pct: float         # Upper travel stop [%]
    actual_angle_pct: float = 0.0

    def step(self, cmd_angle_pct: float, dt: float) -> float:
        """Advance servo state one timestep toward cmd; returns new actual angle."""
        error_pct = cmd_angle_pct - self.actual_angle_pct
        rate = float(np.clip(
            self.kp * error_pct,
            -self.max_rate_pct_per_s,
            self.max_rate_pct_per_s,
        ))
        self.actual_angle_pct = float(np.clip(
            self.actual_angle_pct + rate * dt,
            self.angle_min_pct,
            self.angle_max_pct,
        ))
        return self.actual_angle_pct


# ============================================================
# MTV SIMULATION CLASS
# ============================================================

class MTVSimulation:
    """
    Self-contained MTV hybrid-rocket closed-loop simulation.

    Pass a pre-built CEA grid (expensive to compute; shared across runs) and a
    SimConfig.  Call run() to execute the time-march and retrieve the history
    dict, then plot() to visualise results.

    Example — PID gain sweep
    ------------------------
    cea_grid       = get_or_build_cea_grid(...)
    throat_area_m2, _ = throat_area_from_design_point(...)

    for kp in [0.5, 1.0, 1.5, 2.0]:
        cfg     = SimConfig(pid_kp=kp, pid_ki=0.3)
        sim     = MTVSimulation(cfg, cea_grid, throat_area_m2)
        history = sim.run()
        sim.plot(history)
    """

    def __init__(
        self,
        config: SimConfig,
        cea_grid: dict,
        throat_area_m2: float,
    ) -> None:
        self.config = config
        self.cea_grid = cea_grid
        self.throat_area_m2 = throat_area_m2
        self.feed_model = FeedSideModel(
            tank_pressure_bar=config.pt_bar,
            cda_pfs_m2=config.cda_pfs_m2,
            cda_mtv_max_m2=config.cda_mtv_max_m2,
            angle_data_pct=ANGLE_DATA_PCT_DEFAULT,
            cv_data=CV_DATA_DEFAULT,
            fluid=FLUID,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, list]:
        """Execute the time-marching simulation; return the full history dict."""
        return self._run_time_march()

    def plot(self, history: Dict[str, list]) -> None:
        """Convenience wrapper around plot_time_history."""
        plot_time_history(history)

    # ------------------------------------------------------------------
    # Internal time-march
    # ------------------------------------------------------------------

    def _run_time_march(self) -> Dict[str, list]:  # noqa: C901
        cfg = self.config
        n_steps = int(np.floor(cfg.burn_time_s / cfg.dt_s)) + 1
        d_port_m = cfg.d_port_initial_m

        pid = PIDController(
            kp=cfg.pid_kp,
            ki=cfg.pid_ki,
            kd=cfg.pid_kd,
            setpoint=DESIGN_PC_BAR,
            output_min=cfg.pid_trim_min_pct,
            output_max=cfg.pid_trim_max_pct,
            integral_min=cfg.pid_integral_min,
            integral_max=cfg.pid_integral_max,
            rate_limit_pct_per_s=cfg.pid_rate_limit_pct_per_s,
        )

        servo = ServoActuator(
            kp=cfg.servo_kp,
            max_rate_pct_per_s=cfg.servo_max_rate_pct_per_s,
            angle_min_pct=cfg.pid_valve_min_pct,
            angle_max_pct=cfg.pid_valve_max_pct,
            actual_angle_pct=float(
                np.clip(cfg.pid_initial_valve_pct, cfg.pid_valve_min_pct, cfg.pid_valve_max_pct)
            ),
        )

        init_result = solve_coupled_operating_point(
            angle_pct=servo.actual_angle_pct,
            temperature_C=cfg.temperature_c,
            d_port_m=d_port_m,
            feed_model=self.feed_model,
            throat_area_m2=self.throat_area_m2,
            cea_grid=self.cea_grid,
            mdot_min=M_DOT_MIN,
            mdot_max=M_DOT_MAX,
            tol=M_DOT_RES_TOL,
            max_iter=MAX_OUTER_ITER,
        )

        if not init_result["converged"]:
            raise RuntimeError(
                "Could not initialise simulation from a valid steady operating point. "
                f"Feed error: {init_result['feed'].get('error', 'none')} | "
                f"Engine error: {init_result['engine'].get('error', 'none')}"
            )

        init_feed = init_result["feed"]
        init_eng = init_result["engine"]
        pc_bar = init_eng["pc_bar"]
        m_dot_o_actual = init_eng["m_dot_o"]
        cf_est = init_eng["cf"]

        pid.reset(initial_output=0.0)

        ff_map = build_pc_to_angle_feedforward_map(
            temperature_C=cfg.temperature_c,
            d_port_m=d_port_m,
            feed_model=self.feed_model,
            throat_area_m2=self.throat_area_m2,
            cea_grid=self.cea_grid,
        )
        ff_last_build_t_s = 0.0

        # ---- t = 0 controller step ----
        thrust_target0_n = cfg.thrust_schedule(0.0)
        pc_target0_bar = float(np.clip(
            thrust_to_pc_target_bar(thrust_target0_n, cf_est, self.throat_area_m2),
            CEA_PC_MIN_BAR, CEA_PC_MAX_BAR,
        ))
        angle_ff0_pct = feedforward_angle_from_pc_target(pc_target0_bar, ff_map)

        if cfg.use_pid:
            pid_state0 = pid.update(measurement=pc_bar, dt=cfg.dt_s, setpoint=pc_target0_bar)
            angle_trim0_pct = pid_state0["output"]
            pid_error0 = pid_state0["error"]
            pid_integral0 = pid_state0["integral"]
            pid_derivative0 = pid_state0["derivative"]
        else:
            angle_trim0_pct = 0.0
            pid_error0 = pid_integral0 = pid_derivative0 = np.nan

        angle_cmd0_pct = float(np.clip(
            angle_ff0_pct + angle_trim0_pct, cfg.pid_valve_min_pct, cfg.pid_valve_max_pct
        ))
        actual_angle0_pct = servo.step(angle_cmd0_pct, cfg.dt_s)

        history: Dict[str, list] = {
            "time_s": [],
            "thrust_target_n": [],
            "pc_target_bar": [],
            "angle_ff_pct": [],
            "angle_trim_pct": [],
            "angle_cmd_pct": [],    # total command from FF + PID, before servo
            "angle_pct": [],        # actual valve position after servo dynamics
            "d_port_m": [],
            "p_tank_bar": [], "p1_bar": [], "p2_bar": [],
            "pc_bar": [], "pc_steady_bar": [],
            "dP_pfs_bar": [], "dP_mtv_bar": [], "dP_inj_bar": [],
            "m_dot_o": [], "m_dot_f": [], "m_dot_total": [],
            "of_ratio": [], "gox": [], "r_dot_m_s": [], "r_dot_mm_s": [],
            "cstar": [], "cf": [], "isp_s": [], "ivac_s": [], "tc_K": [],
            "thrust_N": [],
            "pid_error_bar": [], "pid_integral": [], "pid_derivative": [],
            "solver_converged": [], "solver_message": [],
        }

        # -- seed step 0 --
        history["time_s"].append(0.0)
        history["thrust_target_n"].append(thrust_target0_n)
        history["pc_target_bar"].append(pc_target0_bar)
        history["angle_ff_pct"].append(angle_ff0_pct)
        history["angle_trim_pct"].append(angle_trim0_pct)
        history["angle_cmd_pct"].append(angle_cmd0_pct)
        history["angle_pct"].append(actual_angle0_pct)
        history["d_port_m"].append(d_port_m)
        history["p_tank_bar"].append(init_feed["p_tank_bar"])
        history["p1_bar"].append(init_feed["p1_bar"])
        history["p2_bar"].append(init_feed["p2_bar"])
        history["pc_bar"].append(pc_bar)
        history["pc_steady_bar"].append(pc_bar)
        history["dP_pfs_bar"].append(init_feed["dP_pfs_bar"])
        history["dP_mtv_bar"].append(init_feed["dP_mtv_bar"])
        history["dP_inj_bar"].append(init_feed["p2_bar"] - pc_bar)
        history["m_dot_o"].append(init_eng["m_dot_o"])
        history["m_dot_f"].append(init_eng["m_dot_f"])
        history["m_dot_total"].append(init_eng["m_dot_total"])
        history["of_ratio"].append(init_eng["of_ratio"])
        history["gox"].append(init_eng["gox"])
        history["r_dot_m_s"].append(init_eng["r_dot"])
        history["r_dot_mm_s"].append(init_eng["r_dot_mm"])
        history["cstar"].append(init_eng["cstar"])
        history["cf"].append(init_eng["cf"])
        history["isp_s"].append(init_eng["isp"])
        history["ivac_s"].append(init_eng["ivac"])
        history["tc_K"].append(init_eng["tc"])
        history["thrust_N"].append(init_eng["thrust"])
        history["pid_error_bar"].append(pid_error0)
        history["pid_integral"].append(pid_integral0)
        history["pid_derivative"].append(pid_derivative0)
        history["solver_converged"].append(True)
        history["solver_message"].append("Initialized from steady-state solution")

        start_wall = time.time()
        last_progress_print = start_wall

        for k in range(1, n_steps):
            t_s = k * cfg.dt_s

            if SHOW_PROGRESS:
                now = time.time()
                should_print = (
                    (k % PROGRESS_PRINT_EVERY_N_STEPS == 0)
                    and (
                        (now - last_progress_print) >= PROGRESS_PRINT_MIN_INTERVAL_S
                        or k == n_steps - 1
                    )
                )
                if should_print:
                    elapsed = now - start_wall
                    progress = k / n_steps
                    est_total = elapsed / max(progress, 1e-9)
                    print(
                        f"[{k:6d}/{n_steps:6d}] "
                        f"{100.0 * progress:6.2f}% | "
                        f"Sim t = {t_s:7.3f} s | "
                        f"Elapsed = {elapsed:7.1f} s | "
                        f"ETA = {max(est_total - elapsed, 0.0):7.1f} s"
                    )
                    last_progress_print = now

            try:
                if cfg.use_feedforward and (t_s - ff_last_build_t_s) >= FF_REBUILD_INTERVAL_S:
                    ff_map = build_pc_to_angle_feedforward_map(
                        temperature_C=cfg.temperature_c,
                        d_port_m=d_port_m,
                        feed_model=self.feed_model,
                        throat_area_m2=self.throat_area_m2,
                        cea_grid=self.cea_grid,
                    )
                    ff_last_build_t_s = t_s

                thrust_target_n = cfg.thrust_schedule(t_s)

                if not np.isfinite(cf_est) or cf_est <= 0.0:
                    cf_est = max(init_eng["cf"], 1e-6)

                pc_target_bar = float(np.clip(
                    thrust_to_pc_target_bar(thrust_target_n, cf_est, self.throat_area_m2),
                    CEA_PC_MIN_BAR, CEA_PC_MAX_BAR,
                ))

                if cfg.use_feedforward:
                    angle_ff_pct = feedforward_angle_from_pc_target(pc_target_bar, ff_map)
                else:
                    angle_ff_pct = cfg.pid_initial_valve_pct

                if cfg.use_pid:
                    pid_state = pid.update(measurement=pc_bar, dt=cfg.dt_s, setpoint=pc_target_bar)
                    angle_trim_pct = pid_state["output"]
                    pid_error = pid_state["error"]
                    pid_integral = pid_state["integral"]
                    pid_derivative = pid_state["derivative"]
                else:
                    angle_trim_pct = 0.0
                    pid_error = pid_integral = pid_derivative = np.nan

                angle_cmd_pct = float(np.clip(
                    angle_ff_pct + angle_trim_pct, cfg.pid_valve_min_pct, cfg.pid_valve_max_pct
                ))
                # Servo P-controller: physical valve position lags the command
                actual_angle_pct = servo.step(angle_cmd_pct, cfg.dt_s)

                feed = self.feed_model.evaluate_from_mdot(
                    angle_pct=actual_angle_pct,
                    mdot_kg_s=m_dot_o_actual,
                    temperature_C=cfg.temperature_c,
                )

                props = NitrousProperties(fluid=FLUID)
                rho2 = props.density_from_PT(feed["p2_bar"], cfg.temperature_c)
                dP_inj_pa_old = bar_to_pa(max(feed["p2_bar"] - pc_bar, 0.0))
                m_dot_o_target = (
                    CDA_INJ_M2 * safe_sqrt(2.0 * rho2 * dP_inj_pa_old)
                    if dP_inj_pa_old > 0.0 else 0.0
                )

                alpha_m = min(max(cfg.dt_s / cfg.tau_mdot_s if cfg.tau_mdot_s > 0.0 else 1.0, 0.0), 1.0)
                m_dot_o_actual += alpha_m * (m_dot_o_target - m_dot_o_actual)

                eng = compute_chamber_state_from_mdot(
                    m_dot_o=m_dot_o_actual,
                    pc_bar=pc_bar,
                    d_port_m=d_port_m,
                    throat_area_m2=self.throat_area_m2,
                    cea_grid=self.cea_grid,
                )

                if not np.isfinite(eng["pc_steady_bar"]):
                    raise ValueError("pc_steady_bar became non-finite.")

                alpha_pc = min(max(cfg.dt_s / cfg.tau_pc_s if cfg.tau_pc_s > 0.0 else 1.0, 0.0), 1.0)
                pc_bar += alpha_pc * (eng["pc_steady_bar"] - pc_bar)

                eng = compute_chamber_state_from_mdot(
                    m_dot_o=m_dot_o_actual,
                    pc_bar=pc_bar,
                    d_port_m=d_port_m,
                    throat_area_m2=self.throat_area_m2,
                    cea_grid=self.cea_grid,
                )
                eng["dP_inj_bar"] = feed["p2_bar"] - pc_bar
                cf_est = eng["cf"]

                history["time_s"].append(t_s)
                history["thrust_target_n"].append(thrust_target_n)
                history["pc_target_bar"].append(pc_target_bar)
                history["angle_ff_pct"].append(angle_ff_pct)
                history["angle_trim_pct"].append(angle_trim_pct)
                history["angle_cmd_pct"].append(angle_cmd_pct)
                history["angle_pct"].append(actual_angle_pct)
                history["d_port_m"].append(d_port_m)
                history["p_tank_bar"].append(feed["p_tank_bar"])
                history["p1_bar"].append(feed["p1_bar"])
                history["p2_bar"].append(feed["p2_bar"])
                history["pc_bar"].append(pc_bar)
                history["pc_steady_bar"].append(eng["pc_steady_bar"])
                history["dP_pfs_bar"].append(feed["dP_pfs_bar"])
                history["dP_mtv_bar"].append(feed["dP_mtv_bar"])
                history["dP_inj_bar"].append(eng["dP_inj_bar"])
                history["m_dot_o"].append(eng["m_dot_o"])
                history["m_dot_f"].append(eng["m_dot_f"])
                history["m_dot_total"].append(eng["m_dot_total"])
                history["of_ratio"].append(eng["of_ratio"])
                history["gox"].append(eng["gox"])
                history["r_dot_m_s"].append(eng["r_dot"])
                history["r_dot_mm_s"].append(eng["r_dot_mm"])
                history["cstar"].append(eng["cstar"])
                history["cf"].append(eng["cf"])
                history["isp_s"].append(eng["isp"])
                history["ivac_s"].append(eng["ivac"])
                history["tc_K"].append(eng["tc"])
                history["thrust_N"].append(eng["thrust"])
                history["pid_error_bar"].append(pid_error)
                history["pid_integral"].append(pid_integral)
                history["pid_derivative"].append(pid_derivative)
                history["solver_converged"].append(True)
                history["solver_message"].append("OK")

                d_port_m += 2.0 * eng["r_dot"] * cfg.dt_s

                if cfg.d_port_max_m is not None and d_port_m >= cfg.d_port_max_m:
                    print(f"\nTime marching stopped at t = {t_s:.3f} s")
                    print(f"Reason: port diameter reached limit {cfg.d_port_max_m:.6f} m")
                    break

            except Exception as e:
                history["time_s"].append(t_s)
                history["thrust_target_n"].append(np.nan)
                history["pc_target_bar"].append(np.nan)
                history["angle_ff_pct"].append(np.nan)
                history["angle_trim_pct"].append(np.nan)
                history["angle_cmd_pct"].append(np.nan)
                history["angle_pct"].append(servo.actual_angle_pct)
                history["d_port_m"].append(d_port_m)
                for key in [
                    "p_tank_bar", "p1_bar", "p2_bar", "pc_bar", "pc_steady_bar",
                    "dP_pfs_bar", "dP_mtv_bar", "dP_inj_bar",
                    "m_dot_o", "m_dot_f", "m_dot_total", "of_ratio", "gox",
                    "r_dot_m_s", "r_dot_mm_s", "cstar", "cf", "isp_s",
                    "ivac_s", "tc_K", "thrust_N",
                    "pid_error_bar", "pid_integral", "pid_derivative",
                ]:
                    history[key].append(np.nan)
                history["solver_converged"].append(False)
                history["solver_message"].append(str(e))
                print(f"\nTime marching stopped at t = {t_s:.3f} s")
                print(f"Reason: {e}")
                if STOP_ON_SOLVER_FAILURE:
                    break

        return history


# This module is intended to be imported, not run directly.
# Use pid_tuning_runner.py to execute single runs, parameter sweeps, and plots.