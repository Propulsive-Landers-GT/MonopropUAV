import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


FLIGHT_PLANS = (
    "direct_descent",
    "direct_limited_descent",
    "offset_descent",
    "offset_limited_descent",
)
RUN_TYPE_MAP = (
    ("very_coarse", "extreme_short"),
    ("coarse", "ultra_short"),
    ("fine", "short"),
    ("very_fine", "long"),
)
METHODS = ("zoh", "cgl")
METHOD_ORDER = {"zoh": 0, "cgl": 1}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compute max absolute thrust rate-of-change from all trajectory runs for "
            "the selected resolution/run-type buckets and render a summary table."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Root directory to scan (default: {default_root})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_root / "max_thrust_rate_change.csv",
        help=f"Output CSV path (default: {default_root / 'max_thrust_rate_change.csv'})",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=default_root / "figures" / "tables" / "max_thrust_rate_change_table.png",
        help=(
            "Output table image path "
            f"(default: {default_root / 'figures' / 'tables' / 'max_thrust_rate_change_table.png'})"
        ),
    )
    return parser.parse_args()


def parse_float(raw_value: str) -> float | None:
    text = (raw_value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_run_id(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.stem)
    if not match:
        return -1
    return int(match.group(1))


def humanize(text: str) -> str:
    return " ".join(part.capitalize() for part in text.split("_"))


def extract_run_name_from_trajectory(path: Path) -> str:
    stem = path.stem
    if stem.startswith("trajectory_"):
        return stem[len("trajectory_") :]
    return stem


def read_time_of_flight_by_run(run_dir: Path) -> dict[str, float]:
    solve_metrics_path = run_dir / "solve_metrics.csv"
    if not solve_metrics_path.exists() or not solve_metrics_path.is_file():
        return {}

    with solve_metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        if "run_name" not in reader.fieldnames or "time_of_flight_s" not in reader.fieldnames:
            return {}

        pass_priority = {"total": 0, "fine": 1, "coarse": 2}
        selected: dict[str, tuple[int, float]] = {}
        has_pass_column = "pass" in reader.fieldnames

        for row in reader:
            run_name = (row.get("run_name") or "").strip()
            if not run_name:
                continue
            tof = parse_float(row.get("time_of_flight_s", ""))
            if tof is None or tof <= 0:
                continue

            pass_name = (row.get("pass") or "").strip().lower() if has_pass_column else ""
            priority = pass_priority.get(pass_name, 10)
            current = selected.get(run_name)
            if current is None or priority < current[0]:
                selected[run_name] = (priority, tof)

    return {run_name: tof for run_name, (_, tof) in selected.items()}


def compute_max_abs_dthrust_dt(path: Path, time_of_flight_s: float | None) -> tuple[float | None, int]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        required = {"t", "thrust_force"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        t_values: list[float] = []
        thrust_values: list[float] = []

        for row in reader:
            t_value = parse_float(row.get("t", ""))
            thrust_value = parse_float(row.get("thrust_force", ""))
            if t_value is None or thrust_value is None:
                continue

            t_values.append(t_value)
            thrust_values.append(thrust_value)

    parsed_rows = len(t_values)
    if parsed_rows < 2:
        return None, parsed_rows

    raw_time_span = t_values[-1] - t_values[0]
    time_scale = 1.0
    if time_of_flight_s is not None and time_of_flight_s > 0 and raw_time_span > 0:
        # Convert index-like timeline to real seconds while keeping compatibility
        # with future CSVs that might already use real-time t.
        time_scale = time_of_flight_s / raw_time_span

    max_rate: float | None = None
    for idx in range(1, parsed_rows):
        dt_raw = t_values[idx] - t_values[idx - 1]
        dt = dt_raw * time_scale
        if dt <= 0:
            continue
        dthrust = thrust_values[idx] - thrust_values[idx - 1]
        rate = abs(dthrust / dt)
        if max_rate is None or rate > max_rate:
            max_rate = rate

    return max_rate, parsed_rows


def collect_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    resolution_order = {name: idx for idx, (name, _) in enumerate(RUN_TYPE_MAP)}
    flight_order = {name: idx for idx, name in enumerate(FLIGHT_PLANS)}
    tof_cache: dict[Path, dict[str, float]] = {}

    for flight_plan in FLIGHT_PLANS:
        for resolution_label, raw_run_type in RUN_TYPE_MAP:
            for method in METHODS:
                run_dir = root / flight_plan / raw_run_type / method
                if not run_dir.exists() or not run_dir.is_dir():
                    continue

                if run_dir not in tof_cache:
                    tof_cache[run_dir] = read_time_of_flight_by_run(run_dir)
                time_of_flight_by_run = tof_cache[run_dir]

                for trajectory_csv in sorted(run_dir.glob("trajectory_*.csv")):
                    run_name = extract_run_name_from_trajectory(trajectory_csv)
                    time_of_flight_s = time_of_flight_by_run.get(run_name)
                    max_rate, parsed_rows = compute_max_abs_dthrust_dt(
                        trajectory_csv, time_of_flight_s
                    )
                    if max_rate is None:
                        continue
                    run_id = parse_run_id(trajectory_csv)
                    rows.append(
                        {
                            "flight_plan": humanize(flight_plan),
                            "resolution": resolution_label,
                            "run_type": raw_run_type,
                            "method": method,
                            "run_id": str(run_id if run_id >= 0 else ""),
                            "trajectory_csv": str(trajectory_csv.relative_to(root)),
                            "num_samples": str(parsed_rows),
                            "time_of_flight_s": f"{time_of_flight_s:.6f}" if time_of_flight_s else "",
                            "max_abs_dthrust_dt_n_per_s": f"{max_rate:.9f}",
                            # Legacy compatibility for existing consumers.
                            "max_abs_dthrust_dt": f"{max_rate:.9f}",
                        }
                    )

    rows.sort(
        key=lambda row: (
            flight_order.get(row["flight_plan"].replace(" ", "_").lower(), len(flight_order)),
            resolution_order.get(row["resolution"], len(resolution_order)),
            METHOD_ORDER.get(row["method"], len(METHOD_ORDER)),
            int(row["run_id"]) if row["run_id"].isdigit() else 10**9,
            row["trajectory_csv"],
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "flight_plan",
                "resolution",
                "run_type",
                "method",
                "run_id",
                "num_samples",
                "time_of_flight_s",
                "max_abs_dthrust_dt_n_per_s",
                "max_abs_dthrust_dt",
                "trajectory_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def render_table_image(rows: list[dict[str, str]], output_path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 9.5,
            "axes.titlesize": 13,
        }
    )

    height = max(3.0, 1.15 + 0.34 * len(rows))
    fig, ax = plt.subplots(figsize=(11.8, height))
    ax.axis("off")

    cell_rows = []
    for row in rows:
        cell_rows.append(
            [
                row["flight_plan"],
                row["resolution"].replace("_", " ").title(),
                row["run_type"],
                row["method"].upper(),
                row["run_id"] or "N/A",
                f"{float(row['max_abs_dthrust_dt_n_per_s']):.6f}",
            ]
        )

    table = ax.table(
        cellText=cell_rows,
        colLabels=[
            "Flight Plan",
            "Resolution",
            "Raw Run Type",
            "Method",
            "Run ID",
            "Max |dT/dt| (N/s)",
        ],
        colWidths=[0.25, 0.13, 0.18, 0.10, 0.10, 0.14],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.28)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#2E2E2E")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#EAEFF7")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#FFFFFF" if row_idx % 2 == 1 else "#F7F7F7")
        if col_idx == 0 and row_idx > 0:
            cell.get_text().set_ha("left")

    ax.set_title("Maximum Thrust Rate-of-Change by Run (Real Time)", pad=10, fontweight="semibold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_csv = args.output_csv.resolve()
    output_table = args.output_table.resolve()

    rows = collect_rows(root)
    if not rows:
        raise RuntimeError(f"No trajectory rows found under {root}")

    write_csv(output_csv, rows)
    render_table_image(rows, output_table)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved table: {output_table}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
