import argparse
import csv
import math
import re
import sys
from pathlib import Path


METHOD_ORDER = {"zoh": 0, "cgl": 1}
RUN_TYPE_REMAP = {
    "extreme_short": "very_coarse",
    "ultra_short": "coarse",
    "very_short": "less_coarse",
    "short": "fine",
    "med": "less_fine",
    "long": "very_fine",
}
RUN_TYPE_ORDER = {
    "very_coarse": 0,
    "coarse": 1,
    "less_coarse": 2,
    "less_fine": 3,
    "fine": 4,
    "very_fine": 5,
    "extreme_long": 6,
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Collect all group_comparison_results.csv files into a clean graphing CSV "
            "with columns: group_name, run_type, method, rmse_xyz, rmse_sigma, rmse_uxyz, "
            "avg_solve_time, std_error_solve_time, time_of_flight_s, "
            "zoh_fine_dt_s, cgl_fine_nodes."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Root directory to scan (default: {default_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "clean_group_graph_data.csv",
        help=(
            "Output CSV path "
            f"(default: {default_root / 'clean_group_graph_data.csv'})"
        ),
    )
    return parser.parse_args()


def method_from_candidate(candidate_csv: str) -> str:
    match = re.match(r"^trajectory_([^_]+)_", candidate_csv)
    if not match:
        raise ValueError(f"Could not parse method from candidate CSV '{candidate_csv}'")
    return match.group(1)


def humanize_group_name(group_name: str) -> str:
    return " ".join(part.capitalize() for part in group_name.replace("_", " ").split())


def normalize_run_type(run_type: str) -> str:
    raw = (run_type or "").strip().lower()
    return RUN_TYPE_REMAP.get(raw, raw)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header row")

        required = {"run_type", "candidate_csv", "rmse_xyz", "rmse_sigma", "rmse_uxyz"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )

        return list(reader)


def sort_key(row: dict[str, str]) -> tuple[int, str, int, int]:
    method = row["method"]
    run_type = row["run_type"]
    return (
        METHOD_ORDER.get(method, len(METHOD_ORDER)),
        row["group_name"],
        METHOD_ORDER.get(method, len(METHOD_ORDER)),
        RUN_TYPE_ORDER.get(run_type, len(RUN_TYPE_ORDER)),
    )


def standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def read_simple_solve_stats(
    run_type_dir: Path,
) -> dict[str, dict[str, str]]:
    exact = run_type_dir / "simple_solve_metrics.csv"
    if exact.exists() and exact.is_file():
        target = exact
    else:
        candidates = sorted(run_type_dir.glob("simple_solve_metrics*.csv"))
        if not candidates:
            return {}
        target = candidates[0]

    with target.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        required = {"group_name", "run_name", "fine_clarabel_solve_time_s"}
        if not required.issubset(reader.fieldnames):
            return {}
        rows = list(reader)

    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        group = (row.get("group_name") or "").strip()
        run_name = (row.get("run_name") or "").strip()
        if not group:
            continue
        entry = grouped.setdefault(
            group,
            {
                "values": [],
                "time_of_flight_s": "",
                "zoh_fine_dt_s": "",
                "cgl_fine_nodes": "",
            },
        )

        for column in ("time_of_flight_s", "zoh_fine_dt_s", "cgl_fine_nodes"):
            value = (row.get(column) or "").strip()
            if value and not entry[column]:
                entry[column] = value

        if run_name.startswith("set_") and run_name.endswith("_avg"):
            continue

        fine_time_raw = (row.get("fine_clarabel_solve_time_s") or "").strip()
        if not fine_time_raw:
            continue
        try:
            fine_time = float(fine_time_raw)
        except ValueError:
            continue
        entry["values"].append(fine_time)

    stats: dict[str, dict[str, str]] = {}
    for group, entry in grouped.items():
        values = entry["values"]
        if values:
            avg = sum(values) / len(values)
            stderr = standard_error(values)
            avg_str = f"{avg:.6f}"
            stderr_str = f"{stderr:.6f}"
        else:
            avg_str = ""
            stderr_str = ""
        stats[group] = {
            "avg_solve_time": avg_str,
            "std_error_solve_time": stderr_str,
            "time_of_flight_s": str(entry["time_of_flight_s"]),
            "zoh_fine_dt_s": str(entry["zoh_fine_dt_s"]),
            "cgl_fine_nodes": str(entry["cgl_fine_nodes"]),
        }
    return stats


def resolve_simple_group_data(
    simple_stats: dict[str, dict[str, str]],
    method: str,
    group_name: str,
    raw_run_type: str,
    normalized_run_type: str,
) -> dict[str, str]:
    for run_type_candidate in (raw_run_type, normalized_run_type):
        key = f"{method}_{group_name}_{run_type_candidate}"
        data = simple_stats.get(key, {})
        if data:
            return data
    return {}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_path = args.output.resolve()

    collected: list[dict[str, str]] = []
    skipped_files = 0
    simple_stats_cache: dict[tuple[Path, str], dict[str, dict[str, str]]] = {}

    comparison_files = sorted(root.rglob("group_comparison_results.csv"))
    for comparison_csv in comparison_files:
        if not comparison_csv.is_file():
            continue

        raw_group_name = comparison_csv.parent.name
        display_group_name = humanize_group_name(raw_group_name)
        try:
            rows = read_rows(comparison_csv)
            for row in rows:
                candidate_csv = (row.get("candidate_csv") or "").strip()
                raw_run_type = (row.get("run_type") or "").strip()
                run_type = normalize_run_type(raw_run_type)
                method = method_from_candidate(candidate_csv)

                simple_cache_key = (comparison_csv.parent, raw_run_type)
                if simple_cache_key not in simple_stats_cache:
                    simple_stats_cache[simple_cache_key] = read_simple_solve_stats(
                        comparison_csv.parent / raw_run_type
                    )
                simple_stats = simple_stats_cache[simple_cache_key]

                simple_data = resolve_simple_group_data(
                    simple_stats,
                    method,
                    raw_group_name,
                    raw_run_type,
                    run_type,
                )
                zoh_data = resolve_simple_group_data(
                    simple_stats,
                    "zoh",
                    raw_group_name,
                    raw_run_type,
                    run_type,
                )
                cgl_data = resolve_simple_group_data(
                    simple_stats,
                    "cgl",
                    raw_group_name,
                    raw_run_type,
                    run_type,
                )

                avg_solve_time = simple_data.get("avg_solve_time", "")
                std_error_solve_time = simple_data.get("std_error_solve_time", "")
                time_of_flight_s = simple_data.get("time_of_flight_s", "")
                if not time_of_flight_s:
                    time_of_flight_s = zoh_data.get("time_of_flight_s", "") or cgl_data.get(
                        "time_of_flight_s", ""
                    )
                zoh_fine_dt_s = zoh_data.get("zoh_fine_dt_s", "") or simple_data.get(
                    "zoh_fine_dt_s", ""
                )
                cgl_fine_nodes = cgl_data.get("cgl_fine_nodes", "") or simple_data.get(
                    "cgl_fine_nodes", ""
                )

                collected.append(
                    {
                        "group_name": display_group_name,
                        "run_type": run_type,
                        "method": method,
                        "rmse_xyz": (row.get("rmse_xyz") or "").strip(),
                        "rmse_sigma": (row.get("rmse_sigma") or "").strip(),
                        "rmse_uxyz": (row.get("rmse_uxyz") or "").strip(),
                        "avg_solve_time": avg_solve_time,
                        "std_error_solve_time": std_error_solve_time,
                        "time_of_flight_s": time_of_flight_s,
                        "zoh_fine_dt_s": zoh_fine_dt_s,
                        "cgl_fine_nodes": cgl_fine_nodes,
                    }
                )
        except ValueError as exc:
            skipped_files += 1
            print(f"Skipping {comparison_csv}: {exc}", file=sys.stderr)

    if not collected:
        raise RuntimeError(f"No rows collected from {root}")

    collected.sort(key=sort_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_name",
                "run_type",
                "method",
                "rmse_xyz",
                "rmse_sigma",
                "rmse_uxyz",
                "avg_solve_time",
                "std_error_solve_time",
                "time_of_flight_s",
                "zoh_fine_dt_s",
                "cgl_fine_nodes",
            ],
        )
        writer.writeheader()
        writer.writerows(collected)

    print(
        f"Saved {len(collected)} rows from {len(comparison_files)} files to {output_path}"
    )
    if skipped_files:
        print(f"Skipped {skipped_files} files due to format issues.", file=sys.stderr)


if __name__ == "__main__":
    main()
