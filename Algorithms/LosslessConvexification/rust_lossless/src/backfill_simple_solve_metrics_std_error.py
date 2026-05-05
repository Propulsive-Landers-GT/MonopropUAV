import argparse
import csv
import math
from pathlib import Path


STD_ERROR_COLUMN = "set_std_error_fine_clarabel_solve_time_s"


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Find simple_solve_metrics*.csv files and add standard error for set rows "
            "when the standard-error column is missing."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Root directory to scan (default: {default_root})",
    )
    return parser.parse_args()


def standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def process_file(path: Path) -> bool:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return False
        fieldnames = reader.fieldnames
        rows = list(reader)

    if STD_ERROR_COLUMN in fieldnames:
        return False

    required = {
        "group_name",
        "run_name",
        "fine_clarabel_solve_time_s",
        "set_avg_fine_clarabel_solve_time_s",
    }
    if not required.issubset(fieldnames):
        return False

    buffers_by_group: dict[str, list[float]] = {}
    for row in rows:
        group_name = (row.get("group_name") or "").strip()
        run_name = (row.get("run_name") or "").strip()
        fine_time_raw = (row.get("fine_clarabel_solve_time_s") or "").strip()
        is_set_row = run_name.startswith("set_") and run_name.endswith("_avg")

        if is_set_row:
            values = buffers_by_group.get(group_name, [])
            row[STD_ERROR_COLUMN] = f"{standard_error(values):.6f}" if values else ""
            buffers_by_group[group_name] = []
            continue

        row[STD_ERROR_COLUMN] = ""
        if fine_time_raw:
            try:
                fine_time = float(fine_time_raw)
            except ValueError as exc:
                raise ValueError(
                    f"{path}: invalid fine_clarabel_solve_time_s '{fine_time_raw}'"
                ) from exc
            buffers_by_group.setdefault(group_name, []).append(fine_time)

    new_fieldnames = list(fieldnames) + [STD_ERROR_COLUMN]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return True


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    updated = 0
    skipped = 0

    for csv_path in sorted(root.rglob("simple_solve_metrics*.csv")):
        if not csv_path.is_file():
            continue
        changed = process_file(csv_path)
        if changed:
            updated += 1
            print(f"Updated: {csv_path}")
        else:
            skipped += 1

    print(f"Done. Updated {updated} files, skipped {skipped} files.")


if __name__ == "__main__":
    main()
