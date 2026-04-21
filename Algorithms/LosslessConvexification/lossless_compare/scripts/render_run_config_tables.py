import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


RUN_TYPE_REMAP = {
    "extreme_short": "very_coarse",
    "ultra_short": "coarse",
    "short": "fine",
    "long": "very_fine",
}
RUN_TYPE_ORDER = {
    "very_coarse": 0,
    "coarse": 1,
    "fine": 2,
    "very_fine": 3,
    "very_short": 4,
    "med": 5,
    "extreme_long": 6,
}
TARGET_RUN_TYPES = {"very_coarse", "coarse", "fine", "very_fine"}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Render a single consolidated table image for all flight plan + resolution entries "
            "using clean_group_graph_data.csv "
            "with time_of_flight_s, zoh_fine_dt_s, and cgl_fine_nodes."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "clean_group_graph_data.csv",
        help=f"Input clean CSV path (default: {default_root / 'clean_group_graph_data.csv'})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "figures" / "tables" / "run_config_summary_table.png",
        help=(
            "Output image path "
            f"(default: {default_root / 'figures' / 'tables' / 'run_config_summary_table.png'})"
        ),
    )
    return parser.parse_args()


def clean_text(raw_value: str) -> str:
    return (raw_value or "").strip()


def parse_float(raw_value: str) -> float | None:
    text = clean_text(raw_value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(raw_value: str) -> int | None:
    value = parse_float(raw_value)
    if value is None:
        return None
    return int(round(value))


def pick_preferred_value(values: list[str]) -> str:
    for value in values:
        if clean_text(value):
            return clean_text(value)
    return ""


def format_time(value: str, decimals: int) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.{decimals}f}"


def format_nodes(value: str) -> str:
    parsed = parse_int(value)
    if parsed is None:
        return "N/A"
    return str(parsed)


def normalize_run_type(run_type: str) -> str:
    raw = clean_text(run_type).lower()
    return RUN_TYPE_REMAP.get(raw, raw)


def load_group_run_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        required = {"group_name", "run_type", "time_of_flight_s", "zoh_fine_dt_s", "cgl_fine_nodes"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
        for row in reader:
            group_name = clean_text(row.get("group_name", ""))
            run_type = normalize_run_type(row.get("run_type", ""))
            if not group_name or run_type not in TARGET_RUN_TYPES:
                continue

            key = (group_name, run_type)
            entry = grouped.setdefault(
                key,
                {
                    "time_of_flight_s": [],
                    "zoh_fine_dt_s": [],
                    "cgl_fine_nodes": [],
                },
            )
            for column in ("time_of_flight_s", "zoh_fine_dt_s", "cgl_fine_nodes"):
                entry[column].append(clean_text(row.get(column, "")))

    rows = []
    for (group_name, run_type), values in grouped.items():
        rows.append(
            {
                "group_name": group_name,
                "run_type": run_type,
                "time_of_flight_s": pick_preferred_value(values["time_of_flight_s"]),
                "zoh_fine_dt_s": pick_preferred_value(values["zoh_fine_dt_s"]),
                "cgl_fine_nodes": pick_preferred_value(values["cgl_fine_nodes"]),
            }
        )

    rows.sort(key=lambda row: (row["group_name"], RUN_TYPE_ORDER.get(row["run_type"], len(RUN_TYPE_ORDER)), row["run_type"]))
    return rows


def render_table_image(rows: list[dict[str, str]], output_path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.titlesize": 13,
        }
    )

    height = max(3.0, 1.2 + 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(11.2, height))
    ax.axis("off")

    cell_rows = []
    for row in rows:
        run_label = f"{row['group_name']} | {row['run_type'].replace('_', ' ').title()}"
        cell_rows.append(
            [
                run_label,
                format_time(row["time_of_flight_s"], 3),
                format_time(row["zoh_fine_dt_s"], 4),
                format_nodes(row["cgl_fine_nodes"]),
            ]
        )

    table = ax.table(
        cellText=cell_rows,
        colLabels=["Flight Plan | Resolution", "Time of Flight (s)", "ZOH dt (s)", "CGL Nodes"],
        colWidths=[0.46, 0.18, 0.18, 0.18],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.48)

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

    ax.set_title("Flight Plan / Resolution Configuration Summary", pad=10, fontweight="semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    rows = load_group_run_rows(input_path)
    if not rows:
        raise RuntimeError(f"No usable rows found in {input_path}")

    render_table_image(rows, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
