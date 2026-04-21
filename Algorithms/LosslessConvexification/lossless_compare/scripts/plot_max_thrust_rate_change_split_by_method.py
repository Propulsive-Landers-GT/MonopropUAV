import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = ("zoh", "cgl")
FLIGHT_PLANS = (
    "Direct Descent",
    "Direct Limited Descent",
    "Offset Descent",
    "Offset Limited Descent",
)
RESOLUTIONS = ("very_coarse", "coarse", "fine", "very_fine")
RESOLUTION_LABELS = {
    "very_coarse": "Very Coarse",
    "coarse": "Coarse",
    "fine": "Fine",
    "very_fine": "Very Fine",
}
RESOLUTION_COLORS = {
    "very_coarse": "#4C78A8",
    "coarse": "#1B9E77",
    "fine": "#7570B3",
    "very_fine": "#D95F02",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Create max thrust rate-of-change bar chart split by method "
            "(left=ZOH, right=CGL), grouped by flight plan, with resolution bars."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "max_thrust_rate_change.csv",
        help=f"Input CSV path (default: {default_root / 'max_thrust_rate_change.csv'})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "figures" / "max_thrust_rate_change_split_by_method.png",
        help=(
            "Output figure path "
            f"(default: {default_root / 'figures' / 'max_thrust_rate_change_split_by_method.png'})"
        ),
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional fixed y-axis max. If omitted, uses data-driven scaling.",
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


def normalize_flight_plan(name: str) -> str:
    cleaned = " ".join((name or "").strip().replace("_", " ").split())
    return cleaned.title()


def load_data(path: Path) -> dict[tuple[str, str, str], float]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        required = {"flight_plan", "resolution", "method"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        sums: dict[tuple[str, str, str], float] = {}
        counts: dict[tuple[str, str, str], int] = {}

        for row in reader:
            flight_plan = normalize_flight_plan(row.get("flight_plan", ""))
            resolution = (row.get("resolution") or "").strip().lower()
            method = (row.get("method") or "").strip().lower()
            value = parse_float(row.get("max_abs_dthrust_dt_n_per_s", ""))
            if value is None:
                # Backward compatibility with older CSV exports.
                value = parse_float(row.get("max_abs_dthrust_dt", ""))

            if (
                flight_plan not in FLIGHT_PLANS
                or resolution not in RESOLUTIONS
                or method not in METHODS
                or value is None
            ):
                continue

            key = (method, flight_plan, resolution)
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1

    means: dict[tuple[str, str, str], float] = {}
    for key, total in sums.items():
        means[key] = total / counts[key]
    return means


def make_plot(
    values: dict[tuple[str, str, str], float],
    output_path: Path,
    y_max_override: float | None = None,
) -> None:
    if not values:
        raise RuntimeError("No usable rows found for plotting")

    groups = [group for group in FLIGHT_PLANS if any(k[1] == group for k in values.keys())]
    if not groups:
        raise RuntimeError("No recognized flight-plan groups were found in input data")

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
        }
    )

    n_groups = len(groups)
    group_spacing = 1.55
    split_gap = 2.70
    bar_width = 0.30
    res_offsets = {
        "very_coarse": -1.5 * bar_width,
        "coarse": -0.5 * bar_width,
        "fine": 0.5 * bar_width,
        "very_fine": 1.5 * bar_width,
    }

    half_span = (n_groups - 1) * group_spacing
    left_start = -(split_gap / 2.0 + half_span)
    right_start = split_gap / 2.0
    left_centers = [left_start + idx * group_spacing for idx in range(n_groups)]
    right_centers = [right_start + idx * group_spacing for idx in range(n_groups)]
    x_limit = split_gap / 2.0 + half_span + 0.95

    fig_width = max(12.0, 1.2 * n_groups + 8.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    bar_containers: list[tuple[str, object]] = []

    for resolution in RESOLUTIONS:
        color = RESOLUTION_COLORS[resolution]
        offset = res_offsets[resolution]

        zoh_vals = [values.get(("zoh", group, resolution), math.nan) for group in groups]
        cgl_vals = [values.get(("cgl", group, resolution), math.nan) for group in groups]

        zoh_container = ax.bar(
            [x + offset for x in left_centers],
            zoh_vals,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=RESOLUTION_LABELS[resolution],
        )
        cgl_container = ax.bar(
            [x + offset for x in right_centers],
            cgl_vals,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        bar_containers.append(("zoh", zoh_container))
        bar_containers.append(("cgl", cgl_container))

    tick_positions = left_centers + right_centers
    tick_labels = [f"{group}\nZOH" for group in groups] + [f"{group}\nCGL" for group in groups]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_xlim(-x_limit, x_limit)

    split_x = (left_centers[-1] + right_centers[0]) / 2.0
    ax.axvline(split_x, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)

    positive_values = [value for value in values.values() if math.isfinite(value) and value > 0.0]
    if not positive_values:
        raise RuntimeError("No finite positive values available for log-scale plotting")

    y_bottom_limit = max(min(positive_values) * 0.8, 1e-9)
    if y_max_override is not None and y_max_override > 0:
        y_top_limit = y_max_override
    else:
        y_top_limit = max(positive_values) * 6
    if y_top_limit <= y_bottom_limit:
        y_top_limit = y_bottom_limit * 10.0

    ax.set_yscale("log")
    ax.set_ylim(bottom=y_bottom_limit, top=y_top_limit)
    y_top = ax.get_ylim()[1]
    ax.text(
        sum(left_centers) / len(left_centers),
        0.98,
        "Method: ZOH",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        sum(right_centers) / len(right_centers),
        0.98,
        "Method: CGL",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title("Maximum Thrust Rate-of-Change by Method, Flight Plan, and Resolution")
    ax.set_ylabel("Max |dT/dt| (N/s, log scale)")
    ax.set_xlabel("Flight Plan and Method")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    _, y_max = ax.get_ylim()
    for method, container in bar_containers:
        for bar in container:
            height = bar.get_height()
            if not math.isfinite(height) or height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(height * 1.03, y_max * 0.98),
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                rotation=60,
            )

    ax.legend(
        title="Resolution",
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="black",
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    values = load_data(input_path)
    make_plot(values, output_path, y_max_override=args.y_max)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
