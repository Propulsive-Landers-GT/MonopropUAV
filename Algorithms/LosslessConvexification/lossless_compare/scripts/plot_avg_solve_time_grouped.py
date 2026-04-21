import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = ("zoh", "cgl")
METHOD_COLORS = {"zoh": "#1F77B4", "cgl": "#E66101"}
RUN_TYPE_REMAP = {
    "extreme_short": "very_coarse",
    "ultra_short": "coarse",
    "short": "fine",
    "long": "very_fine",
}
TARGET_RUN_TYPES = ("very_coarse", "coarse", "fine", "very_fine")
RUN_TYPE_ORDER = {
    "very_coarse": 0,
    "coarse": 1,
    "fine": 2,
    "very_fine": 3,
}
RUN_TYPE_LABELS = {
    "very_coarse": "Very Coarse",
    "coarse": "Coarse",
    "fine": "Fine",
    "very_fine": "Very Fine",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Create avg solve-time bar chart grouped by flight plan, then resolution, "
            "with side-by-side method bars and std-error error bars."
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
        default=default_root / "figures" / "avg_solve_time_grouped.png",
        help=(
            "Output figure path "
            f"(default: {default_root / 'figures' / 'avg_solve_time_grouped.png'})"
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


def normalize_run_type(run_type: str) -> str:
    raw = (run_type or "").strip().lower()
    return RUN_TYPE_REMAP.get(raw, raw)


def load_data(
    path: Path,
) -> tuple[dict[tuple[str, str, str], tuple[float, float]], list[str], dict[str, list[str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        required = {"group_name", "run_type", "method", "avg_solve_time", "std_error_solve_time"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        data: dict[tuple[str, str, str], tuple[float, float]] = {}
        group_to_run_types: dict[str, set[str]] = {}

        for row in reader:
            group = (row.get("group_name") or "").strip()
            run_type = normalize_run_type(row.get("run_type", ""))
            method = (row.get("method") or "").strip().lower()
            if not group or run_type not in TARGET_RUN_TYPES or method not in METHODS:
                continue

            avg = parse_float(row.get("avg_solve_time", ""))
            stderr = parse_float(row.get("std_error_solve_time", ""))
            if avg is None:
                continue
            if stderr is None:
                stderr = 0.0

            data[(group, run_type, method)] = (avg, stderr)
            group_to_run_types.setdefault(group, set()).add(run_type)

    groups = sorted(group_to_run_types.keys())
    ordered_group_run_types: dict[str, list[str]] = {}
    for group in groups:
        run_types = sorted(
            group_to_run_types[group],
            key=lambda item: RUN_TYPE_ORDER.get(item, len(RUN_TYPE_ORDER)),
        )
        ordered_group_run_types[group] = run_types

    return data, groups, ordered_group_run_types


def make_plot(
    data: dict[tuple[str, str, str], tuple[float, float]],
    groups: list[str],
    group_run_types: dict[str, list[str]],
    output_path: Path,
) -> None:
    if not groups:
        raise RuntimeError("No usable rows found for avg solve-time plotting")

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

    bar_width = 0.44
    run_type_spacing = 1.15
    group_spacing = 0.95
    method_offsets = {"zoh": -bar_width / 2.0, "cgl": bar_width / 2.0}

    cluster_positions: list[tuple[float, str, str]] = []
    group_ranges: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for group in groups:
        run_types = group_run_types[group]
        if not run_types:
            continue
        start_x = cursor
        for run_type in run_types:
            cluster_positions.append((cursor, group, run_type))
            cursor += run_type_spacing
        end_x = cursor - run_type_spacing
        group_ranges[group] = (start_x, end_x)
        cursor += group_spacing

    # Use a stable near-16:9 canvas to avoid extra-wide output as clusters grow.
    fig, ax = plt.subplots(figsize=(12.0, 6.75))
    all_y_values: list[float] = []
    bar_containers = []

    for method in METHODS:
        x_vals = []
        y_vals = []
        y_errs_lower = []
        y_errs_upper = []
        for x_center, group, run_type in cluster_positions:
            pair = data.get((group, run_type, method))
            if pair is None:
                continue
            avg, stderr = pair
            if avg <= 0:
                continue
            x_vals.append(x_center + method_offsets[method])
            y_vals.append(avg)
            all_y_values.append(avg)

            # Keep lower error above zero for log-scale rendering.
            lower_err = min(stderr, avg * 0.95)
            y_errs_lower.append(lower_err)
            y_errs_upper.append(stderr)

        if not x_vals:
            continue

        container = ax.bar(
            x_vals,
            y_vals,
            width=bar_width,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            yerr=[y_errs_lower, y_errs_upper],
            capsize=4,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
            label=method.upper(),
        )
        bar_containers.append(container)

    if not all_y_values:
        raise RuntimeError("No positive avg_solve_time values available for log-scale plotting")

    xticks = [x for x, _, _ in cluster_positions]
    xlabels = [RUN_TYPE_LABELS.get(run_type, run_type) for _, _, run_type in cluster_positions]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")

    ax.set_title("Average Solve Time by Flight Plan, Resolution, and Method")
    ax.set_ylabel("Average Solve Time (s, log scale)")
    ax.set_xlabel("Resolution")
    ax.set_yscale("log")
    y_min = min(all_y_values) * 0.8
    y_max = max(all_y_values) * 4
    ax.set_ylim(bottom=max(y_min, 1e-9), top=y_max)

    # Add explicit headroom so bars and labels do not crowd the top border.
    y_bottom, y_top = ax.get_ylim()
    ax.set_ylim(y_bottom, y_top * 1.18)
    y_max = ax.get_ylim()[1]

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    # Put numeric values just above each bar with a multiplicative offset for log axes.
    for container in bar_containers:
        for bar in container:
            height = bar.get_height()
            if height <= 0:
                continue
            label_y = min(height * 1.07, y_max * 0.98)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=0,
            )

    # Annotate pairwise percent difference between methods at each cluster.
    for x_center, group, run_type in cluster_positions:
        zoh_pair = data.get((group, run_type, "zoh"))
        cgl_pair = data.get((group, run_type, "cgl"))
        if zoh_pair is None or cgl_pair is None:
            continue

        zoh_avg, _ = zoh_pair
        cgl_avg, _ = cgl_pair
        if zoh_avg <= 0 or cgl_avg <= 0:
            continue

        slow_avg = max(zoh_avg, cgl_avg)
        fast_avg = min(zoh_avg, cgl_avg)
        percent_diff = ((slow_avg - fast_avg) / fast_avg) * 100.0
        label_y = min(slow_avg * 1.24, y_max * 0.96)
        ax.text(
            x_center,
            label_y,
            f"Δ{percent_diff:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="black",
            fontweight="semibold",
        )

    ax.legend(title="Method", loc="upper right")

    y_bottom, y_top = ax.get_ylim()
    group_label_y = y_bottom * (10 ** (-0.20 * (math.log10(y_top) - math.log10(y_bottom))))
    for idx, group in enumerate(groups):
        if group not in group_ranges:
            continue
        x_start, x_end = group_ranges[group]
        x_center = (x_start + x_end) / 2.0
        ax.text(
            x_center,
            group_label_y,
            group,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="semibold",
            clip_on=False,
        )
        if idx < len(groups) - 1:
            split_x = x_end + (group_spacing / 2.0) + (run_type_spacing / 2.0)
            ax.axvline(split_x, color="gray", linestyle=":", linewidth=0.9, alpha=0.75)

    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_max_method_percent_difference(
    data: dict[tuple[str, str, str], tuple[float, float]],
    groups: list[str],
    group_run_types: dict[str, list[str]],
) -> None:
    max_record: tuple[float, str, str, str, float, str, float] | None = None

    for group in groups:
        for run_type in group_run_types.get(group, []):
            method_avgs: list[tuple[str, float]] = []
            for method in METHODS:
                pair = data.get((group, run_type, method))
                if pair is None:
                    continue
                avg, _ = pair
                if avg <= 0:
                    continue
                method_avgs.append((method, avg))

            if len(method_avgs) < 2:
                continue

            low_method, low_avg = min(method_avgs, key=lambda item: item[1])
            high_method, high_avg = max(method_avgs, key=lambda item: item[1])
            if low_avg <= 0:
                continue

            percent_diff = ((high_avg - low_avg) / low_avg) * 100.0
            if max_record is None or percent_diff > max_record[0]:
                max_record = (
                    percent_diff,
                    group,
                    run_type,
                    low_method,
                    low_avg,
                    high_method,
                    high_avg,
                )

    if max_record is None:
        print("No method pairs with positive avg_solve_time were found for percent-difference summary.")
        return

    (
        percent_diff,
        group,
        run_type,
        low_method,
        low_avg,
        high_method,
        high_avg,
    ) = max_record
    print(
        "Max method percent difference "
        f"(Resolution: {RUN_TYPE_LABELS.get(run_type, run_type)}, Flight Plan: {group}): "
        f"{percent_diff:.2f}% "
        f"(between {low_avg:.6f}s and {high_avg:.6f}s)"
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    data, groups, group_run_types = load_data(input_path)
    print_max_method_percent_difference(data, groups, group_run_types)
    make_plot(data, groups, group_run_types, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
