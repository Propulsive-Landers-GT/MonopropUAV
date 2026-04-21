import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = ("zoh", "cgl")
RUN_TYPE_REMAP = {
    "extreme_short": "very_coarse",
    "ultra_short": "coarse",
    "short": "fine",
    "long": "very_fine",
}
RUN_TYPES = ("very_coarse", "coarse", "fine", "very_fine")
RUN_TYPE_LABELS = {
    "very_coarse": "Very Coarse",
    "coarse": "Coarse",
    "fine": "Fine",
    "very_fine": "Very Fine",
}
RUN_TYPE_COLORS = {
    "very_coarse": "#4C78A8",
    "coarse": "#1B9E77",
    "fine": "#7570B3",
    "very_fine": "#D95F02",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Create RMSE bar charts split by method (left=ZOH, right=CGL), "
            "grouped by flight plan, with bars for each resolution."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "clean_group_graph_data.csv",
        help=f"Input clean CSV path (default: {default_root / 'clean_group_graph_data.csv'})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "figures",
        help=f"Directory for output figures (default: {default_root / 'figures'})",
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


def load_data(path: Path) -> dict[str, dict[tuple[str, str, str], float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        required = {"group_name", "run_type", "method", "rmse_xyz", "rmse_sigma", "rmse_uxyz"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        sums: dict[str, dict[tuple[str, str, str], float]] = {
            "rmse_xyz": {},
            "rmse_sigma": {},
            "rmse_uxyz": {},
        }
        counts: dict[str, dict[tuple[str, str, str], int]] = {
            "rmse_xyz": {},
            "rmse_sigma": {},
            "rmse_uxyz": {},
        }

        for row in reader:
            method = (row.get("method") or "").strip().lower()
            run_type = normalize_run_type(row.get("run_type", ""))
            group_name = (row.get("group_name") or "").strip()
            if method not in METHODS or run_type not in RUN_TYPES or not group_name:
                continue

            key = (method, group_name, run_type)
            for metric in ("rmse_xyz", "rmse_sigma", "rmse_uxyz"):
                value = parse_float(row.get(metric, ""))
                if value is None:
                    continue
                sums[metric][key] = sums[metric].get(key, 0.0) + value
                counts[metric][key] = counts[metric].get(key, 0) + 1

    means: dict[str, dict[tuple[str, str, str], float]] = {
        "rmse_xyz": {},
        "rmse_sigma": {},
        "rmse_uxyz": {},
    }
    for metric in ("rmse_xyz", "rmse_sigma", "rmse_uxyz"):
        for key, total in sums[metric].items():
            count = counts[metric][key]
            means[metric][key] = total / count
    return means


def plot_metric(
    metric_name: str,
    values: dict[tuple[str, str, str], float],
    output_path: Path,
) -> None:
    groups = sorted({key[1] for key in values.keys()})
    if not groups:
        raise RuntimeError(f"No usable rows found for {metric_name}")

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
    group_spacing = 1.45
    split_gap = 2.40
    bar_width = 0.30
    run_offsets = {
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
    x_limit = split_gap / 2.0 + half_span + 0.9

    fig_width = max(12.0, 1.2 * n_groups + 8.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    bar_containers = []

    for run_type in RUN_TYPES:
        color = RUN_TYPE_COLORS[run_type]
        offset = run_offsets[run_type]

        zoh_vals = [
            values.get(("zoh", group, run_type), math.nan)
            for group in groups
        ]
        cgl_vals = [
            values.get(("cgl", group, run_type), math.nan)
            for group in groups
        ]

        zoh_container = ax.bar(
            [x + offset for x in left_centers],
            zoh_vals,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=RUN_TYPE_LABELS[run_type],
        )
        cgl_container = ax.bar(
            [x + offset for x in right_centers],
            cgl_vals,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        bar_containers.extend([zoh_container, cgl_container])

    tick_positions = left_centers + right_centers
    tick_labels = [f"{group}\nZOH" for group in groups] + [f"{group}\nCGL" for group in groups]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_xlim(-x_limit, x_limit)

    split_x = (left_centers[-1] + right_centers[0]) / 2.0
    ax.axvline(split_x, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)

    y_min, y_max = ax.get_ylim()
    # if metric_name == "rmse_sigma":
    ax.set_ylim(y_min, y_max * 1.18)
    if metric_name == "rmse_uxyz":
        ax.set_ylim(y_min, y_max * 1.5)
    y_top = ax.get_ylim()[1]
    ax.text(
        sum(left_centers) / len(left_centers),
        y_top * 0.98,
        "Method: ZOH",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
    )
    ax.text(
        sum(right_centers) / len(right_centers),
        y_top * 0.98,
        "Method: CGL",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
    )

    metric_label_map = {
        "rmse_xyz": "RMSE XYZ",
        "rmse_sigma": "RMSE Sigma",
        "rmse_uxyz": "RMSE UXYZ",
    }
    metric_units_map = {
        "rmse_xyz": "m",
        "rmse_sigma": "m/s^2",
        "rmse_uxyz": "m/s^2",
    }
    metric_label = metric_label_map.get(metric_name, metric_name)
    metric_units = metric_units_map.get(metric_name)
    ax.set_title(f"{metric_label} by Method, Flight Plan, and Resolution")
    if metric_units:
        ax.set_ylabel(f"{metric_label} ({metric_units})")
    else:
        ax.set_ylabel(metric_label)
    ax.set_xlabel("Flight Plan and Method")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.012
    for container in bar_containers:
        for bar in container:
            height = bar.get_height()
            if not math.isfinite(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + y_offset,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                rotation=60,
            )

    legend_loc = "upper left" if metric_name == "rmse_uxyz" else "upper right"
    legend_anchor = (0.015, 0.985) if metric_name == "rmse_uxyz" else (0.985, 0.985)
    ax.legend(
        title="Resolution",
        ncol=1,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
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
    output_dir = args.output_dir.resolve()

    values_by_metric = load_data(input_path)

    xyz_path = output_dir / "rmse_xyz_split_by_method.png"
    sigma_path = output_dir / "rmse_sigma_split_by_method.png"
    uxyz_path = output_dir / "rmse_uxyz_split_by_method.png"
    plot_metric("rmse_xyz", values_by_metric["rmse_xyz"], xyz_path)
    plot_metric("rmse_sigma", values_by_metric["rmse_sigma"], sigma_path)
    plot_metric("rmse_uxyz", values_by_metric["rmse_uxyz"], uxyz_path)

    print(f"Saved {xyz_path}")
    print(f"Saved {sigma_path}")
    print(f"Saved {uxyz_path}")


if __name__ == "__main__":
    main()
