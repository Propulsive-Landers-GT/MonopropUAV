import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_clean_output = project_root / "clean_group_graph_data.csv"
    default_figures_dir = project_root / "figures"
    default_table_output = project_root / "figures" / "tables" / "run_config_summary_table.png"
    default_avg_plot_output = project_root / "figures" / "avg_solve_time_grouped.png"

    parser = argparse.ArgumentParser(
        description=(
            "Update clean_group_graph_data.csv and regenerate table outputs "
            "and figures from the latest group comparison results."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root,
        help=f"Root directory to scan for group_comparison_results.csv (default: {project_root})",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=default_clean_output,
        help=f"Output path for clean_group_graph_data.csv (default: {default_clean_output})",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=default_figures_dir,
        help=f"Output directory for RMSE figures (default: {default_figures_dir})",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=default_table_output,
        help=f"Output path for consolidated table image (default: {default_table_output})",
    )
    parser.add_argument(
        "--avg-plot-output",
        type=Path,
        default=default_avg_plot_output,
        help=f"Output path for avg solve-time figure (default: {default_avg_plot_output})",
    )
    return parser.parse_args()


def run_step(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(f'"{part}"' if " " in part else part for part in cmd))
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    prepare_script = project_root / "src" / "prepare_group_comparison_graph_data.py"
    rmse_plot_script = project_root / "src" / "plot_rmse_split_by_method.py"
    table_script = project_root / "src" / "render_run_config_tables.py"
    avg_plot_script = project_root / "src" / "plot_avg_solve_time_grouped.py"

    run_step(
        [
            sys.executable,
            str(prepare_script),
            "--root",
            str(args.root.resolve()),
            "--output",
            str(args.clean_output.resolve()),
        ],
        cwd=project_root,
    )

    run_step(
        [
            sys.executable,
            str(rmse_plot_script),
            "--input",
            str(args.clean_output.resolve()),
            "--output-dir",
            str(args.figures_dir.resolve()),
        ],
        cwd=project_root,
    )

    run_step(
        [
            sys.executable,
            str(avg_plot_script),
            "--input",
            str(args.clean_output.resolve()),
            "--output",
            str(args.avg_plot_output.resolve()),
        ],
        cwd=project_root,
    )

    run_step(
        [
            sys.executable,
            str(table_script),
            "--input",
            str(args.clean_output.resolve()),
            "--output",
            str(args.table_output.resolve()),
        ],
        cwd=project_root,
    )

    print("Done.")
    print(f"Clean data: {args.clean_output.resolve()}")
    print(f"RMSE figures dir: {args.figures_dir.resolve()}")
    print(f"Avg solve-time figure: {args.avg_plot_output.resolve()}")
    print(f"Table: {args.table_output.resolve()}")


if __name__ == "__main__":
    main()
