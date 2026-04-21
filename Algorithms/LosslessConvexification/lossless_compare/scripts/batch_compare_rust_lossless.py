import argparse
from pathlib import Path

from compare_rust_lossless import compare_csvs, resolve_csv_path, write_results


def resolve_directory_path(raw_path: str, project_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.exists() and candidate.is_dir():
        return candidate

    project_candidate = project_root / raw_path
    if project_candidate.exists() and project_candidate.is_dir():
        return project_candidate

    raise FileNotFoundError(
        f"Could not find directory '{raw_path}' as a direct path or under {project_root}"
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare a baseline trajectory against all compatible CSVs under a directory "
            "and save results to comparison_results.csv in that directory."
        )
    )
    parser.add_argument(
        "directory",
        help="Directory to scan recursively (example: direct_descent/long2)",
    )
    parser.add_argument("baseline_csv", help="Baseline trajectory CSV path")
    parser.add_argument(
        "--nodes",
        type=int,
        default=100,
        help=(
            "Max number of equally spaced rows to sample per comparison "
            "(capped by the shorter CSV, default: 100)"
        ),
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern used for recursive candidate discovery (default: *.csv)",
    )
    parser.add_argument(
        "--output-name",
        default="comparison_results.csv",
        help="Output filename inside the target directory (default: comparison_results.csv)",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to an existing output file instead of replacing it",
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="Fail if any candidate CSV is skipped due to incompatible format",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help=f"Project root used for path resolution (default: {project_root})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    target_dir = resolve_directory_path(args.directory, project_root).resolve()
    baseline_csv = resolve_csv_path(args.baseline_csv, project_root).resolve()
    output_path = (target_dir / args.output_name).resolve()

    if not args.keep_existing and output_path.exists():
        output_path.unlink()

    compared = 0
    skipped: list[tuple[Path, str]] = []

    for candidate_csv in sorted(target_dir.rglob(args.glob)):
        if not candidate_csv.is_file():
            continue
        if candidate_csv.suffix.lower() != ".csv":
            continue
        if candidate_csv.resolve() == baseline_csv:
            continue
        if candidate_csv.resolve() == output_path:
            continue

        try:
            metrics = compare_csvs(baseline_csv, candidate_csv, args.nodes)
            write_results(output_path, baseline_csv, candidate_csv, metrics)
            compared += 1
        except ValueError as exc:
            skipped.append((candidate_csv, str(exc)))

    if compared == 0:
        raise RuntimeError(
            f"No compatible CSVs were compared under {target_dir}. "
            f"Skipped {len(skipped)} files."
        )

    print(f"Saved {compared} comparisons to {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible CSVs:")
        for path, reason in skipped:
            print(f"  - {path}: {reason}")
        if args.fail_on_skip:
            raise RuntimeError("Some CSV files were skipped (see messages above).")


if __name__ == "__main__":
    main()
