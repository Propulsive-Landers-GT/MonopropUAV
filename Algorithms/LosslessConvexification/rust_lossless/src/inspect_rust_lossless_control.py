import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_trajectory(csv_path: Path) -> None:
    t = []
    ux, uy, uz = [], [], []
    sigma = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            ux.append(float(row["ux"]))
            uy.append(float(row["uy"]))
            uz.append(float(row["uz"]))
            sigma.append(float(row["sigma"]))

    fig, ax = plt.subplots(figsize=(14, 3.2))
    fig.suptitle(f"{csv_path.name} (control)")

    ax.plot(t, ux, label="ux")
    ax.plot(t, uy, label="uy")
    ax.plot(t, uz, label="uz")
    ax.plot(t, sigma, label="sigma", linestyle="--", color="black")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Thrust [N] / Sigma")
    ax.set_aspect("auto")
    ax.grid(True)

    # Keep legend fixed on the right side.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    plt.tight_layout(rect=(0.0, 0.0, 0.84, 0.95))

    output_path = csv_path.with_name(f"{csv_path.stem}_control.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path.name}")
    plt.close(fig)


def resolve_csv_files(args: list[str], project_root: Path) -> list[Path]:
    if args:
        return [Path(arg).resolve() for arg in args]

    inspect_dir = project_root / "to_inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    return sorted(inspect_dir.glob("trajectory*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rust lossless control-history CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="Optional CSV paths. Defaults to to_inspect/trajectory*.csv",
    )
    parsed = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_files = resolve_csv_files(parsed.csv_files, project_root)

    if not csv_files:
        print(f"No files found matching trajectory*.csv in {project_root / 'to_inspect'}")
        return

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue
        plot_trajectory(csv_path)


if __name__ == "__main__":
    main()
