import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_trajectory(csv_path: Path) -> None:
    t, x, y, z = [], [], [], []
    vx, vy, vz = [], [], []
    mass = []
    ux, uy, uz = [], [], []
    sigma = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            x.append(float(row["x"]))
            y.append(float(row["y"]))
            z.append(float(row["z"]))
            vx.append(float(row["vx"]))
            vy.append(float(row["vy"]))
            vz.append(float(row["vz"]))
            mass.append(float(row["mass"]))
            ux.append(float(row["ux"]))
            uy.append(float(row["uy"]))
            uz.append(float(row["uz"]))
            sigma.append(float(row["sigma"]))

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(csv_path.name)

    axes[0].plot(t, x, label="x")
    axes[0].plot(t, y, label="y")
    axes[0].plot(t, z, label="z")
    axes[0].set_ylabel("Position [m]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, vx, label="vx")
    axes[1].plot(t, vy, label="vy")
    axes[1].plot(t, vz, label="vz")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, mass, label="mass", color="purple")
    axes[2].set_ylabel("Mass [kg]")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(t, ux, label="ux")
    axes[3].plot(t, uy, label="uy")
    axes[3].plot(t, uz, label="uz")
    axes[3].plot(t, sigma, label="sigma", linestyle="--", color="black")
    axes[3].set_xlabel("Time step")
    axes[3].set_ylabel("Thrust [N] / Sigma")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()

    output_path = csv_path.with_suffix(".png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path.name}")

    # plt.show()
    plt.close(fig)


def resolve_csv_files(args: list[str], project_root: Path) -> list[Path]:
    if args:
        return [Path(arg).resolve() for arg in args]

    inspect_dir = project_root / "to_inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    return sorted(inspect_dir.glob("trajectory*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rust lossless trajectory CSV files."
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
