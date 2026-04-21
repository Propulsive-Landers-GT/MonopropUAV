import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_trajectory_3d(csv_path: Path) -> None:
    x, y, z = [], [], []
    thrust = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["x"]))
            y.append(float(row["y"]))
            z.append(float(row["z"]))
            thrust_value = (row.get("thrust_force") or "").strip()
            if thrust_value:
                thrust.append(float(thrust_value))
            else:
                ux = float(row["ux"])
                uy = float(row["uy"])
                uz = float(row["uz"])
                thrust.append((ux**2 + uy**2 + uz**2) ** 0.5)

    if not x:
        print(f"Skipping {csv_path.name}: no rows")
        return

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    # fig.suptitle(csv_path.name)

    # Color trajectory by thrust so high/low control effort is visible in 3D.
    scatter = ax.scatter(x, y, z, c=thrust, cmap="YlOrRd", s=14, label="Trajectory")
    ax.plot(x, y, z, color="gray", linewidth=1.0, alpha=0.6)

    ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=50, label="Start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=50, label="End")

    max_num = max(max(x), max(y), max(z))
    # max_num = max(max(x), max(y), max(z)) / 10
    # ax.set_xlim(-max_num/4, max_num/2)
    ax.set_xlim(-max_num/2, max_num)
    # ax.set_ylim(-max_num/4, max_num/2)
    ax.set_ylim(-max_num/2, max_num)
    ax.set_zlim(0, max_num)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.colorbar(scatter, ax=ax, pad=0.1, label="Thrust Force [N]")

    ax.view_init(elev=5, azim=-45)  # change these two numbers  

    plt.tight_layout()

    output_path = csv_path.with_name(f"{csv_path.stem}_3d.png")
    # plt.show()
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
        description="Plot rust lossless trajectory CSV files in 3D."
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
        plot_trajectory_3d(csv_path)


if __name__ == "__main__":
    main()
