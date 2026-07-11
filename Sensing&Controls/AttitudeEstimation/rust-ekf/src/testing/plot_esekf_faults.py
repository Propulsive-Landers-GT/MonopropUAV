"""Plot the fault-injection scenarios from esekf_fault_output.csv.

Run the driver first: `cargo run --bin esekf_fault_test`, then this script.
Four rows: attitude error under dropout/NaN faults, the innovation gate demo
(log scale), attitude error split by axis, and position error split by axis
with and without the VN-200 barometer during the GNSS dropout.
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "esekf_fault_output.csv")
OUT_DIR = os.path.join(HERE, "outputs")

# Fault schedule — keep in sync with esekf_fault_test.rs.
DROPOUT = (8.0, 16.0)
GPS_OUTLIERS = (4.0, 6.0)
MAG_OUTLIERS = (10.0, 10.5)
IMU_NAN = (12.0, 12.2)
MAG_NAN = (14.0, 14.2)
GPS_NAN = (18.0, 19.0)

# One fixed color per scenario, used identically in every panel.
COLOR = {
    "baseline": "#2a78d6",
    "gnss_dropout": "#1baf7a",
    "gnss_dropout_nobaro": "#e87ba4",
    "outliers_gated": "#4a3aa7",
    "outliers_ungated": "#e34948",
    "nan_burst": "#eda100",
    "kitchen_sink": "#eb6834",
}
LABEL = {
    "baseline": "baseline",
    "gnss_dropout": "GPS+UWB dropout",
    "gnss_dropout_nobaro": "dropout, no baro",
    "outliers_gated": "outliers, gate on",
    "outliers_ungated": "outliers, gate off",
    "nan_burst": "NaN bursts",
    "kitchen_sink": "all faults at once",
}
# Axis colors for the per-axis panels (a separate encoding from scenarios).
AXIS_COLOR = {"x": "#2a78d6", "y": "#1baf7a", "z": "#eb6834"}


def shade(ax, window, color, label=None):
    ax.axvspan(window[0], window[1], color=color, alpha=0.12, lw=0, label=label)


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    runs = {name: g.reset_index(drop=True) for name, g in df.groupby("scenario")}

    fig = plt.figure(figsize=(11, 15))
    gs = fig.add_gridspec(4, 2, hspace=0.34, wspace=0.16)
    fig.suptitle(
        "ES-EKF Fault Injection: dropout, outliers, and NaN robustness",
        fontweight="bold", y=0.995,
    )

    # ---- Panel 1: attitude error under dropout + NaN faults ----
    ax = fig.add_subplot(gs[0, :])
    shade(ax, DROPOUT, "gray", label="GPS+UWB out (8-16 s)")
    shade(ax, IMU_NAN, "#eda100")
    shade(ax, MAG_NAN, "#eda100")
    shade(ax, GPS_NAN, "#eda100")
    for name in ("baseline", "gnss_dropout", "nan_burst", "kitchen_sink"):
        g = runs[name]
        ax.plot(g["time"], g["att_err_deg"], color=COLOR[name], lw=1.1,
                label=LABEL[name])
    ax.set_ylabel("Attitude error (deg)")
    ax.set_title("Attitude error: faults barely move it off baseline",
                 fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.annotate("NaN bursts (IMU / mag / GPS)", xy=(8.4, 1.15), fontsize=8,
                color="#8a6100",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    style(ax)

    # ---- Panel 2: the innovation gate, log scale ----
    ax = fig.add_subplot(gs[1, :])
    shade(ax, GPS_OUTLIERS, "#e34948", label="GPS says ±300 km (4-6 s)")
    shade(ax, MAG_OUTLIERS, "#e34948")
    for name in ("baseline", "outliers_gated", "outliers_ungated"):
        g = runs[name]
        err = np.maximum(g["att_err_deg"], 1e-4)
        ax.plot(g["time"], err, color=COLOR[name], lw=1.1, label=LABEL[name])
    ax.set_yscale("log")
    ax.set_ylabel("Attitude error (deg, log)")
    ax.set_title(
        "Chi-square innovation gate: same garbage data, fused blindly vs rejected",
        fontsize=10, loc="left",
    )
    peak = float(runs["outliers_ungated"]["att_err_deg"].max())
    ax.annotate(
        f"gate off: {peak:.0f}° peak, never recovers",
        xy=(12.0, 2.5), fontsize=8, color="#e34948",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )
    ax.annotate(
        "gate on: indistinguishable from baseline", xy=(11.5, 0.012),
        fontsize=8, color="#4a3aa7",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.annotate("mag 1000x (10-10.5 s)", xy=(9.2, 3e-4), fontsize=8, color="#b53837")
    style(ax)

    # ---- Panel 3: attitude error by axis (dropout run) ----
    ax = fig.add_subplot(gs[2, :])
    shade(ax, DROPOUT, "gray")
    g = runs["gnss_dropout"]
    for axis, label in (("x", "roll err (x)"), ("y", "pitch err (y)"),
                        ("z", "yaw err (z)")):
        ax.plot(g["time"], g[f"att_err_{axis}_deg"], color=AXIS_COLOR[axis],
                lw=1.0, label=label)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.3)
    ax.set_ylabel("Attitude error (deg)")
    ax.set_title(
        "Attitude error by axis (dropout run): tilt is accel+mag limited, "
        "yaw is mag limited",
        fontsize=10, loc="left",
    )
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    style(ax)

    # ---- Panel 4: position error by axis, without vs with barometer ----
    g_nb = runs["gnss_dropout_nobaro"]
    g_b = runs["gnss_dropout"]
    lim = 1.15 * max(
        max(abs(g_nb[f"pos_err_{a}_m"].min()), g_nb[f"pos_err_{a}_m"].max())
        for a in "xyz"
    )
    for col, g, sub in ((0, g_nb, "no barometer"), (1, g_b, "with VN-200 barometer, 25 Hz")):
        ax = fig.add_subplot(gs[3, col])
        shade(ax, DROPOUT, "gray")
        for axis in "xyz":
            label = {"x": "x", "y": "y", "z": "z (altitude)"}[axis]
            ax.plot(g["time"], g[f"pos_err_{axis}_m"], color=AXIS_COLOR[axis],
                    lw=1.0, label=label)
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.3)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("time (s)")
        ax.set_title(f"Position error by axis: {sub}", fontsize=10, loc="left")
        if col == 0:
            ax.set_ylabel("Position error (m)")
            ax.legend(loc="lower left", fontsize=8)
        else:
            ax.tick_params(labelleft=False)
            zpk_nb = g_nb["pos_err_z_m"].abs().max()
            zpk_b = g_b["pos_err_z_m"].abs().max()
            ax.annotate(
                f"altitude pinned by baro:\n|z| peak {zpk_nb:.2f} m -> {zpk_b:.2f} m",
                xy=(8.6, -0.78 * lim), fontsize=8, color="#b04a17",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            )
        style(ax)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "esekf_fault_injection.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
