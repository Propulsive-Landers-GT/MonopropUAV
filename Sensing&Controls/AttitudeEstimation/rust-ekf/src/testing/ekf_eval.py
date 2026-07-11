"""Compare ES-EKF attitude estimate against the truth quaternion in
flight_data.csv and render a 3D visualization of the estimated trajectory."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import os
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
truth = pd.read_csv(f"{BASE}/flight_data.csv")
est = pd.read_csv(f"{BASE}/esekf_output.csv")

n = min(len(truth), len(est))
truth, est = truth.iloc[:n], est.iloc[:n]

# Truth CSV stores q as (x, y, z, w); EKF output stores (w, x, y, z).
q_true = truth[["q_w", "q_x", "q_y", "q_z"]].to_numpy()
q_est = est[["qw", "qx", "qy", "qz"]].to_numpy()

# Normalize and resolve the q ~ -q double-cover ambiguity.
q_true /= np.linalg.norm(q_true, axis=1, keepdims=True)
q_est /= np.linalg.norm(q_est, axis=1, keepdims=True)
dots = np.sum(q_true * q_est, axis=1)
q_est[dots < 0] *= -1.0
dots = np.abs(dots).clip(0.0, 1.0)

# Geodesic attitude error angle between estimate and truth.
ang_err_deg = np.degrees(2.0 * np.arccos(dots))

# Percent deviation: quaternion chordal distance ||q_est - q_true|| relative to
# ||q_true|| (= 1 for a unit quaternion), per sample.
pct_dev = np.linalg.norm(q_est - q_true, axis=1) / np.linalg.norm(q_true, axis=1) * 100.0

# Per-component deviation relative to each component's RMS magnitude.
comp_names = ["qw", "qx", "qy", "qz"]
comp_pct = {
    name: np.mean(np.abs(q_est[:, i] - q_true[:, i])) / np.sqrt(np.mean(q_true[:, i] ** 2)) * 100.0
    for i, name in enumerate(comp_names)
}

t = truth["time"].to_numpy()
print(f"samples compared              : {n}")
print(f"avg percent deviation (norm)  : {pct_dev.mean():.4f} %")
print(f"max percent deviation (norm)  : {pct_dev.max():.4f} %")
print(f"avg attitude error            : {ang_err_deg.mean():.4f} deg")
print(f"max attitude error            : {ang_err_deg.max():.4f} deg  at t={t[ang_err_deg.argmax()]:.2f}s")
for name, v in comp_pct.items():
    print(f"avg |d{name}| / rms({name})        : {v:.4f} %")

# ---------------------------------------------------------------- 3D figure
pos = est[["px", "py", "pz"]].to_numpy()

fig = plt.figure(figsize=(14, 7))

ax = fig.add_subplot(1, 2, 1, projection="3d")
pts = pos.reshape(-1, 1, 3)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = Line3DCollection(segs, cmap="viridis", linewidths=2)
lc.set_array(ang_err_deg[:-1])
ax.add_collection3d(lc)
ax.scatter(*pos[0], color="green", s=60, label="start")
ax.scatter(*pos[-1], color="red", s=60, label="end")

# Draw body Z-axis arrows (truth green, EKF orange) at intervals along the path.
def rotate_z(q):
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

span = np.ptp(pos, axis=0).max() or 1.0
alen = 0.08 * span
for i in range(0, n, n // 25):
    for q, c in ((q_true[i], "tab:green"), (q_est[i], "tab:orange")):
        d = rotate_z(q) * alen
        ax.plot(
            [pos[i, 0], pos[i, 0] + d[0]],
            [pos[i, 1], pos[i, 1] + d[1]],
            [pos[i, 2], pos[i, 2] + d[2]],
            color=c, alpha=0.8, linewidth=1.2,
        )
ax.plot([], [], color="tab:green", label="truth body-Z")
ax.plot([], [], color="tab:orange", label="EKF body-Z")

lo, hi = pos.min(axis=0), pos.max(axis=0)
mid, r = (lo + hi) / 2, np.max(hi - lo) / 2 or 1.0
ax.set_xlim(mid[0] - r, mid[0] + r)
ax.set_ylim(mid[1] - r, mid[1] + r)
ax.set_zlim(mid[2] - r, mid[2] + r)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_title("ES-EKF estimated trajectory\n(color = attitude error [deg])")
ax.legend(loc="upper left", fontsize=8)
fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.1, label="attitude error [deg]")

ax2 = fig.add_subplot(2, 2, 2)
for i, name in enumerate(comp_names):
    ax2.plot(t, q_true[:, i], linewidth=1.0, label=f"{name} truth")
    ax2.plot(t, q_est[:, i], "--", linewidth=1.0, label=f"{name} EKF")
ax2.set_ylabel("quaternion component")
ax2.legend(fontsize=6, ncol=4)
ax2.set_title("Attitude: truth (solid) vs EKF (dashed)")
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(t, ang_err_deg, color="tab:red", linewidth=1.0, label="attitude error [deg]")
ax3.plot(t, pct_dev, color="tab:blue", linewidth=1.0, label="percent deviation [%]")
ax3.set_xlabel("time [s]")
ax3.legend(fontsize=8)
ax3.set_title(
    f"avg deviation {pct_dev.mean():.3f}% | avg attitude error {ang_err_deg.mean():.3f} deg"
)
ax3.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(OUT_DIR, "ekf_vs_truth_3d.png")
fig.savefig(out, dpi=140)
print(f"saved: {out}")
