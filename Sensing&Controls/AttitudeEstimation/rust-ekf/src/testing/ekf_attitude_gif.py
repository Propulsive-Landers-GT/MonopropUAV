"""Animated GIF: rocket attitude from the ES-EKF (blue) vs flight-data truth
(red), rendered as a 3D stick rocket with fins, with t / r-p-y / omega readouts."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "ekf_vs_data_attitude.gif")

truth = pd.read_csv(f"{BASE}/flight_data.csv")
est = pd.read_csv(f"{BASE}/esekf_output.csv")
n = min(len(truth), len(est))
truth, est = truth.iloc[:n], est.iloc[:n]

t = truth["time"].to_numpy()
gyro = truth[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
q_true = truth[["q_w", "q_x", "q_y", "q_z"]].to_numpy()
q_ekf = est[["qw", "qx", "qy", "qz"]].to_numpy()
q_true /= np.linalg.norm(q_true, axis=1, keepdims=True)
q_ekf /= np.linalg.norm(q_ekf, axis=1, keepdims=True)

STEP = 5          # subsample: 100 Hz data -> 20 fps == real-time playback
FPS = 20
frames = range(0, n, STEP)


def rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rpy_deg(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


# Rocket geometry in the body frame: a rod along body-Z with a fin cross at
# the base, plus faint body X/Y axes at the center.
ROD = np.array([[0, 0, -0.6], [0, 0, 0.9]])
FIN_X = np.array([[-0.35, 0, -0.6], [0.35, 0, -0.6]])
FIN_Y = np.array([[0, -0.35, -0.6], [0, 0.35, -0.6]])
AX_X = np.array([[0, 0, 0], [0.6, 0, 0]])
AX_Y = np.array([[0, 0, 0], [0, 0.6, 0]])
BODY_PARTS = [ROD, FIN_X, FIN_Y]

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(projection="3d")
ax.set_xlim(-2, 2), ax.set_ylim(-2, 2), ax.set_zlim(-2, 2)
ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
fig.suptitle("Rocket Attitude: EKF (blue) vs. Flight Data (red)", y=0.97)

# EKF drawn as a thick translucent outline, truth as a thin core on top so
# both stay visible when they overlap (which is most of the flight).
lines_ekf = [ax.plot([], [], [], color="dodgerblue", linewidth=lw, alpha=0.6, zorder=4)[0]
             for lw in (6.0, 4.5, 4.5)]
lines_true = [ax.plot([], [], [], color="red", linewidth=lw, alpha=0.95, zorder=5)[0]
              for lw in (2.2, 1.6, 1.6)]
axis_x = ax.plot([], [], [], color="salmon", linewidth=1.2, alpha=0.55)[0]
axis_y = ax.plot([], [], [], color="lightgreen", linewidth=1.2, alpha=0.55)[0]

txt_t = fig.text(0.06, 0.90, "", fontsize=10, color="black")
txt_ekf = fig.text(0.06, 0.865, "", fontsize=9, color="dodgerblue")
txt_dat = fig.text(0.06, 0.83, "", fontsize=9, color="red")
txt_w = fig.text(0.06, 0.795, "", fontsize=9, color="gray")


def set_line(line, seg):
    line.set_data(seg[:, 0], seg[:, 1])
    line.set_3d_properties(seg[:, 2])


def update(i):
    r_true, r_ekf = rotmat(q_true[i]), rotmat(q_ekf[i])
    for line, part in zip(lines_true, BODY_PARTS):
        set_line(line, part @ r_true.T)
    for line, part in zip(lines_ekf, BODY_PARTS):
        set_line(line, part @ r_ekf.T)
    set_line(axis_x, AX_X @ r_true.T)
    set_line(axis_y, AX_Y @ r_true.T)

    re_, pe, ye = rpy_deg(q_ekf[i])
    rd, pd_, yd = rpy_deg(q_true[i])
    txt_t.set_text(f"t = {t[i]:.2f} s")
    txt_ekf.set_text(f"[EKF]  r/p/y = {re_:.1f}°, {pe:.1f}°, {ye:.1f}°")
    txt_dat.set_text(f"[DATA] r/p/y = {rd:.1f}°, {pd_:.1f}°, {yd:.1f}°")
    txt_w.set_text(
        f"ω = ({gyro[i,0]:.3f}, {gyro[i,1]:.3f}, {gyro[i,2]:.3f}) rad/s"
    )
    return []


anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS)
anim.save(OUT, writer=PillowWriter(fps=FPS), dpi=80)
print(f"saved: {OUT}  ({len(list(frames))} frames @ {FPS} fps, real-time speed)")
