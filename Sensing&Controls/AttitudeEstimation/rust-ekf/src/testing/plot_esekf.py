"""Plot ES-EKF inputs (IMU) and outputs (estimated state) from esekf_output.csv.

Run the driver first: `cargo run --bin esekf_test`, then `python3 plot_esekf.py`.
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "esekf_output.csv")
OUT_DIR = os.path.join(HERE, "outputs")
FLIGHT_CSV = os.path.join(HERE, "flight_data.csv")


def quat_mul(a, b):
    """Hamilton product of (w,x,y,z) quaternions, arrays of shape (N,4)."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def attitude_error_deg(q_est, q_ref):
    """Angle of the relative rotation q_ref^-1 (x) q_est, in degrees (N,)."""
    q_ref_inv = q_ref.copy()
    q_ref_inv[:, 1:] *= -1.0  # conjugate of a unit quaternion
    q_err = quat_mul(q_ref_inv, q_est)
    w = np.clip(np.abs(q_err[:, 0]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def quat_to_euler(qw, qx, qy, qz):
    """Hamilton quaternion (w,x,y,z) -> roll, pitch, yaw in degrees."""
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def main():
    df = pd.read_csv(CSV)
    t = df["time"].to_numpy()

    # ---- Figure 1: INPUTS (raw IMU fed to the filter) ----
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("ES-EKF Inputs (IMU from flight_data.csv)", fontweight="bold")

    ax[0].plot(t, df["accel_x"], label="accel_x", lw=0.8)
    ax[0].plot(t, df["accel_y"], label="accel_y", lw=0.8)
    ax[0].plot(t, df["accel_z"], label="accel_z", lw=0.8)
    ax[0].set_ylabel("Accel (m/s^2)")
    ax[0].legend(loc="upper right", ncol=3)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, df["gyro_x"], label="gyro_x", lw=0.8)
    ax[1].plot(t, df["gyro_y"], label="gyro_y", lw=0.8)
    ax[1].plot(t, df["gyro_z"], label="gyro_z", lw=0.8)
    ax[1].set_ylabel("Gyro (rad/s)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(loc="upper right", ncol=3)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    inputs_path = os.path.join(OUT_DIR, "esekf_inputs.png")
    fig.savefig(inputs_path, dpi=120)

    # ---- Figure 2: OUTPUTS (estimated nominal state) ----
    roll, pitch, yaw = quat_to_euler(df["qw"], df["qx"], df["qy"], df["qz"])

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        "ES-EKF Outputs (estimated state; synthetic zero-GPS smoke test)",
        fontweight="bold",
    )

    ax[0].plot(t, df["px"], label="px", lw=1.0)
    ax[0].plot(t, df["py"], label="py", lw=1.0)
    ax[0].plot(t, df["pz"], label="pz", lw=1.0)
    ax[0].set_ylabel("Position (m)")
    ax[0].legend(loc="upper left", ncol=3)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, df["vx"], label="vx", lw=1.0)
    ax[1].plot(t, df["vy"], label="vy", lw=1.0)
    ax[1].plot(t, df["vz"], label="vz", lw=1.0)
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].legend(loc="upper left", ncol=3)
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t, roll, label="roll", lw=1.0)
    ax[2].plot(t, pitch, label="pitch", lw=1.0)
    ax[2].plot(t, yaw, label="yaw", lw=1.0)
    ax[2].set_ylabel("Attitude (deg)")
    ax[2].set_xlabel("time (s)")
    ax[2].legend(loc="upper left", ncol=3)
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    outputs_path = os.path.join(OUT_DIR, "esekf_outputs.png")
    fig.savefig(outputs_path, dpi=120)

    print("wrote", inputs_path)
    print("wrote", outputs_path)

    # ---- Figure 3: VERIFICATION - estimated attitude vs reference quaternion ----
    # flight_data.csv carries a reference/ground-truth attitude (q_x,q_y,q_z,q_w).
    # Compare the filter's gyro-integrated attitude against it.
    flight = pd.read_csv(FLIGHT_CSV)
    n = min(len(flight), len(df))
    q_est = df[["qw", "qx", "qy", "qz"]].to_numpy()[:n]
    q_ref = flight[["q_w", "q_x", "q_y", "q_z"]].to_numpy()[:n]  # reorder to (w,x,y,z)

    err = attitude_error_deg(q_est, q_ref)
    rms = float(np.sqrt(np.mean(err**2)))
    final = float(err[-1])
    peak = float(np.max(err))

    roll_r, pitch_r, yaw_r = quat_to_euler(
        q_ref[:, 0], q_ref[:, 1], q_ref[:, 2], q_ref[:, 3]
    )
    roll_e, pitch_e, yaw_e = quat_to_euler(
        q_est[:, 0], q_est[:, 1], q_est[:, 2], q_est[:, 3]
    )
    tn = t[:n]

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        "ES-EKF Verification: estimated attitude vs reference quaternion",
        fontweight="bold",
    )

    ax[0].plot(tn, roll_e, label="roll est", lw=1.0)
    ax[0].plot(tn, roll_r, label="roll ref", lw=1.0, ls="--")
    ax[0].plot(tn, pitch_e, label="pitch est", lw=1.0)
    ax[0].plot(tn, pitch_r, label="pitch ref", lw=1.0, ls="--")
    ax[0].plot(tn, yaw_e, label="yaw est", lw=1.0)
    ax[0].plot(tn, yaw_r, label="yaw ref", lw=1.0, ls="--")
    ax[0].set_ylabel("Euler (deg)")
    ax[0].legend(loc="upper left", ncol=3, fontsize=8)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(tn, err, color="crimson", lw=1.0)
    ax[1].set_ylabel("Total attitude error (deg)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_title(f"RMS = {rms:.3f} deg   |   peak = {peak:.3f} deg   |   final = {final:.3f} deg")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    verify_path = os.path.join(OUT_DIR, "esekf_attitude_error.png")
    fig.savefig(verify_path, dpi=120)
    print("wrote", verify_path)
    print(f"attitude error vs reference: RMS={rms:.3f} deg, peak={peak:.3f} deg, final={final:.3f} deg")


if __name__ == "__main__":
    main()
