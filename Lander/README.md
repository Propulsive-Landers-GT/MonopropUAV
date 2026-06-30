# Lander Flight Software (FSW)

This directory contains the flight control loop, sensor fusion, and guidance algorithms for the Monopropellant UAV Lander. The software is written in Rust and is designed to run in real-time on the flight computer.

## Software Architecture

The flight software operates a multi-rate control loop:
*   **Sensor Fusion (500 Hz)**: Collects telemetry from the onboard sensors and updates a 12-state Unified Extended Kalman Filter (EKF) to estimate position, velocity, attitude (Euler angles/quaternion), and gyroscope biases.
*   **Guidance & Navigation (1 Hz)**: Operates a phase-aware trajectory planner. It dynamically computes fuel-optimal paths to the hover and landing targets using a Lossless Convexification solver (powered by the `Clarabel` SOCP solver).
*   **Control Loop (50 Hz)**: Uses a real-time Non-linear Model Predictive Controller (NMPC) powered by the PANOC solver (`optimization_engine`) to track the reference trajectory.

```
+-----------------------------------------------------------------+
|                                                                 |
|                         Flight Computer                         |
|                                                                 |
|   +------------------+                   +------------------+   |
|   |                  |                   |                  |   |
|   |   Sensors (IMU)  |                   |   Operator (CLI) |   |
|   |                  |                   |                  |   |
|   +--------+---------+                   +--------+---------+   |
|            |                                      |             |
|            | 500 Hz (Raw telemetry)               | Stdin channel
|            v                                      v             |
|   +--------+---------+                   +--------+---------+   |
|   |                  |                   |                  |   |
|   |  Sensor Fusion   |                   |  Command Handler |   |
|   |      (EKF)       |                   |  (Arm / Launch)  |   |
|   |                  |                   |                  |   |
|   +--------+---------+                   +--------+---------+   |
|            |                                      |             |
|            | Fused state                          | State transitions
|            +-----------------+--------------------+             |
|                              |                                  |
|                              v                                  |
|                    +---------+---------+                        |
|                    |                   |                        |
|                    |   Flight State    |                        |
|                    |     Machine       |                        |
|                    |                   |                        |
|                    +---------+---------+                        |
|                              |                                  |
|            +-----------------+-----------------+                |
|            | 1 Hz                              | 50 Hz          |
|            v                                   v                |
|   +--------+---------+                +--------+---------+      |
|   |                  |                |                  |      |
|   |    Trajectory    | Reference path |       NMPC       |      |
|   |    Generator     +--------------->|    Controller    |      |
|   |    (Lossless)    |                |     (PANOC)      |      |
|   |                  |                |                  |      |
|   +------------------+                +--------+---------+      |
|                                                |                |
|                                                | Commanded forces
|                                                v                |
|                                       +--------+---------+      |
|                                       |                  |      |
|                                       | Actuator Mapping |      |
|                                       |   (Slew limits)  |      |
|                                       |                  |      |
|                                       +--------+---------+      |
|                                                |                |
|                                                | TVC / Throttle |
|                                                v                |
|                                       +--------+---------+      |
|                                       |   MCAP Logger    |      |
|                                       |  (flight_log)    |      |
|                                       +------------------+      |
+-----------------------------------------------------------------+
```

---

## Hardware Configuration (VectorNav VN-200)

The state estimator is tuned for the **VectorNav VN-200** Tactical-Grade GPS/INS unit. 
*   **Telemetry**: Accel, Gyro, and Mag measurements are read at 500 Hz. GNSS updates are fused at 5 Hz.
*   **Sensor Metadata**: Registered directly within the logged MCAP file schemas for traceability during analysis.
*   **Dynamic Mass Depletion**: Fuel mass is calculated at runtime by integrating proper acceleration measurements from the VN-200 to model mass flow rate ($T / (I_{sp} \cdot g_0)$ where $I_{sp} = 180\,\text{s}$), clamping at a dry mass floor of $50.0\,\text{kg}$.

---

## Flight Phases & Transitions

The Lander transitions through the following discrete flight phases:
1.  **`Standby`**: The resting state on the launch pad. Trajectory generation is pre-calculated. Thrust is locked to zero. Awaiting manual operator arm command.
2.  **`Armed`**: Safety interlock removed. Actuators are ready. Awaiting manual operator launch command.
3.  **`Ascent`**: Thrust is active. The vehicle climbs to the target altitude (50m) tracking the lossless convex trajectory.
4.  **`Hover`**: Triggered when altitude reaches 95% of target (47.5m). Holds position for 20 seconds.
5.  **`Descent`**: Triggers a guided descent trajectory using the Lossless solver to perform a controlled decelerating landing.
6.  **`Landed`**: Touchdown detected ($z \le 0$). Thrust is disabled, controls zeroed, and logging is finalized.

---

## Actuator Safety & Fallbacks

*   **NaN / Infinity Check**: Instantly detects optimization failures or numerical errors in the NMPC output and falls back to the last valid command sequence.
*   **Gimbal & Thrust Slew-Rate Limiting**: Commands sent to the physical TVC (Thrust Vector Control) actuators are rate-limited to protect hardware:
    *   **Gimbal Servos**: Limited to $100^\circ/\text{s}$ (max $2^\circ$ change per 20ms step).
    *   **Throttle Valve**: Limited to $2000\,\text{N}/\text{s}$ (max $40\,\text{N}$ change per 20ms step).
*   **Thrust Cost Penalty**: The NMPC diagonal cost matrix includes a weight penalty of `0.05` on thrust magnitude changes to prevent high-frequency actuator slamming and bang-bang behaviors.

---

## Interactive Command Console

The simulation runs a non-blocking console interface. Operators can type the following commands directly into the terminal to transition the flight phase:
*   `arm`: Arms the vehicle (Standby $\rightarrow$ Armed).
*   `disarm`: Disarms the vehicle back to Standby (Armed $\rightarrow$ Standby).
*   `launch`: Triggers liftoff (Armed $\rightarrow$ Ascent).

---

## Telemetry Logging (MCAP)

During flight, the computer writes real-time telemetry into `flight_log.mcap` using the standard MCAP robotics container format.
*   **JSON Encoding**: Channel schemas are serialized as JSON payloads, natively compatible with visualization software like **Foxglove Studio**.
*   **Available Channels**:
    *   `telemetry/sensor_data`: Raw IMU, GPS, UWB, and pressure readings.
    *   `telemetry/vehicle_state`: Fused position, velocity, attitude (Euler & quaternion), angular velocity, and mass.
    *   `telemetry/control_output`: Gimbal angles ($\theta, \phi$), thrust (Newtons), and world-frame Cartesian force vector.
    *   `telemetry/flight_phase`: Current phase transitions and elapsed time.

---

## How to Build and Run

1.  **Compile the code**:
    ```bash
    cargo build
    ```
2.  **Run the simulation**:
    ```bash
    cargo run
    ```
3.  **Commanding**: Type `arm` followed by `launch` in the terminal to takeoff. 
