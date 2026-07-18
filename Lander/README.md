# Lander Flight Software

This directory contains the flight control loop, navigation estimation, and guidance tracking algorithms for the Lander. The software is written in Rust and is designed to run in real-time on the onboard flight computer.

---

## Software Architecture

The flight software is structured into four distinct, modular components designed to run within a high-frequency real-time loop:

```
+-------------------------------------------------------------------------+
|                                                                         |
|                             Flight Computer                             |
|                                                                         |
|     +------------------+                   +------------------+         |
|     |                  |                   |                  |         |
|     |   Sensors (IMU)  |                   |   Operator (CLI) |         |
|     |                  |                   |                  |         |
|     +--------+---------+                   +--------+---------+         |
|              |                                      |                   |
|              | 500 Hz Telemetry                     | CLI Commands      |
|              v                                      v                   |
|     +--------+--------------------------------------+---------+         |
|     |                                                         |         |
|     |                 Flight State Machine                    |         |
|     |           (Path determination & transitions)            |         |
|     |                                                         |         |
|     +---+----------------------+--------------------------+---+         |
|         |                      |                          |             |
|         | Rates                | Fused State / Traj       | Step        |
|         v                      v                          v             |
|     +---+------+       +-------+--------+         +-------+-------+     |
|     |          |       |                |         |               |     |
|     |          |       |   Autopilot    |         |   Actuator    |     |
|     |          |       |  - Navigator   |         |  Controller   |     |
|     |Scheduler |       |  - Guidance    |         | - RCS Control |     |
|     |          |       |  - MPC         |         | - Slew/Clamp  |     |
|     |          |       |                |         |               |     |
|     +----------+       +-------+--------+         +-------+-------+     |
|                                |                          |             |
|                                | Path Reference           | TVC / RCS   |
|                                v                          v             |
|                        +-------+--------------------------+-------+     |
|                        |                MCAP Logger               |     |
|                        +------------------------------------------+     |
+-------------------------------------------------------------------------+
```

### 1. Flight State Manager (FSM Coordinator)
*   **Orchestration**: Directs the main loop execution, handles checking safety boundaries, and manages state transitions.
*   **Decoupled State Transitions**: Implements a pure path determination function (`next_phase`) to return the target phase, and a unified state modification hook (`on_transition`) to execute transitions safely.

### 2. Scheduler
*   **Rate Management**: Tracks monotonic mission time and decides when tasks are due to run:
    *   **Navigation Updates (500 Hz)**: Triggers sensor estimation updates.
    *   **Guidance Planner (1 Hz)**: Triggers trajectory re-generation.
    *   **MPC Tracking Control (50 Hz)**: Triggers optimization controller updates.

### 3. Autopilot
*   **Navigator (EKF)**: Fuses VN-200 IMU, GPS, and UWB telemetry inside a 15-state Error-State Kalman Filter (ES-EKF) to estimate vehicle position, velocity, attitude (quaternion), and sensor biases.
*   **Guidance (Lossless)**: Dynamically plans fuel-optimal reference trajectories using lossless convexification algorithms.
*   **MPC Controller (PANOC)**: Tracks the reference trajectory using a Non-linear Model Predictive Control optimization solver.

### 4. Actuator Controller
*   **Roll Control**: Runs the reaction control system (RCS) controller to regulate vehicle roll.
*   **Slew-Rate Clamping**: Limits maximum change rate on gimbal and thrust command signals to prevent actuator slamming and protect flight hardware.

---

## Flight Phases & Transitions

The Lander transitions through the following discrete flight phases:
1.  **`Standby`**: Resting on the launch pad. Trajectory generation is pre-calculated. Thrust is locked to zero. Awaiting manual operator arm command.
2.  **`Armed`**: Safety interlocks removed. Actuators are ready. Awaiting manual operator launch command.
3.  **`Ascent`**: Thrust is active. The vehicle climbs to the target altitude (50m) tracking the lossless convex trajectory.
4.  **`Hover`**: Holds target hover position for 20 seconds.
5.  **`Descent`**: Triggers a guided descent trajectory to perform a controlled decelerating landing back at the pad.
6.  **`Landed`**: Touchdown detected (altitude z <= 0.1m and estimated speed ||v|| < 0.2m/s). Thrust is disabled, controls zeroed, and logging is finalized.

---

## Actuator Safety & Fallbacks

*   **NaN / Infinity Check**: Instantly detects optimization failures or numerical errors in the MPC output and falls back to the last valid control commands.
*   **Actuator Clamping Limits**:
    *   **Gimbal Servos**: TVC angles are rate-limited to $100^\circ/\text{s}$ (max $2^\circ$ change per 20ms step).
    *   **Throttle Valve**: Thrust change rates are limited to $2000\,\text{N}/\text{s}$ (max $40\,\text{N}$ change per 20ms step).
*   **Thrust Cost Penalty**: The MPC cost formulation penalizes high-frequency thrust changes to prevent actuator oscillations.

---

## Real-Time Execution & Safety Contingencies

*   **Real-Time Priority**: Thread scheduling on Linux targets is configured to `SCHED_FIFO` with a priority of `80` to minimize loop execution jitter.
*   **GPS/UWB Denial Emergency Landing**: If absolute position data (GPS/UWB) is lost for more than 5 seconds during active flight, the FSM transitions directly to `Descent` to attempt an emergency soft landing using EKF dead reckoning. If absolute position denial exceeds 15 seconds, a hard safety flight termination is triggered.
*   **Decoupled Master Timing**: Governed by a dedicated monotonic `Clock` struct owned by the FSM loop, ensuring timing references are independent of logging or OS scheduling anomalies.

---

## Interactive Command Console

The flight computer binary runs a non-blocking console interface. Operators can type commands directly into the terminal to transition the flight phase:
*   `arm`: Arms the vehicle (Standby $\rightarrow$ Armed).
*   `disarm`: Disarms the vehicle back to Standby (Armed $\rightarrow$ Standby).
*   `launch`: Triggers liftoff (Armed $\rightarrow$ Ascent).

---

## Telemetry Logging (MCAP)

During flight, the computer writes real-time telemetry into `flight_log_*.mcap` files using the standard MCAP robotics container format.
*   **Asynchronous Logging**: Disk writes and JSON serialization run on a dedicated worker thread decoupled from the 500 Hz control loop via a lock-free channel.
*   **Visualizing**: Log files can be directly dragged and dropped into visualization platforms like **Foxglove Studio**.
*   **Logged Channels**:
    *   `telemetry/sensor_data` (500 Hz): Raw IMU, GPS, UWB, and pressure readings.
    *   `telemetry/vehicle_state` (500 Hz): Fused position, velocity, attitude (Euler & quaternion), angular velocity, and mass.
    *   `telemetry/control_output` (50 Hz / Event-driven): TVC gimbal angles, thrust, and RCS command inputs.
    *   `telemetry/flight_phase` (On-transition): Recorded only on flight phase transition updates to save disk space.
    *   `telemetry/diagnostics` (Event-driven): Phase transitions, solver alerts, fallbacks, and flight termination causes.

---

## How to Build and Run

1.  **Compile the code**:
    ```bash
    cargo build
    ```
2.  **Run the FSM console wrapper**:
    ```bash
    cargo run
    ```
3.  **Commanding**: Type `arm` followed by `launch` in the terminal to takeoff.
