# Rust Extended Kalman Filter (EKF) Library

A Rust implementation of the Extended Kalman Filter. This library provides a generic EKF implementation along with specific models for attitude, altitude, and position estimation. It also includes an Error-State EKF (ES-EKF) for full 15-state inertial navigation with quaternion attitude.

## Features

- **Generic EKF Implementation**: A flexible, trait-based EKF that can be extended for various state estimation problems. This is the core of the EKF implementation handling prediction and update logic.
- **Pre-built Models**: Includes models for common aerospace estimation tasks:
  - **Altitude Estimation**: For vertical position and velocity estimation
  - **Attitude Estimation**: For orientation estimation
  - **XY Position**: For 2D position tracking
- **Error-State EKF (ES-EKF)**: A quaternion-based error-state filter for full inertial navigation (position, velocity, attitude, and IMU biases). Avoids the gimbal-lock and unit-norm problems of the Euler-angle EKF. See [ES_EKF.md](ES_EKF.md) for the full theory, math, and usage.

## Project Structure

```
src/
├── lib.rs                   # Library root and public API
├── ekf/
│   ├── mod.rs               # EKF module exports
│   ├── filter.rs            # Core EKF implementation
│   └── model.rs             # EKFModel trait definition
├── es_ekf/                  # Error-State EKF (see ES_EKF.md)
│   ├── mod.rs               # ES-EKF module exports
│   ├── filter.rs            # ErrorStateKalmanFilter implementation
│   └── model.rs             # ESEKFModel trait definition
├── models/                  # Pre-built models
│   ├── mod.rs               # Model exports and type aliases
│   ├── altitude.rs          # Altitude estimation model
│   ├── attitude.rs          # Attitude estimation model
│   ├── xy_position.rs       # 2D position tracking model
│   └── full_state_esekf.rs  # RocketState: 15-state INS model for the ES-EKF
└── testing/
    ├── attitude_test.rs     # Attitude EKF driver (binary: attitude_test)
    └── esekf_test.rs        # ES-EKF smoke-test driver (binary: esekf_test)
```

## Core Components

### 1. ExtendedKalmanFilter

The main EKF implementation that works with any model implementing the `EKFModel` trait. Key methods:

- `new()`: Create a new EKF instance
- `predict()`: Predict the next state using the process model
- `update()`: Update the state estimate with new measurements
- `get_state()`: Get the current state estimate
- `get_covariance()`: Get the current state covariance

### 2. EKFModel Trait

Implement this trait to create custom EKF models. Required methods:

- `parse_data()`: Convert raw sensor data to a measurement vector
- `state_transition_function()`: Define the process model
- `state_transition_jacobian()`: Compute the process model Jacobian
- `measurement_prediction_function()`: Define the measurement model
- `measurement_prediction_jacobian()`: Compute the measurement model Jacobian

## Pre-built Models

### 1. AltitudeModel

Estimates vertical position and velocity using altitude measurements from sensors like barometers or GPS.

**State Vector**: `[altitude]`

**Input Data Format**: `[timestamp, lidar_measurement] `

**Example Usage**:
```rust
use rust_ekf::{AltitudeModel, AltitudeEKF};
use ndarray::array;

// Initialize with 10ms timestep
let model = AltitudeModel::new(0.01);
let initial_state = array![0.0];      // Starting altitude (meters)
let initial_measurement = array![0.0];

// Create EKF instance
let mut ekf = AltitudeEKF::new(
    initial_state,
    initial_measurement,
    0.01,  // dt (seconds)
    array![[0.1]],  // Process noise covariance Q (1x1 matrix)
    array![[0.01]], // Measurement noise covariance R (1x1 matrix)
    array![[1.0]],  // Initial state covariance P (1x1 matrix)
    model,
);

// In your main loop:
loop {
    // 1. Get sensor data (time in seconds, altitude in meters)
    let sensor_data = [current_time, altitude_measurement];
    
    // 2. Prediction step
    ekf.predict();
    
    // 3. Update with new measurement
    ekf.update(&sensor_data);
    
    // 4. Get current state estimate
    let state = ekf.get_state();
    let altitude = state[0];
}
```

### 2. AttitudeModel

Estimates 3D orientation and body angular velocity using a fast 9-axis IMU model (gyroscope, accelerometer, magnetometer).

**Dynamics model**: Euler angle kinematics + constant body-rate angular dynamics between updates. This keeps the predict step cheap while the gyro, accelerometer, and magnetometer measurements continuously pull the state back toward the observed motion.

**State Vector**: `[φ, θ, ψ, ωx, ωy, ωz]`

**Input Data Format**: `[timestamp, gx, gy, gz, ax, ay, az, mx, my, mz]`
- `gx,gy,gz`: Gyroscope measurements (rad/s)
- `ax,ay,az`: Accelerometer measurements (normalised internally to a direction vector)
- `mx,my,mz`: Magnetometer measurements (normalised internally to a direction vector)

**Example Usage**:
```rust
use rust_ekf::{AttitudeModel, AttitudeEKF};
use ndarray::array;

// Fast default model with built-in gravity and magnetic reference vectors
let model = AttitudeModel::new(0.01);

// Or provide local field references explicitly:
// let model = AttitudeModel::with_reference_vectors(
//     0.01,
//     [0.0, 0.0, -1.0],
//     [0.45, 0.0, -0.89],
// )?;

// Initial state: [φ, θ, ψ, ωx, ωy, ωz]
let initial_state = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
let initial_measurement = array![0.0; 9];

let mut ekf = AttitudeEKF::new(
    initial_state,
    initial_measurement,
    0.01,  // dt (seconds)
    Array2::eye(6) * 0.1,   // Process noise covariance Q (6x6 matrix)
    Array2::eye(9) * 0.01,  // Measurement noise covariance R (9x9 matrix)
    Array2::eye(6) * 1.0,   // Initial state covariance P (6x6 matrix)
    model,
);

// In your main loop:
loop {
    // Get IMU data (time, gyro, accel, mag)
    let imu_data = [
        current_time,
        gx, gy, gz,   // Gyroscope (rad/s)
        ax, ay, az,   // Accelerometer (normalised)
        mx, my, mz,   // Magnetometer (normalised)
    ];

    // `step()` uses the timestamp in `imu_data` to update dt before prediction.
    ekf.step(&imu_data);

    let state = ekf.get_state();
    let (roll, pitch, yaw)          = (state[0], state[1], state[2]);
    let (omega_x, omega_y, omega_z) = (state[3], state[4], state[5]);
}
```

**Process noise tuning note**: The Q matrix allows you to specify different noise levels for each state variable. For uniform noise, use `Array2::eye(n) * scalar` where `n` is the state dimension.

### 3. XYPositionModel

Tracks 2D position and velocity in the horizontal plane using GPS or other positioning data.

**State Vector**: `[x, y, vx, vy]`
- `x, y`: Position in meters (relative to initial position)
- `vx, vy`: Velocity in m/s

**Input Data Format**: `[longitude, latitude]` (degrees)

**Features**:
- Automatically handles conversion from lat/lon to local tangent plane coordinates
- Maintains velocity estimates for position prediction
- Uses a constant velocity motion model

**Example Usage**:
```rust
use rust_ekf::{XYPositionModel, XYPositionEKF};
use ndarray::array;

// Initialize with 1Hz update rate
let model = XYPositionModel::new(1.0);
let initial_state = array![0.0, 0.0, 0.0, 0.0];  // x, y, vx, vy
let initial_measurement = array![0.0, 0.0];      // Initial position

let mut ekf = XYPositionEKF::new(
    initial_state,
    initial_measurement,
    1.0,    // dt (1 second between updates)
    Array2::eye(4) * 0.1,    // Process noise covariance Q (4x4 matrix)
    Array2::eye(2) * 1.0,    // Measurement noise covariance R (2x2 matrix)
    Array2::eye(4) * 10.0,   // Initial state covariance P (4x4 matrix)
    model,
);

// In your main loop:
loop {
    // Get position data (longitude, latitude in degrees)
    let position_data = [longitude, latitude];
    
    ekf.predict();
    ekf.update(&position_data);
    
    let state = ekf.get_state();
    let (x, y, vx, vy) = (state[0], state[1], state[2], state[3]);
}
```

## Error-State EKF (ES-EKF)

For full inertial navigation (position, velocity, quaternion attitude, and accel/gyro
biases) the library provides a separate **Error-State EKF** in `src/es_ekf/`, with the
`RocketState` model in `src/models/full_state_esekf.rs`.

Unlike the Euler-angle `AttitudeModel`, the ES-EKF tracks attitude as a unit quaternion
in a 16D **nominal state** while running the Kalman filter on a minimal 15D **error
state** (`[dp, dv, dtheta, da_b, dw_b]`). This avoids gimbal lock and the quaternion
unit-norm / singular-covariance issues of a direct EKF.

- Fast loop: `predict(&imu, dt)` integrates accel + gyro every step.
- Slow loop: `update(&gps, &r)` fuses a GPS position fix when available.
- Convenience alias: `RocketESEKF = ErrorStateKalmanFilter<RocketState>`.

Run the smoke-test driver with `cargo run --bin esekf_test`.

See **[ES_EKF.md](ES_EKF.md)** for the full theory, the math and its code mapping,
usage examples, current limitations, and the roadmap to a production implementation.

## Model Tuning Tips

1. **Process Noise Covariance (Q)**:
   - Higher values make the filter more responsive to measurements
   - Lower values make the filter trust the model more
   - Can specify different noise levels for each state variable
   - For uniform noise: `Array2::eye(n) * scalar` where `n` is state dimension
   - Typical diagonal values: 0.01 to 1.0

2. **Measurement Noise Covariance (R)**:
   - Should match your sensor's expected error characteristics
   - Lower values indicate more trust in the measurements
   - Can specify different noise levels for each measurement
   - For uniform noise: `Array2::eye(m) * scalar` where `m` is measurement dimension
   - Typical diagonal values: 0.001 to 1.0

3. **Initial State Covariance (P)**:
   - Represents uncertainty in initial state
   - Larger values allow faster initial convergence
   - Can specify different uncertainty for each state variable
   - For uniform uncertainty: `Array2::eye(n) * scalar`
   - Typical diagonal values: 0.1 to 10.0

## Adding a Custom Model

1. Create a new struct implementing the `EKFModel` trait
2. Define your state transition and measurement models
3. Create a type alias for convenience

Example:
```rust
use ndarray::{Array1, Array2};
use rust_ekf::ekf::model::EKFModel;

struct MyCustomModel {
    // Model parameters
}

impl EKFModel for MyCustomModel {
    // Implement required trait methods
}

// Type alias for convenience
pub type MyCustomEKF = crate::ekf::filter::ExtendedKalmanFilter<MyCustomModel>;
```
## References

- [Wikipedia: Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter)
- [Math Behind EKF](https://www.alanzucconi.com/2022/07/24/extended-kalman-filter/)
- [EKF for Attitude Estimation](https://ahrs.readthedocs.io/en/latest/filters/ekf.html)
- [Joan Sola, "Quaternion kinematics for the error-state Kalman filter"](https://arxiv.org/abs/1711.02508) (basis for the ES-EKF; see [ES_EKF.md](ES_EKF.md))
