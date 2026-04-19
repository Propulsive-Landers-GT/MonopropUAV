use std::time::Instant;
use ndarray::{Array1, Array2, s};
use nalgebra::{Vector3, UnitQuaternion};
use rust_ekf::ExtendedKalmanFilter;
use rust_ekf::AttitudeModel;
use rust_lossless::LosslessSolver;

// TODO: Probably make a global MPC wrapper in the MPC algorithm instead of having a unique wrapper for the Lander and Simulation
#[derive(Debug, Clone)]
pub struct MPC {
    pub n: usize,        // state dimension [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    pub m: usize,        // control dimension [thrust_x, thrust_y, thrust_z]
    pub n_steps: usize,  // MPC horizon steps
    pub dt: f64,         // time step
    pub mass: f64,       // vehicle mass
    pub min_thrust: f64, // minimum thrust
    pub max_thrust: f64, // maximum thrust
    pub q: Array2<f64>,  // state cost matrix
    pub r: Array2<f64>,  // control cost matrix
    pub qn: Array2<f64>, // terminal state cost matrix
    pub smoothing_weight: f64, // weight for exponential smoothing
    pub system_time: f64, // internal time tracking for MPC updates
    pub update_rate: f64, // rate at which MPC updates (e.g., 50 Hz)
    previous_control: Vec<Array1<f64>>, // previous control input for smoothing
    pub last_solve_time: f64,
}

impl MPC {
    pub fn new() -> Self {
        let n = 13; // [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        let m = 3;  // [thrust_x, thrust_y, thrust_z]
        let n_steps = 10;
        let dt = 0.02; // 50 Hz
        
        // Cost matrices (diagonal)
        let mut q = Array2::zeros((n, n));
        let mut r = Array2::zeros((m, m));
        let mut qn = Array2::zeros((n, n));
        
        // Position costs (high priority)
        q[(0, 0)] = 100.0; // x
        q[(1, 1)] = 100.0; // y  
        q[(2, 2)] = 200.0; // z (altitude most important)
        
        // Attitude costs
        q[(3, 3)] = 50.0; // qx
        q[(4, 4)] = 50.0; // qy
        q[(5, 5)] = 50.0; // qz
        q[(6, 6)] = 50.0; // qw
        
        // Velocity costs
        q[(7, 7)] = 10.0; // vx
        q[(8, 8)] = 10.0; // vy
        q[(9, 9)] = 20.0; // vz
        
        // Angular velocity costs
        q[(10, 10)] = 5.0; // wx
        q[(11, 11)] = 5.0; // wy
        q[(12, 12)] = 5.0; // wz
        
        // Control costs (penalize excessive thrust)
        r[(0, 0)] = 0.1; // thrust_x
        r[(1, 1)] = 0.1; // thrust_y
        r[(2, 2)] = 0.1; // thrust_z
        
        // Terminal cost (same as state cost)
        qn = q.clone();
        
        Self {
            n,
            m,
            n_steps,
            dt,
            mass: 10.0, // Default mass
            min_thrust: 0.0,
            max_thrust: 50.0,
            q,
            r,
            qn,
            smoothing_weight: 0.01,
            system_time: 0.0,
            update_rate: 50.0,
            previous_control: vec![Array1::zeros(m); n_steps],
            last_solve_time: 0.0,
        }
    }
    
    pub fn update(&mut self, current_state: &Array1<f64>, reference_trajectory: &Vec<Array1<f64>>, 
                  warm_start: &Vec<Array1<f64>>, mass: f64) -> Result<Vec<Array1<f64>>, String> {
        // Rate limiting: only update at specified rate
        if self.system_time < 1.0 / self.update_rate {
            self.system_time += self.dt;
            return Ok(warm_start.clone());
        }
        self.system_time = 0.0;
        
        self.mass = mass;
        
        // Simple MPC implementation using gradient descent
        let mut control_sequence = warm_start.clone();
        
        for _iteration in 0..50 { // Max 50 iterations
            let mut gradient = vec![Array1::zeros(self.m); self.n_steps];
            let mut cost = 0.0;
            
            // Forward simulate and compute cost
            let mut state = current_state.clone();
            for k in 0..self.n_steps {
                let control = &control_sequence[k];
                let ref_state = if k < reference_trajectory.len() {
                    &reference_trajectory[k]
                } else {
                    // Use last reference state if trajectory is shorter
                    &reference_trajectory[reference_trajectory.len() - 1]
                };
                
                // State cost
                let state_error = &state - ref_state;
                let state_cost = 0.5 * state_error.dot(&self.q.dot(&state_error));
                cost += state_cost;
                
                // Control cost
                let control_cost = 0.5 * control.dot(&self.r.dot(control));
                cost += control_cost;
                
                // Compute gradient (simplified)
                let state_grad = self.q.dot(&state_error);
                gradient[k] = self.r.dot(control) + 0.1 * &state_grad.slice(s![0..3]).to_owned();
                
                // Simple dynamics update
                state = self.dynamics_step(&state, control);
            }
            
            // Terminal cost
            let ref_terminal = if reference_trajectory.len() > 0 {
                &reference_trajectory[reference_trajectory.len() - 1]
            } else {
                current_state
            };
            let terminal_error = &state - ref_terminal;
            cost += 0.5 * terminal_error.dot(&self.qn.dot(&terminal_error));
            
            // Gradient descent update
            let step_size = 0.01;
            for k in 0..self.n_steps {
                control_sequence[k] = &control_sequence[k] - &(step_size * &gradient[k]);
                
                // Apply thrust constraints
                for i in 0..self.m {
                    control_sequence[k][i] = control_sequence[k][i].clamp(self.min_thrust, self.max_thrust);
                }
            }
            
            // Check convergence
            if cost < 1e-6 {
                break;
            }
        }
        
        Ok(control_sequence)
    }
    
    fn dynamics_step(&self, state: &Array1<f64>, control: &Array1<f64>) -> Array1<f64> {
        let mut next_state = state.clone();
        
        // Simple dynamics: position and velocity integration
        // Acceleration from thrust
        let ax = control[0] / self.mass;
        let ay = control[1] / self.mass;
        let az = control[2] / self.mass - 9.81; // Include gravity
        
        // Update velocities
        next_state[7] += ax * self.dt; // vx
        next_state[8] += ay * self.dt; // vy
        next_state[9] += az * self.dt; // vz
        
        // Update positions
        next_state[0] += next_state[7] * self.dt; // x
        next_state[1] += next_state[8] * self.dt; // y
        next_state[2] += next_state[9] * self.dt; // z
        
        // Keep attitude and angular velocity unchanged (simplified)
        next_state[3] = state[3]; // qx
        next_state[4] = state[4]; // qy
        next_state[5] = state[5]; // qz
        next_state[6] = state[6]; // qw
        next_state[10] = state[10]; // wx
        next_state[11] = state[11]; // wy
        next_state[12] = state[12]; // wz
        
        next_state
    }
}

// Lossless wrapper for lander
#[derive(Debug, Clone)]
pub struct Lossless {
    pub max_velocity: f64,
    pub dry_mass: f64,
    pub alpha: f64,
    pub lower_thrust_bound: f64,
    pub upper_thrust_bound: f64,
    pub tvc_range_rad: f64,
    pub coarse_delta_t: f64,
    pub fine_delta_t: f64,
    pub glide_slope: f64,
    pub use_glide_slope: bool,
    pub flip_glide_slope: bool,
    pub system_time: f64,
    pub update_rate: f64,
    last_solution: rust_lossless::TrajectoryResult, // Store the last solution for use when not updating
    pub last_solve_time: f64,
}

impl Lossless {
    pub fn new() -> Self {
        let max_velocity = 5.0;
        let dry_mass = 10.0; // Lander-specific mass
        let alpha = 1.0 / (9.81 * 180.0);
        let lower_thrust_bound = 0.0; // Lander can throttle to zero
        let upper_thrust_bound = 50.0; // Lander-specific thrust range
        let tvc_range_rad = 15_f64.to_radians();
        let coarse_delta_t = 0.25;
        let fine_delta_t = 0.1;
        let glide_slope = 0.05_f64.to_radians();
        let use_glide_slope = true;
        let flip_glide_slope = true;
        let system_time = -1.0;
        let update_rate = 1.0; // Update at 1 Hz for trajectory planning

        Self {
            max_velocity,
            dry_mass,
            alpha,
            lower_thrust_bound,
            upper_thrust_bound,
            tvc_range_rad,
            coarse_delta_t,
            fine_delta_t,
            glide_slope,
            use_glide_slope,
            flip_glide_slope,
            system_time,
            update_rate,
            last_solution: rust_lossless::TrajectoryResult {
                positions: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                velocities: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                masses: vec![dry_mass; 20], // Placeholder for 20 steps
                thrusts: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                sigmas: vec![0.0; 20], // Placeholder for 20 steps
                time_of_flight_s: 0.0, // Placeholder for time of flight
            },
            last_solve_time: 0.0,
        }
    }

    pub fn update(&mut self, current_position: [f64; 3], current_velocity: [f64; 3], target_position: [f64; 3], propellant_mass: f64, system_time: f64) -> rust_lossless::TrajectoryResult {
        let elapsed_time = system_time - self.system_time;
        if elapsed_time < 1.0 / self.update_rate {
            // Not time to update yet, return previous solution
            return self.last_solution.clone();
        } else {
            self.system_time = system_time;
            self.last_solve_time = system_time;

            // Call the lossless convexification solver to get the optimal trajectory
            let trajectory = self.solve(current_position, current_velocity, target_position, propellant_mass);
            self.last_solution = trajectory;
            return self.last_solution.clone();
        }
    }

    pub fn solve(&mut self, current_position: [f64; 3], current_velocity: [f64; 3], target_position: [f64; 3], propellant_mass: f64) -> rust_lossless::TrajectoryResult {
        let mut solver = rust_lossless::LosslessSolver {
            landing_point: target_position,
            initial_position: current_position,
            initial_velocity: current_velocity,
            max_velocity: self.max_velocity,
            dry_mass: self.dry_mass,
            fuel_mass: propellant_mass,
            alpha: self.alpha,
            lower_thrust_bound: self.lower_thrust_bound,
            upper_thrust_bound: self.upper_thrust_bound,
            tvc_range_rad: self.tvc_range_rad,
            coarse_delta_t: self.coarse_delta_t,
            fine_delta_t: self.fine_delta_t,
            use_glide_slope: self.use_glide_slope,
            glide_slope: self.glide_slope,
            N: 20, // will be overridden by the solver based on the problem setup
            ..Default::default()
        };

        let result = solver.solve();

        if result.trajectory.is_none() {
            return self.last_solution.clone();
        } else {
            return result.trajectory.expect("Failed to solve trajectory optimization problem");
        }
    }
}

// Sensor Fusion implementation
pub struct SensorFusion {
    ekf_attitude: ExtendedKalmanFilter<AttitudeModel>,
    // TODO: Need EKFs for GPS and UWB as well
    sensor_fusion_rate: f64,
    last_update: Option<Instant>,
}

impl SensorFusion {
    pub fn new() -> Self {
        let ekf_attitude = ExtendedKalmanFilter::new(
            Array1::from(vec![0.0; 6]),
            Array1::from(vec![0.0; 6]),
            0.002,
            Array2::eye(6) * 0.01,  // Process noise
            Array2::eye(6) * 0.1,   // Measurement noise
            Array2::eye(6) * 1.0,   // Initial covariance
            AttitudeModel::new(0.002)
        );
        
        Self {
            ekf_attitude,
            sensor_fusion_rate: 500.0,  // 500 Hz
            last_update: None,
        }
    }
    
    /// Run the EKF prediction and update steps for every available filter
    /// 
    /// # Arguments
    /// * `sensor_data` - The SensorData struct containing all sensor readings
    /// * `dt` - The time step
    /// 
    /// # Returns
    /// * `Option<VehicleState>` - The fused vehicle state, or None if not enough data
    pub fn update(&mut self, sensor_data: &SensorData, dt: f64) -> Option<VehicleState> {
        let now = Instant::now();
        
        // Rate limiting: only update at specified frequency
        if let Some(last_time) = self.last_update {
            if now.duration_since(last_time).as_secs_f64() < 1.0 / self.sensor_fusion_rate {
                return None; // Not time to update yet
            }
        }
        self.last_update = Some(now);

        let mut new_state = VehicleState::default();
        // TODO: update mass based on fuel consumption
        
        // Update EKF with IMU data if available
        if let Some(imu_data) = &sensor_data.imu_data {
            // Predict step
            self.ekf_attitude.predict();
            
            // Update step with accelerometer and gyroscope data
            let measurement = [
                imu_data.accel[0], imu_data.accel[1], imu_data.accel[2], // Accelerometer
                imu_data.gyro[0], imu_data.gyro[1], imu_data.gyro[2],          // Gyroscope
                imu_data.mag[0], imu_data.mag[1], imu_data.mag[2], // Magnetometer
            ];
            
            self.ekf_attitude.update(&measurement);
            let updated_state = self.ekf_attitude.get_state();
            
            new_state.attitude = UnitQuaternion::from_euler_angles(updated_state[0], updated_state[1], updated_state[2]);
            new_state.angular_velocity = Vector3::new(updated_state[3], updated_state[4], updated_state[5]);
        }

        if let Some(uwb_data) = &sensor_data.uwb_data {
            // TODO: Update UWB EKF
            // self.ekf_uwb_position.predict();
            // self.ekf_uwb_position.update(&[uwb_data]);

            // let updated_state = self.ekf_uwb_position.get_state();

            // Placeholder until UWB EKF is implemented
            let updated_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            new_state.position = Vector3::new(updated_state[0], updated_state[1], updated_state[2]);
            new_state.velocity = Vector3::new(updated_state[3], updated_state[4], updated_state[5]);

        } else if let Some(gps_data) = &sensor_data.gps_data {
            // TODO: Update GPS EKF
            // self.ekf_gps_position.predict();
            // self.ekf_gps_position.update(&[gps_data]);

            // let updated_state = self.ekf_gps_position.get_state();

            // Placeholder until GPS EKF is implemented
            let updated_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            new_state.position = Vector3::new(updated_state[0], updated_state[1], updated_state[2]);
            new_state.velocity = Vector3::new(updated_state[3], updated_state[4], updated_state[5]);
        }
        None
    }
}

// Data structures for sensor fusion
#[derive(Debug, Clone)]
pub struct SensorData {
    pub timestamp: f64,
    pub imu_data: Option<ImuData>,
    pub gps_data: Option<[f64; 3]>,
    pub uwb_data: Option<[f64; 3]>, // Ultra-wideband position
    pub chamber_pressure: Option<f64>,
    pub tank_pressure: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ImuData {
    pub accel: [f64; 3],    // Accelerometer measurements [x, y, z]
    pub gyro: [f64; 3],     // Gyroscope measurements [wx, wy, wz]
    pub mag: [f64; 3],      // Magnetometer measurements [mx, my, mz]
}

// Vehicle state data structures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlightPhase {
    PreLaunch,
    Ascent,
    Hover,
    Descent,
    Landed
}

#[derive(Debug, Clone)]
pub struct VehicleState {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub attitude: UnitQuaternion<f64>,
    pub angular_velocity: Vector3<f64>,
    pub mass: f64,
    pub dry_mass: f64,
}

#[derive(Debug, Clone)]
pub struct ControlLoopState {
    pub sensor_fusion_state: Option<Array1<f64>>,
    pub trajectory_state: Option<rust_lossless::TrajectoryResult>,
    pub last_sensor_update: Instant,
    pub last_navigation_update: Instant,
    pub last_mpc_update: Instant,
    pub start_time: Instant,
    pub vehicle_state: VehicleState,
    pub flight_terminated: bool,
    pub flight_phase: FlightPhase,
    pub last_state_time: Instant,
}

impl Default for ControlLoopState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            sensor_fusion_state: None,
            trajectory_state: None,
            last_sensor_update: now,
            last_navigation_update: now,
            last_mpc_update: now,
            start_time: now,
            vehicle_state: VehicleState {
                position: Vector3::new(0.0, 0.0, 0.0),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                attitude: UnitQuaternion::identity(),
                angular_velocity: Vector3::new(0.0, 0.0, 0.0),
                mass: 80.0,
                dry_mass: 50.0,
            },
            flight_terminated: false,
            flight_phase: FlightPhase::PreLaunch,
            last_state_time: now,
        }
    }
}
