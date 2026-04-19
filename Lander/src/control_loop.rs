use std::time::{Duration, Instant};
use ndarray::{Array1, Array2, s};
use nalgebra::{Vector3, UnitQuaternion};

// Import algorithms module
use crate::algorithms::{MPC, Lossless, SensorFusion, SensorData, VehicleState, FlightPhase, ControlLoopState};


// TODO: We need to add the rest of our EKFs, maybe a different struct to contain each of the EKFs
pub struct ControlLoop {
    sensor_fusion: SensorFusion,
    lossless: Lossless,
    mpc: MPC,
    state: ControlLoopState,
    goal: [f64; 3],
    sensor_fusion_rate: f64,
    navigation_rate: f64,
    mpc_rate: f64,
    previous_control: Vec<Array1<f64>>,
}

impl ControlLoop {
    pub fn new() -> Self {
        let sensor_fusion = SensorFusion::new();
        let lossless = Lossless::new();
        let mpc = MPC::new();
        let previous_control = vec![Array1::zeros(3); 10]; // 3 control inputs, 10 steps
        
        Self {
            sensor_fusion,
            lossless,
            mpc,
            state: ControlLoopState::default(),
            goal: [0.0, 0.0, 50.0],
            sensor_fusion_rate: 500.0,  // 500 Hz
            navigation_rate: 1.0,       // 1 Hz
            mpc_rate: 50.0,              // 50 Hz
            previous_control,
        }
    }
    
    pub fn initialize(&mut self) {
        let now = Instant::now();
        self.state.start_time = now;
        self.state.last_sensor_update = now;
        self.state.last_navigation_update = now;
        self.state.last_mpc_update = now;

        println!("Control loop initialized");
    }
    
    pub fn get_state(&self) -> &ControlLoopState {
        &self.state
    }
    
    pub fn step(&mut self, sensor_data: &SensorData) -> Option<[f64; 3]> {
        if self.state.flight_terminated {
            return None;
        }
        
        // Check flight termination conditions
        if self.check_flight_termination(sensor_data) {
            self.state.flight_terminated = true;
            println!("Flight terminated!");
            return None;
        }
        
        // Update sensor fusion (500 Hz)
        self.update_sensor_fusion(sensor_data);

        self.determine_flight_phase();
        
        // Update navigation (1 Hz)
        self.update_navigation(goal_position);
        
        // Update MPC (50 Hz)
        if let Some(control_output) = self.update_mpc() {
            return Some(control_output);
        }
        
        None
    }
    
    fn update_sensor_fusion(&mut self, sensor_data: &SensorData) -> bool {
        let now = Instant::now();
        if now.duration_since(self.state.last_sensor_update) >= Duration::from_secs_f64(1.0 / self.sensor_fusion_rate) {
            // Update sensor fusion using the SensorFusion struct
            if let Some(updated_state) = self.sensor_fusion.update(sensor_data, 0.002) {
                self.state.sensor_fusion_state = Some(updated_state);
                
                // Update vehicle state with fused sensor data
                // TODO: Implement update_vehicle_state method
                
                // TODO: We need to update the vehicle state with the EKF state. The Flight Termination Slides has information on What sensors to priortize for determining our location and pose.
            }
            
            self.state.last_sensor_update = now;
            return true;
        }
        false
    }
    
    fn update_navigation(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.state.last_navigation_update) >= Duration::from_secs_f64(1.0 / self.navigation_rate) {
            match self.flight_phase {
                FlightPhase::PreLaunch => {
                    if self.state.trajectory_state.is_none() {
                        self.generate_ascent_trajectory();
                    }
                }
                FlightPhase::Ascent => {
                    self.generate_ascent_trajectory();
                }
                FlightPhase::Descent => {
                    self.generate_descent_trajectory();
                }
                FlightPhase::Landed => {
                    // Do nothing
                }
            }
        }
    }
    
    fn update_mpc(&mut self) -> Option<[f64; 3]> {
        let now = Instant::now();
        if now.duration_since(self.state.last_mpc_update) >= Duration::from_secs_f64(1.0 / self.mpc_rate) {
            if let Some(trajectory) = &self.state.trajectory_state {
                // Convert vehicle state to MPC format
                let mpc_state = self.vehicle_state_to_mpc_state(&self.state.trajectory_state);
                
                // Generate reference trajectory
                let (xref_traj, uref_traj) = self.generate_mpc_reference(trajectory);
                
                // Update MPC using the algorithm from simulation
                if let Ok(control_sequence) = self.mpc.update(&mpc_state, &xref_traj, &self.previous_control, self.state.vehicle_state.mass) {
                    if let Some(first_control) = control_sequence.first() {
                        let control_output = [first_control[0], first_control[1], first_control[2]];
                        self.state.last_mpc_update = now;
                        return Some(control_output);
                    }
                }
            }
        }
        None
    }
    
    fn determine_flight_phase(&self, goal_position: [f64; 3]) -> FlightPhase {
        let current_altitude = self.state.vehicle_state.position.z;
        let goal_altitude = goal_position[2];

        // Use tracked phase with transitions based on altitude and time
        match self.state.flight_phase {
            FlightPhase::Ascent => {
                // Transition to hover when reaching target altitude
                if current_altitude >= goal_altitude{
                    self.state.last_state_time = Instant::now();
                    FlightPhase::Hover
                } else {
                    FlightPhase::Ascent
                }
            },
            FlightPhase::Hover => {
                // Hover for 20 seconds in this phase, then transition to descent
                if self.state.last_state_time.elapsed().as_secs_f64() >= 20.0 {
                    self.state.last_state_time = Instant::now();
                    FlightPhase::Descent
                } else {
                    FlightPhase::Hover
                }
            },
            FlightPhase::Descent => {
                // Stay in descent until landing
                if self.state.vehicle_state.position.z <= 0.0 {
                    FlightPhase::Landed
                } else {
                    FlightPhase::Descent
                }
            }
            FlightPhase::PreLaunch => {
                // Stay in initialized until we start the flight
                // TODO: Change this logic to transition to Ascent when we're ready to start
                if self.state.last_state_time.elapsed().as_secs_f64() >= 20.0 {
                    self.state.last_state_time = Instant::now();
                    FlightPhase::Ascent
                } else {
                    FlightPhase::PreLaunch
                }
            }
        }
    }
    
    fn generate_ascent_trajectory(&mut self) {
        let current_pos = self.state.vehicle_state.position;
        let current_vel = self.state.vehicle_state.velocity;
        let goal_position =  // Target 50m altitude
        let propellant_mass = self.state.vehicle_state.mass - self.state.vehicle_state.dry_mass;
        let system_time = self.state.start_time.elapsed().as_secs_f64();
        
        // Generate ascent trajectory
        let trajectory = self.lossless.solve(
            [current_pos.x, current_pos.y, current_pos.z],
            [current_vel.x, current_vel.y, current_vel.z],
            goal_position,
            propellant_mass
        );
        
        self.state.trajectory_state = Some(trajectory.clone());
        println!("Ascent trajectory generated: {:.2}s flight time", trajectory.time_of_flight_s);
    }
    
    fn generate_descent_trajectory(&mut self) {
        let current_pos = self.state.vehicle_state.position;
        let current_vel = self.state.vehicle_state.velocity;
        let landing_point = [0.0, 0.0, 0.0]; // Land at origin
        let propellant_mass = self.state.vehicle_state.mass - self.state.vehicle_state.dry_mass;
        
        // Generate descent trajectory
        let trajectory = self.lossless.solve(
            [current_pos.x, current_pos.y, current_pos.z],
            [current_vel.x, current_vel.y, current_vel.z],
            landing_point,
            propellant_mass
        );
        
        self.state.trajectory_state = Some(trajectory.clone());
        println!("Descent trajectory generated: {:.2}s flight time", trajectory.time_of_flight_s);
    }
    
    
    // TODO: This Might be a Avionics thing later, but Fill out the different failure modes and their handling
    //       More information in the Flight Termination System documentation 
    fn check_flight_termination(&self, sensor_data: &SensorData) -> bool {
        // 1. IMU failure - terminate immediately
        if sensor_data.imu_data.is_none() {
            return true;
        }
        
        // 2. GPS/UWB failure - terminate if no position data for 5 seconds
        if sensor_data.uwb_data.is_none() {
            // TODO: Add timeout check
            return true;
        } else if sensor_data.gps_data.is_none() {
            // TODO: Add timeout check
            // Attempt Landing
            return true;
        }
        
        // 3. Pressure sensor failure - terminate if no altitude data
        if sensor_data.chamber_pressure.is_none() || sensor_data.tank_pressure.is_none() {
            return true;
        }
        
        // 4. Angle constraints - terminate if tilt angle exceeds maximum
        let euler_attitude = self.state.vehicle_state.attitude.euler_angles();
        let tilt_angle = euler_attitude.0.abs(); // Simplified tilt calculation using roll
        if tilt_angle > 30.0_f64.to_radians() { // 30 degree max tilt
            return true;
        }
        
        // 5. Position deviation - terminate if too far from intended trajectory
        if let Some(ref _trajectory) = self.state.trajectory_state {
            let current_pos = &self.state.vehicle_state.position;
            let distance_from_trajectory = current_pos.norm();
            if distance_from_trajectory > 10.0 {
                return true;
            }
        }
        
        false
    }
    
    fn vehicle_state_to_mpc_state(&self, _sensor_state: &Option<Array1<f64>>) -> Array1<f64> {
        // Convert vehicle state to MPC state format
        // Order: [x, y, z, qx, qy, qz, qw, x_dot, y_dot, z_dot, wx, wy, wz]
        let quat = self.state.vehicle_state.attitude.quaternion();
        
        Array1::from(vec![
            self.state.vehicle_state.position.x,
            self.state.vehicle_state.position.y,
            self.state.vehicle_state.position.z,
            quat.i, quat.j, quat.k, quat.w, // Quaternion components
            self.state.vehicle_state.velocity.x,
            self.state.vehicle_state.velocity.y,
            self.state.vehicle_state.velocity.z,
            self.state.vehicle_state.angular_velocity.x,
            self.state.vehicle_state.angular_velocity.y,
            self.state.vehicle_state.angular_velocity.z
        ])
    }
    
    fn generate_mpc_reference(&self, trajectory: &rust_lossless::TrajectoryResult) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut xref_traj = Vec::new();
        let mut uref_traj = Vec::new();
        
        // Generate reference trajectory using simulation algorithm
        let current_time = self.state.start_time.elapsed().as_secs_f64();
        let time_since_trajectory = current_time; // Assuming trajectory was generated at time 0
        
        for i in 0..=self.mpc.n_steps {
            let t_target = time_since_trajectory + (i as f64) * self.mpc.dt;
            
            let mut interp_p = [0.0; 3];
            let mut interp_v = [0.0; 3];
            let mut interp_u = [0.0; 3];
            
            if t_target >= trajectory.time_of_flight_s || trajectory.positions.is_empty() {
                // Use last known values or hover
                if let Some(&last_pos) = trajectory.positions.last() {
                    interp_p = last_pos;
                    interp_v = [0.0, 0.0, 0.0];
                    interp_u = [0.0, 0.0, self.state.vehicle_state.mass * 9.81];
                } else {
                    interp_p = [0.0, 0.0, 0.0];
                    interp_v = [0.0, 0.0, 0.0];
                    interp_u = [0.0, 0.0, self.state.vehicle_state.mass * 9.81];
                }
            } else {
                // Interpolate between trajectory points
                let exact_idx = t_target / 0.1; // Assuming 0.1s time step in trajectory
                let base_idx = exact_idx.floor() as usize;
                let safe_idx = base_idx.min(trajectory.positions.len().saturating_sub(2));
                let clamped_frac = (exact_idx - safe_idx as f64).clamp(0.0, 1.0);
                
                let p0 = trajectory.positions[safe_idx];
                let p1 = trajectory.positions[safe_idx + 1];
                let v0 = if safe_idx < trajectory.velocities.len() { trajectory.velocities[safe_idx] } else { [0.0, 0.0, 0.0] };
                let v1 = if safe_idx + 1 < trajectory.velocities.len() { trajectory.velocities[safe_idx + 1] } else { [0.0, 0.0, 0.0] };
                
                for j in 0..3 {
                    interp_p[j] = p0[j] + clamped_frac * (p1[j] - p0[j]);
                    interp_v[j] = v0[j] + clamped_frac * (v1[j] - v0[j]);
                }
                
                // Get thrust from trajectory
                let thrust_idx = base_idx.min(trajectory.thrusts.len().saturating_sub(1));
                interp_u = trajectory.thrusts[thrust_idx];
            }
            
            // Create reference state for MPC
            let xref = Array1::from(vec![
                interp_p[0], interp_p[1], interp_p[2], // position
                0.0, 0.0, 0.0, 1.0,                  // quaternion (upright)
                interp_v[0], interp_v[1], interp_v[2], // velocity
                0.0, 0.0, 0.0                          // angular velocity
            ]);
            
            let uref = Array1::from(vec![interp_u[0], interp_u[1], interp_u[2]]);
            
            xref_traj.push(xref);
            if i < self.mpc.n_steps {
                uref_traj.push(uref);
            }
        }
        
        (xref_traj, uref_traj)
    }
}

