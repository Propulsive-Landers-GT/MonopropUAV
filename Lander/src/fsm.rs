use ndarray::Array1;
use crate::state::{SensorData, FlightPhase, ControlLoopState};
use crate::algorithms::{SensorFusionEstimator, GuidancePlanner, Controller};
use crate::algorithms::sensor_fusion::SensorFusion;
use crate::algorithms::guidance::Lossless;
use crate::algorithms::control::MPC;

pub struct FlightStateMachine {
    sensor_fusion: Box<dyn SensorFusionEstimator>,
    guidance: Box<dyn GuidancePlanner>,
    controller: Box<dyn Controller>,
    state: ControlLoopState,
    goal: [f64; 3],
    sensor_fusion_rate: f64,
    navigation_rate: f64,
    mpc_rate: f64,
    previous_control: Vec<Array1<f64>>,
}

impl FlightStateMachine {
    pub fn new() -> Self {
        let mut lossless = Lossless::new();
        let mut mpc = MPC::new();
        
        // Configure default parameters
        lossless.dry_mass = 50.0;
        lossless.upper_thrust_bound = 1000.0;
        lossless.max_velocity = 15.0;
        mpc.mass = 80.0;
        mpc.max_thrust = 1000.0;
        
        let previous_control = vec![Array1::from(vec![0.0, 0.0, 80.0 * 9.81]); 10];
        
        Self {
            sensor_fusion: Box::new(SensorFusion::new()),
            guidance: Box::new(lossless),
            controller: Box::new(mpc),
            state: ControlLoopState::default(),
            goal: [0.0, 0.0, 50.0], // The ascent target. The descent targets the origin [0.0, 0.0, 0.0] as the landing pad.
            sensor_fusion_rate: 500.0,
            navigation_rate: 1.0,
            mpc_rate: 50.0,
            previous_control,
        }
    }
    
    #[allow(dead_code)]
    pub fn new_with_algorithms(
        sensor_fusion: Box<dyn SensorFusionEstimator>,
        guidance: Box<dyn GuidancePlanner>,
        controller: Box<dyn Controller>,
    ) -> Self {
        let previous_control = vec![Array1::from(vec![0.0, 0.0, 80.0 * 9.81]); 10];
        Self {
            sensor_fusion,
            guidance,
            controller,
            state: ControlLoopState::default(),
            goal: [0.0, 0.0, 50.0],
            sensor_fusion_rate: 500.0,
            navigation_rate: 1.0,
            mpc_rate: 50.0,
            previous_control,
        }
    }
    
    pub fn initialize(&mut self) {
        self.state.start_time = 0.0;
        self.state.last_sensor_update = 0.0;
        self.state.last_navigation_update = 0.0;
        self.state.last_mpc_update = 0.0;
        self.state.last_state_time = 0.0;
        self.state.last_position_update = 0.0;
        self.state.mass = 80.0;
        self.state.flight_terminated = false;
        self.state.flight_phase = FlightPhase::Standby;

        println!("Flight State Machine initialized");
    }
    
    pub fn arm(&mut self, now: f64) {
        // TODO: Implement a comprehensive suite of pre-launch health checks (e.g. check sensor telemetry values are in range, verify EKF variance convergence, check tank/battery levels) that must all pass before we can arm.
        if self.state.flight_phase == FlightPhase::Standby {
            self.state.flight_phase = FlightPhase::Armed;
            self.state.last_state_time = now;
            println!("Command received: ARMED");
        }
    }
    
    pub fn disarm(&mut self, now: f64) {
        if self.state.flight_phase == FlightPhase::Armed {
            self.state.flight_phase = FlightPhase::Standby;
            self.state.last_state_time = now;
            println!("Command received: DISARMED");
        }
    }
    
    pub fn launch(&mut self, now: f64) {
        if self.state.flight_phase == FlightPhase::Armed {
            self.state.flight_phase = FlightPhase::Ascent;
            self.state.last_state_time = now;
            println!("Command received: LAUNCHED");
        }
    }
    
    #[inline]
    fn due(last: f64, now: f64, rate: f64) -> bool {
        now - last >= (1.0 / rate) - 1e-6
    }

    pub fn get_state(&self) -> &ControlLoopState {
        &self.state
    }

    pub fn get_state_mut(&mut self) -> &mut ControlLoopState {
        &mut self.state
    }
    
    pub fn step(&mut self, sensor_data: &SensorData) -> Option<[f64; 3]> {
        if self.state.flight_terminated {
            return None;
        }
        
        let now = sensor_data.timestamp;

        // Centralized scheduling for Sensor Fusion (500 Hz)
        let sensor_fusion_due = Self::due(self.state.last_sensor_update, now, self.sensor_fusion_rate);
        let sensor_dt = if self.state.last_sensor_update > 0.0 {
            now - self.state.last_sensor_update
        } else {
            1.0 / self.sensor_fusion_rate
        };

        if sensor_fusion_due {
            self.update_sensor_fusion(sensor_data, now, sensor_dt);
        }
        
        // Mass depletion model (based on last_thrust from operator / MPC and actual elapsed sensor_dt)
        if self.state.flight_phase == FlightPhase::Standby || self.state.flight_phase == FlightPhase::Armed {
            self.state.mass = 80.0;
        } else if self.state.flight_phase != FlightPhase::Landed && sensor_fusion_due {
            // TODO: Implement a more accurate mass estimation model as propellant depletes
            let mass_flow = self.state.last_thrust / (180.0 * 9.81);
            
            // TODO: Map chamber pressure to thrust ratio (thrust = chamber_pressure * K_thrust) to verify actuator telemetry
            self.state.mass -= mass_flow * sensor_dt;
            self.state.mass = self.state.mass.max(50.0);
        }
        
        // Run termination check on the freshest state (after sensor fusion update)
        if let Some(reason) = self.check_flight_termination(sensor_data, now) {
            self.state.flight_terminated = true;
            self.state.termination_reason = Some(reason.clone());
            self.state.diagnostics_queue.push(format!("Flight terminated! Reason: {}", reason));
            println!("Flight terminated! Reason: {}", reason);
            return None;
        }
        
        let old_phase = self.state.flight_phase;
        self.state.flight_phase = self.determine_flight_phase(self.goal, now);
        if self.state.flight_phase != old_phase {
            self.state.diagnostics_queue.push(format!("Flight phase transition: {:?} -> {:?}", old_phase, self.state.flight_phase));
        }
        
        // Centralized scheduling for Guidance planner (1 Hz)
        if Self::due(self.state.last_navigation_update, now, self.navigation_rate) {
            self.update_guidance(now);
        }
        
        // Centralized scheduling for MPC (50 Hz)
        if self.state.flight_phase == FlightPhase::Standby || self.state.flight_phase == FlightPhase::Armed || self.state.flight_phase == FlightPhase::Landed {
            if Self::due(self.state.last_mpc_update, now, self.mpc_rate) {
                self.state.last_mpc_update = now;
                return Some([0.0, 0.0, 0.0]); // zero gimbal/thrust
            }
            return None;
        }
        
        if Self::due(self.state.last_mpc_update, now, self.mpc_rate) {
            self.state.last_mpc_update = now;
            if let Some(control_output) = self.update_mpc(now) {
                return Some(control_output);
            }
        }
        
        None
    }
    
    fn update_sensor_fusion(&mut self, sensor_data: &SensorData, now: f64, dt: f64) -> bool {
        let in_prelaunch = self.state.flight_phase == FlightPhase::Standby || self.state.flight_phase == FlightPhase::Armed;
        if let Some(mut updated_state) = self.sensor_fusion.update(sensor_data, dt, in_prelaunch) {
            updated_state.mass = self.state.mass;
            self.state.sensor_fusion_state = self.sensor_fusion.get_state_vector();
            self.state.vehicle_state = updated_state;
        }
        
        if sensor_data.uwb_data.is_some() || sensor_data.gps_data.is_some() {
            self.state.last_position_update = now;
        }

        self.state.last_sensor_update = now;
        true
    }
    
    // Trajectory generation and guidance re-planning (1 Hz)
    fn update_guidance(&mut self, now: f64) {
        match self.state.flight_phase {
            FlightPhase::Standby | FlightPhase::Armed => {
                if self.state.trajectory_state.is_none() {
                    self.generate_ascent_trajectory(now);
                }
            }
            FlightPhase::Ascent => {
                // Dynamically re-plan trajectory from current EKF state to hover target
                self.generate_ascent_trajectory(now);
            }
            FlightPhase::Hover => {
                self.state.trajectory_state = None;
            }
            FlightPhase::Descent => {
                // Dynamically re-plan trajectory from current EKF state to landing target
                self.generate_descent_trajectory(now);
            }
            FlightPhase::Landed => {
                self.state.trajectory_state = None;
            }
        }
        self.state.last_navigation_update = now;
    }
    
    fn update_mpc(&mut self, now: f64) -> Option<[f64; 3]> {
        if let Some(trajectory) = &self.state.trajectory_state {
            let mpc_state = self.vehicle_state_to_mpc_state();
            let (xref_traj, uref_traj) = self.generate_mpc_reference(trajectory, now);
            
            if let Ok(control_sequence) = self.controller.update(&mpc_state, &xref_traj, &uref_traj, &self.previous_control, self.state.mass) {
                if let Some(first_control) = control_sequence.first() {
                    let mut gimbal_theta = first_control[0];
                    let mut gimbal_phi = first_control[1];
                    let mut thrust = first_control[2];
                    
                    if gimbal_theta.is_nan() || gimbal_theta.is_infinite() ||
                       gimbal_phi.is_nan() || gimbal_phi.is_infinite() ||
                       thrust.is_nan() || thrust.is_infinite() {
                        println!("Warning: MPC returned NaN/Inf! Falling back to last valid control.");
                        gimbal_theta = self.state.last_gimbal_theta;
                        gimbal_phi = self.state.last_gimbal_phi;
                        thrust = self.state.last_thrust;
                    }
                    
                    let max_gimbal_step = 2.0_f64.to_radians();
                    let max_thrust_step = 40.0;
                    
                    let delta_theta = (gimbal_theta - self.state.last_gimbal_theta).clamp(-max_gimbal_step, max_gimbal_step);
                    gimbal_theta = self.state.last_gimbal_theta + delta_theta;
                    
                    let delta_phi = (gimbal_phi - self.state.last_gimbal_phi).clamp(-max_gimbal_step, max_gimbal_step);
                    gimbal_phi = self.state.last_gimbal_phi + delta_phi;
                    
                    let delta_thrust = (thrust - self.state.last_thrust).clamp(-max_thrust_step, max_thrust_step);
                    thrust = self.state.last_thrust + delta_thrust;
                    
                    self.state.last_gimbal_theta = gimbal_theta;
                    self.state.last_gimbal_phi = gimbal_phi;
                    self.state.last_thrust = thrust;
                    
                    if control_sequence.len() > 1 {
                        let mut shifted = control_sequence[1..].to_vec();
                        if let Some(last) = shifted.last().cloned() {
                            shifted.push(last);
                        }
                        self.previous_control = shifted;
                    } else {
                        self.previous_control = control_sequence;
                    }
                    
                    return Some([gimbal_theta, gimbal_phi, thrust]);
                }
            }
        }
        None
    }
    
    fn determine_flight_phase(&mut self, goal_position: [f64; 3], now: f64) -> FlightPhase {
        let current_altitude = self.state.vehicle_state.position.z;
        let goal_altitude = goal_position[2];

        // Emergency landing contingency check on GPS/UWB sensor denial
        // TODO: Intergrate this into the contigencies or into the flight_phase transitions below. Kinda sticks out right now
        if self.state.flight_phase == FlightPhase::Ascent || self.state.flight_phase == FlightPhase::Hover {
            if self.state.last_position_update > 0.0 && now - self.state.last_position_update > 5.0 {
                let msg = format!("Emergency landing triggered: absolute position data (GPS/UWB) denied for {:.2}s", now - self.state.last_position_update);
                self.state.diagnostics_queue.push(msg.clone());
                println!("{}", msg);
                self.state.last_state_time = now;
                return FlightPhase::Descent;
            }
        }

        match self.state.flight_phase {
            FlightPhase::Ascent => {
                if current_altitude >= goal_altitude {
                    self.state.last_state_time = now;
                    FlightPhase::Hover
                } else {
                    FlightPhase::Ascent
                }
            },
            FlightPhase::Hover => {
                if now - self.state.last_state_time >= 20.0 {
                    self.state.last_state_time = now;
                    FlightPhase::Descent
                } else {
                    FlightPhase::Hover
                }
            },
            FlightPhase::Descent => {
                let v = &self.state.vehicle_state.velocity;
                let speed = (v.x.powi(2) + v.y.powi(2) + v.z.powi(2)).sqrt();
                if current_altitude <= 0.1 && speed < 0.2 {
                    FlightPhase::Landed
                } else {
                    FlightPhase::Descent
                }
            }
            FlightPhase::Standby => FlightPhase::Standby,
            FlightPhase::Armed => FlightPhase::Armed,
            FlightPhase::Landed => FlightPhase::Landed,
        }
    }
    
    fn generate_ascent_trajectory(&mut self, now: f64) {
        let current_pos = self.state.vehicle_state.position;
        let current_vel = self.state.vehicle_state.velocity;
        let goal_position = self.goal;
        let propellant_mass = self.state.mass - self.state.vehicle_state.dry_mass;
        
        if let Some(trajectory) = self.guidance.solve(
            [current_pos.x, current_pos.y, current_pos.z],
            [current_vel.x, current_vel.y, current_vel.z],
            goal_position,
            propellant_mass
        ) {
            self.state.trajectory_state = Some(trajectory.clone());
            self.state.trajectory_generation_time = now;
            let msg = format!("Ascent trajectory regenerated: {:.2}s flight time", trajectory.time_of_flight_s);
            println!("{}", msg);
            self.state.diagnostics_queue.push(msg);
        } else {
            let msg = "Warning: Ascent trajectory regeneration failed! Falling back to last valid trajectory.".to_string();
            println!("{}", msg);
            self.state.diagnostics_queue.push(msg);
        }
    }
    
    fn generate_descent_trajectory(&mut self, now: f64) {
        let current_pos = self.state.vehicle_state.position;
        let current_vel = self.state.vehicle_state.velocity;
        let landing_point = [0.0, 0.0, 0.0];
        let propellant_mass = self.state.mass - self.state.vehicle_state.dry_mass;
        
        if let Some(trajectory) = self.guidance.solve(
            [current_pos.x, current_pos.y, current_pos.z],
            [current_vel.x, current_vel.y, current_vel.z],
            landing_point,
            propellant_mass
        ) {
            self.state.trajectory_state = Some(trajectory.clone());
            self.state.trajectory_generation_time = now;
            let msg = format!("Descent trajectory regenerated: {:.2}s flight time", trajectory.time_of_flight_s);
            println!("{}", msg);
            self.state.diagnostics_queue.push(msg);
        } else {
            let msg = "Warning: Descent trajectory regeneration failed! Falling back to last valid trajectory.".to_string();
            println!("{}", msg);
            self.state.diagnostics_queue.push(msg);
        }
    }
    
    fn check_flight_termination(&self, sensor_data: &SensorData, now: f64) -> Option<String> {
        if sensor_data.imu_data.is_none() {
            return Some("IMU data missing".to_string());
        }
        
        // TODO: Decide if we want to keep this block or not, we already shifted to dead reckoning and landing, if we don't trust IMU at all we can just cut off, makes sense
        if sensor_data.uwb_data.is_none() && sensor_data.gps_data.is_none() {
            let elapsed_since_pos = now - self.state.last_position_update;
            // TODO: Probably want to get some more realistic numbers of how bad we think the IMU would drift in a given amount of time to decide the amount of error we want to permit
            if elapsed_since_pos > 15.0 {
                return Some(format!("No GPS or UWB position data for {:.2}s (exceeded dead-reckoning safety limit)", elapsed_since_pos));
            }
        }
        
        if sensor_data.chamber_pressure.is_none() || sensor_data.tank_pressure.is_none() {
            return Some("Pressure data missing".to_string());
        }
        
        let euler_attitude = self.state.vehicle_state.attitude.euler_angles();
        let tilt_angle = euler_attitude.0.abs();
        if tilt_angle > 30.0_f64.to_radians() {
            return Some(format!("Tilt angle {:.2} deg exceeds maximum (30 deg)", tilt_angle.to_degrees()));
        }
        
        if let Some(ref trajectory) = self.state.trajectory_state {
            let current_pos = &self.state.vehicle_state.position;
            let mut min_dist = f64::MAX;
            for pos in &trajectory.positions {
                let dist = ((current_pos.x - pos[0]).powi(2) + 
                            (current_pos.y - pos[1]).powi(2) + 
                            (current_pos.z - pos[2]).powi(2)).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            if !trajectory.positions.is_empty() && min_dist > 10.0 {
                return Some(format!("Deviation from trajectory {:.2}m exceeds 10m limit", min_dist));
            }
        }
        
        None
    }
    
    fn vehicle_state_to_mpc_state(&self) -> Array1<f64> {
        let quat = self.state.vehicle_state.attitude.quaternion();
        
        Array1::from(vec![
            self.state.vehicle_state.position.x,
            self.state.vehicle_state.position.y,
            self.state.vehicle_state.position.z,
            quat.i, quat.j, quat.k, quat.w,
            self.state.vehicle_state.velocity.x,
            self.state.vehicle_state.velocity.y,
            self.state.vehicle_state.velocity.z,
            self.state.vehicle_state.angular_velocity.x,
            self.state.vehicle_state.angular_velocity.y,
            self.state.vehicle_state.angular_velocity.z
        ])
    }
    
    fn generate_mpc_reference(&self, trajectory: &rust_lossless::TrajectoryResult, now: f64) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut xref_traj = Vec::new();
        let mut uref_traj = Vec::new();
        
        let time_since_trajectory = now - self.state.trajectory_generation_time;
        let horizon_steps = self.controller.get_horizon_steps();
        let controller_dt = self.controller.get_time_step();
        
        for i in 0..=horizon_steps {
            let t_target = time_since_trajectory + (i as f64) * controller_dt;
            
            let mut interp_p = [0.0; 3];
            let mut interp_v = [0.0; 3];
            let interp_u;
            
            if t_target >= trajectory.time_of_flight_s || trajectory.positions.is_empty() {
                if let Some(&last_pos) = trajectory.positions.last() {
                    interp_p = last_pos;
                    interp_v = [0.0, 0.0, 0.0];
                    interp_u = [0.0, 0.0, self.state.mass * 9.81];
                } else {
                    interp_p = [0.0, 0.0, 0.0];
                    interp_v = [0.0, 0.0, 0.0];
                    interp_u = [0.0, 0.0, self.state.mass * 9.81];
                }
            } else {
                let traj_dt = if trajectory.positions.len() > 1 {
                    (trajectory.time_of_flight_s / (trajectory.positions.len() - 1) as f64).max(1e-4)
                } else {
                    0.1
                };
                let exact_idx = t_target / traj_dt;
                let base_idx = exact_idx.floor() as usize;
                
                if trajectory.positions.len() < 2 {
                    if let Some(&pos) = trajectory.positions.first() {
                        interp_p = pos;
                        interp_v = [0.0, 0.0, 0.0];
                    }
                } else {
                    let safe_idx = base_idx.min(trajectory.positions.len() - 2);
                    let clamped_frac = (exact_idx - safe_idx as f64).clamp(0.0, 1.0);
                    
                    let p0 = trajectory.positions[safe_idx];
                    let p1 = trajectory.positions[safe_idx + 1];
                    let v0 = if safe_idx < trajectory.velocities.len() { trajectory.velocities[safe_idx] } else { [0.0, 0.0, 0.0] };
                    let v1 = if safe_idx + 1 < trajectory.velocities.len() { trajectory.velocities[safe_idx + 1] } else { [0.0, 0.0, 0.0] };
                    
                    for j in 0..3 {
                        interp_p[j] = p0[j] + clamped_frac * (p1[j] - p0[j]);
                        interp_v[j] = v0[j] + clamped_frac * (v1[j] - v0[j]);
                    }
                }
                
                let thrust_idx = base_idx.min(trajectory.thrusts.len().saturating_sub(1));
                if thrust_idx < trajectory.thrusts.len() {
                    interp_u = trajectory.thrusts[thrust_idx];
                } else {
                    interp_u = [0.0, 0.0, self.state.mass * 9.81];
                }
            }
            
            let xref = Array1::from(vec![
                interp_p[0], interp_p[1], interp_p[2],
                0.0, 0.0, 0.0, 1.0,
                interp_v[0], interp_v[1], interp_v[2],
                0.0, 0.0, 0.0
            ]);
            
            let uref = Array1::from(vec![interp_u[0], interp_u[1], interp_u[2]]);
            
            xref_traj.push(xref);
            if i < horizon_steps {
                uref_traj.push(uref);
            }
        }
        
        (xref_traj, uref_traj)
    }
}
