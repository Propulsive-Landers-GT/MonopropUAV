use ndarray::Array1;
use crate::state::{SensorData, ControlLoopState, FlightPhase};
use crate::algorithms::{Navigator, GuidancePlanner, Controller};
use crate::algorithms::navigator;
use crate::algorithms::guidance::Lossless;
use crate::algorithms::control::MPC;

pub struct Autopilot {
    navigator: Box<dyn Navigator>,
    guidance: Box<dyn GuidancePlanner>,
    controller: Box<dyn Controller>,
    previous_control: Vec<Array1<f64>>,
}

impl Autopilot {
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
            navigator: Box::new(navigator::Navigator::new()),
            guidance: Box::new(lossless),
            controller: Box::new(mpc),
            previous_control,
        }
    }

    pub fn new_with_algorithms(
        navigator: Box<dyn Navigator>,
        guidance: Box<dyn GuidancePlanner>,
        controller: Box<dyn Controller>,
    ) -> Self {
        let previous_control = vec![Array1::from(vec![0.0, 0.0, 80.0 * 9.81]); 10];
        Self {
            navigator,
            guidance,
            controller,
            previous_control,
        }
    }

    pub fn set_flight_phase(&mut self, phase: FlightPhase) {
        self.controller.set_flight_phase(phase);
    }

    pub fn get_horizon_steps(&self) -> usize {
        self.controller.get_horizon_steps()
    }

    pub fn get_time_step(&self) -> f64 {
        self.controller.get_time_step()
    }

    pub fn update_navigator(&mut self, state: &mut ControlLoopState, sensor_data: &SensorData, now: f64, dt: f64) -> bool {
        let in_prelaunch = state.flight_phase == FlightPhase::Standby || state.flight_phase == FlightPhase::Armed;
        if let Some(mut updated_state) = self.navigator.update(sensor_data, dt, in_prelaunch) {
            updated_state.mass = state.mass;
            updated_state.dry_mass = state.vehicle_state.dry_mass;
            state.sensor_fusion_state = self.navigator.get_state_vector();
            state.vehicle_state = updated_state;
        }
        
        if sensor_data.uwb_data.is_some() || sensor_data.gps_data.is_some() {
            state.last_position_update = now;
        }

        state.last_sensor_update = now;
        true
    }

    pub fn update_guidance(&mut self, state: &mut ControlLoopState, goal: [f64; 3], now: f64) {
        match state.flight_phase {
            FlightPhase::Standby | FlightPhase::Armed => {
                if state.trajectory_state.is_none() {
                    self.generate_ascent_trajectory(state, goal, now);
                }
            }
            FlightPhase::Ascent => {
                self.generate_ascent_trajectory(state, goal, now);
            }
            FlightPhase::Hover => {
                state.trajectory_state = None;
            }
            FlightPhase::Descent => {
                self.generate_descent_trajectory(state, now);
            }
            FlightPhase::Landed => {
                state.trajectory_state = None;
            }
        }
        state.last_navigation_update = now;
    }

    pub fn update_mpc(&mut self, state: &mut ControlLoopState, goal: [f64; 3], now: f64) -> Option<[f64; 3]> {
        let mpc_state = self.vehicle_state_to_mpc_state(state);

        let (xref_traj, uref_traj) = if let Some(trajectory) = &state.trajectory_state {
            self.generate_mpc_reference(state, trajectory, now)
        } else if state.flight_phase == FlightPhase::Hover {
            let horizon_steps = self.controller.get_horizon_steps();
            let mut xrefs = Vec::new();
            let mut urefs = Vec::new();
            for _ in 0..=horizon_steps {
                xrefs.push(Array1::from(vec![
                    goal[0], goal[1], goal[2],
                    0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0
                ]));
                urefs.push(Array1::from(vec![0.0, 0.0, state.mass * 9.81]));
            }
            urefs.truncate(horizon_steps);
            (xrefs, urefs)
        } else {
            return None;
        };
        
        if let Ok(control_sequence) = self.controller.update(&mpc_state, &xref_traj, &uref_traj, &self.previous_control, state.mass) {
            if let Some(first_control) = control_sequence.first() {
                println!("FSM: MPC state z: {:.3}, att: [{:.3}, {:.3}, {:.3}, {:.3}], vel: [{:.3}, {:.3}, {:.3}]", 
                         mpc_state[2], mpc_state[3], mpc_state[4], mpc_state[5], mpc_state[6], mpc_state[7], mpc_state[8], mpc_state[9]);
                println!("FSM: MPC output control: {:?}", first_control);
                
                let out = [first_control[0], first_control[1], first_control[2]];
                
                if control_sequence.len() > 1 {
                    let mut shifted = control_sequence[1..].to_vec();
                    if let Some(last) = shifted.last().cloned() {
                        shifted.push(last);
                    }
                    self.previous_control = shifted;
                } else {
                    self.previous_control = control_sequence;
                }
                
                return Some(out);
            }
        }
        None
    }

    fn vehicle_state_to_mpc_state(&self, state: &ControlLoopState) -> Array1<f64> {
        let quat = state.vehicle_state.attitude.quaternion();
        
        Array1::from(vec![
            state.vehicle_state.position.x,
            state.vehicle_state.position.y,
            state.vehicle_state.position.z,
            quat.i, quat.j, quat.k, quat.w,
            state.vehicle_state.velocity.x,
            state.vehicle_state.velocity.y,
            state.vehicle_state.velocity.z,
            state.vehicle_state.angular_velocity.x,
            state.vehicle_state.angular_velocity.y,
            state.vehicle_state.angular_velocity.z
        ])
    }

    fn generate_mpc_reference(&self, state: &ControlLoopState, trajectory: &rust_lossless::TrajectoryResult, now: f64) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut xref_traj = Vec::new();
        let mut uref_traj = Vec::new();
        
        let time_since_trajectory = now - state.trajectory_generation_time;
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
                    interp_u = [0.0, 0.0, state.mass * 9.81];
                } else {
                    interp_p = [0.0, 0.0, 0.0];
                    interp_v = [0.0, 0.0, 0.0];
                    interp_u = [0.0, 0.0, state.mass * 9.81];
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
                    let u = trajectory.thrusts[thrust_idx];
                    let m = if thrust_idx < trajectory.masses.len() { trajectory.masses[thrust_idx] } else { state.mass };
                    let thrust_n = (u[0].powi(2) + u[1].powi(2) + u[2].powi(2)).sqrt() * m;
                    interp_u = [0.0, 0.0, thrust_n];
                } else {
                    interp_u = [0.0, 0.0, state.mass * 9.81];
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

    fn generate_ascent_trajectory(&mut self, state: &mut ControlLoopState, goal: [f64; 3], now: f64) {
        let current_pos = state.vehicle_state.position;
        let current_vel = state.vehicle_state.velocity;
        let propellant_mass = state.mass - state.vehicle_state.dry_mass;
        
        self.guidance.configure(50.0, 300.0, state.vehicle_state.dry_mass);
        
        if let Some(trajectory) = self.guidance.solve(
            [current_pos.x, current_pos.y, current_pos.z],
            [current_vel.x, current_vel.y, current_vel.z],
            goal,
            propellant_mass
        ) {
            state.trajectory_state = Some(trajectory.clone());
            state.trajectory_generation_time = now;
            let msg = format!("Ascent trajectory regenerated: {:.2}s flight time", trajectory.time_of_flight_s);
            println!("{}", msg);
            state.diagnostics_queue.push(msg);
        } else {
            let msg = "Warning: Ascent trajectory regeneration failed! Falling back to last valid trajectory.".to_string();
            println!("{}", msg);
            state.diagnostics_queue.push(msg);
        }
    }

    fn generate_descent_trajectory(&mut self, state: &mut ControlLoopState, now: f64) {
        let current_pos = state.vehicle_state.position;
        let current_vel = state.vehicle_state.velocity;
        let landing_point = [0.0, 0.0, 1.0];
        let propellant_mass = state.mass - state.vehicle_state.dry_mass;
        
        self.guidance.configure(15.0, 300.0, state.vehicle_state.dry_mass);
        
        if let Some(trajectory) = self.guidance.solve(
            [current_pos.x, current_pos.y, current_pos.z],
            [current_vel.x, current_vel.y, current_vel.z],
            landing_point,
            propellant_mass
        ) {
            state.trajectory_state = Some(trajectory.clone());
            state.trajectory_generation_time = now;
            let msg = format!("Descent trajectory regenerated: {:.2}s flight time", trajectory.time_of_flight_s);
            println!("{}", msg);
            state.diagnostics_queue.push(msg);
        } else {
            let msg = "Warning: Descent trajectory regeneration failed! Falling back to last valid trajectory.".to_string();
            println!("{}", msg);
            state.diagnostics_queue.push(msg);
        }
    }
}
