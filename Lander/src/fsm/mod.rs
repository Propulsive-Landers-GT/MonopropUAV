use crate::state::{SensorData, FlightPhase, ControlLoopState};
use crate::algorithms::{Navigator, GuidancePlanner, Controller};

mod scheduler;
mod actuator;
mod autopilot;

pub use scheduler::Scheduler;
pub use actuator::ActuatorController;
pub use autopilot::Autopilot;

pub struct FlightStateMachine {
    autopilot: Autopilot,
    actuator_controller: ActuatorController,
    scheduler: Scheduler,
    state: ControlLoopState,
    goal: [f64; 3],
}

impl FlightStateMachine {
    pub fn new() -> Self {
        Self {
            autopilot: Autopilot::new(),
            actuator_controller: ActuatorController::new(),
            scheduler: Scheduler::new(500.0, 1.0, 50.0),
            state: ControlLoopState::default(),
            goal: [0.0, 0.0, 50.0], // The ascent target. The descent targets the origin [0.0, 0.0, 0.0] as the landing pad.
        }
    }
    
    pub fn new_with_algorithms(
        navigator: Box<dyn Navigator>,
        guidance: Box<dyn GuidancePlanner>,
        controller: Box<dyn Controller>,
    ) -> Self {
        Self {
            autopilot: Autopilot::new_with_algorithms(navigator, guidance, controller),
            actuator_controller: ActuatorController::new(),
            scheduler: Scheduler::new(500.0, 1.0, 50.0),
            state: ControlLoopState::default(),
            goal: [0.0, 0.0, 50.0],
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
        if self.state.flight_phase == FlightPhase::Standby {
            self.on_transition(FlightPhase::Standby, FlightPhase::Armed, now);
            println!("Command received: ARMED");
        }
    }
    
    pub fn disarm(&mut self, now: f64) {
        if self.state.flight_phase == FlightPhase::Armed {
            self.on_transition(FlightPhase::Armed, FlightPhase::Standby, now);
            println!("Command received: DISARMED");
        }
    }
    
    pub fn launch(&mut self, now: f64) {
        if self.state.flight_phase == FlightPhase::Armed {
            self.on_transition(FlightPhase::Armed, FlightPhase::Ascent, now);
            println!("Command received: LAUNCHED");
        }
    }

    pub fn set_flight_phase(&mut self, phase: FlightPhase, now: f64) {
        let from = self.state.flight_phase;
        self.on_transition(from, phase, now);
    }

    pub fn get_state(&self) -> &ControlLoopState {
        &self.state
    }

    pub fn get_state_mut(&mut self) -> &mut ControlLoopState {
        &mut self.state
    }

    fn next_phase(&self, goal_position: [f64; 3], now: f64) -> FlightPhase {
        let current_altitude = self.state.vehicle_state.position.z;
        let goal_altitude = goal_position[2];

        if self.state.flight_phase == FlightPhase::Ascent || self.state.flight_phase == FlightPhase::Hover {
            if self.state.last_position_update > 0.0 && now - self.state.last_position_update > 5.0 {
                return FlightPhase::Descent;
            }
        }

        match self.state.flight_phase {
            FlightPhase::Ascent => {
                if current_altitude >= goal_altitude {
                    FlightPhase::Hover
                } else {
                    FlightPhase::Ascent
                }
            },
            FlightPhase::Hover => {
                if now - self.state.last_state_time >= 20.0 {
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

    fn on_transition(&mut self, from: FlightPhase, to: FlightPhase, now: f64) {
        self.state.flight_phase = to;
        self.state.last_state_time = now;
        self.state.diagnostics_queue.push(format!("Flight phase transition: {:?} -> {:?}", from, to));
        self.autopilot.set_flight_phase(to);

        // Emergency landing contingency log if GPS/UWB denied
        if (from == FlightPhase::Ascent || from == FlightPhase::Hover) && to == FlightPhase::Descent {
            if self.state.last_position_update > 0.0 && now - self.state.last_position_update > 5.0 {
                let msg = format!(
                    "Emergency landing triggered: absolute position data (GPS/UWB) denied for {:.2}s",
                    now - self.state.last_position_update
                );
                self.state.diagnostics_queue.push(msg.clone());
                println!("{}", msg);
            }
        }
    }
    
    pub fn step(&mut self, sensor_data: &SensorData) -> Option<[f64; 4]> {
        if self.state.flight_terminated {
            return None;
        }
        
        let now = sensor_data.timestamp;

        // Centralized scheduling for Sensor Fusion (500 Hz)
        let sensor_fusion_due = self.scheduler.is_sensor_fusion_due(self.state.last_sensor_update, now);
        let sensor_dt = self.scheduler.sensor_dt(self.state.last_sensor_update, now);

        if sensor_fusion_due {
            self.autopilot.update_navigator(&mut self.state, sensor_data, now, sensor_dt);
        }
        
        // Mass depletion model (based on last_thrust from operator / MPC and actual elapsed sensor_dt)
        if self.state.flight_phase == FlightPhase::Standby || self.state.flight_phase == FlightPhase::Armed {
            self.state.mass = 80.0;
        } else if self.state.flight_phase != FlightPhase::Landed && sensor_fusion_due {
            let mass_flow = self.state.last_thrust / (180.0 * 9.81);
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
        let new_phase = self.next_phase(self.goal, now);
        if new_phase != old_phase {
            self.on_transition(old_phase, new_phase, now);
        }
        
        // Centralized scheduling for Guidance planner (1 Hz)
        if self.scheduler.is_navigation_due(self.state.last_navigation_update, now) {
            self.autopilot.update_guidance(&mut self.state, self.goal, now);
        }
        
        // Centralized scheduling for MPC (50 Hz) and Actuator output
        let mut mpc_control_output = None;
        if self.state.flight_phase == FlightPhase::Standby || self.state.flight_phase == FlightPhase::Armed || self.state.flight_phase == FlightPhase::Landed {
            self.state.last_gimbal_theta = 0.0;
            self.state.last_gimbal_phi = 0.0;
            self.state.last_thrust = 0.0;
        } else if self.scheduler.is_mpc_due(self.state.last_mpc_update, now) {
            self.state.last_mpc_update = now;
            mpc_control_output = self.autopilot.update_mpc(&mut self.state, self.goal, now);
        }

        // Run actuator controller (gimbal step/clamping, roll control, thrust clamping)
        let control_signals = self.actuator_controller.update(&mut self.state, mpc_control_output, now);

        Some(control_signals)
    }
    
    fn check_flight_termination(&self, sensor_data: &SensorData, now: f64) -> Option<String> {
        if sensor_data.imu_data.is_none() {
            return Some("IMU data missing".to_string());
        }
        
        if sensor_data.uwb_data.is_none() && sensor_data.gps_data.is_none() {
            let elapsed_since_pos = now - self.state.last_position_update;
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
}
