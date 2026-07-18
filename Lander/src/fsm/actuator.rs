use crate::state::ControlLoopState;

pub struct ActuatorController {
    rcs_controller: crate::algorithms::rcs::RcsController,
}

impl ActuatorController {
    pub fn new() -> Self {
        Self {
            rcs_controller: crate::algorithms::rcs::RcsController::new(),
        }
    }

    pub fn update(
        &mut self,
        state: &mut ControlLoopState,
        mpc_control_output: Option<[f64; 3]>,
        now: f64,
    ) -> [f64; 4] {
        if let Some(first_control) = mpc_control_output {
            let mut gimbal_theta = first_control[0];
            let mut gimbal_phi = first_control[1];
            let mut thrust = first_control[2];
            
            if gimbal_theta.is_nan() || gimbal_theta.is_infinite() ||
               gimbal_phi.is_nan() || gimbal_phi.is_infinite() ||
               thrust.is_nan() || thrust.is_infinite() {
                println!("Warning: MPC returned NaN/Inf! Falling back to last valid control.");
                gimbal_theta = state.last_gimbal_theta;
                gimbal_phi = state.last_gimbal_phi;
                thrust = state.last_thrust;
            }
            
            let max_gimbal_step = 2.0_f64.to_radians();
            let max_thrust_step = 40.0;
            
            let delta_theta = (gimbal_theta - state.last_gimbal_theta).clamp(-max_gimbal_step, max_gimbal_step);
            gimbal_theta = state.last_gimbal_theta + delta_theta;
            
            let delta_phi = (gimbal_phi - state.last_gimbal_phi).clamp(-max_gimbal_step, max_gimbal_step);
            gimbal_phi = state.last_gimbal_phi + delta_phi;
            
            let delta_thrust = (thrust - state.last_thrust).clamp(-max_thrust_step, max_thrust_step);
            thrust = state.last_thrust + delta_thrust;
            
            // TODO: Implement gain/calibration lookup tables to map target thrust (Newtons) to Main Throttle Valve (MTV) position commands (valve angle/CdA)
            
            state.last_gimbal_theta = gimbal_theta;
            state.last_gimbal_phi = gimbal_phi;
            state.last_thrust = thrust;
        }
        
        let (_, _, roll) = state.vehicle_state.attitude.euler_angles();
        let roll_rate = state.vehicle_state.angular_velocity.z;
        let rcs_command = self.rcs_controller.update(roll, roll_rate, now);
        
        [
            state.last_gimbal_theta,
            state.last_gimbal_phi,
            state.last_thrust,
            rcs_command,
        ]
    }
}
