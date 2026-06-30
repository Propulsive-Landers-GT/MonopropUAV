use ndarray::Array1;
use crate::state::{SensorData, VehicleState};

pub trait SensorFusionEstimator {
    fn update(&mut self, sensor_data: &SensorData, dt: f64, in_prelaunch: bool) -> Option<VehicleState>;
    fn get_state_vector(&self) -> Option<Array1<f64>>;
}

pub trait GuidancePlanner {
    fn solve(
        &mut self,
        current_position: [f64; 3],
        current_velocity: [f64; 3],
        target_position: [f64; 3],
        propellant_mass: f64,
    ) -> rust_lossless::TrajectoryResult;
}

pub trait Controller {
    fn update(
        &mut self,
        current_state: &Array1<f64>,
        reference_trajectory: &[Array1<f64>],
        uref_trajectory: &[Array1<f64>],
        warm_start: &[Array1<f64>],
        mass: f64,
    ) -> Result<Vec<Array1<f64>>, String>;
    
    fn get_horizon_steps(&self) -> usize;
    fn get_time_step(&self) -> f64;
}

// Module declarations
pub mod sensor_fusion;
pub mod guidance;
pub mod control;
