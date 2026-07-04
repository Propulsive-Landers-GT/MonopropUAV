use ndarray::Array1;
use nalgebra::{Vector3, UnitQuaternion};

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImuData {
    pub accel: [f64; 3],
    pub gyro: [f64; 3],
    pub mag: [f64; 3],
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SensorData {
    pub timestamp: f64,
    pub imu_data: Option<ImuData>,
    pub gps_data: Option<[f64; 3]>,
    pub uwb_data: Option<[f64; 3]>,
    pub chamber_pressure: Option<f64>,
    pub tank_pressure: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub enum FlightPhase {
    Standby,
    Armed,
    Ascent,
    Hover,
    Descent,
    Landed,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VehicleState {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub attitude: UnitQuaternion<f64>,
    pub angular_velocity: Vector3<f64>,
    pub mass: f64,
    pub dry_mass: f64,
}

impl Default for VehicleState {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            attitude: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
            mass: 80.0,
            dry_mass: 50.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ControlLoopState {
    pub sensor_fusion_state: Option<Array1<f64>>,
    pub trajectory_state: Option<rust_lossless::TrajectoryResult>,
    pub last_sensor_update: f64,
    pub last_navigation_update: f64,
    pub last_mpc_update: f64,
    pub last_position_update: f64,
    pub start_time: f64,
    pub vehicle_state: VehicleState,
    pub flight_terminated: bool,
    pub flight_phase: FlightPhase,
    pub last_state_time: f64,
    pub trajectory_generation_time: f64,
    pub last_gimbal_theta: f64,
    pub last_gimbal_phi: f64,
    pub last_thrust: f64,
    pub mass: f64,
    pub termination_reason: Option<String>,
    pub diagnostics_queue: Vec<String>,
}

impl Default for ControlLoopState {
    fn default() -> Self {
        Self {
            sensor_fusion_state: None,
            trajectory_state: None,
            last_sensor_update: 0.0,
            last_navigation_update: 0.0,
            last_mpc_update: 0.0,
            last_position_update: 0.0,
            start_time: 0.0,
            vehicle_state: VehicleState::default(),
            flight_terminated: false,
            flight_phase: FlightPhase::Standby,
            last_state_time: 0.0,
            trajectory_generation_time: 0.0,
            last_gimbal_theta: 0.0,
            last_gimbal_phi: 0.0,
            last_thrust: 80.0 * 9.81,
            mass: 80.0,
            termination_reason: None,
            diagnostics_queue: Vec::new(),
        }
    }
}
