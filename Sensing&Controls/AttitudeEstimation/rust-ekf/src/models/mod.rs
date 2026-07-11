pub mod xy_position;
pub mod altitude;
pub mod attitude;
pub mod full_state_esekf;

pub use xy_position::XYPositionModel;
pub use altitude::AltitudeModel;
pub use attitude::AttitudeModel;
pub use full_state_esekf::RocketState;

// Type aliases for convenience
pub type XYPositionEKF = crate::ekf::filter::ExtendedKalmanFilter<XYPositionModel>;
pub type AltitudeEKF   = crate::ekf::filter::ExtendedKalmanFilter<AltitudeModel>;
pub type AttitudeEKF   = crate::ekf::filter::ExtendedKalmanFilter<AttitudeModel>;
pub type RocketESEKF   = crate::es_ekf::filter::ErrorStateKalmanFilter<RocketState>;
