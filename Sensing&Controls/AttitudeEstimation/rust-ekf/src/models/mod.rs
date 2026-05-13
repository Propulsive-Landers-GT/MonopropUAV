pub mod xy_position;
pub mod altitude;
pub mod attitude;
pub mod lidar;
pub mod gps;

pub use xy_position::XYPositionModel;
pub use altitude::AltitudeModel;
pub use attitude::AttitudeModel;
pub use lidar::LidarModel;
pub use gps::GpsModel;

// Type aliases for convenience
pub type XYPositionEKF = crate::ekf::filter::ExtendedKalmanFilter<XYPositionModel>;
pub type AltitudeEKF   = crate::ekf::filter::ExtendedKalmanFilter<AltitudeModel>;
pub type AttitudeEKF   = crate::ekf::filter::ExtendedKalmanFilter<AttitudeModel>;
pub type LidarEKF     = crate::ekf::filter::ExtendedKalmanFilter<LidarModel>;
pub type GpsEKF       = crate::ekf::filter::ExtendedKalmanFilter<GpsModel>;