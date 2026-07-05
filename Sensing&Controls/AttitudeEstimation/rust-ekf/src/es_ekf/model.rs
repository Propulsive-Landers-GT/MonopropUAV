use ndarray::{Array1, Array2};

/// Trait defining the behavior for an Error-State EKF implementation
pub trait ESEKFModel {
    /// High-frequency non-linear prediction of the true state using IMU data (e.g., 16D)
    fn nominal_prediction(&self, nominal_state: &Array1<f64>, imu_data: &[f64], dt: f64) -> Array1<f64>;

    /// Jacobian of the error-state transition function F (e.g., 15x15)
    fn error_transition_jacobian(&self, nominal_state: &Array1<f64>, imu_data: &[f64], dt: f64) -> Array2<f64>;

    /// Expected measurement h(x) based on the current nominal state
    fn measurement_prediction(&self, nominal_state: &Array1<f64>) -> Array1<f64>;

    /// Jacobian of the measurement model H with respect to the ERROR state (e.g., Zx15)
    fn measurement_jacobian(&self, nominal_state: &Array1<f64>) -> Array2<f64>;

    /// Inject the calculated error state (e.g., 15D) into the nominal state (e.g., 16D)
    /// This is where quaternion unit-length constraints are enforced mathematically.
    fn inject_error(&self, nominal_state: &Array1<f64>, error_state: &Array1<f64>) -> Array1<f64>;
}