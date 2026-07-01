use crate::es_ekf::model::ESEKFModel;
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

pub struct ErrorStateKalmanFilter<T: ESEKFModel> {
    pub nominal_state: Array1<f64>,
    pub error_covariance: Array2<f64>, // Size of Error State (e.g., 15x15)
    pub process_noise: Array2<f64>,    // Size of Error State (e.g., 15x15)
    pub model: T,
}

impl<T: ESEKFModel> ErrorStateKalmanFilter<T> {
    pub fn new(
        initial_nominal: Array1<f64>,
        initial_p: Array2<f64>,
        q: Array2<f64>,
        model: T,
    ) -> Self {
        Self {
            nominal_state: initial_nominal,
            error_covariance: initial_p,
            process_noise: q,
            model,
        }
    }

    /// Fast loop: integrate IMU data forward
    pub fn predict(&mut self, imu_data: &[f64], dt: f64) {
        // 1. Advance the nominal state non-linearly
        self.nominal_state = self.model.nominal_prediction(&self.nominal_state, imu_data, dt);

        // 2. Advance the error covariance linearly
        let f = self.model.error_transition_jacobian(&self.nominal_state, imu_data, dt);
        
        self.error_covariance = f.dot(&self.error_covariance).dot(&f.t()) + &self.process_noise;
    }

    /// Slow loop: apply measurement (e.g., GPS, UWB) to update and inject errors
    pub fn update(&mut self, measurement: &Array1<f64>, r_matrix: &Array2<f64>) {
        let prediction = self.model.measurement_prediction(&self.nominal_state);
        let residual = measurement - &prediction;

        let h = self.model.measurement_jacobian(&self.nominal_state);

        // S = H * P * H^T + R
        let s = h.dot(&self.error_covariance).dot(&h.t()) + r_matrix;

        // Invert S using nalgebra fallback
        let s_data: Vec<f64> = s.iter().copied().collect();
        let s_matrix = DMatrix::from_row_slice(s.nrows(), s.ncols(), &s_data);
        let s_inv = match s_matrix.try_inverse() {
            Some(m) => Array2::from_shape_vec((s.nrows(), s.ncols()), m.iter().copied().collect())
                .expect("inverse shape must match"),
            None => {
                // Prevent panic on singular matrix by adding small diagonal
                self.error_covariance = &self.error_covariance + &(Array2::<f64>::eye(self.error_covariance.nrows()) * 1e-6); 
                return;
            }
        };

        // K = P * H^T * S^-1
        let k = self.error_covariance.dot(&h.t()).dot(&s_inv);

        // Calculate the Error State: delta_x = K * residual
        let error_state = k.dot(&residual);

        // Inject the error state into the nominal state, resetting error state to 0 implicitly
        self.nominal_state = self.model.inject_error(&self.nominal_state, &error_state);

        // P = (I - K * H) * P
        let identity = Array2::eye(self.error_covariance.nrows());
        self.error_covariance = (identity - k.dot(&h)).dot(&self.error_covariance);
    }
}