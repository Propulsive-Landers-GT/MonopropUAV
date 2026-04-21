use crate::ekf::model::EKFModel;
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

/// Generic Extended Kalman Filter implementation
pub struct ExtendedKalmanFilter<T: EKFModel> {
    pub state: Array1<f64>,
    pub previous_state: Array1<f64>,
    pub error_covariance: Array2<f64>,
    pub process_noise_covariance: Array2<f64>,
    pub measurement_noise_covariance: Array2<f64>,
    pub delta_time: f64,
    pub current_time: f64,
    pub previous_time: f64,
    pub model: T,
}

impl<T: EKFModel> ExtendedKalmanFilter<T> {
    pub fn new(
        initial_state: Array1<f64>,
        _initial_measurements: Array1<f64>,
        delta_time: f64,
        q: Array2<f64>, // Should be the size of the state
        r: Array2<f64>, // Should be the size of the measurement
        initial_p: Array2<f64>, // Should be the size of the state
        model: T,
    ) -> Self {
        Self {
            previous_state: initial_state.clone(),
            state: initial_state,
            error_covariance: initial_p,
            process_noise_covariance: q,
            measurement_noise_covariance: r,
            delta_time,
            current_time: -delta_time,
            previous_time: -2.0 * delta_time,
            model,
        }
    }

    pub fn predict(&mut self) {
        self.previous_state = self.state.clone();
        self.state =
            self.model
                .state_transition_function(&self.state, self.delta_time);

        let transition_jacobian = self
            .model
            .state_transition_jacobian(&self.previous_state, self.delta_time);

        self.error_covariance = transition_jacobian.dot(&self.error_covariance).dot(&transition_jacobian.t())
            + &self.process_noise_covariance;
    }

    pub fn update(&mut self, data: &[f64]) {
        let measurement = self.model.parse_data(data);

        let prediction = self.model.measurement_prediction_function(&self.state);
        let residual = &measurement - &prediction;

        let prediction_jacobian = self.model.measurement_prediction_jacobian(&self.state);

        let s = prediction_jacobian.dot(&self.error_covariance).dot(&prediction_jacobian.t())
            + &self.measurement_noise_covariance;

        let s_data: Vec<f64> = s.iter().copied().collect();
        let s_matrix = DMatrix::from_row_slice(s.nrows(), s.ncols(), &s_data);
        let s_inv = match s_matrix.try_inverse() {
            Some(m) => Array2::from_shape_vec((s.nrows(), s.ncols()), m.iter().copied().collect())
                .expect("inverse shape must match"),
            None => {
                self.error_covariance =
                    &self.error_covariance + &(Array2::<f64>::eye(self.state.len()) * 1e-6); // Prevent panic on singular matrix by adding small diagonal before falling back to prediction
                return;
            } 
        };

        let k = self.error_covariance.dot(&prediction_jacobian.t()).dot(&s_inv);

        self.previous_state = self.state.clone();
        self.state = &self.state + &k.dot(&residual);

        let identity = Array2::eye(self.state.len());
        self.error_covariance = (identity - k.dot(&prediction_jacobian)).dot(&self.error_covariance);
    }

    pub fn get_state(&self) -> &Array1<f64> {
        &self.state
    }

    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.error_covariance
    }
}