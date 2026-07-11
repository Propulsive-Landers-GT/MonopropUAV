use crate::es_ekf::model::ESEKFModel;
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

/// What happened to a measurement handed to the filter. A single filter is
/// expected to ride through sensor faults on its own (dropouts, outliers,
/// NaNs from firmware), so rejected measurements are a normal, reportable
/// outcome rather than a panic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpdateOutcome {
    /// Measurement fused; `nis` is the normalized innovation squared
    /// (Mahalanobis distance of the residual), useful for consistency checks.
    Fused { nis: f64 },
    /// Measurement contained NaN/Inf and was ignored.
    RejectedNonFinite,
    /// Innovation failed the chi-square gate and was ignored.
    RejectedGate { nis: f64 },
    /// Innovation covariance S was singular; measurement ignored.
    RejectedSingular,
}

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

    /// Fast loop: integrate IMU data forward. Returns false (and leaves the
    /// state untouched) if the IMU sample or dt is non-finite — a NaN here
    /// would otherwise poison the whole state vector in one step.
    pub fn predict(&mut self, imu_data: &[f64], dt: f64) -> bool {
        if !dt.is_finite() || dt <= 0.0 || imu_data.iter().any(|v| !v.is_finite()) {
            return false;
        }

        // 1. Linearize the error dynamics about the state at the START of the
        // step (the same state the nonlinear prediction integrates from).
        let f = self.model.error_transition_jacobian(&self.nominal_state, imu_data, dt);

        // 2. Advance the nominal state non-linearly
        self.nominal_state = self.model.nominal_prediction(&self.nominal_state, imu_data, dt);

        // 3. Advance the error covariance linearly. `process_noise` must be
        // the DISCRETE Q for this dt (see RocketState::process_noise, which
        // scales the datasheet noise densities by the sample period).
        self.error_covariance = f.dot(&self.error_covariance).dot(&f.t()) + &self.process_noise;
        true
    }

    /// Slow loop: apply the model's default measurement (e.g., GPS position)
    /// to update and inject errors
    pub fn update(&mut self, measurement: &Array1<f64>, r_matrix: &Array2<f64>) -> UpdateOutcome {
        let prediction = self.model.measurement_prediction(&self.nominal_state);
        let h = self.model.measurement_jacobian(&self.nominal_state);
        self.update_with(measurement, &prediction, &h, r_matrix)
    }

    /// Apply an arbitrary linearized measurement given its predicted value and
    /// Jacobian with respect to the error state. This lets callers fuse
    /// additional sensors beyond the model's default measurement (e.g., a
    /// magnetometer for yaw observability) without changing the model trait.
    pub fn update_with(
        &mut self,
        measurement: &Array1<f64>,
        prediction: &Array1<f64>,
        h: &Array2<f64>,
        r_matrix: &Array2<f64>,
    ) -> UpdateOutcome {
        self.update_gated(measurement, prediction, h, r_matrix, None)
    }

    /// Like `update_with`, but with an optional chi-square gate on the
    /// normalized innovation squared (NIS = r^T S^-1 r). A measurement whose
    /// NIS exceeds `gate` is statistically inconsistent with the filter's own
    /// uncertainty (wild outlier, stuck sensor, wrong frame) and is dropped
    /// instead of fused. For a 3D measurement, gate = 16.27 keeps 99.9% of
    /// genuine measurements (chi-square, 3 degrees of freedom).
    pub fn update_gated(
        &mut self,
        measurement: &Array1<f64>,
        prediction: &Array1<f64>,
        h: &Array2<f64>,
        r_matrix: &Array2<f64>,
        gate: Option<f64>,
    ) -> UpdateOutcome {
        // Firmware faults show up as NaN/Inf; fusing one would corrupt the
        // entire state and covariance in a single update.
        if measurement.iter().any(|v| !v.is_finite()) {
            return UpdateOutcome::RejectedNonFinite;
        }

        let residual = measurement - prediction;

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
                return UpdateOutcome::RejectedSingular;
            }
        };

        let nis = residual.dot(&s_inv.dot(&residual));
        if let Some(g) = gate {
            if !nis.is_finite() || nis > g {
                return UpdateOutcome::RejectedGate { nis };
            }
        }

        // K = P * H^T * S^-1
        let k = self.error_covariance.dot(&h.t()).dot(&s_inv);

        // Calculate the Error State: delta_x = K * residual
        let error_state = k.dot(&residual);

        // Inject the error state into the nominal state, resetting error state to 0 implicitly
        self.nominal_state = self.model.inject_error(&self.nominal_state, &error_state);

        // P = (I - K * H) * P
        let identity = Array2::eye(self.error_covariance.nrows());
        self.error_covariance = (identity - k.dot(h)).dot(&self.error_covariance);

        // TODO(covariance-reset): after injecting the attitude error, the error frame
        // rotates. A rigorous ES-EKF applies P <- G P G^T with
        // G = blockdiag(I, I, I - 0.5*skew(dtheta), I, I). Omitted here because the
        // correction is small; add if attitude corrections become large.

        UpdateOutcome::Fused { nis }
    }
}