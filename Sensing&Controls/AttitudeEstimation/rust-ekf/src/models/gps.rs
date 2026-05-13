//! GPS / local XYZ position EKF with measurement-based velocity for prediction.
//!
//! State [x, y, z, vx, vy, vz] — position (m) and velocity (m/s) in a fixed local frame.
//!
//! Measurement [x, y, z] from each fix (caller converts geodetic to ENU/NED if needed).
//!


use ndarray::{arr1, Array1, Array2};
use crate::ekf::model::EKFModel;

pub struct GpsModel {
    pub current_time: f64,
    pub previous_time: f64,
    pub delta_time: f64,
    pub prev_meas: [f64; 3],
    pub prev_time: f64,
    pub fix_count: u32,
    /// GPS finite-difference velocity (m/s)
    pub v_est: [f64; 3],
}

impl GpsModel {
    pub fn new(delta_time: f64) -> Self {
        Self {
            current_time: -delta_time,
            previous_time: -2.0 * delta_time,
            delta_time,
            prev_meas: [0.0; 3],
            prev_time: 0.0,
            fix_count: 0,
            v_est: [0.0; 3],
        }
    }
}

impl EKFModel for GpsModel {
    fn delta_time(&self) -> Option<f64> {
        Some(self.delta_time)
    }

    /// `[t, x, y, z]` — positions in meters
    fn parse_data(&mut self, data: &[f64]) -> Array1<f64> {
        assert!(
            data.len() >= 4,
            "GpsModel::parse_data: expected at least 4 values [t, x, y, z]"
        );

        self.previous_time = self.current_time;
        self.current_time = data[0];
        let parsed_dt = self.current_time - self.previous_time;
        if parsed_dt.is_finite() && parsed_dt > 0.0 {
            self.delta_time = parsed_dt;
        }

        let p = [data[1], data[2], data[3]];

        // First two measurements: no motion model from history.
        if self.fix_count >= 2 {
            let dt_gps = self.current_time - self.prev_time;
            if dt_gps.is_finite() && dt_gps > 0.0 {
                self.v_est = [
                    (p[0] - self.prev_meas[0]) / dt_gps,
                    (p[1] - self.prev_meas[1]) / dt_gps,
                    (p[2] - self.prev_meas[2]) / dt_gps,
                ];
            } else {
                self.v_est = [0.0; 3];
            }
        } else {
            self.v_est = [0.0; 3];
        }

        self.prev_meas = p;
        self.prev_time = self.current_time;
        self.fix_count = self.fix_count.saturating_add(1);

        arr1(&[p[0], p[1], p[2]])
    }

    fn state_transition_function(&self, state: &Array1<f64>, dt: f64) -> Array1<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return state.clone();
        }
        let vx = self.v_est[0];
        let vy = self.v_est[1];
        let vz = self.v_est[2];
        arr1(&[
            state[0] + vx * dt,
            state[1] + vy * dt,
            state[2] + vz * dt,
            vx,
            vy,
            vz,
        ])
    }

    /// `v_est` treated as independent of `state` for linearization (known input from GPS history).
    fn state_transition_jacobian(&self, _state: &Array1<f64>, _dt: f64) -> Array2<f64> {
        if !_dt.is_finite() || _dt <= 0.0 {
            return Array2::eye(6);
        }
        let mut f = Array2::<f64>::zeros((6, 6));
        for i in 0..3 {
            f[[i, i]] = 1.0;
        }
        // ∂p_next/∂v_state = 0 because we use v_est, not state velocity, for position increment.
        // ∂v_next/∂v_state = 0 because v_next is set to v_est.
        f
    }

    fn measurement_prediction_function(&self, state: &Array1<f64>) -> Array1<f64> {
        arr1(&[state[0], state[1], state[2]])
    }

    fn measurement_prediction_jacobian(&self, _state: &Array1<f64>) -> Array2<f64> {
        let mut h = Array2::<f64>::zeros((3, 6));
        h[[0, 0]] = 1.0;
        h[[1, 1]] = 1.0;
        h[[2, 2]] = 1.0;
        h
    }
}
