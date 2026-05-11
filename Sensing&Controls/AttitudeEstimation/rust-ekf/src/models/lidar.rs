//! Lidar range EKF: estimates vertical height above a flat ground plane and vertical velocity.
//!
//! **State** `[d, v]`: `d` is perpendicular distance from the vehicle to the ground plane (world +Z
//! up, ground at `z = 0`), `v` is the vertical velocity (positive upward).
//!
//! **Process model**: constant-acceleration in the vertical direction over each step, with
//! acceleration inferred from the IMU specific-force vector rotated into the world frame and
//! corrected by gravity magnitude

use ndarray::{arr1, Array1, Array2};
use crate::ekf::model::EKFModel;

pub const DEFAULT_GRAVITY_MS2: f64 = 9.80665;

pub struct LidarModel {
    pub current_time: f64,
    pub previous_time: f64,
    pub delta_time: f64,
    /// Unit vector in body frame from sensor toward the ground along the lidar beam.
    pub lidar_dir_body: Array1<f64>,
    pub gravity_magnitude: f64,
    /// Last accelerometer sample (m/s², body frame).
    pub accel_body: [f64; 3],
    /// Euler angles (rad) from the upstream attitude EKF: roll φ, pitch θ, yaw ψ (ZYX, same as `AttitudeModel`).
    pub euler: [f64; 3],
}

impl LidarModel {
    pub fn new(delta_time: f64) -> Self {
        Self::with_boresight(delta_time, [0.0, 0.0, -1.0]).expect("default boresight is non-zero")
    }

    /// `boresight_body` is a non-zero vector in the body frame pointing along the lidar beam
    /// toward the ground (it is normalized internally).
    pub fn with_boresight(
        delta_time: f64,
        boresight_body: [f64; 3],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let v = arr1(&boresight_body);
        let n = v.mapv(|x| x * x).sum().sqrt();
        if n <= f64::EPSILON {
            return Err("lidar boresight body vector must be non-zero".into());
        }
        Ok(Self {
            current_time: -delta_time,
            previous_time: -2.0 * delta_time,
            delta_time,
            lidar_dir_body: &v / n,
            gravity_magnitude: DEFAULT_GRAVITY_MS2,
            accel_body: [0.0; 3],
            euler: [0.0; 3],
        })
    }

    /// ZYX rotation matrix `R`: transforms a vector from world to body (`v_body = R * v_world`),
    /// matching [`crate::models::AttitudeModel`].
    fn euler_to_rotation_matrix(euler: &Array1<f64>) -> Array2<f64> {
        let (phi, theta, psi) = (euler[0], euler[1], euler[2]);
        let (cr, sr) = (phi.cos(), phi.sin());
        let (cp, sp) = (theta.cos(), theta.sin());
        let (cy, sy) = (psi.cos(), psi.sin());

        Array2::from_shape_vec(
            (3, 3),
            vec![
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
                -sp,
                cp * sr,
                cp * cr,
            ],
        )
        .expect("rotation matrix shape")
    }

    /// Unit direction of the lidar ray in the world frame (from vehicle toward the ground).
    fn beam_world(&self) -> Array1<f64> {
        let euler = arr1(&self.euler);
        let r = Self::euler_to_rotation_matrix(&euler);
        r.t().dot(&self.lidar_dir_body)
    }

    /// World +Z linear acceleration (m/s²) from specific force, using `a_z ≈ (Rᵀ f_b)_z − g`.
    fn vertical_linear_accel(&self) -> f64 {
        let euler = arr1(&self.euler);
        let r = Self::euler_to_rotation_matrix(&euler);
        let f_b = arr1(&self.accel_body);
        let f_w = r.t().dot(&f_b);
        f_w[2] - self.gravity_magnitude
    }

    /// Clamp beam world Z so the measurement stays finite when the ray is nearly horizontal.
    fn clamp_beam_wz(wz: f64) -> f64 {
        wz.min(-1e-3)
    }
}

impl EKFModel for LidarModel {
    fn delta_time(&self) -> Option<f64> {
        Some(self.delta_time)
    }

    /// `[t, range_m, ax, ay, az, φ, θ, ψ]` — attitude Euler radians must come from the upstream attitude EKF.
    fn parse_data(&mut self, data: &[f64]) -> Array1<f64> {
        assert!(
            data.len() >= 8,
            "LidarModel::parse_data: expected at least 8 values [t, range_m, ax, ay, az, roll, pitch, yaw]"
        );
        self.previous_time = self.current_time;
        self.current_time = data[0];
        let parsed_dt = self.current_time - self.previous_time;
        if parsed_dt.is_finite() && parsed_dt > 0.0 {
            self.delta_time = parsed_dt;
        }

        self.accel_body = [data[2], data[3], data[4]];
        self.euler = [data[5], data[6], data[7]];

        arr1(&[data[1]])
    }

    fn state_transition_function(&self, state: &Array1<f64>, dt: f64) -> Array1<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return state.clone();
        }
        let d = state[0];
        let v = state[1];
        let a = self.vertical_linear_accel();
        arr1(&[
            d + v * dt + 0.5 * a * dt * dt,
            v + a * dt,
        ])
    }

    fn state_transition_jacobian(&self, _state: &Array1<f64>, dt: f64) -> Array2<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return Array2::eye(2);
        }
        Array2::from_shape_vec((2, 2), vec![1.0, dt, 0.0, 1.0]).expect("F shape")
    }

    fn measurement_prediction_function(&self, state: &Array1<f64>) -> Array1<f64> {
        let d = state[0];
        let w = self.beam_world();
        let wz = Self::clamp_beam_wz(w[2]);
        arr1(&[-d / wz])
    }

    fn measurement_prediction_jacobian(&self, _state: &Array1<f64>) -> Array2<f64> {
        let w = self.beam_world();
        let wz = Self::clamp_beam_wz(w[2]);
        let dh_dd = -1.0 / wz;
        Array2::from_shape_vec((1, 2), vec![dh_dd, 0.0]).expect("H shape")
    }
}