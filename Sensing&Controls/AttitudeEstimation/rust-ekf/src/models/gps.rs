use ndarray::{arr1, Array1, Array2};
use crate::ekf::model::EKFModel;

pub const DEFAULT_GRAVITY_MS2: f64 = 9.80665;

pub struct GpsModel {
    pub current_time: f64,
    pub previous_time: f64,
    pub delta_time: f64,
    pub gravity_magnitude: f64,
    /// Accelerometer sample (m/s², body frame).
    pub accel_body: [f64; 3],
    /// Euler angles (rad) from the upstream attitude EKF: roll φ, pitch θ, yaw ψ (ZYX).
    pub euler: [f64; 3],
}

impl GpsModel {
    pub fn new(delta_time: f64) -> Self {
        Self {
            current_time: -delta_time,
            previous_time: -2.0 * delta_time,
            delta_time,
            gravity_magnitude: DEFAULT_GRAVITY_MS2,
            accel_body: [0.0; 3],
            euler: [0.0; 3],
        }
    }

    fn advance_time(&mut self, time: f64) {
        self.previous_time = self.current_time;
        self.current_time = time;
        let parsed_dt = self.current_time - self.previous_time;
        if parsed_dt.is_finite() && parsed_dt > 0.0 {
            self.delta_time = parsed_dt;
        }
    }

    /// Refresh IMU and attitude between GPS fixes
    pub fn set_imu_and_attitude(&mut self, accel_body: [f64; 3], euler_rad: [f64; 3]) {
        self.accel_body = accel_body;
        self.euler = euler_rad;
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

    /// World-frame linear acceleration (m/s²): `a_w = Rᵀ f_b − [0, 0, g]`.
    fn world_linear_accel(&self) -> [f64; 3] {
        let euler = arr1(&self.euler);
        let r = Self::euler_to_rotation_matrix(&euler);
        let f_w = r.t().dot(&arr1(&self.accel_body));
        [
            f_w[0],
            f_w[1],
            f_w[2] - self.gravity_magnitude,
        ]
    }
}

impl EKFModel for GpsModel {
    fn delta_time(&self) -> Option<f64> {
        Some(self.delta_time)
    }

    /// `[t, x, y, z, ax, ay, az, roll, pitch, yaw]`.
    fn parse_data(&mut self, data: &[f64]) -> Array1<f64> {
        assert!(
            data.len() >= 10,
            "GpsModel::parse_data: expected at least 10 values [t, x, y, z, ax, ay, az, roll, pitch, yaw]"
        );

        self.advance_time(data[0]);
        self.accel_body = [data[4], data[5], data[6]];
        self.euler = [data[7], data[8], data[9]];

        arr1(&[data[1], data[2], data[3]])
    }

    fn state_transition_function(&self, state: &Array1<f64>, dt: f64) -> Array1<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return state.clone();
        }
        let a = self.world_linear_accel();
        arr1(&[
            state[0] + state[3] * dt + 0.5 * a[0] * dt * dt,
            state[1] + state[4] * dt + 0.5 * a[1] * dt * dt,
            state[2] + state[5] * dt + 0.5 * a[2] * dt * dt,
            state[3] + a[0] * dt,
            state[4] + a[1] * dt,
            state[5] + a[2] * dt,
        ])
    }

    fn state_transition_jacobian(&self, _state: &Array1<f64>, dt: f64) -> Array2<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return Array2::eye(6);
        }
        Array2::from_shape_vec(
            (6, 6),
            vec![
                1.0, 0.0, 0.0, dt,  0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, dt,  0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, dt,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .expect("F shape")
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
