use std::io::{Error as IoError, ErrorKind};
use ndarray::{arr1, Array1, Array2, s};
use crate::ekf::model::EKFModel;

// State vector: [roll, pitch, yaw, gyro_x, gyro_y, gyro_z]
// Measurements: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
// Input data row: [t, gx, gy, gz, ax, ay, az]

pub struct ImuModel6Axis {
    pub current_time: f64,
    pub previous_time: f64,
    pub delta_time: f64,
    gravity_reference: Array1<f64>,
}

impl ImuModel6Axis {
    pub fn new(delta_time: f64) -> Self {
        Self::with_gravity_reference(delta_time, [0.0, 0.0, 1.0])
            .expect("default gravity reference vector must be valid")
    }

    /// Construct the model with a caller-provided gravity reference vector.
    ///
    /// `gravity_reference` should match the accelerometer reference direction
    /// in the world frame.
    pub fn with_gravity_reference(
        delta_time: f64,
        gravity_reference: [f64; 3],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let gravity_reference = Self::normalize_vector(&arr1(&gravity_reference));

        if gravity_reference.iter().all(|value| value.abs() <= f64::EPSILON) {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "gravity reference vector must be non-zero",
            ).into());
        }

        Ok(Self {
            current_time: -delta_time,
            previous_time: -2.0 * delta_time,
            delta_time,
            gravity_reference,
        })
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// ZYX Euler kinematic equation: [roll_dot, pitch_dot, yaw_dot] = W * omega
    #[inline]
    fn euler_angle_rates(phi: f64, theta: f64, omega: &[f64; 3]) -> [f64; 3] {
        let (sp, cp) = (phi.sin(), phi.cos());
        let (tt, ct) = (theta.tan(), theta.cos());
        [
            omega[0] + omega[1] * sp * tt + omega[2] * cp * tt,
            omega[1] * cp - omega[2] * sp,
            (omega[1] * sp + omega[2] * cp) / ct,
        ]
    }

    /// ZYX rotation matrix R(φ, θ, ψ): transforms vectors from world to body frame.
    fn euler_to_rotation_matrix(state: &Array1<f64>) -> Array2<f64> {
        let (phi, theta, psi) = (state[0], state[1], state[2]);
        let (cr, sr) = (phi.cos(), phi.sin());
        let (cp, sp) = (theta.cos(), theta.sin());
        let (cy, sy) = (psi.cos(), psi.sin());

        Array2::from_shape_vec(
            (3, 3),
            vec![
                cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr,
                sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr,
                -sp,      cp * sr,                 cp * cr,
            ],
        )
        .unwrap()
    }

    pub fn normalize_vector(v: &Array1<f64>) -> Array1<f64> {
        let norm = v.mapv(|x| x * x).sum().sqrt();
        if norm <= f64::EPSILON {
            Array1::zeros(v.len())
        } else {
            v / norm
        }
    }

    #[inline]
    fn safe_pitch(theta: f64) -> f64 {
        theta.clamp(
            -std::f64::consts::FRAC_PI_2 + 1e-4,
             std::f64::consts::FRAC_PI_2 - 1e-4,
        )
    }

    #[inline]
    fn wrap_angle(angle: f64) -> f64 {
        let wrapped = (angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        if wrapped == -std::f64::consts::PI {
            std::f64::consts::PI
        } else {
            wrapped
        }
    }

    fn dcm_angle_derivatives(state: &Array1<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let (phi, theta, psi) = (state[0], state[1], state[2]);
        let (cr, sr) = (phi.cos(), phi.sin());
        let (cp, sp) = (theta.cos(), theta.sin());
        let (cy, sy) = (psi.cos(), psi.sin());

        let dr_dphi = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, cy * sp * cr + sy * sr, -cy * sp * sr + sy * cr,
                0.0, sy * sp * cr - cy * sr, -sy * sp * sr - cy * cr,
                0.0, cp * cr,                -cp * sr,
            ],
        ).unwrap();

        let dr_dtheta = Array2::from_shape_vec(
            (3, 3),
            vec![
                -cy * sp, cy * cp * sr, cy * cp * cr,
                -sy * sp, sy * cp * sr, sy * cp * cr,
                -cp,      -sp * sr,     -sp * cr,
            ],
        ).unwrap();

        let dr_dpsi = Array2::from_shape_vec(
            (3, 3),
            vec![
                -sy * cp, -sy * sp * sr - cy * cr, -sy * sp * cr + cy * sr,
                 cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr,
                 0.0,      0.0,                      0.0,
            ],
        ).unwrap();

        (dr_dphi, dr_dtheta, dr_dpsi)
    }

    fn theta_is_clamped(theta: f64) -> bool {
        (theta - Self::safe_pitch(theta)).abs() > f64::EPSILON
    }
}

impl EKFModel for ImuModel6Axis {
    fn delta_time(&self) -> Option<f64> {
        Some(self.delta_time)
    }

    /// Parse a 7-element data row `[t, gx, gy, gz, ax, ay, az]`.
    fn parse_data(&mut self, data: &[f64]) -> Array1<f64> {
        self.previous_time = self.current_time;
        self.current_time = data[0];
        let parsed_dt = self.current_time - self.previous_time;
        if parsed_dt.is_finite() && parsed_dt > 0.0 {
            self.delta_time = parsed_dt;
        }

        let accel = Self::normalize_vector(&arr1(&[data[4], data[5], data[6]]));

        Array1::from(vec![
            data[1], data[2], data[3], // gyro  (rad/s)
            accel[0], accel[1], accel[2],
        ])
    }

    /// Discrete state transition x[k+1] = x[k] + dt * f(x[k]).
    fn state_transition_function(&self, state: &Array1<f64>, dt: f64) -> Array1<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return state.clone();
        }

        let phi = state[0];
        let theta = Self::safe_pitch(state[1]);
        let omega = [state[3], state[4], state[5]];
        let euler_dot = Self::euler_angle_rates(phi, theta, &omega);

        arr1(&[
            Self::wrap_angle(state[0] + dt * euler_dot[0]),
            Self::safe_pitch(state[1] + dt * euler_dot[1]),
            Self::wrap_angle(state[2] + dt * euler_dot[2]),
            state[3],
            state[4],
            state[5],
        ])
    }

    fn state_transition_jacobian(&self, state: &Array1<f64>, dt: f64) -> Array2<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return Array2::eye(state.len());
        }

        let phi = state[0];
        let theta = Self::safe_pitch(state[1]);
        let omega = [state[3], state[4], state[5]];

        let (sp, cp) = (phi.sin(), phi.cos());
        let (tt, ct) = (theta.tan(), theta.cos());
        let st = theta.sin();

        let mut f = Array2::<f64>::eye(6);

        f[[0, 0]] += dt * (omega[1] * cp * tt - omega[2] * sp * tt);
        if !Self::theta_is_clamped(state[1]) {
            f[[0, 1]] += dt * (omega[1] * sp + omega[2] * cp) / (ct * ct);
        }
        f[[0, 3]] = dt;
        f[[0, 4]] = dt * sp * tt;
        f[[0, 5]] = dt * cp * tt;

        f[[1, 0]] += dt * (-omega[1] * sp - omega[2] * cp);
        f[[1, 4]] = dt * cp;
        f[[1, 5]] = -dt * sp;

        f[[2, 0]] += dt * (omega[1] * cp - omega[2] * sp) / ct;
        if !Self::theta_is_clamped(state[1]) {
            f[[2, 1]] += dt * (omega[1] * sp + omega[2] * cp) * st / (ct * ct);
        }
        f[[2, 4]] = dt * sp / ct;
        f[[2, 5]] = dt * cp / ct;

        f
    }

    /// Measurement prediction h(x) for the 6-axis IMU.
    fn measurement_prediction_function(&self, state: &Array1<f64>) -> Array1<f64> {
        let euler = state.slice(s![0..3]).to_owned();
        let r = Self::euler_to_rotation_matrix(&euler);
        let gyro_pred = state.slice(s![3..6]).to_owned();
        let accel_pred = r.dot(&self.gravity_reference);

        let mut z = Array1::zeros(6);
        z.slice_mut(s![0..3]).assign(&gyro_pred);
        z.slice_mut(s![3..6]).assign(&accel_pred);
        z
    }

    fn measurement_prediction_jacobian(&self, state: &Array1<f64>) -> Array2<f64> {
        let (dr_dphi, dr_dtheta, dr_dpsi) = Self::dcm_angle_derivatives(state);
        let accel_dphi   = dr_dphi.dot(&self.gravity_reference);
        let accel_dtheta = dr_dtheta.dot(&self.gravity_reference);
        let accel_dpsi   = dr_dpsi.dot(&self.gravity_reference);

        let mut h = Array2::<f64>::zeros((6, 6));
        h[[0, 3]] = 1.0;
        h[[1, 4]] = 1.0;
        h[[2, 5]] = 1.0;

        h.slice_mut(s![3..6, 0]).assign(&accel_dphi);
        h.slice_mut(s![3..6, 1]).assign(&accel_dtheta);
        h.slice_mut(s![3..6, 2]).assign(&accel_dpsi);

        h
    }
}
