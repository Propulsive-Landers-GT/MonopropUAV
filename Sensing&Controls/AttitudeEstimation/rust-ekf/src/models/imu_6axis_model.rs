use ndarray::{Array1, Array2};
use crate::ekf::model::EKFModel;

// State vector: [roll, pitch, yaw, bias_x, bias_y, bias_z]
// Measurements: [accel_x, accel_y, accel_z]
// Gyro is used as a control input in state transition (not a measurement)

pub struct ImuModel6Axis {
    pub delta_time: f64,
    pub current_time: f64,
    pub previous_time: f64,
    gyro: Array1<f64>,
}

impl ImuModel6Axis {
    pub fn new(delta_time: f64) -> Self {
        Self {
            delta_time,
            current_time: -delta_time,
            previous_time: -2.0 * delta_time,
            gyro: Array1::zeros(3),
        }
    }
}

impl EKFModel for ImuModel6Axis {
    fn parse_data(&mut self, data: &[f64]) -> Array1<f64> {
        self.previous_time = self.current_time;
        self.current_time = data[0];
        self.delta_time = self.current_time - self.previous_time;

        // Store gyro internally — used in state transition, not as a measurement
        self.gyro = Array1::from(vec![data[1], data[2], data[3]]);

        // Only accel is returned as the measurement vector
        Array1::from(vec![data[4], data[5], data[6]])
    }

    fn state_transition_function(&self, state: &Array1<f64>, dt: f64) -> Array1<f64> {
        let roll  = state[0];
        let pitch = state[1];
        let yaw   = state[2];
        let bx    = state[3];
        let by    = state[4];
        let bz    = state[5];

        // Subtract bias from raw gyro to get corrected angular rate
        let wx = self.gyro[0] - bx;
        let wy = self.gyro[1] - by;
        let wz = self.gyro[2] - bz;

        // Euler integration to propagate angles forward
        // Biases modeled as constant (random walk — no change until accel corrects them)
        Array1::from(vec![
            roll  + wx * dt,
            pitch + wy * dt,
            yaw   + wz * dt,
            bx,
            by,
            bz,
        ])
    }

    fn state_transition_jacobian(&self, _state: &Array1<f64>, dt: f64) -> Array2<f64> {
        // Analytical Jacobian of state_transition_function w.r.t. state
        // d(roll)/d(bias_x) = -dt, same for pitch/bias_y and yaw/bias_z
        // Everything else is identity
        let mut f = Array2::eye(6);
        f[[0, 3]] = -dt;
        f[[1, 4]] = -dt;
        f[[2, 5]] = -dt;
        f
    }

    fn measurement_prediction_function(&self, state: &Array1<f64>) -> Array1<f64> {
        // Predict what the accelerometer should read given current attitude
        // Rotate gravity vector into body frame, then normalize
        let rotation_matrix = Self::euler_to_rotation_matrix(state);
        let gravity = Array1::from(vec![0.0, 0.0, -9.81]);
        Self::normalize_vector(&rotation_matrix.dot(&gravity))
    }

    fn measurement_prediction_jacobian(&self, state: &Array1<f64>) -> Array2<f64> {
        // Numerical Jacobian via central finite differences
        // Analytical form is complex — this is clean and accurate enough
        let eps = 1e-6;
        let h0 = self.measurement_prediction_function(state);
        let m = h0.len(); // 3 (accel x/y/z)
        let n = state.len(); // 6
        let mut jac = Array2::zeros((m, n));

        for i in 0..n {
            let mut perturbed = state.clone();
            perturbed[i] += eps;
            let h1 = self.measurement_prediction_function(&perturbed);
            for j in 0..m {
                jac[[j, i]] = (h1[j] - h0[j]) / eps;
            }
        }
        jac
    }
}

impl ImuModel6Axis {
    fn euler_to_rotation_matrix(euler: &Array1<f64>) -> Array2<f64> {
        let (roll, pitch, yaw) = (euler[0], euler[1], euler[2]);

        let cr = roll.cos();
        let sr = roll.sin();
        let cp = pitch.cos();
        let sp = pitch.sin();
        let cy = yaw.cos();
        let sy = yaw.sin();

        Array2::from_shape_vec(
            (3, 3),
            vec![
                cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                -sp,     cp * sr,                cp * cr,
            ],
        )
        .unwrap()
    }

    fn normalize_vector(v: &Array1<f64>) -> Array1<f64> {
        let norm = v.mapv(|x| x * x).sum().sqrt();
        if norm == 0.0 { v.clone() } else { v / norm }
    }
}