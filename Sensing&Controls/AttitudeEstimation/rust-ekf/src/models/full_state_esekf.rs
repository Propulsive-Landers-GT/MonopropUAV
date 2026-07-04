use crate::es_ekf::model::ESEKFModel;
use ndarray::{array, Array1, Array2};
use nalgebra::{UnitQuaternion, Vector3, Quaternion};

pub struct RocketState;

impl ESEKFModel for RocketState {
    /// Nominal state: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, abx, aby, abz, wbx, wby, wbz] (16D)
    fn nominal_prediction(&self, state: &Array1<f64>, imu: &[f64], dt: f64) -> Array1<f64> {
        let mut next_state = state.clone();

        // Extract current state vectors
        let pos = Vector3::new(state[0], state[1], state[2]);
        let vel = Vector3::new(state[3], state[4], state[5]);
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(state[6], state[7], state[8], state[9]));
        let a_bias = Vector3::new(state[10], state[11], state[12]);
        let w_bias = Vector3::new(state[13], state[14], state[15]);

        // Extract IMU inputs
        let a_measured = Vector3::new(imu[0], imu[1], imu[2]);
        let w_measured = Vector3::new(imu[3], imu[4], imu[5]);

        // Unbias IMU
        let a_body = a_measured - a_bias;
        let w_body = w_measured - w_bias;

        // Kinematics using STRICT Body-to-World convention
        // a_world = R * a_body - g (assuming Z is up)
        let gravity = Vector3::new(0.0, 0.0, 9.81); 
        let a_world = quat.transform_vector(&a_body) - gravity;

        let next_pos = pos + vel * dt + 0.5 * a_world * dt * dt;
        let next_vel = vel + a_world * dt;

        // Quaternion integration (using small angle approximation for omega)
        let w_norm = w_body.norm();
        let q_update = if w_norm > 1e-6 {
            let axis = w_body / w_norm;
            UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), w_norm * dt)
        } else {
            UnitQuaternion::identity()
        };
        let next_quat = quat * q_update; // Post-multiply for local frame rotation

        // Update state array
        next_state[0..3].copy_from_slice(next_pos.as_slice());
        next_state[3..6].copy_from_slice(next_vel.as_slice());
        next_state[6] = next_quat.w;
        next_state[7] = next_quat.i;
        next_state[8] = next_quat.j;
        next_state[9] = next_quat.k;
        // Biases remain constant in prediction

        next_state
    }

    /// Error Jacobian F (15x15)
    fn error_transition_jacobian(&self, state: &Array1<f64>, imu: &[f64], dt: f64) -> Array2<f64> {
        let mut f = Array2::<f64>::eye(15);
        
        // This is a simplified placeholder. In reality, you must compute the derivatives 
        // of position w.r.t velocity, velocity w.r.t attitude error, etc.
        // Example: Position changes due to velocity
        f[[0, 3]] = dt; f[[1, 4]] = dt; f[[2, 5]] = dt;
        
        // Example: Attitude error updates based on body rates and gyro bias
        // Requires forming the skew-symmetric matrix of the un-biased body rates.

        f
    }

    /// Example Measurement: GPS Position (3D)
    fn measurement_prediction(&self, state: &Array1<f64>) -> Array1<f64> {
        // Assuming GPS antenna is exactly at IMU center for simplicity here.
        // If there is a lever arm, apply: p + R * r_arm
        array![state[0], state[1], state[2]] 
    }

    /// Measurement Jacobian H (3x15)
    fn measurement_jacobian(&self, _state: &Array1<f64>) -> Array2<f64> {
        let mut h = Array2::<f64>::zeros((3, 15));
        // Direct map from position error (indices 0, 1, 2) to measurement
        h[[0, 0]] = 1.0;
        h[[1, 1]] = 1.0;
        h[[2, 2]] = 1.0;

        // If using a lever arm, you must populate h[[0..3, 6..9]] with the 
        // skew-symmetric matrix of the rotated lever arm.
        h
    }

    /// Error Injection (15D -> 16D)
    fn inject_error(&self, nominal: &Array1<f64>, error: &Array1<f64>) -> Array1<f64> {
        let mut injected = nominal.clone();

        // 1. Standard linear additions (pos, vel, biases)
        for i in 0..6 { injected[i] += error[i]; }
        for i in 10..16 { injected[i] += error[i - 1]; } // Map 15D indices to 16D indices

        // 2. Quaternion multiplicative correction
        let q_nom = UnitQuaternion::from_quaternion(Quaternion::new(nominal[6], nominal[7], nominal[8], nominal[9]));
        
        // Error state 6, 7, 8 is delta_theta. Create a local error quaternion.
        let half_dtheta = Vector3::new(error[6], error[7], error[8]) * 0.5;
        let q_err = Quaternion::new(1.0, half_dtheta.x, half_dtheta.y, half_dtheta.z);
        
        // Normalize the error quaternion (handles cases where error is slightly large)
        let q_err_unit = UnitQuaternion::from_quaternion(q_err).normalize();

        // Apply error rotation to nominal
        let q_corrected = q_nom * q_err_unit;

        injected[6] = q_corrected.w;
        injected[7] = q_corrected.i;
        injected[8] = q_corrected.j;
        injected[9] = q_corrected.k;

        injected
    }
}