use crate::es_ekf::model::ESEKFModel;
use ndarray::{array, Array1, Array2};
use nalgebra::{UnitQuaternion, Vector3, Quaternion};

/// IMU noise characteristics for the VectorNav VN-200, taken from the
/// datasheet (https://www.vectornav.com/products/detail/vn-200):
///
///   - Accel noise density:         < 0.14 mg/sqrt(Hz)
///   - Accel in-run bias stability: < 0.04 mg
///   - Gyro noise density:            0.0035 deg/s/sqrt(Hz)
///   - Gyro in-run bias stability:  < 10 deg/hr (5 deg/hr typical)
///
/// These match the simulated IMU in
/// Algorithms/RocketSimulation/rust_rocket_sim/src/device_sim.rs, which uses
/// accel sigma 0.00137 m/s^2/sqrt(Hz) and gyro sigma 6.1e-5 rad/s/sqrt(Hz).
///
/// IMU sample rate: the datasheet supports IMU output up to 1 kHz; the
/// simulated IMU runs at 300 Hz (device_sim.rs) and the recorded
/// flight_data.csv is logged at 100 Hz. Because the rate varies by source,
/// nothing here hardcodes it — pass the actual sample period `dt` to
/// [`RocketState::process_noise`].
pub mod vn200 {
    /// Accel noise density [m/s^2/sqrt(Hz)]: 0.14 mg/sqrt(Hz) * 9.80665.
    pub const ACCEL_NOISE_DENSITY: f64 = 0.14e-3 * 9.80665;

    /// Gyro noise density [rad/s/sqrt(Hz)]: 0.0035 deg/s/sqrt(Hz).
    pub const GYRO_NOISE_DENSITY: f64 = 0.0035 * std::f64::consts::PI / 180.0;

    /// Accel in-run bias stability [m/s^2]: 0.04 mg * 9.80665.
    pub const ACCEL_BIAS_INSTABILITY: f64 = 0.04e-3 * 9.80665;

    /// Gyro in-run bias stability [rad/s]: 10 deg/hr (datasheet worst case).
    pub const GYRO_BIAS_INSTABILITY: f64 = 10.0 / 3600.0 * std::f64::consts::PI / 180.0;

    /// Assumed correlation time [s] of the in-run bias wander. The datasheet
    /// gives bias *instability*, not a random-walk density; the standard
    /// first-order Gauss-Markov approximation converts one to the other as
    /// sigma_rw = instability / sqrt(tau).
    pub const BIAS_CORRELATION_TIME: f64 = 100.0;

    /// Accel bias random-walk density [m/s^2/sqrt(s)] = instability / sqrt(tau).
    pub const ACCEL_BIAS_RANDOM_WALK: f64 = ACCEL_BIAS_INSTABILITY / 10.0; // sqrt(100)

    /// Gyro bias random-walk density [rad/s/sqrt(s)] = instability / sqrt(tau).
    pub const GYRO_BIAS_RANDOM_WALK: f64 = GYRO_BIAS_INSTABILITY / 10.0; // sqrt(100)
}

pub struct RocketState;

impl RocketState {
    /// 3x3 skew-symmetric (cross-product) matrix of a vector, such that
    /// `skew(a) * b == a x b`.
    fn skew(v: &Vector3<f64>) -> Array2<f64> {
        Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, -v.z, v.y,
                v.z, 0.0, -v.x,
                -v.y, v.x, 0.0,
            ],
        )
        .expect("3x3 skew matrix shape is fixed")
    }

    /// Rotation matrix (body -> world) from the nominal unit quaternion,
    /// returned as an ndarray for use in the error-state Jacobian blocks.
    fn quat_to_rotation_matrix(q: &UnitQuaternion<f64>) -> Array2<f64> {
        let m = q.to_rotation_matrix();
        let m = m.matrix();
        Array2::from_shape_vec(
            (3, 3),
            vec![
                m[(0, 0)], m[(0, 1)], m[(0, 2)],
                m[(1, 0)], m[(1, 1)], m[(1, 2)],
                m[(2, 0)], m[(2, 1)], m[(2, 2)],
            ],
        )
        .expect("3x3 rotation matrix shape is fixed")
    }

    /// Copy a 3x3 `block` into `target` with its top-left corner at (row, col).
    fn set_block(target: &mut Array2<f64>, row: usize, col: usize, block: &Array2<f64>) {
        for i in 0..3 {
            for j in 0..3 {
                target[[row + i, col + j]] = block[[i, j]];
            }
        }
    }

    /// Discrete process-noise covariance Q (15x15) built from the VN-200
    /// datasheet values in [`vn200`], for an IMU sample period `dt` [s].
    ///
    /// Follows the Sola ES-EKF discretization, where each white-noise density
    /// integrates over one sample period into a covariance of density^2 * dt:
    ///   - dv     <- accel measurement noise:   (accel noise density)^2 * dt
    ///   - dtheta <- gyro measurement noise:    (gyro noise density)^2 * dt
    ///   - da_b   <- accel bias random walk:    (accel bias RW density)^2 * dt
    ///   - dw_b   <- gyro bias random walk:     (gyro bias RW density)^2 * dt
    ///
    /// The position block is zero: position uncertainty grows only through the
    /// velocity coupling in F, not from direct process noise.
    ///
    /// The filter adds this Q once per predict() call, so it must be built
    /// with the same `dt` used for prediction (rebuild it if the IMU rate
    /// changes).
    pub fn process_noise(dt: f64) -> Array2<f64> {
        let q_v = vn200::ACCEL_NOISE_DENSITY.powi(2) * dt;
        let q_theta = vn200::GYRO_NOISE_DENSITY.powi(2) * dt;
        let q_ab = vn200::ACCEL_BIAS_RANDOM_WALK.powi(2) * dt;
        let q_wb = vn200::GYRO_BIAS_RANDOM_WALK.powi(2) * dt;

        let mut q = Array2::<f64>::zeros((15, 15));
        for i in 0..3 {
            q[[3 + i, 3 + i]] = q_v;
            q[[6 + i, 6 + i]] = q_theta;
            q[[9 + i, 9 + i]] = q_ab;
            q[[12 + i, 12 + i]] = q_wb;
        }
        q
    }

    /// Recommended initial error covariance P0 (15x15).
    ///
    /// The bias blocks matter most: an over-inflated bias uncertainty (e.g. a
    /// uniform eye * 0.01, i.e. gyro-bias sigma 0.1 rad/s ~ 5.7 deg/s) lets
    /// measurement updates slam the bias estimates around, and the bias-error
    /// component along any unobservable direction (rotation about the magnetic
    /// field axis, for mag-aided yaw) integrates directly into attitude error.
    /// Bounding the bias blocks near the VN-200's actual bias scale fixed a
    /// 15-19 deg attitude transient in replay testing.
    ///
    ///   - position:   sigma 0.1 m
    ///   - velocity:   sigma 0.1 m/s
    ///   - attitude:   sigma 0.1 rad (~5.7 deg)
    ///   - accel bias: sigma 0.01 m/s^2 (~1 mg; generous vs the 0.04 mg
    ///     in-run stability, leaving room for turn-on bias)
    ///   - gyro bias:  sigma 1e-3 rad/s (~0.06 deg/s; generous vs the
    ///     10 deg/hr in-run stability, leaving room for turn-on bias)
    pub fn initial_covariance() -> Array2<f64> {
        let mut p = Array2::<f64>::eye(15) * 0.01;
        for i in 9..12 {
            p[[i, i]] = 1e-4; // accel bias
        }
        for i in 12..15 {
            p[[i, i]] = 1e-6; // gyro bias
        }
        p
    }

    /// Predicted magnetometer measurement (3D, body frame): the known world
    /// magnetic field rotated into the body frame, m_body = R^T * m_world.
    ///
    /// Fusing this makes yaw observable — GPS position alone leaves rotation
    /// about the world vertical (and gyro-Z bias) unobserved, so yaw drifts.
    pub fn mag_prediction(state: &Array1<f64>, mag_world: &Vector3<f64>) -> Array1<f64> {
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            state[6], state[7], state[8], state[9],
        ));
        let m_body = quat.inverse_transform_vector(mag_world);
        array![m_body.x, m_body.y, m_body.z]
    }

    /// Magnetometer Jacobian (3x15) with respect to the error state.
    ///
    /// Only the attitude block is nonzero. With the local (right-multiplied)
    /// attitude-error convention used throughout this model,
    ///   h(q (x) dq) = (R(q) R(dtheta))^T m_w ~= (I - skew(dtheta)) R^T m_w
    ///               = m_body + skew(m_body) * dtheta,
    /// so H[0..3, 6..9] = skew(R^T * m_world).
    pub fn mag_jacobian(state: &Array1<f64>, mag_world: &Vector3<f64>) -> Array2<f64> {
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            state[6], state[7], state[8], state[9],
        ));
        let m_body = quat.inverse_transform_vector(mag_world);
        let mut h = Array2::<f64>::zeros((3, 15));
        Self::set_block(&mut h, 0, 6, &Self::skew(&m_body));
        h
    }
}

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

        // Kinematics using STRICT Body-to-World convention.
        // FRAME ASSUMPTION: world frame is Z-up, so gravity is +9.81 along world Z
        // and is subtracted from the rotated specific-force to get world acceleration:
        //   a_world = R * a_body - g
        let gravity = Vector3::new(0.0, 0.0, 9.81); 
        let a_world = quat.transform_vector(&a_body) - gravity;

        let next_pos = pos + vel * dt + 0.5 * a_world * dt * dt;
        let next_vel = vel + a_world * dt;

        // Quaternion integration via the exponential map (from_scaled_axis
        // handles the small-rotation limit internally).
        let q_update = UnitQuaternion::from_scaled_axis(w_body * dt);
        let next_quat = quat * q_update; // Post-multiply for local frame rotation

        // Update state array (ndarray Array1 does not support range slice assignment
        // from a foreign slice, so write components explicitly).
        next_state[0] = next_pos.x;
        next_state[1] = next_pos.y;
        next_state[2] = next_pos.z;
        next_state[3] = next_vel.x;
        next_state[4] = next_vel.y;
        next_state[5] = next_vel.z;
        next_state[6] = next_quat.w;
        next_state[7] = next_quat.i;
        next_state[8] = next_quat.j;
        next_state[9] = next_quat.k;
        // Biases remain constant in prediction

        next_state
    }

    /// Error Jacobian F (15x15)
    fn error_transition_jacobian(&self, state: &Array1<f64>, imu: &[f64], dt: f64) -> Array2<f64> {
        // === ES-EKF ERROR-STATE TRANSITION JACOBIAN F (15x15) ===
        // COMPLETED (was a simplified placeholder). Discretized as F = I + A*dt using
        // the Sola error-state convention, with R = body->world from the nominal
        // quaternion. Error-state ordering:
        //   [dp(0..3), dv(3..6), dtheta(6..9), da_b(9..12), dw_b(12..15)]
        // Nonzero off-identity blocks:
        //   F[dp,     dv]     =  I * dt
        //   F[dv,     dtheta] = -R * skew(a_body) * dt
        //   F[dv,     da_b]   = -R * dt
        //   F[dtheta, dtheta] =  I - skew(w_body) * dt
        //   F[dtheta, dw_b]   = -I * dt
        let mut f = Array2::<f64>::eye(15);

        // Nominal attitude and IMU inputs, with biases removed.
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            state[6], state[7], state[8], state[9],
        ));
        let a_bias = Vector3::new(state[10], state[11], state[12]);
        let w_bias = Vector3::new(state[13], state[14], state[15]);
        let a_measured = Vector3::new(imu[0], imu[1], imu[2]);
        let w_measured = Vector3::new(imu[3], imu[4], imu[5]);
        let a_body = a_measured - a_bias;
        let w_body = w_measured - w_bias;

        let r = Self::quat_to_rotation_matrix(&quat);
        let identity3 = Array2::<f64>::eye(3);

        // F[dp, dv] = I * dt
        Self::set_block(&mut f, 0, 3, &(&identity3 * dt));

        // F[dv, dtheta] = -R * skew(a_body) * dt
        let dv_dtheta = r.dot(&Self::skew(&a_body)) * (-dt);
        Self::set_block(&mut f, 3, 6, &dv_dtheta);

        // F[dv, da_b] = -R * dt
        let dv_dab = &r * (-dt);
        Self::set_block(&mut f, 3, 9, &dv_dab);

        // F[dtheta, dtheta] = I - skew(w_body) * dt
        let dtheta_dtheta = &identity3 - &(Self::skew(&w_body) * dt);
        Self::set_block(&mut f, 6, 6, &dtheta_dtheta);

        // F[dtheta, dw_b] = -I * dt
        Self::set_block(&mut f, 6, 12, &(&identity3 * -dt));

        f
    }

    /// Example Measurement: GPS Position (3D)
    fn measurement_prediction(&self, state: &Array1<f64>) -> Array1<f64> {
        // Assuming GPS antenna is exactly at IMU center for simplicity here.
        // TODO(lever-arm): if the antenna is offset from the IMU, predict
        // p + R * r_arm instead of the raw position.
        array![state[0], state[1], state[2]] 
    }

    /// Measurement Jacobian H (3x15)
    fn measurement_jacobian(&self, _state: &Array1<f64>) -> Array2<f64> {
        let mut h = Array2::<f64>::zeros((3, 15));
        // Direct map from position error (indices 0, 1, 2) to measurement
        h[[0, 0]] = 1.0;
        h[[1, 1]] = 1.0;
        h[[2, 2]] = 1.0;

        // TODO(lever-arm): if using a lever arm, populate h[[0..3, 6..9]] with the
        // skew-symmetric matrix of the rotated lever arm (attitude error couples
        // into the predicted antenna position).
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
        
        // Normalize the error quaternion (handles cases where error is slightly large).
        // `from_quaternion` normalizes internally and yields a UnitQuaternion.
        let q_err_unit = UnitQuaternion::from_quaternion(q_err);

        // Apply error rotation to nominal (UnitQuaternion * UnitQuaternion).
        let q_corrected = q_nom * q_err_unit;

        injected[6] = q_corrected.w;
        injected[7] = q_corrected.i;
        injected[8] = q_corrected.j;
        injected[9] = q_corrected.k;

        injected
    }
}

#[cfg(test)]
mod tests {
    use super::RocketState;
    use crate::es_ekf::filter::ErrorStateKalmanFilter;
    use crate::es_ekf::model::ESEKFModel;
    use nalgebra::{Quaternion, UnitQuaternion};
    use ndarray::{array, Array1, Array2};

    fn identity_state() -> Array1<f64> {
        let mut s = Array1::<f64>::zeros(16);
        s[6] = 1.0; // qw
        s
    }

    fn quat_norm(x: &Array1<f64>) -> f64 {
        (x[6] * x[6] + x[7] * x[7] + x[8] * x[8] + x[9] * x[9]).sqrt()
    }

    /// Local (body-frame) boxminus: the 15D error such that
    /// `inject_error(x_ref, boxminus(x_pert, x_ref)) ~= x_pert` for small errors.
    /// Inverse of `inject_error`, needed to numerically differentiate the error dynamics.
    fn boxminus(x_pert: &Array1<f64>, x_ref: &Array1<f64>) -> Array1<f64> {
        let mut d = Array1::<f64>::zeros(15);
        for i in 0..3 {
            d[i] = x_pert[i] - x_ref[i]; // position
            d[3 + i] = x_pert[3 + i] - x_ref[3 + i]; // velocity
        }
        // Attitude error: dtheta = 2 * vec(q_ref^-1 (x) q_pert)  (local convention)
        let q_ref = UnitQuaternion::from_quaternion(Quaternion::new(
            x_ref[6], x_ref[7], x_ref[8], x_ref[9],
        ));
        let q_pert = UnitQuaternion::from_quaternion(Quaternion::new(
            x_pert[6], x_pert[7], x_pert[8], x_pert[9],
        ));
        let q_err = q_ref.inverse() * q_pert;
        let sign = if q_err.w < 0.0 { -1.0 } else { 1.0 }; // shortest rotation
        d[6] = 2.0 * sign * q_err.i;
        d[7] = 2.0 * sign * q_err.j;
        d[8] = 2.0 * sign * q_err.k;
        for i in 0..6 {
            d[9 + i] = x_pert[10 + i] - x_ref[10 + i]; // accel + gyro biases
        }
        d
    }

    /// A representative non-trivial state + IMU input for the Jacobian test.
    fn sample_state() -> (Array1<f64>, [f64; 6], f64) {
        let mut x = identity_state();
        let q = UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3);
        x[6] = q.w;
        x[7] = q.i;
        x[8] = q.j;
        x[9] = q.k;
        x[3] = 1.0;
        x[4] = -2.0;
        x[5] = 0.5; // velocity
        x[10] = 0.05;
        x[11] = -0.02;
        x[12] = 0.1; // accel bias
        x[13] = 0.01;
        x[14] = -0.03;
        x[15] = 0.02; // gyro bias
        let imu = [0.3, -9.0, 2.0, 0.2, -0.1, 0.05]; // accel xyz, gyro xyz
        (x, imu, 1e-3)
    }

    #[test]
    fn quaternion_stays_normalized() {
        let model = RocketState;
        let mut x = identity_state();
        let imu = [0.5, 0.2, 9.9, 0.3, -0.2, 0.1];
        for _ in 0..2000 {
            x = model.nominal_prediction(&x, &imu, 0.01);
        }
        assert!(
            (quat_norm(&x) - 1.0).abs() < 1e-9,
            "quaternion norm drifted during prediction: {}",
            quat_norm(&x)
        );

        // ...and after injecting a sizeable attitude error.
        let mut d = Array1::<f64>::zeros(15);
        d[6] = 0.1;
        d[7] = -0.05;
        d[8] = 0.2;
        let xi = model.inject_error(&x, &d);
        assert!(
            (quat_norm(&xi) - 1.0).abs() < 1e-9,
            "quaternion norm drifted after injection: {}",
            quat_norm(&xi)
        );
    }

    #[test]
    fn measurement_jacobian_matches_finite_difference() {
        let model = RocketState;
        let (x, _imu, _dt) = sample_state();
        let h = model.measurement_jacobian(&x);
        let eps = 1e-6;
        let base = model.measurement_prediction(&x);
        for j in 0..15 {
            let mut dp = Array1::<f64>::zeros(15);
            dp[j] = eps;
            let x_plus = model.inject_error(&x, &dp);
            let pred_plus = model.measurement_prediction(&x_plus);
            for i in 0..3 {
                let numeric = (pred_plus[i] - base[i]) / eps;
                assert!(
                    (h[[i, j]] - numeric).abs() < 1e-4,
                    "H[{i},{j}] analytic {} vs numeric {}",
                    h[[i, j]],
                    numeric
                );
            }
        }
    }

    #[test]
    fn error_transition_jacobian_matches_finite_difference() {
        let model = RocketState;
        let (x, imu, dt) = sample_state();
        let f_analytic = model.error_transition_jacobian(&x, &imu, dt);

        let eps = 1e-6;
        let x1 = model.nominal_prediction(&x, &imu, dt);

        let mut max_err = 0.0f64;
        for j in 0..15 {
            let mut dp = Array1::<f64>::zeros(15);
            dp[j] = eps;
            let mut dm = Array1::<f64>::zeros(15);
            dm[j] = -eps;
            let x1_plus = model.nominal_prediction(&model.inject_error(&x, &dp), &imu, dt);
            let x1_minus = model.nominal_prediction(&model.inject_error(&x, &dm), &imu, dt);
            let d_plus = boxminus(&x1_plus, &x1);
            let d_minus = boxminus(&x1_minus, &x1);
            for i in 0..15 {
                let numeric = (d_plus[i] - d_minus[i]) / (2.0 * eps);
                max_err = max_err.max((f_analytic[[i, j]] - numeric).abs());
            }
        }
        // At dt = 1e-3 the first-order F should match the exact nonlinear
        // propagation to O(dt^2) ~ 1e-6, well inside this tolerance.
        assert!(
            max_err < 1e-4,
            "F disagrees with finite-difference Jacobian: max abs error = {max_err}"
        );
    }

    /// World magnetic field used by the simulator (device_sim.rs), in teslas.
    fn sim_mag_world() -> nalgebra::Vector3<f64> {
        nalgebra::Vector3::new(-2.0e-6, 22.0e-6, -44.3e-6)
    }

    #[test]
    fn mag_jacobian_matches_finite_difference() {
        let model = RocketState;
        let (x, _imu, _dt) = sample_state();
        let m_w = sim_mag_world();
        let h = RocketState::mag_jacobian(&x, &m_w);
        let eps = 1e-6;
        let base = RocketState::mag_prediction(&x, &m_w);
        for j in 0..15 {
            let mut dp = Array1::<f64>::zeros(15);
            dp[j] = eps;
            let x_plus = model.inject_error(&x, &dp);
            let pred_plus = RocketState::mag_prediction(&x_plus, &m_w);
            for i in 0..3 {
                let numeric = (pred_plus[i] - base[i]) / eps;
                assert!(
                    (h[[i, j]] - numeric).abs() < 1e-9,
                    "H_mag[{i},{j}] analytic {} vs numeric {}",
                    h[[i, j]],
                    numeric
                );
            }
        }
    }

    #[test]
    fn mag_updates_correct_observable_attitude_error() {
        // Truth attitude is identity; the filter starts with a 0.2 rad yaw
        // error. A single reference vector cannot observe rotation ABOUT that
        // vector, so mag updates must (a) remove the error component
        // perpendicular to the field and (b) leave only a residual rotation
        // aligned with the field axis. With the sim's 26.5-degree field tilt
        // that still removes ~20% of the yaw error per full correction; the
        // rest is recovered in the full filter by the roll/pitch information
        // in the GPS/accel channel.
        let m_w = sim_mag_world();
        let z_mag = array![m_w.x, m_w.y, m_w.z]; // body frame of identity truth

        let mut init = identity_state();
        let q0 = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.2);
        init[6] = q0.w;
        init[7] = q0.i;
        init[8] = q0.j;
        init[9] = q0.k;

        let mut ekf = ErrorStateKalmanFilter::new(
            init,
            Array2::<f64>::eye(15) * 0.01,
            RocketState::process_noise(0.01),
            RocketState,
        );
        let r_mag = Array2::<f64>::eye(3) * (2.0e-7_f64).powi(2);

        for _ in 0..20 {
            let pred = RocketState::mag_prediction(&ekf.nominal_state, &m_w);
            let h = RocketState::mag_jacobian(&ekf.nominal_state, &m_w);
            ekf.update_with(&z_mag, &pred, &h, &r_mag);
        }

        // Remaining attitude error (truth is identity, so the state IS the error).
        let s = &ekf.nominal_state;
        let q_err = UnitQuaternion::from_quaternion(Quaternion::new(s[6], s[7], s[8], s[9]));
        let err_angle = q_err.angle();

        // (a) The total error shrank (perpendicular component removed):
        // residual should be ~0.2 * cos(field tilt) = 0.179 rad, not 0.2.
        assert!(
            err_angle < 0.19,
            "mag updates removed no attitude error: {err_angle} (started at 0.2)"
        );

        // (b) The residual rotation is about the field axis (unobservable
        // direction) — its axis must align with the world field direction.
        let axis = q_err
            .axis()
            .expect("residual error should be a nonzero rotation");
        let field_dir = m_w / m_w.norm();
        let alignment = axis.dot(&field_dir).abs();
        assert!(
            alignment > 0.99,
            "residual error axis not aligned with field direction: dot = {alignment}"
        );
    }

    #[test]
    fn process_noise_is_diagonal_psd_and_scales_with_dt() {
        let dt = 1.0 / 300.0; // simulated VN-200 rate in device_sim.rs
        let q = RocketState::process_noise(dt);
        assert_eq!(q.dim(), (15, 15));
        for i in 0..15 {
            for j in 0..15 {
                if i == j {
                    assert!(q[[i, j]] >= 0.0, "Q[{i},{i}] negative");
                } else {
                    assert_eq!(q[[i, j]], 0.0, "Q[{i},{j}] should be zero");
                }
            }
        }
        // Position rows carry no direct process noise; all driven blocks do.
        for i in 0..3 {
            assert_eq!(q[[i, i]], 0.0);
            assert!(q[[3 + i, 3 + i]] > 0.0);
            assert!(q[[6 + i, 6 + i]] > 0.0);
            assert!(q[[9 + i, 9 + i]] > 0.0);
            assert!(q[[12 + i, 12 + i]] > 0.0);
        }
        // White-noise densities integrate linearly in dt.
        let q2 = RocketState::process_noise(2.0 * dt);
        assert!((q2[[3, 3]] / q[[3, 3]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn covariance_stays_symmetric_and_psd() {
        let mut ekf = ErrorStateKalmanFilter::new(
            identity_state(),
            Array2::<f64>::eye(15) * 0.01,
            RocketState::process_noise(0.01),
            RocketState,
        );
        let imu = [0.2, -0.1, 9.85, 0.05, -0.02, 0.03];
        let r = Array2::<f64>::eye(3) * 1.0;
        let z = array![0.0, 0.0, 0.0];

        for k in 0..500 {
            ekf.predict(&imu, 0.01);
            if k % 50 == 0 {
                ekf.update(&z, &r);
            }
        }

        let p = &ekf.error_covariance;

        // Positive semi-definiteness of the symmetric part (the physically
        // meaningful validity check for a covariance).
        let data: Vec<f64> = p.iter().copied().collect();
        let m = nalgebra::DMatrix::from_row_slice(15, 15, &data);
        let sym = (&m + m.transpose()) * 0.5;
        let eig = nalgebra::SymmetricEigen::new(sym);
        let min_eig = eig.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min_eig > -1e-6, "P is not PSD: min eigenvalue = {min_eig}");

        // Documented limitation: the (I-KH)P update is not Joseph-stabilized, so P
        // is only approximately symmetric. This bounds how bad that is in practice.
        let mut max_asym = 0.0f64;
        for i in 0..15 {
            for j in 0..15 {
                max_asym = max_asym.max((p[[i, j]] - p[[j, i]]).abs());
            }
        }
        assert!(max_asym < 1e-2, "P asymmetry unexpectedly large: {max_asym}");
    }

    #[test]
    fn measurement_update_does_not_increase_uncertainty() {
        let mut ekf = ErrorStateKalmanFilter::new(
            identity_state(),
            Array2::<f64>::eye(15) * 0.5,
            Array2::<f64>::eye(15) * 1e-4,
            RocketState,
        );
        let imu = [0.0, 0.0, 9.81, 0.0, 0.0, 0.0];
        ekf.predict(&imu, 0.01);
        let trace_before: f64 = (0..15).map(|i| ekf.error_covariance[[i, i]]).sum();
        ekf.update(&array![0.0, 0.0, 0.0], &(Array2::<f64>::eye(3) * 1.0));
        let trace_after: f64 = (0..15).map(|i| ekf.error_covariance[[i, i]]).sum();
        assert!(
            trace_after <= trace_before + 1e-9,
            "covariance trace grew after a measurement update: {trace_before} -> {trace_after}"
        );
    }
}