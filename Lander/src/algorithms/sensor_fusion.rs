use ndarray::{Array1, Array2, s};
use nalgebra::{Vector3, UnitQuaternion};
use rust_ekf::{ExtendedKalmanFilter, EKFModel};
use crate::state::{SensorData, VehicleState};
use super::SensorFusionEstimator;

// Unified Lander EKF Model
#[allow(dead_code)]
pub struct UnifiedLanderModel {
    pub current_time: f64,
    pub previous_time: f64,
    pub delta_time: f64,
    pub latest_accel: Array1<f64>,
    pub latest_gyro: Array1<f64>,
    gravity_reference: Array1<f64>,
    magnetic_reference: Array1<f64>,
}

impl UnifiedLanderModel {
    pub fn new(delta_time: f64) -> Self {
        Self {
            current_time: -delta_time,
            previous_time: -2.0 * delta_time,
            delta_time,
            latest_accel: Array1::from(vec![0.0, 0.0, 9.81]),
            latest_gyro: Array1::zeros(3),
            gravity_reference: Array1::from(vec![0.0, 0.0, 1.0]),
            magnetic_reference: Array1::from(vec![-0.04, 0.44, -0.89]),
        }
    }

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

    fn euler_to_rotation_matrix(euler: &Array1<f64>) -> Array2<f64> {
        let (phi, theta, psi) = (euler[0], euler[1], euler[2]);
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
}

impl EKFModel for UnifiedLanderModel {
    fn parse_data(&mut self, data: &[f64]) -> Array1<f64> {
        self.latest_gyro = Array1::from(vec![data[0], data[1], data[2]]);
        self.latest_accel = Array1::from(vec![data[3], data[4], data[5]]);

        let ax = data[3]; let ay = data[4]; let az = data[5];
        let a_norm = (ax*ax + ay*ay + az*az).sqrt();
        let (ax_u, ay_u, az_u) = if a_norm > 1e-3 { (ax/a_norm, ay/a_norm, az/a_norm) } else { (0.0, 0.0, 1.0) };

        let mx = data[6]; let my = data[7]; let mz = data[8];
        let m_norm = (mx*mx + my*my + mz*mz).sqrt();
        let (mx_u, my_u, mz_u) = if m_norm > 1e-3 { (mx/m_norm, my/m_norm, mz/m_norm) } else { (0.0, 1.0, 0.0) };

        Array1::from(vec![
            data[9], data[10], data[11],
            mx_u, my_u, mz_u,
            ax_u, ay_u, az_u,
        ])
    }

    fn state_transition_function(&self, state: &Array1<f64>, dt: f64) -> Array1<f64> {
        if !dt.is_finite() || dt <= 0.0 {
            return state.clone();
        }

        let p = state.slice(s![0..3]);
        let v = state.slice(s![3..6]);
        let euler = state.slice(s![6..9]);
        let bg = state.slice(s![9..12]);

        let p_next = p.to_owned() + &v.to_owned() * dt;

        let r_world_to_body = Self::euler_to_rotation_matrix(&euler.to_owned());
        let r_body_to_world = r_world_to_body.t();
        let a_world = r_body_to_world.dot(&self.latest_accel) + Array1::from(vec![0.0, 0.0, -9.81]);
        let v_next = v.to_owned() + &a_world * dt;

        let w_corrected = &self.latest_gyro - &bg;
        let w_corr_arr = [w_corrected[0], w_corrected[1], w_corrected[2]];
        let euler_dot = Self::euler_angle_rates(euler[0], euler[1], &w_corr_arr);

        let roll_next = Self::wrap_angle(euler[0] + dt * euler_dot[0]);
        let pitch_next = Self::safe_pitch(euler[1] + dt * euler_dot[1]);
        let yaw_next = Self::wrap_angle(euler[2] + dt * euler_dot[2]);

        let bg_next = bg.to_owned();

        let mut next_state = Array1::zeros(12);
        next_state.slice_mut(s![0..3]).assign(&p_next);
        next_state.slice_mut(s![3..6]).assign(&v_next);
        next_state.slice_mut(s![6..9]).assign(&ndarray::arr1(&[roll_next, pitch_next, yaw_next]));
        next_state.slice_mut(s![9..12]).assign(&bg_next);

        next_state
    }

    fn state_transition_jacobian(&self, state: &Array1<f64>, dt: f64) -> Array2<f64> {
        let n = state.len();
        let mut F = Array2::zeros((n, n));
        let eps = 1e-6;

        let f0 = self.state_transition_function(state, dt);
        for i in 0..n {
            let mut perturbed = state.clone();
            perturbed[i] += eps;
            let fi = self.state_transition_function(&perturbed, dt);
            let diff = (&fi - &f0) / eps;
            for j in 0..n {
                F[[j, i]] = diff[j];
            }
        }
        F
    }

    fn measurement_prediction_function(&self, state: &Array1<f64>) -> Array1<f64> {
        let p = state.slice(s![0..3]);
        let euler = state.slice(s![6..9]);

        let p_pred = p.to_owned();
        let r_world_to_body = Self::euler_to_rotation_matrix(&euler.to_owned());
        let mag_pred = r_world_to_body.dot(&self.magnetic_reference);
        let accel_pred = r_world_to_body.dot(&self.gravity_reference);

        let mut z = Array1::zeros(9);
        z.slice_mut(s![0..3]).assign(&p_pred);
        z.slice_mut(s![3..6]).assign(&mag_pred);
        z.slice_mut(s![6..9]).assign(&accel_pred);
        z
    }

    fn measurement_prediction_jacobian(&self, state: &Array1<f64>) -> Array2<f64> {
        let n = state.len();
        let m = 9;
        let mut H = Array2::zeros((m, n));
        let eps = 1e-6;

        let h0 = self.measurement_prediction_function(state);
        for i in 0..n {
            let mut perturbed = state.clone();
            perturbed[i] += eps;
            let hi = self.measurement_prediction_function(&perturbed);
            let diff = (&hi - &h0) / eps;
            for j in 0..m {
                H[[j, i]] = diff[j];
            }
        }
        H
    }
}

pub struct SensorFusion {
    pub ekf_unified: ExtendedKalmanFilter<UnifiedLanderModel>,
    last_update: Option<f64>,
}

impl SensorFusion {
    pub fn new() -> Self {
        let initial_state = Array1::from(vec![0.0; 12]);
        
        let mut q = Array2::eye(12);
        for i in 0..3 { q[(i, i)] = 0.01; }
        for i in 3..6 { q[(i, i)] = 0.05; }
        for i in 6..9 { q[(i, i)] = 0.01; }
        for i in 9..12 { q[(i, i)] = 0.001; }
        q = q * 0.1;

        let mut r = Array2::eye(9);
        for i in 0..3 { r[(i, i)] = 0.1; }
        for i in 3..6 { r[(i, i)] = 0.5; }
        for i in 6..9 { r[(i, i)] = 0.1; }
        
        let initial_p = Array2::eye(12) * 1.0;
        
        let ekf_unified = ExtendedKalmanFilter::new(
            initial_state,
            Array1::zeros(9),
            0.002,
            q,
            r,
            initial_p,
            UnifiedLanderModel::new(0.002)
        );

        Self {
            ekf_unified,
            last_update: None,
        }
    }
    
    pub fn update(&mut self, sensor_data: &SensorData, dt: f64, in_prelaunch: bool) -> Option<VehicleState> {
        let now = sensor_data.timestamp;
        if let Some(last) = self.last_update {
            if now - last < dt - 1e-6 {
                return None;
            }
        }
        self.last_update = Some(now);

        let gyro = sensor_data.imu_data.as_ref().map(|d| d.gyro).unwrap_or([0.0, 0.0, 0.0]);
        let accel = sensor_data.imu_data.as_ref().map(|d| d.accel).unwrap_or([0.0, 0.0, 9.81]);
        let mag = sensor_data.imu_data.as_ref().map(|d| d.mag).unwrap_or([-0.04, 0.44, -0.89]);
        
        let (pos, pos_noise) = match (&sensor_data.uwb_data, &sensor_data.gps_data) {
            (Some(uwb), _) => {
                ([uwb[0], uwb[1], uwb[2]], 0.01)
            }
            (None, Some(gps)) => {
                ([gps[0], gps[1], gps[2]], 0.2)
            }
            (None, None) => {
                ([0.0, 0.0, 0.0], 1e10)
            }
        };

        for i in 0..3 {
            self.ekf_unified.measurement_noise_covariance[(i, i)] = pos_noise;
        }
        
        let mag_noise = if sensor_data.imu_data.is_none() { 1e10 } else { 0.5 };
        for i in 3..6 {
            self.ekf_unified.measurement_noise_covariance[(i, i)] = mag_noise;
        }

        let accel_noise = if in_prelaunch && sensor_data.imu_data.is_some() {
            0.1
        } else {
            1e10
        };
        for i in 6..9 {
            self.ekf_unified.measurement_noise_covariance[(i, i)] = accel_noise;
        }

        let ekf_input = vec![
            gyro[0], gyro[1], gyro[2],
            accel[0], accel[1], accel[2],
            mag[0], mag[1], mag[2],
            pos[0], pos[1], pos[2],
        ];

        self.ekf_unified.predict();
        self.ekf_unified.update(&ekf_input);

        let state_vec = self.ekf_unified.get_state();
        
        let position = Vector3::new(state_vec[0], state_vec[1], state_vec[2]);
        let velocity = Vector3::new(state_vec[3], state_vec[4], state_vec[5]);
        let attitude = UnitQuaternion::from_euler_angles(state_vec[6], state_vec[7], state_vec[8]);
        let bg = [state_vec[9], state_vec[10], state_vec[11]];
        let angular_velocity = Vector3::new(gyro[0] - bg[0], gyro[1] - bg[1], gyro[2] - bg[2]);

        Some(VehicleState {
            position,
            velocity,
            attitude,
            angular_velocity,
            mass: 80.0,
            dry_mass: 50.0,
        })
    }
}

impl SensorFusionEstimator for SensorFusion {
    fn update(&mut self, sensor_data: &SensorData, dt: f64, in_prelaunch: bool) -> Option<VehicleState> {
        self.update(sensor_data, dt, in_prelaunch)
    }

    fn get_state_vector(&self) -> Option<Array1<f64>> {
        Some(self.ekf_unified.get_state().clone())
    }
}
