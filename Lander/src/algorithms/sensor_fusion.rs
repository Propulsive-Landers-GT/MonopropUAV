use ndarray::{Array1, Array2};
use nalgebra::{Vector3, UnitQuaternion, Quaternion};
use rust_ekf::es_ekf::filter::ErrorStateKalmanFilter;
use rust_ekf::models::full_state_esekf::RocketState;
use crate::state::{SensorData, VehicleState};
use super::SensorFusionEstimator;

pub struct SensorFusion {
    pub ekf_es: ErrorStateKalmanFilter<RocketState>,
    last_update: Option<f64>,
}

impl SensorFusion {
    pub fn new() -> Self {
        // Nominal state: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, abx, aby, abz, wbx, wby, wbz] (16D)
        let initial_nominal = Array1::from(vec![
            0.0, 0.0, 0.0, // pos
            0.0, 0.0, 0.0, // vel
            1.0, 0.0, 0.0, 0.0, // quat (w, x, y, z)
            0.0, 0.0, 0.0, // ab
            0.0, 0.0, 0.0, // wb
        ]);
        
        let initial_p = RocketState::initial_covariance();
        
        let mut q = Array2::eye(15);
        for i in 0..3 { q[(i, i)] = 0.01; } // pos
        for i in 3..6 { q[(i, i)] = 0.05; } // vel
        for i in 6..9 { q[(i, i)] = 0.01; } // att
        for i in 9..12 { q[(i, i)] = 0.001; } // ab
        for i in 12..15 { q[(i, i)] = 0.001; } // wb
        q = q * 0.1;
        
        let ekf_es = ErrorStateKalmanFilter::new(
            initial_nominal,
            initial_p,
            q,
            RocketState,
        );

        Self {
            ekf_es,
            last_update: None,
        }
    }
    
    pub fn update(&mut self, sensor_data: &SensorData, dt: f64, _in_prelaunch: bool) -> Option<VehicleState> {
        let now = sensor_data.timestamp;
        if let Some(last) = self.last_update {
            if now - last < dt - 1e-6 {
                return None;
            }
        }
        self.last_update = Some(now);

        let gyro = sensor_data.imu_data.as_ref().map(|d| d.gyro).unwrap_or([0.0, 0.0, 0.0]);
        let accel = sensor_data.imu_data.as_ref().map(|d| d.accel).unwrap_or([0.0, 0.0, 9.81]);
        
        // 1. Prediction step using high-frequency IMU
        let imu_input = [accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]];
        self.ekf_es.predict(&imu_input, dt);

        // 2. Correction step if GPS/UWB is available
        let (pos, pos_noise) = match (&sensor_data.uwb_data, &sensor_data.gps_data) {
            (Some(uwb), _) => {
                (Some([uwb[0], uwb[1], uwb[2]]), 0.01)
            }
            (None, Some(gps)) => {
                (Some([gps[0], gps[1], gps[2]]), 0.2)
            }
            (None, None) => {
                (None, 1e10)
            }
        };

        if let Some(pos_val) = pos {
            let measurement = Array1::from(vec![pos_val[0], pos_val[1], pos_val[2]]);
            let r_matrix = Array2::eye(3) * pos_noise;
            self.ekf_es.update(&measurement, &r_matrix);
        }

        // 3. Magnetometer update step to resolve yaw/attitude observability
        if let Some(imu) = &sensor_data.imu_data {
            let measurement = Array1::from(vec![imu.mag[0], imu.mag[1], imu.mag[2]]);
            let mag_world = RocketState::mag_world();
            let prediction = RocketState::mag_prediction(&self.ekf_es.nominal_state, &mag_world);
            let h = RocketState::mag_jacobian(&self.ekf_es.nominal_state, &mag_world);
            let r_mag = Array2::eye(3) * (14.0e-9_f64).powi(2);
            self.ekf_es.update_with(&measurement, &prediction, &h, &r_mag);
        }

        // 3. Extract and construct VehicleState
        let state_vec = &self.ekf_es.nominal_state;
        
        let position = Vector3::new(state_vec[0], state_vec[1], state_vec[2]);
        let velocity = Vector3::new(state_vec[3], state_vec[4], state_vec[5]);
        let attitude = UnitQuaternion::new_normalize(Quaternion::new(state_vec[6], state_vec[7], state_vec[8], state_vec[9]));
        let wb = [state_vec[13], state_vec[14], state_vec[15]];
        let angular_velocity = Vector3::new(gyro[0] - wb[0], gyro[1] - wb[1], gyro[2] - wb[2]);

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
        Some(self.ekf_es.nominal_state.clone())
    }
}
