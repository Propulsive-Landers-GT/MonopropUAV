use ndarray::{Array1, Array2};
extern crate MPC as mpc_crate;
use super::Controller;
use crate::state::FlightPhase;

pub struct MPC {
    pub n: usize,
    pub m: usize,
    pub n_steps: usize,
    pub dt: f64,
    pub mass: f64,
    pub min_thrust: f64,
    pub max_thrust: f64,
    pub q: Array2<f64>,
    pub r: Array2<f64>,
    pub qn: Array2<f64>,
    pub smoothing_weight: f64,
    pub system_time: f64,
    pub update_rate: f64,
    pub last_solve_time: f64,
    pub flight_phase: FlightPhase,
    pub reset_warm_start: bool,
    pub panoc_cache: optimization_engine::panoc::PANOCCache,
}

impl Clone for MPC {
    fn clone(&self) -> Self {
        let n_dim_u = self.m * self.n_steps;
        Self {
            n: self.n,
            m: self.m,
            n_steps: self.n_steps,
            dt: self.dt,
            mass: self.mass,
            min_thrust: self.min_thrust,
            max_thrust: self.max_thrust,
            q: self.q.clone(),
            r: self.r.clone(),
            qn: self.qn.clone(),
            smoothing_weight: self.smoothing_weight,
            system_time: self.system_time,
            update_rate: self.update_rate,
            last_solve_time: self.last_solve_time,
            flight_phase: self.flight_phase,
            reset_warm_start: self.reset_warm_start,
            panoc_cache: optimization_engine::panoc::PANOCCache::new(n_dim_u, 1e-4, 20),
        }
    }
}

impl std::fmt::Debug for MPC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MPC")
            .field("n", &self.n)
            .field("m", &self.m)
            .field("n_steps", &self.n_steps)
            .field("dt", &self.dt)
            .field("mass", &self.mass)
            .field("min_thrust", &self.min_thrust)
            .field("max_thrust", &self.max_thrust)
            .finish()
    }
}

impl MPC {
    pub fn new() -> Self {
        let n = 13;
        let m = 3;
        let n_steps = 10;
        let dt = 0.2;
        
        let q_vec = vec![
            20.0, 20.0, 70.0,   // position x, y, z
            6000.0, 6000.0, 6000.0, 0.0, // quaternion qx, qy, qz, qw
            30.0, 30.0, 50.0,        // linear velocities x_dot, y_dot, z_dot
            100.0, 100.0, 100.0          // angular velocities wx, wy, wz
        ];
        let q = Array2::<f64>::from_diag(&Array1::from(q_vec));
        
        let r = Array2::<f64>::from_diag(&Array1::from(vec![50.0, 50.0, 1.0]));
        
        let qn_vec = vec![
            40.0, 40.0, 80.0,   // position x, y, z
            10000.0, 10000.0, 10000.0, 0.0, // quaternion qx, qy, qz, qw
            20.0, 20.0, 100.0,        // linear velocities x_dot, y_dot, z_dot
            50.0, 50.0, 50.0          // angular velocities wx, wy, wz
        ];
        let qn = Array2::<f64>::from_diag(&Array1::from(qn_vec));
        
        let panoc_cache = optimization_engine::panoc::PANOCCache::new(m * n_steps, 1e-4, 20);
        
        Self {
            n,
            m,
            n_steps,
            dt,
            mass: 80.0,
            min_thrust: 300.0,
            max_thrust: 1200.0,
            q,
            r,
            qn,
            smoothing_weight: 0.01,
            system_time: 0.0,
            update_rate: 50.0,
            last_solve_time: 0.0,
            flight_phase: FlightPhase::Standby,
            reset_warm_start: false,
            panoc_cache,
        }
    }
    
    pub fn update(&mut self, current_state: &Array1<f64>, reference_trajectory: &Vec<Array1<f64>>, 
                  uref_trajectory: &Vec<Array1<f64>>, warm_start: &Vec<Array1<f64>>, mass: f64) -> Result<Vec<Array1<f64>>, String> {
        self.mass = mass;
        
        let mut u_warm = warm_start.clone();
        if self.reset_warm_start {
            self.reset_warm_start = false;
            for i in 0..self.n_steps.min(uref_trajectory.len()) {
                u_warm[i] = uref_trajectory[i].clone();
            }
        } else {
            for i in 0..self.n_steps.min(uref_trajectory.len()) {
                if u_warm[i][2].abs() < 1e-3 {
                    u_warm[i] = uref_trajectory[i].clone();
                }
            }
        }
        
        let mut moi = Array2::<f64>::zeros((3,3));
        moi[(0,0)] = 6.65;
        moi[(1,1)] = 6.65;
        moi[(2,2)] = 2.3;
        
        let gimbal_limit = 15.0_f64.to_radians();
        let thrust_min = 300.0;
        let thrust_max = self.max_thrust;

        let u_min = Array1::from(vec![-gimbal_limit, -gimbal_limit, thrust_min]);
        let u_max = Array1::from(vec![gimbal_limit, gimbal_limit, thrust_max]);
        
        let (u_warm_seq, _, u_apply) = mpc_crate::mpc_main(
            current_state,
            &mut u_warm,
            reference_trajectory,
            &self.q,
            &self.r,
            &self.qn,
            &u_min,
            &u_max,
            3,
            2e-6,
            self.mass,
            &moi,
            self.dt,
        );
        
        let mut U_optimal = vec![u_apply];
        U_optimal.extend(u_warm_seq[0..self.n_steps - 1].to_vec());
        
        Ok(U_optimal)
    }
}

impl Controller for MPC {
    fn update(
        &mut self,
        current_state: &Array1<f64>,
        reference_trajectory: &[Array1<f64>],
        uref_trajectory: &[Array1<f64>],
        warm_start: &[Array1<f64>],
        mass: f64,
    ) -> Result<Vec<Array1<f64>>, String> {
        self.update(
            current_state,
            &reference_trajectory.to_vec(),
            &uref_trajectory.to_vec(),
            &warm_start.to_vec(),
            mass,
        )
    }
    
    fn get_horizon_steps(&self) -> usize {
        self.n_steps
    }
    
    fn get_time_step(&self) -> f64 {
        self.dt
    }

    fn set_flight_phase(&mut self, phase: FlightPhase) {
        self.flight_phase = phase;
        self.reset_warm_start = true;
        match phase {
            FlightPhase::Ascent => {
                self.q = Array2::<f64>::from_diag(&Array1::from(vec![
                    150.0, 150.0, 600.0,
                    40000.0, 40000.0, 0.0, 0.0,
                    100.0, 100.0, 2000.0,
                    500.0, 500.0, 500.0
                ]));
                self.r = Array2::<f64>::from_diag(&Array1::from(vec![50.0, 50.0, 0.005]));
                self.qn = Array2::<f64>::from_diag(&Array1::from(vec![
                    150.0, 150.0, 600.0,
                    50000.0, 50000.0, 0.0, 0.0,
                    100.0, 100.0, 200.0,
                    1000.0, 1000.0, 1000.0
                ]));
            }
            FlightPhase::Hover => {
                self.q = Array2::<f64>::from_diag(&Array1::from(vec![
                    200.0, 200.0, 800.0,
                    60000.0, 60000.0, 0.0, 0.0,
                    100.0, 100.0, 250.0,
                    500.0, 500.0, 500.0
                ]));
                self.r = Array2::<f64>::from_diag(&Array1::from(vec![50.0, 50.0, 1.0]));
                self.qn = Array2::<f64>::from_diag(&Array1::from(vec![
                    200.0, 200.0, 1000.0,
                    100000.0, 100000.0, 0.0, 0.0,
                    100.0, 100.0, 300.0,
                    1000.0, 1000.0, 1000.0
                ]));
            }
            FlightPhase::Descent => {
                self.q = Array2::<f64>::from_diag(&Array1::from(vec![
                    300.0, 300.0, 800.0,
                    100000.0, 100000.0, 0.0, 0.0,
                    100.0, 100.0, 6000.0,
                    500.0, 500.0, 500.0
                ]));
                self.r = Array2::<f64>::from_diag(&Array1::from(vec![50.0, 50.0, 0.005]));
                self.qn = Array2::<f64>::from_diag(&Array1::from(vec![
                    150.0, 150.0, 50000.0,
                    120000.0, 120000.0, 0.0, 0.0,
                    100.0, 100.0, 200.0,
                    1000.0, 1000.0, 1000.0
                ]));
            }
            _ => {}
        }
    }
}
