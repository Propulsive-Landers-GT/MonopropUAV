use ndarray::{Array1, Array2};
extern crate MPC as mpc_crate;
use super::Controller;

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
        let dt = 0.02;
        
        let q_vec = vec![
            10.0, 10.0, 80.0,
            220.0, 220.0, 220.0, 220.0,
            30.0, 30.0, 10.0,
            10.0, 10.0, 10.0
        ];
        let q = Array2::<f64>::from_diag(&Array1::from(q_vec));
        
        let r = Array2::<f64>::from_diag(&Array1::from(vec![5.0, 5.0, 0.05]));
        
        let qn_vec = vec![
            10.0, 10.0, 80.0,
            300.0, 300.0, 300.0, 300.0,
            40.0, 40.0, 20.0,
            5.0, 5.0, 5.0
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
            panoc_cache,
        }
    }
    
    pub fn update(&mut self, current_state: &Array1<f64>, reference_trajectory: &Vec<Array1<f64>>, 
                  uref_trajectory: &Vec<Array1<f64>>, warm_start: &Vec<Array1<f64>>, mass: f64) -> Result<Vec<Array1<f64>>, String> {
        self.mass = mass;
        
        let mut u_warm = warm_start.clone();
        for i in 0..self.n_steps.min(uref_trajectory.len()) {
            if u_warm[i][2].abs() < 1e-3 || u_warm[i][2] == self.mass * 9.81 {
                u_warm[i] = uref_trajectory[i].clone();
            }
        }
        
        let smoothing_weight = Array1::from(vec![1500.0, 1500.0, 0.02]);
        let mut moi = Array2::<f64>::zeros((3,3));
        moi[(0,0)] = 0.5;
        moi[(1,1)] = 0.5;
        moi[(2,2)] = 0.8;
        
        let gimbal_limit = 15.0_f64.to_radians();
        let thrust_min = 300.0;
        let thrust_max = self.max_thrust;
        
        let (_u_apply, u_warm_seq) = mpc_crate::OpEnSolve(
            current_state,
            &u_warm,
            reference_trajectory,
            &self.q,
            &self.r,
            &self.qn,
            &smoothing_weight,
            &mut self.panoc_cache,
            self.mass,
            &moi,
            thrust_min,
            thrust_max,
            gimbal_limit,
            self.dt,
        );
        
        let mut u_opt_vec = Vec::new();
        for row in u_warm_seq.axis_iter(ndarray::Axis(0)) {
            u_opt_vec.push(row.to_owned());
        }
        
        Ok(u_opt_vec)
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
}
