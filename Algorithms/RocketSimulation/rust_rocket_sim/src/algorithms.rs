#[path="../../../MPC/src/mpc_crate.rs"]
pub mod mpc_crate;
#[path="../../../LosslessConvexification/rust_lossless/src/lossless.rs"]
pub mod lossless;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use clarabel::algebra::*;
use clarabel::solver::*;
use std::time::Instant;
use crate::mpc_crate::MPCDebugInfo;
use crate::device_sim::{RefreshUpdater, noise_1_d};
use crate::lossless::*;

#[derive(Debug, Clone)]
pub struct MPC {
    pub n: usize, // state dimension [x, y, z, qx, qy, qz, qw, x_dot, y_dot, z_dot, wx, wy, wz]
    pub m: usize, // control dimension [gimbal_theta, gimbal_phi, thrust]
    pub n_steps: usize, // MPC horizon steps
    pub dt: f64, // time step
    pub integral_gains: (f64, f64, f64), // integral gains for x, y, z
    pub integral_values: (f64, f64, f64), // integral values for x, y, z
    pub q: Array2<f64>, // state cost matrix
    pub r: Array2<f64>, // control cost matrix
    pub qn: Array2<f64>, // terminal state cost matrix
    pub smoothing_weight: Array1<f64>, // weight for exponential smoothing
    pub panoc_cache_tolerance: f64, // tolerance for OpEn solver
    pub panoc_cache_lbfgs_memory: usize, // memory for L-BFGS in OpEn solver
    pub min_thrust: f64, // minimum thrust
    pub max_thrust: f64, // maximum thrust
    pub gimbal_limit: f64, // maximum gimbal angle in radians
    pub system_time: f64, // internal time tracking for MPC updates
    pub refresh_updater: RefreshUpdater,
    current_control: Vec<Array1<f64>>, // current control input
    cached_control: Vec<Array1<f64>>, // previous control input for smoothing
    pub last_solve_time: f64,
    pub cached_solve_time: f64,
}

impl MPC {
    pub fn new(n: usize, m: usize, n_steps: usize, dt: f64, integral_gains: (f64, f64, f64), q: Array2<f64>, r: Array2<f64>, qn: Array2<f64>, smoothing_weight: Array1<f64>, panoc_cache_tolerance: f64, panoc_cache_lbfgs_memory: usize, min_thrust: f64, max_thrust: f64, gimbal_limit: f64, system_time: f64, refresh_updater: RefreshUpdater) -> Self {
        Self {
            n,
            m,
            n_steps,
            dt,
            integral_gains,
            integral_values: (0.0, 0.0, 0.0),
            q,
            r,
            qn,
            smoothing_weight,
            panoc_cache_tolerance,
            panoc_cache_lbfgs_memory,
            min_thrust,
            max_thrust,
            gimbal_limit,
            system_time: 0.0,
            refresh_updater,
            // update_rate,
            current_control: vec![Array1::zeros(m); n_steps],
            cached_control: vec![Array1::zeros(m); n_steps],
            last_solve_time: 0.0,
            cached_solve_time: 0.0,
        }
    }

    pub fn default() -> Self {
        let n = 13; // [x, y, z, qx, qy, qz, qw, x_dot, y_dot, z_dot, wx, wy, wz]
        let m = 3;  // [gimbal_theta, gimbal_phi, thrust]
        let n_steps = 10;
        let dt = 0.2;
        // let integral_gains = (0.001, 0.001, 0.002);
        let integral_gains = (0.0, 0.0, 0.0);
        let q = Array2::<f64>::from_diag(&Array1::from(vec![
            150.0, 150.0, 200.0,   // position x, y, z
            40000.0, 40000.0, 0.0, 0.0, // quaternion qx, qy, qz, qw
            100.0, 100.0, 6000.0,        // linear velocities x_dot, y_dot, z_dot
            500.0, 500.0, 500.0          // angular velocities wx, wy, wz
        ]));
        let r = Array2::<f64>::from_diag(&Array1::from(vec![50.0, 50.0, 0.005]));
        let qn = Array2::<f64>::from_diag(&Array1::from(vec![
            150.0, 150.0, 200.0,   // position x, y, z
            50000.0, 50000.0, 0.0, 0.0, // quaternion qx, qy, qz, qw
            100.0, 100.0, 80000.0,        // linear velocities x_dot, y_dot, z_dot
            1000.0, 1000.0, 1000.0          // angular velocities wx, wy, wz
        ]));
        // let smoothing_weight = Array1::from(vec![150.0, 150.0, -2.0]);
        let smoothing_weight = Array1::from(vec![0.0, 0.0, 0.0]);
        let panoc_cache_tolerance = 1e-4;
        let panoc_cache_lbfgs_memory = 20;
        let min_thrust = 400.0;
        let max_thrust = 1000.0;
        let gimbal_limit = 15_f64.to_radians();
        let system_time = -1.0;
        // let update_rate = 50.0;
        // let refresh_updater = RefreshUpdater::new(0.005, 0.001, 0.00025, 0.00025);
        let refresh_updater = RefreshUpdater::new(0.01, 0.0, 0.0, 0.0);

        Self::new(n, m, n_steps, dt, integral_gains, q, r, qn, smoothing_weight, panoc_cache_tolerance, panoc_cache_lbfgs_memory, min_thrust, max_thrust, gimbal_limit, system_time, refresh_updater)
    }

    pub fn update(&mut self, x0: &Array1<f64>, xref_traj: &Vec<Array1<f64>>, u_warm: &Vec<Array1<f64>>, mass: f64, moi: &Array2<f64>, system_time: f64) -> Vec<Array1<f64>> {
        // basically, if the refresh updater allows us to update, we update previous control and calculate a new wait time with the number of itertions. otherwise, we return the previous control.

        if self.refresh_updater.update(system_time) {
            // rerun mpc, find iterations, run iter_update() on refresh_updater
            let (control_sequence, mpc_debug_info) = self.solve(x0, xref_traj, u_warm, mass, moi);
            self.current_control = self.cached_control.clone();
            self.last_solve_time = self.cached_solve_time;
            self.cached_control = control_sequence.clone();
            self.cached_solve_time = system_time;
            self.refresh_updater.reset(mpc_debug_info.iterations as f64, system_time);
            // println!("MPC updated at system time {:.2} s with {} iterations", system_time, mpc_debug_info.iterations);
            // std::process::exit(1);
            return self.current_control.clone();
        } else {
            // Not time to update yet, return previous control sequence
            return self.current_control.clone();
        }
    }

    pub fn solve(&mut self, x0: &Array1<f64>, xref_traj: &Vec<Array1<f64>>, u_warm: &Vec<Array1<f64>>, mass: f64, moi: &Array2<f64>) -> (Vec<Array1<f64>>, MPCDebugInfo) {
        let mut x = x0.clone(); // initial state
        let mut xref_traj = xref_traj.clone(); // reference trajectory
        let mut u_warm = u_warm.clone(); // warm start control sequence
        let mut u_prev: Array1<f64> = u_warm[0].clone(); // previous control input for smoothing
        let mut error_integral: (f64, f64, f64) = (0.0, 0.0, 0.0); // integral of errors for x, y, z
        let mut panoc_cache = optimization_engine::panoc::PANOCCache::new(self.m * self.n_steps, self.panoc_cache_tolerance, self.panoc_cache_lbfgs_memory);

        let mut control_sequence: Vec<Array1<f64>> = Vec::new();
        let mut mpc_debug_info = MPCDebugInfo::new();

        for k in 0..self.n_steps {
            let mut xref = xref_traj[k].clone();

            self.integral_values.0 += self.integral_gains.0 * (xref[0] - x[0]) * self.dt;
            self.integral_values.1 += self.integral_gains.1 * (xref[1] - x[1]) * self.dt;
            self.integral_values.2 += self.integral_gains.2 * (xref[2] - x[2]) * self.dt;

            xref[0] += self.integral_values.0; // modify x reference with integral term
            xref[1] += self.integral_values.1; // modify y reference with integral term
            xref[2] += self.integral_values.2; // modify z reference with integral term

            let mut xref_traj_k = xref_traj.clone();
            for i in k..self.n_steps {
                xref_traj_k[i][0] += self.integral_values.0; // modify future x references with integral term
                xref_traj_k[i][1] += self.integral_values.1; // modify future y references with integral term
                xref_traj_k[i][2] += self.integral_values.2; // modify future z references with integral term
            }

            // println!("Integral Values: {:?}", self.integral_values);

            // Solve MPC to get optimal control sequence
            // solve using OpEn
            let (mut u_apply, u_warm, debug_info) = mpc_crate::OpEnSolve(&x, &u_warm, &xref_traj_k, &self.q, &self.r, &self.qn, &self.smoothing_weight, &mut panoc_cache, mass, moi, self.min_thrust, self.max_thrust, self.gimbal_limit, self.dt);
            mpc_debug_info = debug_info;

            // exponential filter
            if k >= 1 {
                let alpha = 0.4; // smoothing factor
                u_apply = alpha * &u_apply + (1.0 - alpha) * &u_prev;
            }
            u_prev = u_apply.clone();

            control_sequence.push(u_apply.clone());
        }

        return (control_sequence, mpc_debug_info);
    }
}

#[derive(Debug, Clone)]
pub struct LosslessRefreshUpdater {
    pub coarse_overhead: f64, // overhead on startup of process, in seconds
    pub coarse_overhead_sigma: f64, // standard deviation of overhead, in seconds
    pub coarse_iteration_time: f64, // time each iteration takes, in seconds
    pub coarse_iteration_sigma: f64, // standard deviation of iteration time, in seconds
    pub fine_overhead: f64, // overhead on startup of process, in seconds
    pub fine_overhead_sigma: f64, // standard deviation of overhead, in seconds
    pub fine_iteration_time: f64, // time each iteration takes, in seconds
    pub fine_iteration_sigma: f64, // standard deviation of iteration time, in seconds
    pub fine_iterations: f64,
    pub coarse_iterations: f64,
    pub start_time: f64, // system time when the last solve started
    current_overhead: f64,
    current_total_iteration_time: f64,
}

impl LosslessRefreshUpdater {
    pub fn new(coarse_overhead: f64, coarse_overhead_sigma: f64, coarse_iteration_time: f64, coarse_iteration_sigma: f64, fine_overhead: f64, fine_overhead_sigma: f64, fine_iteration_time: f64, fine_iteration_sigma: f64) -> Self {
        let mut lossless_refresh_updater = Self {
            coarse_overhead,
            coarse_overhead_sigma,
            coarse_iteration_time,
            coarse_iteration_sigma,
            fine_overhead,
            fine_overhead_sigma,
            fine_iteration_time,
            fine_iteration_sigma,
            coarse_iterations: 1.0, // initial guess
            fine_iterations: 1.0, // initial guess
            start_time: 0.0,
            current_overhead: coarse_overhead + fine_overhead, // initial guess
            current_total_iteration_time: coarse_iteration_time + fine_iteration_time, // initial guess
        };

        lossless_refresh_updater.reset(1.0, 1.0, 0.0);

        lossless_refresh_updater
    }

    pub fn default() -> Self {
        // Self::new(
        //     0.25, 0.05, 0.1, 0.01, // coarse solve: overhead 0.25s ±0.1s, iteration time 0.1s ±0.01s
        //     0.25, 0.05, 0.15, 0.02, // fine solve: overhead 0.25s ±0.3s, iteration time 0.15s ±0.02s
        // )
        Self::new(
            1.2, 0.0, 0.0, 0.0, // coarse solve: overhead 0.25s ±0.1s, iteration time 0.1s ±0.01s
            0.0, 0.0, 0.0, 0.0, // fine solve: overhead 0.25s ±0.3s, iteration time 0.15s ±0.02s
        )
    }

    pub fn reset(&mut self, coarse_iterations: f64, fine_iterations: f64, system_time: f64) {
        self.coarse_iterations = coarse_iterations;
        self.fine_iterations = fine_iterations;
        self.start_time = system_time;

        let mut rng = rand::rng();
        self.current_overhead = self.coarse_overhead + self.fine_overhead +
            noise_1_d(
            (self.coarse_overhead_sigma.powi(2) + self.fine_overhead_sigma.powi(2)).sqrt(), 
            &mut rng
        );

        let coarse_var_total = self.coarse_iteration_sigma.powi(2) * (self.coarse_iterations as f64);
        let fine_var_total = self.fine_iteration_sigma.powi(2) * (self.fine_iterations as f64);
        self.current_total_iteration_time = (self.coarse_iteration_time * self.coarse_iterations) + (self.fine_iteration_time * self.fine_iterations) + noise_1_d((coarse_var_total + fine_var_total).sqrt(), &mut rng);    }

    pub fn update(&mut self, system_time: f64) -> bool {
        if system_time - self.start_time > self.current_overhead + self.current_total_iteration_time {
            return true;
        } else {
            return false;
        }
    }
}

pub struct LosslessDebugInfo {
    pub coarse_iterations: u32,
    pub fine_iterations: u32,
}

#[derive(Debug, Clone)]
pub struct Lossless {
    pub max_velocity: f64,
    pub dry_mass: f64,
    pub alpha: f64,
    pub lower_thrust_bound: f64,
    pub upper_thrust_bound: f64,
    pub tvc_range_rad: f64,
    pub coarse_delta_t: f64,
    pub fine_delta_t: f64,
    pub glide_slope: f64,
    pub use_glide_slope: bool,
    pub flip_glide_slope: bool,
    pub system_time: f64,
    // pub update_rate: f64,
    pub lossless_refresh_updater: LosslessRefreshUpdater,
    pub current_traj: TrajectoryResult, // Store the current trajectory result
    pub cached_traj: TrajectoryResult, // Store the previous trajectory result for smoothing
    pub last_solve_time: f64,
    pub cached_solve_time: f64,
}

impl Lossless {
    pub fn new(max_velocity: f64, dry_mass: f64, alpha: f64, lower_thrust_bound: f64, upper_thrust_bound: f64, tvc_range_rad: f64, coarse_delta_t: f64, fine_delta_t: f64, glide_slope: f64, use_glide_slope: bool, flip_glide_slope: bool, pointing_direction: [f64; 3], system_time: f64, lossless_refresh_updater: LosslessRefreshUpdater) -> Self {
        Self {
            max_velocity,
            dry_mass,
            alpha,
            lower_thrust_bound,
            upper_thrust_bound,
            tvc_range_rad,
            coarse_delta_t,
            fine_delta_t,
            glide_slope,
            use_glide_slope,
            flip_glide_slope,
            system_time,
            // update_rate,
            lossless_refresh_updater,
            current_traj: TrajectoryResult {
                positions: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                velocities: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                masses: vec![dry_mass; 20], // Placeholder for 20 steps
                thrusts: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                sigmas: vec![0.0; 20], // Placeholder for 20 steps
                time_of_flight_s: 0.0, // Placeholder for time of flight
            },
            cached_traj: TrajectoryResult {
                positions: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                velocities: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                masses: vec![dry_mass; 20], // Placeholder for 20 steps
                thrusts: vec![[0.0; 3]; 20], // Placeholder for 20 steps
                sigmas: vec![0.0; 20], // Placeholder for 20 steps
                time_of_flight_s: 0.0, // Placeholder for time of flight
            },
            last_solve_time: 0.0,
            cached_solve_time: 0.0,
        }
    }

    pub fn default() -> Self {
        let max_velocity = 5.0;
        let dry_mass = 61.0;
        let alpha = 1.0 / (9.81 * 180.0);
        let lower_thrust_bound = 400.0;
        let upper_thrust_bound = 900.0;
        let tvc_range_rad = 15_f64.to_radians();
        let coarse_delta_t = 0.25;
        let fine_delta_t = 0.1;
        let glide_slope = 0.05_f64.to_radians();
        let use_glide_slope = true;
        let flip_glide_slope = true;
        let system_time = -1.0;
        // let update_rate = 3.0;
        let lossless_refresh_updater = LosslessRefreshUpdater::default();

        Self::new(max_velocity, dry_mass, alpha, lower_thrust_bound, upper_thrust_bound, tvc_range_rad, coarse_delta_t, fine_delta_t, glide_slope, use_glide_slope, flip_glide_slope, [0.0; 3], system_time, lossless_refresh_updater)
    }

    pub fn update(&mut self, current_position: [f64; 3], current_velocity: [f64; 3], target_position: [f64; 3], propellant_mass: f64, system_time: f64) -> TrajectoryResult {
        // let elapsed_time = system_time - self.system_time;
        // if elapsed_time < 1.0 / self.update_rate {
        //     // Not time to update yet, return previous control input (could be stored in the struct if needed)
        //     return self.last_solution.clone(); // Return the last solution if not time to update
        // } else {
        //     self.system_time = system_time;
        //     self.last_solve_time = system_time;

        //     // Call the lossless convexification solver to get the optimal control input
        //     let trajectory = self.solve(current_position, current_velocity, target_position, propellant_mass);
        //     self.last_solution = trajectory;
        //     return self.last_solution.clone(); // Return the new solution after updating
        // }

        
        if self.lossless_refresh_updater.update(system_time) {
            // rerun mpc, find iterations, run iter_update() on refresh_updater
            let (trajectory, debug_info) = self.solve(current_position, current_velocity, target_position, propellant_mass);
            self.current_traj = self.cached_traj.clone();
            self.last_solve_time = self.cached_solve_time;
            self.cached_traj = trajectory.clone();
            self.cached_solve_time = system_time;
            self.lossless_refresh_updater.reset(debug_info.coarse_iterations as f64, debug_info.fine_iterations as f64, system_time);
            return self.current_traj.clone();
        } else {
            // Not time to update yet, return previous control sequence
            return self.current_traj.clone();
        }
    }

    pub fn solve(&mut self, current_position: [f64; 3], current_velocity: [f64; 3], target_position: [f64; 3], propellant_mass: f64) -> (TrajectoryResult, LosslessDebugInfo) {
        let mut solver = lossless::LosslessSolver {
            landing_point: target_position,
            initial_position: current_position,
            initial_velocity: current_velocity,
            max_velocity: self.max_velocity,
            dry_mass: self.dry_mass,
            fuel_mass: propellant_mass,
            alpha: self.alpha,
            lower_thrust_bound: self.lower_thrust_bound,
            upper_thrust_bound: self.upper_thrust_bound,
            tvc_range_rad: self.tvc_range_rad,
            coarse_delta_t: self.coarse_delta_t,
            fine_delta_t: self.fine_delta_t,
            use_glide_slope: self.use_glide_slope,
            glide_slope: self.glide_slope,
            N: 20, // will be overridden by the solver based on the problem setup
            ..Default::default()
        };

        let result = solver.solve();
        let mut debug_info = LosslessDebugInfo {
            coarse_iterations: 0,
            fine_iterations: 0,
        };

        if result.trajectory.is_none() {
            return (self.current_traj.clone(), debug_info);
        } else {
            debug_info.coarse_iterations = result.coarse_metrics.iterations;
            debug_info.fine_iterations = result.fine_metrics.iterations;
            return (result.trajectory.expect("Failed to solve trajectory optimization problem"), debug_info);
        }

        // return solver.solve().trajectory.expect("Failed to solve trajectory optimization problem");
    }
}


#[derive(Debug, Clone)]
pub struct RcsController {
    pub kp: f64,
    pub kd: f64,
    pub dead_theta: f64,
    pub dead_omega: f64,
    pub threshold: f64,
    pub system_time: f64,
    pub refresh_updater: RefreshUpdater,
}

#[derive(Debug, Clone)]
pub struct RcsCommand {
    pub firing_positive: bool,
    pub firing_negative: bool,
}


impl RcsController {
    pub fn new(kp: f64, kd: f64, dead_theta: f64, dead_omega: f64, threshold: f64, system_time: f64, refresh_updater: RefreshUpdater) -> Self {
        Self{
            kp,
            kd,
            dead_theta,
            dead_omega,
            threshold,
            system_time,
            refresh_updater,
        }
    }

    pub fn default() -> Self {
        let kp = 59.9687;
        let kd = 11.7118;
        let dead_theta = 0.025;
        let dead_omega = 0.08;
        let threshold = 0.8;
        let system_time = 0.0;
        let refresh_updater = RefreshUpdater::new(0.005, 0.001, 0.00025, 0.00025);

        Self::new(kp, kd, dead_theta, dead_omega, threshold, system_time, refresh_updater)
    }

    pub fn update(&mut self, roll_angle: f64, roll_rate: f64) -> RcsCommand {

        if roll_angle.abs() < self.dead_theta && roll_rate.abs() < self.dead_omega {
            return RcsCommand {
                firing_positive: false,
                firing_negative: false,
            };
        }

        let torque = -self.kp * roll_angle - self.kd * roll_rate;

        if torque > self.threshold {
            RcsCommand { firing_positive: true,  firing_negative: false }
        } else if torque < -self.threshold {
            RcsCommand { firing_positive: false, firing_negative: true  }
        } else {
            RcsCommand { firing_positive: false, firing_negative: false }
        }

    }

}