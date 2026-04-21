use clarabel::algebra::*;
use clarabel::solver::*;
use std::time::Instant;

const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];
// Prevents 1 / tan(glide_slope) from blowing up when the configured glide slope is
// extremely small. This keeps the SOC rows numerically well-behaved.
const GLIDE_TAN_FLOOR: f64 = 1.0e-6;
const GROUND_Z_EPSILON: f64 = 1.0e-9;

pub struct LosslessSolver {
    pub landing_point: [f64; 3],
    pub initial_position: [f64; 3],
    pub initial_velocity: [f64; 3],
    pub glide_slope: f64,
    pub use_initial_glide_slope: bool,
    pub use_final_glide_slope: bool,
    pub use_terminal_lateral_hard_tube: bool,
    pub terminal_lateral_hard_tube_time_s: f64,
    pub terminal_lateral_hard_tube_radius_m: f64,
    pub max_velocity: f64,
    pub dry_mass: f64,
    pub fuel_mass: f64,
    pub alpha: f64,
    pub lower_thrust_bound: f64,
    pub upper_thrust_bound: f64,
    pub tvc_range_rad: f64,
    pub coarse_line_search_delta_t: f64,
    pub fine_line_search_delta_t: f64,
    pub coarse_delta_t: f64,
    pub fine_delta_t: f64,
    pub delta_t: f64,
    pub pointing_direction: [f64; 3],
    pub N: i64,
    pub start_time: Instant,
    pub timeout: f64,
}

#[derive(Debug, Clone)]
pub struct TrajectoryResult {
    pub positions: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub masses: Vec<f64>,
    pub thrusts: Vec<[f64; 3]>,
    pub sigmas: Vec<f64>,
    pub time_of_flight_s: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SolveMetrics {
    pub attempts: u32,
    pub successes: u32,
    pub iterations: u32,
    pub clarabel_solve_time_s: f64,
    pub wall_time_s: f64,
    pub last_status: SolverStatus,
}

impl Default for SolveMetrics {
    fn default() -> Self {
        Self {
            attempts: 0,
            successes: 0,
            iterations: 0,
            clarabel_solve_time_s: 0.0,
            wall_time_s: 0.0,
            last_status: SolverStatus::Unsolved,
        }
    }
}

impl SolveMetrics {
    pub fn from_single_attempt(
        status: SolverStatus,
        iterations: u32,
        clarabel_solve_time_s: f64,
        wall_time_s: f64,
    ) -> Self {
        Self {
            attempts: 1,
            successes: if status == SolverStatus::Solved || status == SolverStatus::AlmostSolved {
                1
            } else {
                0
            },
            iterations,
            clarabel_solve_time_s,
            wall_time_s,
            last_status: status,
        }
    }

    pub fn accumulate(&mut self, other: &SolveMetrics) {
        self.attempts += other.attempts;
        self.successes += other.successes;
        self.iterations += other.iterations;
        self.clarabel_solve_time_s += other.clarabel_solve_time_s;
        self.wall_time_s += other.wall_time_s;
        self.last_status = other.last_status;
    }
}

pub struct SolveAttemptResult {
    pub trajectory: Option<TrajectoryResult>,
    pub metrics: SolveMetrics,
}

pub struct SolveRunResult {
    pub trajectory: Option<TrajectoryResult>,
    pub coarse_metrics: SolveMetrics,
    pub fine_metrics: SolveMetrics,
    pub total_metrics: SolveMetrics,
    pub coarse_time_of_flight_s: Option<f64>,
    pub fine_time_of_flight_s: Option<f64>,
}

impl Default for LosslessSolver {
    fn default() -> Self {
        LosslessSolver {
            landing_point: [0.0, 0.0, 0.0],
            initial_position: [0.0, 0.0, 0.0],
            initial_velocity: [0.0, 0.0, 0.0],
            glide_slope: 0.0,
            use_initial_glide_slope: false,
            use_final_glide_slope: false,
            use_terminal_lateral_hard_tube: false,
            terminal_lateral_hard_tube_time_s: 0.0,
            terminal_lateral_hard_tube_radius_m: 0.0,
            max_velocity: 100.0,
            dry_mass: 0.0,
            fuel_mass: 0.0,
            alpha: 0.0,
            lower_thrust_bound: 0.0,
            upper_thrust_bound: 0.0,
            tvc_range_rad: (15.0_f64).to_radians(),
            coarse_line_search_delta_t: 1.0,
            fine_line_search_delta_t: 1.0,
            coarse_delta_t: 1.0,
            fine_delta_t: 1.0,
            delta_t: 1.0,
            pointing_direction: [0.0, 0.0, 1.0],
            N: 1,
            start_time: Instant::now(),
            timeout: 3.0,
        }
    }
}

impl LosslessSolver {
    pub fn new() -> Self {
        LosslessSolver::default()
    }

    pub fn solve_at_current_time(&mut self) -> SolveAttemptResult {
        let attempt_wall_start = Instant::now();
        let m0 = self.dry_mass + self.fuel_mass;
        let use_terminal_lateral_hard_tube = self.use_terminal_lateral_hard_tube
            && self.terminal_lateral_hard_tube_time_s > 0.0
            && self.terminal_lateral_hard_tube_radius_m >= 0.0;
        let terminal_hard_tube_steps = if use_terminal_lateral_hard_tube {
            ((self.terminal_lateral_hard_tube_time_s / self.delta_t).ceil() as i64)
                .clamp(0, self.N)
        } else {
            0
        };
        // Apply the tube near whichever endpoint is on the ground.
        let initial_is_grounded = self.initial_position[2] <= GROUND_Z_EPSILON;
        let landing_is_grounded = self.landing_point[2] <= GROUND_Z_EPSILON;
        let terminal_hard_tube_start = (self.N - terminal_hard_tube_steps).max(0);

        let n_vars = 3 * (self.N + 1) // x
            + 3 * (self.N + 1) // v
            + (self.N + 1) // w
            + 3 * self.N // u
            + self.N; // sigma (thrust slack variable)

        let idx_x = 0;
        let idx_v = idx_x + 3 * (self.N + 1);
        let idx_w = idx_v + 3 * (self.N + 1);
        let idx_u = idx_w + (self.N + 1);
        let idx_sigma = idx_u + 3 * self.N;

        /*
        Equality constraints (Ax = b)
        */

        let mut cones = Vec::new();
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals = Vec::new();
        let mut b = Vec::new();
        let mut row_counter = 0;

        // Setting initial conditions

        // x0 = initial_position
        for i in 0..3 {
            rows.push((row_counter + i) as usize);
            cols.push((idx_x + i) as usize);
            vals.push(1.0);
            b.push(self.initial_position[i as usize]);
        }
        row_counter += 3;
        cones.push(SupportedConeT::ZeroConeT(3));

        // v0 = initial_velocity
        for i in 0..3 {
            rows.push((row_counter + i) as usize);
            cols.push((idx_v + i) as usize); // v0 indices
            vals.push(1.0);
            b.push(self.initial_velocity[i as usize]);
        }
        row_counter += 3;
        cones.push(SupportedConeT::ZeroConeT(3));

        // w0 = ln(m0)
        rows.push(row_counter as usize);
        cols.push(idx_w as usize);
        vals.push(1.0);
        b.push(m0.ln());
        row_counter += 1;
        cones.push(SupportedConeT::ZeroConeT(1));

        // vehicle kinematics constraints
        for k in 0..self.N {
            let curr_x_offset = idx_x + 3 * k;
            let next_x_offset = idx_x + 3 * (k + 1);
            let curr_v_offset = idx_v + 3 * k;
            let next_v_offset = idx_v + 3 * (k + 1);
            let curr_w_offset = idx_w + k;
            let next_w_offset = idx_w + (k + 1);
            let curr_u_offset = idx_u + 3 * k;
            let curr_sigma_offset = idx_sigma + k;

            // x_{k+1} - x_k - dt * v_k = 0
            for i in 0..3 {
                rows.push((row_counter + i) as usize);
                cols.push((next_x_offset + i) as usize);
                vals.push(1.0);

                rows.push((row_counter + i) as usize);
                cols.push((curr_x_offset + i) as usize);
                vals.push(-1.0);

                rows.push((row_counter + i) as usize);
                cols.push((curr_v_offset + i) as usize);
                vals.push(-self.delta_t);

                b.push(0.0);
            }
            row_counter += 3;
            cones.push(SupportedConeT::ZeroConeT(3));

            // v_{k+1} - v_k - dt * u_k = dt * g
            for i in 0..3 {
                rows.push((row_counter + i) as usize);
                cols.push((next_v_offset + i) as usize);
                vals.push(1.0);

                rows.push((row_counter + i) as usize);
                cols.push((curr_v_offset + i) as usize);
                vals.push(-1.0);

                rows.push((row_counter + i) as usize);
                cols.push((curr_u_offset + i) as usize);
                vals.push(-self.delta_t);

                // b = dt * g
                b.push(self.delta_t * GRAVITY[i as usize]);
            }
            row_counter += 3;
            cones.push(SupportedConeT::ZeroConeT(3));

            // w_{k+1} - w_k + dt * alpha * sigma_k = 0
            rows.push(row_counter as usize);
            cols.push(next_w_offset as usize);
            vals.push(1.0);

            rows.push(row_counter as usize);
            cols.push(curr_w_offset as usize);
            vals.push(-1.0);

            rows.push(row_counter as usize);
            cols.push(curr_sigma_offset as usize);
            vals.push(self.delta_t * self.alpha);

            b.push(0.0);
            row_counter += 1;
            cones.push(SupportedConeT::ZeroConeT(1));
        }

        // End constraints

        // x_N = landing_point
        for i in 0..3 {
            rows.push(row_counter as usize);
            cols.push((idx_x + 3 * self.N + i) as usize);
            vals.push(1.0);
            b.push(self.landing_point[i as usize]);
            row_counter += 1;
        }
        cones.push(SupportedConeT::ZeroConeT(3));

        // v_N = 0
        for i in 0..3 {
            rows.push(row_counter as usize);
            cols.push((idx_v + 3 * self.N + i) as usize);
            vals.push(1.0);
            b.push(0.0);
            row_counter += 1;
        }
        cones.push(SupportedConeT::ZeroConeT(3));

        /*
        SOCP constraints
        */

        let sin_theta = self.tvc_range_rad.sin();

        for k in 0..self.N {
            let v_offset = idx_v + 3 * k;
            let w_offset = idx_w + k;
            let u_offset = idx_u + 3 * k;
            let sigma_offset = idx_sigma + k;

            // One change I did make is to make the lower bound only a linear Taylor approximation instead of a quadratic one.
            let z_0 = (m0 - self.alpha * self.upper_thrust_bound * self.delta_t * (k as f64)).ln();
            let exp_neg_z_0 = (-z_0).exp();
            let sigma_min_coefficient = self.lower_thrust_bound * exp_neg_z_0;
            let sigma_min_h_val = sigma_min_coefficient * (1.0 + z_0);
            let sigma_max_coefficient = self.upper_thrust_bound * exp_neg_z_0;
            let sigma_max_h_val = sigma_max_coefficient * (1.0 + z_0);

            // thrust bounds
            // min bound
            cones.push(SupportedConeT::NonnegativeConeT(1));
            rows.push(row_counter as usize);
            cols.push(sigma_offset as usize);
            vals.push(-1.0); // sigma_k - sigma_min
            rows.push(row_counter as usize);
            cols.push(w_offset as usize);
            vals.push(-sigma_min_coefficient);
            b.push(-sigma_min_h_val);
            row_counter += 1;

            // max bound
            cones.push(SupportedConeT::NonnegativeConeT(1));
            rows.push(row_counter as usize);
            cols.push(sigma_offset as usize);
            vals.push(1.0); // -sigma_k + sigma_max
            rows.push(row_counter as usize);
            cols.push(w_offset as usize);
            vals.push(sigma_max_coefficient);
            b.push(sigma_max_h_val);
            row_counter += 1;

            // ---------- Thrust magnitude SOC: ||u_k|| <= sigma_k ----------
            rows.push((row_counter as usize) + 0);
            cols.push(sigma_offset as usize);
            vals.push(-1.0); // -sigma_k
            rows.push((row_counter as usize) + 1);
            cols.push(u_offset as usize);
            vals.push(-1.0); // -u0
            rows.push((row_counter as usize) + 2);
            cols.push((u_offset as usize) + 1);
            vals.push(-1.0); // -u1
            rows.push((row_counter as usize) + 3);
            cols.push((u_offset as usize) + 2);
            vals.push(-1.0); // -u2
            cones.push(SupportedConeT::SecondOrderConeT(4));
            b.push(0.0);
            b.push(0.0);
            b.push(0.0);
            b.push(0.0);
            row_counter += 4;

            // ---------- Thrust pointing SOC: ||u_k - sigma_k*d|| <= sigma_k*sin(theta) ----------
            rows.push((row_counter as usize) + 0);
            cols.push(sigma_offset as usize);
            vals.push(-sin_theta); // top of SOC
            rows.push((row_counter as usize) + 1);
            cols.push(u_offset as usize);
            vals.push(1.0);
            rows.push((row_counter as usize) + 1);
            cols.push(sigma_offset as usize);
            vals.push(-self.pointing_direction[0]);
            rows.push((row_counter as usize) + 2);
            cols.push((u_offset as usize) + 1);
            vals.push(1.0);
            rows.push((row_counter as usize) + 2);
            cols.push(sigma_offset as usize);
            vals.push(-self.pointing_direction[1]);
            rows.push((row_counter as usize) + 3);
            cols.push((u_offset as usize) + 2);
            vals.push(1.0);
            rows.push((row_counter as usize) + 3);
            cols.push(sigma_offset as usize);
            vals.push(-self.pointing_direction[2]);
            cones.push(SupportedConeT::SecondOrderConeT(4));
            b.push(0.0);
            b.push(0.0);
            b.push(0.0);
            b.push(0.0);
            row_counter += 4;

            let apply_initial_hard_tube =
                use_terminal_lateral_hard_tube && initial_is_grounded && k < terminal_hard_tube_steps;
            let apply_landing_hard_tube =
                use_terminal_lateral_hard_tube && landing_is_grounded && k >= terminal_hard_tube_start;

            if apply_initial_hard_tube {
                let x_offset = idx_x + 3 * k;

                // ---------- Hard lateral tube near the grounded initial endpoint ----------
                rows.push((row_counter as usize) + 1);
                cols.push((x_offset as usize) + 0);
                vals.push(1.0);
                rows.push((row_counter as usize) + 2);
                cols.push((x_offset as usize) + 1);
                vals.push(1.0);
                cones.push(SupportedConeT::SecondOrderConeT(3));
                b.push(self.terminal_lateral_hard_tube_radius_m);
                b.push(self.initial_position[0]);
                b.push(self.initial_position[1]);
                row_counter += 3;
            }

            if apply_landing_hard_tube {
                let x_offset = idx_x + 3 * k;

                // ---------- Hard lateral tube near the grounded landing endpoint ----------
                rows.push((row_counter as usize) + 1);
                cols.push((x_offset as usize) + 0);
                vals.push(1.0);
                rows.push((row_counter as usize) + 2);
                cols.push((x_offset as usize) + 1);
                vals.push(1.0);
                cones.push(SupportedConeT::SecondOrderConeT(3));
                b.push(self.terminal_lateral_hard_tube_radius_m);
                b.push(self.landing_point[0]);
                b.push(self.landing_point[1]);
                row_counter += 3;
            }

            // ---------- Max velocity SOC: ||v_k|| <= max_velocity ----------
            // rows.push((row_counter as usize) + 0); cols.push(v_offset as usize);     vals.push(0.0); // -v0
            // No need to push a zero value
            rows.push((row_counter as usize) + 1);
            cols.push(v_offset as usize);
            vals.push(-1.0); // -v0
            rows.push((row_counter as usize) + 2);
            cols.push((v_offset as usize) + 1);
            vals.push(-1.0); // -v1
            rows.push((row_counter as usize) + 3);
            cols.push((v_offset as usize) + 2);
            vals.push(-1.0); // -v2
            cones.push(SupportedConeT::SecondOrderConeT(4));
            b.push(self.max_velocity);
            b.push(0.0);
            b.push(0.0);
            b.push(0.0);
            row_counter += 4;
        }

        if self.use_initial_glide_slope || self.use_final_glide_slope {
            // The glide-slope corridor scales allowable lateral displacement with
            // vertical separation from an anchor point. We use tan(gamma) as the
            // slope parameter, with a floor for numerical safety.
            let glide_tan = self.glide_slope.tan();
            let glide_tan_safe = if glide_tan >= 0.0 {
                glide_tan.max(GLIDE_TAN_FLOOR)
            } else {
                glide_tan.min(-GLIDE_TAN_FLOOR)
            };
            for k in 0..=self.N {
                let x_offset = idx_x + 3 * k;

                if self.use_final_glide_slope {
                    // Final-endpoint cone:
                    //   if z_0 >= z_f: ||[x_k - x_f, y_k - y_f]||_2 <= (z_k - z_f) / tan(gamma)
                    //   else         : ||[x_k - x_f, y_k - y_f]||_2 <= (z_f - z_k) / tan(gamma)
                    Self::push_glide_slope_soc(
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        &mut b,
                        &mut cones,
                        &mut row_counter,
                        x_offset,
                        self.landing_point,
                        glide_tan_safe,
                        self.initial_position[2] < self.landing_point[2],
                    );
                }

                if self.use_initial_glide_slope {
                    // Initial-endpoint cone:
                    //   if z_0 >= z_f: ||[x_k - x_0, y_k - y_0]||_2 <= (z_0 - z_k) / tan(gamma)
                    //   else         : ||[x_k - x_0, y_k - y_0]||_2 <= (z_k - z_0) / tan(gamma)
                    Self::push_glide_slope_soc(
                        &mut rows,
                        &mut cols,
                        &mut vals,
                        &mut b,
                        &mut cones,
                        &mut row_counter,
                        x_offset,
                        self.initial_position,
                        glide_tan_safe,
                        self.initial_position[2] >= self.landing_point[2],
                    );
                }
            }
        }

        // ---------- End mass SOC: w_N >= log(m_dry) ----------
        rows.push(row_counter as usize);
        cols.push((idx_w + self.N) as usize);
        vals.push(-1.0);
        cones.push(SupportedConeT::NonnegativeConeT(1));
        b.push(-self.dry_mass.ln());
        row_counter += 1;

        let A = Self::csc_from_triplets(row_counter as usize, n_vars as usize, &rows, &cols, &vals);

        /*
        Objective: minimize sum of sigma_k * delta_t
        */

        let mut c = vec![0.0; n_vars as usize];
        // sigma variables start at idx_sigma
        for k in 0..self.N {
            c[(idx_sigma + k) as usize] = self.delta_t; // weight = delta_t for discretization
        }

        let prow = Vec::new();
        let pcol = Vec::new();
        let pval = Vec::new();
        let P = LosslessSolver::csc_from_triplets(
            n_vars as usize,
            n_vars as usize,
            &prow,
            &pcol,
            &pval,
        );

        let mut settings = DefaultSettings::default();
        settings.verbose = false;

        // Build the solver
        let mut solver = DefaultSolver::new(
            &P,       // P matrix (or zero for linear problem)
            &c,       // cost vector
            &A,       // equality constraint matrix
            &b,       // equality RHS
            &cones,   // vector of SupportedConeT
            settings, // settings
        );

        solver.solve();
        let metrics = SolveMetrics::from_single_attempt(
            solver.solution.status,
            solver.solution.iterations,
            solver.solution.solve_time,
            attempt_wall_start.elapsed().as_secs_f64(),
        );

        if solver.solution.status == SolverStatus::Solved
            || solver.solution.status == SolverStatus::AlmostSolved
        {
            let traj_result = self.extract_result(&solver.solution);
            SolveAttemptResult {
                trajectory: Some(traj_result),
                metrics,
            }
        } else {
            println!(
                "{}",
                format!("Solver failed with status {:?}", solver.solution.status)
            );
            SolveAttemptResult {
                trajectory: None,
                metrics,
            }
        }
    }

    pub fn csc_from_triplets(
        nrows: usize,
        ncols: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
    ) -> CscMatrix<f64> {
        let mut triplets: Vec<(usize, usize, f64)> = rows
            .iter()
            .zip(cols)
            .zip(vals)
            .map(|((r, c), v)| (*r, *c, *v))
            .collect();
        triplets.sort_by_key(|&(r, c, _)| (c, r));

        let mut colptr = vec![0; ncols + 1];
        let mut rowval = Vec::with_capacity(vals.len());
        let mut nzval = Vec::with_capacity(vals.len());

        let mut current_col = 0;
        let mut count = 0;

        for &(r, c, v) in &triplets {
            while current_col < c {
                colptr[current_col + 1] = count;
                current_col += 1;
            }
            rowval.push(r);
            nzval.push(v);
            count += 1;
        }
        while current_col < ncols {
            colptr[current_col + 1] = count;
            current_col += 1;
        }

        CscMatrix::new(nrows, ncols, colptr, rowval, nzval)
    }

    fn push_glide_slope_soc(
        rows: &mut Vec<usize>,
        cols: &mut Vec<usize>,
        vals: &mut Vec<f64>,
        b: &mut Vec<f64>,
        cones: &mut Vec<SupportedConeT<f64>>,
        row_counter: &mut i64,
        x_offset: i64,
        anchor: [f64; 3],
        glide_tan: f64,
        upper_bound: bool,
    ) {
        // This helper encodes one glide-slope bound in SOC form.
        //
        // Desired constraint about anchor a = [a_x, a_y, a_z]:
        //
        //   ||[x_k - a_x, y_k - a_y]||_2 <= (z_k - a_z) / tan(gamma)   lower-bound cone
        //
        // or
        //
        //   ||[x_k - a_x, y_k - a_y]||_2 <= (a_z - z_k) / tan(gamma)   upper-bound cone
        //
        // Clarabel takes SOC blocks as ||u||_2 <= t. We therefore assemble:
        //
        //   u = [x_k - a_x, y_k - a_y]
        //   t = +/- z_k / tan(gamma)  +/- a_z / tan(gamma)
        //
        // with the sign chosen by `upper_bound`.
        let top_coeff = if upper_bound {
            1.0 / glide_tan
        } else {
            -1.0 / glide_tan
        };
        let top_rhs = anchor[2] / glide_tan;
        let top_rhs = if upper_bound { top_rhs } else { -top_rhs };

        rows.push(*row_counter as usize);
        cols.push((x_offset + 2) as usize);
        vals.push(top_coeff);

        rows.push((*row_counter as usize) + 1);
        cols.push(x_offset as usize);
        vals.push(1.0);

        rows.push((*row_counter as usize) + 2);
        cols.push((x_offset + 1) as usize);
        vals.push(1.0);

        cones.push(SupportedConeT::SecondOrderConeT(3));
        b.push(top_rhs);
        b.push(anchor[0]);
        b.push(anchor[1]);
        *row_counter += 3;
    }

    fn extract_result(&self, result: &DefaultSolution<f64>) -> TrajectoryResult {
        let x = &result.x;

        let idx_x = 0;
        let idx_v = idx_x + 3 * (self.N + 1);
        let idx_w = idx_v + 3 * (self.N + 1);
        let idx_u = idx_w + (self.N + 1);
        let idx_sigma = idx_u + 3 * self.N;

        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        let mut masses = Vec::new();
        let mut thrusts = Vec::new();
        let mut sigmas = Vec::new();

        // positions (x_k)
        for k in 0..=self.N {
            let start: usize = (idx_x + 3 * k) as usize;
            positions.push([x[start], x[start + 1], x[start + 2]]);
        }

        // velocities (v_k)
        for k in 0..=self.N {
            let start: usize = (idx_v + 3 * k) as usize;
            velocities.push([x[start], x[start + 1], x[start + 2]]);
        }

        // log-masses (w_k)
        for k in 0..=self.N {
            masses.push(x[(idx_w + k) as usize].exp()); // convert ln(m) → m
        }

        // thrusts (u_k)
        for k in 0..self.N {
            let start: usize = (idx_u + 3 * k) as usize;
            thrusts.push([x[start], x[start + 1], x[start + 2]]);
        }

        // sigmas (σ_k)
        for k in 0..self.N {
            sigmas.push(x[(idx_sigma + k) as usize]);
        }

        TrajectoryResult {
            positions,
            velocities,
            masses,
            thrusts,
            sigmas,
            time_of_flight_s: (self.N as f64) * self.delta_t,
        }
    }

    pub fn solve(&mut self) -> SolveRunResult {
        let solve_wall_start = Instant::now();
        self.start_time = Instant::now();
        self.delta_t = self.coarse_delta_t;

        let vel_norm = self
            .initial_velocity
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let t_min = self.dry_mass * vel_norm / self.upper_thrust_bound;
        let t_max = self.fuel_mass / (self.alpha * self.lower_thrust_bound);

        let coarse_n_min = (t_min / self.delta_t).ceil() as i64;
        let coarse_n_max = (t_max / self.delta_t).floor() as i64;
        let coarse_n_step =
            ((self.coarse_line_search_delta_t / self.delta_t).round() as i64).max(1);

        let mut traj_result: Option<TrajectoryResult> = None;
        let mut coarse_metrics = SolveMetrics::default();
        let mut fine_metrics = SolveMetrics::default();
        let mut coarse_time_of_flight_s: Option<f64> = None;
        let mut fine_time_of_flight_s: Option<f64> = None;

        for k in (coarse_n_min..=coarse_n_max).step_by(coarse_n_step as usize) {
            println!("Solving step {}...", k);
            self.N = k;
            let attempt = self.solve_at_current_time();
            coarse_metrics.accumulate(&attempt.metrics);
            match attempt.trajectory {
                Some(sol) => {
                    println!(
                        "✅ Step {} converged successfully with tof {:.2} s.",
                        k, sol.time_of_flight_s
                    );
                    coarse_time_of_flight_s = Some(sol.time_of_flight_s);
                    traj_result = Some(sol);
                    break;
                }
                None => println!("Failed to converge at step {}.", k),
            }

            if self.start_time.elapsed().as_secs_f64() > self.timeout {
                break;
            }
        }

        if traj_result.is_none() {
            println!("No successful solve found in the given time bounds.");
            let mut total_metrics = coarse_metrics;
            total_metrics.wall_time_s = solve_wall_start.elapsed().as_secs_f64();
            return SolveRunResult {
                trajectory: None,
                coarse_metrics,
                fine_metrics,
                total_metrics,
                coarse_time_of_flight_s,
                fine_time_of_flight_s,
            };
        }

        let fine_start = ((self.N as f64) * self.coarse_delta_t / self.fine_delta_t).ceil() as i64;
        self.delta_t = self.fine_delta_t;
        let fine_n_min = (t_min / self.delta_t).ceil() as i64;
        let fine_n_step = ((self.fine_line_search_delta_t / self.delta_t).round() as i64).max(1);

        let mut k = fine_start;
        while k >= fine_n_min {
            println!("Solving step {}...", k);
            self.N = k;
            let attempt = self.solve_at_current_time();
            match attempt.trajectory {
                Some(sol) => {
                    println!(
                        "✅ Fine solve converged successfully with tof {:.2} s.",
                        sol.time_of_flight_s
                    );
                    fine_time_of_flight_s = Some(sol.time_of_flight_s);
                    traj_result = Some(sol);
                    fine_metrics.accumulate(&attempt.metrics);

                    // comment this line to perform a line search backwards for the "most optimal" (takes longer)
                    break;
                }
                None => {
                    println!("Failed to converge at step {}.", k);
                    break;
                }
            }
            k -= fine_n_step;

            if self.start_time.elapsed().as_secs_f64() > self.timeout {
                break;
            }
        }

        if traj_result.is_none() {
            println!("No successful solve found in the given time bounds.");
            let mut total_metrics = coarse_metrics;
            total_metrics.accumulate(&fine_metrics);
            total_metrics.wall_time_s = solve_wall_start.elapsed().as_secs_f64();
            return SolveRunResult {
                trajectory: None,
                coarse_metrics,
                fine_metrics,
                total_metrics,
                coarse_time_of_flight_s,
                fine_time_of_flight_s,
            };
        }

        let mut total_metrics = coarse_metrics;
        total_metrics.accumulate(&fine_metrics);
        total_metrics.wall_time_s = solve_wall_start.elapsed().as_secs_f64();

        SolveRunResult {
            trajectory: traj_result,
            coarse_metrics,
            fine_metrics,
            total_metrics,
            coarse_time_of_flight_s,
            fine_time_of_flight_s,
        }
    }
}
