use crate::lossless::*;
use crate::rocket_dynamics::*;
use crate::device_sim::*;
use crate::algorithms::*;
use crate::algorithms::lossless::*;
use crate::sloshing_sim::*;
use crate::fluid_dynamics::*;
use nalgebra::{Matrix3, Vector3, Vector4, UnitQuaternion};
use ndarray::{Array1, Array2};
use rerun::*;

#[derive(Debug)]
pub struct Simulation {
    pub rocket: Rocket,
    pub mpc: MPC,
    pub lossless: Lossless,
    pub debug: bool,
    pub dt: f64,
    pub current_time: f64,
    pub has_exceeded_angle: bool,
    pub min_time: f64,
    pub traj_stage: i32,
    pub traj_timer: f64,
    pub start_state: String,
    pub end_state: String,
    pub end_stage: i32,
    pub rec: Option<RecordingStream>,
}

impl Default for Simulation {
    fn default() -> Self {
        Self {
            rocket: Rocket::default(),
            mpc: MPC::default(),
            lossless: Lossless::default(),
            dt: 0.01,
            current_time: 0.0,
            debug: false,
            has_exceeded_angle: false,
            min_time: 10.0,
            traj_stage: 0,
            traj_timer: 0.0,
            start_state: "ascent".to_string(),
            end_state: "none".to_string(),
            end_stage: 0,
            rec: None,
        }
    }
}

impl Simulation {
    pub fn new(rocket: Rocket, mpc: MPC, lossless: Lossless, dt: f64, current_time: f64, debug: bool, has_exceeded_angle: bool, min_time: f64, traj_stage: i32, traj_timer: f64, start_state: String, end_state: String) -> Self {
        let mut simulation = Self {
            rocket,
            mpc,
            lossless,
            dt,
            current_time,
            debug,
            has_exceeded_angle,
            min_time,
            traj_stage,
            traj_timer,
            start_state,
            end_state,
            end_stage: 0,
            rec: None,
        };

        simulation
    }

    pub fn init(&mut self) {
        if self.debug {
            // use "~/.cargo/bin/rerun --web-viewer" to run rereun viewer if "rerun --web-viewer" doesn't work
            println!("Connecting to Rerun Viewer...");

            // FIX: Use .connect_grpc() as the compiler suggested.
            // This connects to the 'rerun --web-viewer' running in your other terminal.
            self.rec = Some(
                RecordingStreamBuilder::new("rocket_sim")
                    .connect_grpc()
                    .expect("🚨 FATAL: Failed to connect to the Rerun viewer!")
            );

            println!("Connected! Sending data...");

            // --- The Rest of Your Simulation Code ---
            
            // 1. Ground
            let _ = self.rec.as_ref().unwrap().log(
                "world/ground",
                &Boxes3D::from_centers_and_half_sizes(
                    [(0.0, 0.0, -0.05)], // Center: Shift down slightly so y=0 is the top surface
                    [(100.0, 100.0, 0.05)], // Half-sizes: 200x200 wide, 0.1 thick
                )
                .with_colors([Color::from_rgb(40, 40, 40)]) // Dark Grey
                .with_fill_mode(FillMode::Solid), // Make it solid, not wireframe
            );
        }

        self.has_exceeded_angle = false;
        
        match &self.start_state {
            val if *val == "ascent".to_string() => self.traj_stage = 0,
            val if *val == "hover".to_string() => self.traj_stage = 1,
            val if *val == "descent".to_string() => self.traj_stage = 2,
            _ => self.traj_stage = 0,
        }
        match &self.end_state {
            val if *val == "ascent".to_string() => self.end_stage = 0,
            val if *val == "hover".to_string() => self.end_stage = 1,
            val if *val == "descent".to_string() => self.end_stage = 2,
            _ => self.end_stage = -1,
        }

        self.lossless.update([self.rocket.position.x, self.rocket.position.y, self.rocket.position.z], [self.rocket.velocity.x, self.rocket.velocity.y, self.rocket.velocity.z], [0.0, 0.0, 50.0], self.rocket.get_mass() - self.rocket.get_dry_mass(), -20.0);
        self.lossless.update([self.rocket.position.x, self.rocket.position.y, self.rocket.position.z], [self.rocket.velocity.x, self.rocket.velocity.y, self.rocket.velocity.z], [0.0, 0.0, 50.0], self.rocket.get_mass() - self.rocket.get_dry_mass(), -10.0);

        self.mpc.update(&self.rocket.get_state(), &vec![Array1::zeros(13); self.mpc.n_steps + 1], &vec![Array1::zeros(3); self.mpc.n_steps], self.rocket.get_mass(), &self.rocket.get_moi_mpc(), -10.0);
    }

    pub fn step(&mut self) -> bool {
        if self.traj_stage == self.end_stage {
            return false;
        }

        let mass = self.rocket.get_mass();

        let world_z_axis = self.rocket.attitude * Vector3::z();
        let cos_theta = world_z_axis.z;
        let _ = cos_theta.clamp(-1.0, 1.0).acos();
        if cos_theta < 0.965925826289{
            self.has_exceeded_angle = true;
        }

        // TODO: get xref_traj and uref_traj in a way that allows for hovering (i.e. automatically toggling lossless usage on/off, will likely need to set up trajectory state machine)
        let xref_traj;
        let uref_traj;
        (xref_traj, uref_traj) = self.get_ref_traj();

        let mut control_input = Vector4::new(0.0, 0.0, 0.0, 0.0);
        let mut outside_forces = Vector3::new(0.0, 0.0, 0.0);
        let mut outside_torques = Vector3::new(0.0, 0.0, 0.0);

        let control_sequence = self.mpc.update(&self.rocket.get_state(), &xref_traj, &uref_traj, mass, &self.rocket.get_moi_mpc(), self.current_time); // Placeholder reference trajectory and warm start
        
        let last_solve_time = self.mpc.last_solve_time;// 1. Calculate the fractional progress between MPC steps (0.0 to 1.0)
        let time_since_mpc_solve = self.current_time - last_solve_time;
        let raw_index = (time_since_mpc_solve / self.mpc.dt) as usize;
        let max_index = control_sequence.len().saturating_sub(1);
        let current_step_index = raw_index.min(max_index);
        let progress = 0.0;
        // 2. Grab the current command (u0) and the next predicted command (u1)
        let u_0 = &control_sequence[current_step_index];
        // Safety check: ensure the sequence actually has a future step to blend into
        let u_1 = if current_step_index + 1 <= max_index {
            &control_sequence[current_step_index + 1]
        } else {
            u_0
        };
        // 3. Interpolate! 
        // Because u_0 and u_1 are ndarray::Array1<f64>, Rust lets you do the vector math directly.
        let current_control = u_0 + progress * (u_1 - u_0);

        let control_input = Vector4::new(current_control[0], current_control[1], current_control[2], 0.0); // Convert first control input to Vector4 (assuming the 4th component is not used for now)

        if !self.rocket.step(control_input, outside_forces, outside_torques, self.dt) && self.current_time > self.min_time {
            return false;
        }
        
        if self.debug {
            println!("XREF_TRAJ{:?}", xref_traj);
            println!("Rocket State: {:?}", self.rocket.get_state());
            println!("Rocket Mass: {}", mass);
            println!("Control Input: {:?}", control_input);
            println!("Total Force: {:?}", self.rocket.debug_info.total_force);
            println!("Thrust Vector: {:?}", self.rocket.thrust_vector);

            // Log Vehicle
            let normalized_thrust_vector = self.rocket.thrust_vector / 1000.0; // Scale for visualization
            let rocket_color;
            if self.has_exceeded_angle {
                rocket_color = Color::from_rgb(255, 0, 0);
            } else {
                rocket_color = Color::from_rgb(0, 255, 0);
            }
            let rotated_offset = self.rocket.attitude.transform_vector(&self.rocket.com_to_ground);
            let _ = self.rec.as_ref().unwrap().log(
                "world/rocket",
                &Arrows3D::from_vectors([((-2.0 * rotated_offset.x) as f32, (-2.0 * rotated_offset.y) as f32, (-2.0 * rotated_offset.z) as f32)])
                    .with_origins([[(self.rocket.position.x+rotated_offset.x) as f32, (self.rocket.position.y+rotated_offset.y) as f32, (self.rocket.position.z+rotated_offset.z) as f32]])
                    .with_colors([rocket_color]) 
            );

            // Log Thrust Vector
            let _ = self.rec.as_ref().unwrap().log(
                "world/thrust_vector",
                &Arrows3D::from_vectors([((normalized_thrust_vector.x) as f32, (normalized_thrust_vector.y) as f32, (normalized_thrust_vector.z) as f32)])
                    .with_origins([[(self.rocket.position.x+rotated_offset.x) as f32, (self.rocket.position.y+rotated_offset.y) as f32, (self.rocket.position.z+rotated_offset.z) as f32]])
                    .with_colors([Color::from_rgb(0, 0, 255)]) 
            );

            // Log the rocket's position for visualization
            let _ = self.rec.as_ref().unwrap().log(
                "world/rocket",
                &Points3D::new([(self.rocket.position.x as f32, self.rocket.position.y as f32, self.rocket.position.z as f32)])
                    .with_colors([rocket_color]) // Red color for the rocket
                    .with_radii([0.1]) // Size of the point representing the rocket
            );

            // Log the rocket's position for visualization
            let _ = self.rec.as_ref().unwrap().log(
                "world/timer",
                &Points3D::new([(self.current_time as f32, 0.0, 0.0)])
                    .with_colors([Color::from_rgb(0, 255, 0)]) // Green color for the timer
                    .with_radii([0.1]) // Size of the point representing the timer
            );

            // 1. Safely check if we actually have a recording stream configured
            if let Some(rec) = &self.rec {
                // 2. Call the function without the `?` operator. 
                // 3. Match on the Result to handle potential graphics errors gracefully!
                if let Err(e) = Self::plot_trajectory(rec, &self.lossless.current_traj, self.current_time - self.lossless.last_solve_time) {
                    eprintln!("Warning: Failed to plot trajectory to Rerun: {}", e);
                }
            }
        }

        self.current_time += self.dt;

        true
    }

    pub fn finish_sim(&mut self) -> (bool, Array1<f64>) {
        if self.debug {
            let _ = self.rocket.save_debug_to_csv("simulation.csv");

            println!("Nitrogen mass: {}", self.rocket.nitrogen_mass);
            println!("Pressurizing nitrogen mass: {}", self.rocket.pressurizing_nitrogen_mass);
            println!("Nitrous mass: {}", self.rocket.nitrous_mass);
            println!("Fuel grain mass: {}", self.rocket.fuel_grain_mass);
        }

        return (self.has_exceeded_angle, self.rocket.get_state())
    }

    pub fn get_ref_traj(&mut self) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let xref_traj;
        let uref_traj;

        if self.traj_stage == 0 {
            // 1. Stage Cost (Q) - Massive penalty for Z errors
            self.mpc.q = Array2::<f64>::from_diag(&Array1::from(vec![
                150.0, 150.0, 200.0,  // Stiffened Z-Position spring (from 40.0 to 1500.0)
                40000.0, 40000.0, 0.0, 0.0,
                // 100.0, 100.0, 6000.0,  // Stiffened Z-Velocity damper (from 300.0 to 2500.0)
                100.0, 100.0, 1000.0,  // Stiffened Z-Velocity damper (from 300.0 to 2500.0)
                500.0, 500.0, 500.0   
            ]));

            // 2. Control Cost (R) - Remove the fear of using the throttle!
            // Gimbal X/Y stay at 50 (because radians are tiny numbers), Thrust drops to 0.005!
            self.mpc.r = Array2::<f64>::from_diag(&Array1::from(vec![50.0, 50.0, 0.005]));

            // 3. Terminal Cost (QN) - Land softly!
            self.mpc.qn = Array2::<f64>::from_diag(&Array1::from(vec![
                150.0, 150.0, 400.0, 
                50000.0, 50000.0, 0.0, 0.0,
                100.0, 100.0, 1000.0, 
                1000.0, 1000.0, 1000.0 
            ]));
            self.lossless.use_terminal_lateral_hard_tube = true;
            self.lossless.terminal_lateral_hard_tube_time_s = 0.5;
            self.lossless.terminal_lateral_hard_tube_radius_m = 0.05;

            self.lossless.max_velocity = 5.0;
            let mut trajectory;
            if self.rocket.position.z < 40.0 {
                trajectory = self.lossless.update([self.rocket.position.x, self.rocket.position.y, self.rocket.position.z], [self.rocket.velocity.x, self.rocket.velocity.y, self.rocket.velocity.z], [0.0, 0.0, 50.0], self.rocket.get_mass() - self.rocket.get_dry_mass(), self.current_time);
            } else {
                trajectory = self.lossless.current_traj.clone();
            }
            (xref_traj, uref_traj) = self.get_mpc_reference(&trajectory, self.current_time - self.lossless.last_solve_time, self.rocket.attitude, self.mpc.min_thrust, self.mpc.dt, self.lossless.fine_delta_t, self.mpc.n_steps + 1);
            if self.rocket.position.z >= 48.0 {
                self.traj_stage = 1;
                self.traj_timer = self.current_time;
            }
            // uref_traj = vec![Array1::from(vec![0.0, 0.0, mass * 9.81]); mpc.n_steps]; // Warm start with zero control inputs
        } else if self.traj_stage == 1 {
            self.mpc.q = Array2::<f64>::from_diag(&Array1::from(vec![
                200.0, 200.0, 800.0,   // position x, y, z
                60000.0, 60000.0, 0.0, 0.0, // quaternion qx, qy, qz, qw
                100.0, 100.0, 250.0,        // linear velocities x_dot, y_dot, z_dot
                500.0, 500.0, 500.0          // angular velocities wx, wy, wz
            ]));
            self.mpc.r = Array2::<f64>::from_diag(&Array1::from(vec![50.0, 50.0, 1.0]));
            self.mpc.qn = Array2::<f64>::from_diag(&Array1::from(vec![
                200.0, 200.0, 1000.0,   // position x, y, z
                100000.0, 100000.0, 0.0, 0.0, // quaternion qx, qy, qz, qw
                100.0, 100.0, 300.0,        // linear velocities x_dot, y_dot, z_dot
                1000.0, 1000.0, 1000.0          // angular velocities wx, wy, wz
            ]));
            xref_traj = vec![Array1::from(vec![0.0, 0.0, 50.0, // x, y, z
                                                0.0, 0.0, 0.0, 1.0, // qx, qy, qz, qw (upright)
                                                0.0, 0.0, 0.0, // x_dot, y_dot, z_dot
                                                0.0, 0.0, 0.0]); // wx, wy, wz
                                                self.mpc.n_steps + 1];
            uref_traj = vec![Array1::from(vec![0.0, 0.0, self.rocket.get_mass() * 9.81]); self.mpc.n_steps]; // Warm start with zero control inputs
            self.lossless.update([self.rocket.position.x, self.rocket.position.y, self.rocket.position.z], [0.0, 0.0, 0.0], [0.0, 0.0, 1.5], self.rocket.get_mass() - self.rocket.get_dry_mass(), self.current_time);
            if self.current_time - self.traj_timer >= 10.0 {
                self.traj_stage = 2;
                self.traj_timer = self.current_time;
            }
        } else if self.traj_stage == 2 {
            // TODO: fix the vertical (and horizontal) drift during hovering

            // 1. Stage Cost (Q) - Massive penalty for Z errors
            self.mpc.q = Array2::<f64>::from_diag(&Array1::from(vec![
                150.0, 150.0, 1000.0,  // Stiffened Z-Position spring (from 40.0 to 1500.0)
                50000.0, 50000.0, 0.0, 0.0,
                100.0, 100.0, 200.0,  // Stiffened Z-Velocity damper (from 300.0 to 2500.0)
                500.0, 500.0, 500.0   
            ]));

            // 2. Control Cost (R) - Remove the fear of using the throttle!
            // Gimbal X/Y stay at 50 (because radians are tiny numbers), Thrust drops to 0.005!
            self.mpc.r = Array2::<f64>::from_diag(&Array1::from(vec![50.0, 50.0, 0.005]));

            // 3. Terminal Cost (QN) - Land softly!
            self.mpc.qn = Array2::<f64>::from_diag(&Array1::from(vec![
                150.0, 150.0, 3000.0, 
                50000.0, 50000.0, 0.0, 0.0,
                100.0, 100.0, 500.0, 
                1000.0, 1000.0, 1000.0 
            ]));
            self.lossless.lower_thrust_bound = 400.0;
            self.lossless.flip_glide_slope = false;
            self.lossless.use_glide_slope = true;
            self.lossless.glide_slope = 0.005_f64.to_radians();
            // self.lossless.max_velocity = 5.0;
            // if self.rocket.velocity.norm() >= 5.0 {
            //     self.lossless.max_velocity = self.rocket.velocity.norm() * 1.25;
            // }
            let trajectory = self.lossless.update([self.rocket.position.x, self.rocket.position.y, self.rocket.position.z], [self.rocket.velocity.x, self.rocket.velocity.y, self.rocket.velocity.z], [0.0, 0.0, 1.5], self.rocket.get_mass() - self.rocket.get_dry_mass(), self.current_time);
            (xref_traj, uref_traj) = self.get_mpc_reference(&trajectory, self.current_time - self.lossless.last_solve_time, self.rocket.attitude, self.mpc.min_thrust, self.mpc.dt, self.lossless.fine_delta_t, self.mpc.n_steps + 1);
            if self.current_time - self.traj_timer > 30.0 {
                self.traj_stage = -1;
                self.end_stage = -1;
            }
        } else {
            self.traj_stage = 1;
            self.traj_timer = self.current_time;
            self.end_stage = 2;
            return self.get_ref_traj();
        }

        (xref_traj, uref_traj)
    }

    // TODO: make all the print satements locked behind an if self.debug {} statement
    pub fn get_mpc_reference(
        &mut self,
        traj: &lossless::TrajectoryResult,
        current_time: f64,
        initial_attitude: UnitQuaternion<f64>,
        default_thrust: f64,
        mpc_dt: f64,
        lossless_dt: f64,
        n_steps: usize,
    ) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut xref_traj = Vec::with_capacity(n_steps);
        let num_points = traj.positions.len();
        if num_points == 0 {
            panic!("TrajectoryResult is empty!");
        }

        let mut uref_traj = Vec::with_capacity(n_steps - 1);
        let real_thrusts: Vec<[f64; 1]> = traj.thrusts.iter()
            .zip(&traj.masses)
            .map(|(u, &m)| {
                let world_thrust = Vector3::new(u[0] * m, u[1] * m, u[2] * m);
                [world_thrust.norm()] 
            })
            .collect();
        let num_thrusts = traj.thrusts.len();

        // Generate the state for the current time, plus each future step in the horizon
        for i in 0..=n_steps {
            // 1. What exact time is the MPC predicting for this step?
            let t_target = current_time + (i as f64) * mpc_dt;

            let mut interp_p = [0.0; 3];
            let mut interp_v = [0.0; 3];
            let mut interp_u = [0.0; 3];

            // // 2. Are we past the end of the flight profile?
            // if t_target >= traj.time_of_flight_s || num_points == 1 {
            //     // Park the rocket at the final coordinate
            //     interp_p = traj.positions[num_points - 1];
            //     interp_v = [0.0, 0.0, 0.0]; // Force target velocity to 0 to hold the hover
            //     interp_u = [0.0, 0.0, default_thrust];
            if t_target >= traj.time_of_flight_s || num_points <= 1 {
                if num_points <= 1 {
                    // 🚨 TRAJECTORY CRASHED: Fallback to a safe, slow vertical drop
                    interp_p = [0.0, 0.0, -2.0];
                    interp_v = [0.0, 0.0, -2.0]; 
                    interp_u = [0.0, 0.0, default_thrust];
                } else {
                    // Normal completion: Extrapolate the final state smoothly!
                    // Get the final known position and velocity from the solver
                    let final_p = traj.positions[num_points - 1];
                    let final_v = traj.velocities[num_points - 1];
                    
                    // Calculate how much time has passed SINCE the trajectory ended
                    let dt_overfill = t_target - traj.time_of_flight_s;
                    
                    // Extrapolate position: p = p_final + v_final * dt
                    interp_p = [
                        final_p[0] + final_v[0] * dt_overfill,
                        final_p[1] + final_v[1] * dt_overfill,
                        final_p[2] + final_v[2] * dt_overfill,
                    ];
                    
                    // Keep asking for the final velocity to prevent the MPC from braking too early
                    interp_v = final_v; 
                    
                    // If the trajectory ended, it likely commanded max or min thrust. 
                    // We'll just ask for enough thrust to maintain the hover mass as a neutral baseline.
                    interp_u = [0.0, 0.0, default_thrust];
                }
            } else if t_target <= 0.0 {
                // We haven't launched yet (or t_target is 0)
                interp_p = traj.positions[0];
                interp_v = traj.velocities[0];
                if num_thrusts > 0 {
                    interp_u = [0.0, 0.0, real_thrusts[0][0]];
                }
                
            } else {
                // 3. Interpolate perfectly between the two closest trajectory points
                let exact_idx = t_target / lossless_dt;
                let base_idx = exact_idx.floor() as usize;
                
                // Prevent out-of-bounds on the array lookup
                let safe_idx = base_idx.min(num_points.saturating_sub(2));
                
                // Calculate the fraction (0.0 to 1.0) between the two points
                // let frac = (t_target - (safe_idx as f64) * lossless_dt) / lossless_dt;
                // let clamped_frac = frac.clamp(0.0, 1.0);
                // 3. Subtract to get the float portion, then clamp to lock out-of-bounds to 1.0
                let clamped_frac = (exact_idx - safe_idx as f64).clamp(0.0, 1.0);

                let p0 = traj.positions[safe_idx];
                let p1 = traj.positions[safe_idx + 1];
                let v0 = traj.velocities[safe_idx];
                let v1 = traj.velocities[safe_idx + 1];

                let max_u_idx = num_thrusts.saturating_sub(2);
                let idx_u = base_idx.min(max_u_idx);
                let frac_u = (exact_idx - idx_u as f64).clamp(0.0, 1.0);

                let u0 = if num_thrusts > 0 { [0.0, 0.0, real_thrusts[idx_u][0]] } else { [0.0, 0.0, default_thrust] };
                let u1 = if num_thrusts > 1 { [0.0, 0.0, real_thrusts[idx_u + 1][0]] } else { u0 };

                for j in 0..3 {
                    interp_p[j] = p0[j] + clamped_frac * (p1[j] - p0[j]);
                    interp_v[j] = v0[j] + clamped_frac * (v1[j] - v0[j]);
                    interp_u[j] = u0[j] + clamped_frac * (u1[j] - u0[j]);
                }
            }

            // 4. Map the interpolated data into the 13-element MPC state array
            // Order: [x, y, z, qx, qy, qz, qw, x_dot, y_dot, z_dot, wx, wy, wz]
            let state = Array1::from(vec![
                interp_p[0], interp_p[1], interp_p[2], // Target Position
                0.0, 0.0, 0.0, 1.0,                    // Target Attitude (Upright)
                interp_v[0], interp_v[1], interp_v[2], // Target Velocity
                0.0, 0.0, 0.0                          // Target Angular Velocity (Zero spin)
            ]);

            xref_traj.push(state);
            // Push controls ONLY for the current step and future steps, excluding the terminal boundary
            if i < n_steps - 1 {
                uref_traj.push(Array1::from(vec![interp_u[0], interp_u[1], interp_u[2]]));
            }
        }
        
        if self.debug {
            println!("UREF_TRAJ: {:?}", uref_traj);
        }
        
        (xref_traj, uref_traj)
    }

    pub fn plot_trajectory(
        rec: &RecordingStream, 
        traj: &TrajectoryResult, 
        elapsed_trajectory_time: f64 // 🚀 NEW: How long ago was this trajectory solved?
    ) -> Result<(), Box<dyn std::error::Error>> {
        
        let num_nodes = traj.positions.len();
        if num_nodes < 2 { return Ok(()); } // Need at least 2 points to draw a line

        // 1. Calculate the time spacing between each node
        // (Divide total flight time by the number of gaps between nodes)
        let dt = traj.time_of_flight_s / (num_nodes - 1) as f64;

        // 2. Calculate how many nodes we have already flown past
        let nodes_passed = (elapsed_trajectory_time / dt).floor() as usize;
        
        // Safety clamp to ensure we don't skip past the end of the array
        let start_index = nodes_passed.min(num_nodes - 1);

        // 3. Cast to f32 AND filter out the past using `.skip()`
        let graphics_positions: Vec<[f32; 3]> = traj.positions
            .iter()
            .skip(start_index) // 🚀 MAGIC: Ignores the first N elements!
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();

        // If we only have 1 point left, we can't draw a line, so just exit cleanly
        if graphics_positions.len() < 2 {
            return Ok(());
        }

        // 4. Log the future path
        rec.log(
            "rocket/trajectory/path",
            &LineStrips3D::new([graphics_positions.clone()])
                .with_colors([Color::from_rgb(0, 150, 255)]) 
                .with_radii([0.05]), 
        )?;

        // 5. Log the future nodes
        rec.log(
            "rocket/trajectory/nodes",
            &Points3D::new(graphics_positions)
                .with_colors([Color::from_rgb(255, 100, 0)]) 
                .with_radii([0.1]), // Beautiful, small dots
        )?;

        Ok(())
    }
}