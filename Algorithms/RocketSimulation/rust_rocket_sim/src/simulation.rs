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

impl std::fmt::Debug for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Simulation")
            .field("current_time", &self.current_time)
            .field("start_state", &self.start_state)
            .finish()
    }
}

pub struct Simulation {
    pub rocket: Rocket,
    pub fsm: Lander::fsm::FlightStateMachine,
    pub rcs_controller: RcsController,
    pub held_control: Vector3<f64>,
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
            fsm: Lander::fsm::FlightStateMachine::new(),
            rcs_controller: RcsController::default(),
            held_control: Vector3::new(0.0, 0.0, 80.0 * 9.81),
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
        
        self.fsm.initialize();
        self.fsm.get_state_mut().mass = self.rocket.get_mass();
        self.fsm.get_state_mut().vehicle_state.dry_mass = self.rocket.get_dry_mass();

        match self.start_state.as_str() {
            "ascent" => {
                self.fsm.arm(0.0);
                self.fsm.launch(0.0);
            }
            "hover" => {
                self.fsm.set_flight_phase(Lander::state::FlightPhase::Hover, 0.0);
            }
            "descent" => {
                self.fsm.set_flight_phase(Lander::state::FlightPhase::Descent, 0.0);
            }
            _ => {
                self.fsm.arm(0.0);
                self.fsm.launch(0.0);
            }
        }

        let (_, _, yaw) = self.rocket.attitude.euler_angles();
        self.rcs_controller.update(yaw, self.rocket.ang_vel.z, -10.0);
    }

    pub fn step(&mut self) -> bool {
        if self.fsm.get_state().flight_terminated {
            println!("Simulation stopped: FSM terminated. Reason: {:?}", self.fsm.get_state().termination_reason);
            return false;
        }
        if self.fsm.get_state().flight_phase == Lander::state::FlightPhase::Landed {
            println!("Simulation stopped: Lander successfully landed!");
            return false;
        }

        let mass = self.rocket.get_mass();

        if (self.current_time * 100.0).round() as usize % 100 == 0 {
            println!(
                "Time: {:.2}s | Phase: {:?} | Pos: [{:.2}, {:.2}, {:.2}] | Vel: [{:.2}, {:.2}, {:.2}] | Mass: {:.2}kg",
                self.current_time,
                self.fsm.get_state().flight_phase,
                self.rocket.position.x, self.rocket.position.y, self.rocket.position.z,
                self.rocket.velocity.x, self.rocket.velocity.y, self.rocket.velocity.z,
                mass
            );
        }

        let world_z_axis = self.rocket.attitude * Vector3::z();
        let cos_theta = world_z_axis.z;
        let _ = cos_theta.clamp(-1.0, 1.0).acos();
        if cos_theta < 0.965925826289 {
            self.has_exceeded_angle = true;
        }

        // Construct SensorData from simulated sensors
        let imu_reading = &self.rocket.imu.last_reading;
        let gps_reading = &self.rocket.gps.last_reading;
        let uwb_reading = &self.rocket.uwb.last_reading;

        let sensor_data = Lander::state::SensorData {
            timestamp: self.current_time,
            imu_data: Some(Lander::state::ImuData {
                accel: [imu_reading.accel.x, imu_reading.accel.y, imu_reading.accel.z],
                gyro: [imu_reading.gyro.x, imu_reading.gyro.y, imu_reading.gyro.z],
                mag: [imu_reading.mag.x, imu_reading.mag.y, imu_reading.mag.z],
            }),
            gps_data: Some([gps_reading.position.x, gps_reading.position.y, gps_reading.position.z]),
            uwb_data: if (self.rocket.position - self.rocket.uwb.origin).norm() <= self.rocket.uwb.range {
                Some([uwb_reading.position.x, uwb_reading.position.y, uwb_reading.position.z])
            } else {
                None
            },
            chamber_pressure: Some(self.rocket.m2_pt.last_reading.pressure_bar),
            tank_pressure: Some(self.rocket.o_pt.last_reading.pressure_bar),
            true_attitude: Some([self.rocket.attitude.coords[3], self.rocket.attitude.coords[0], self.rocket.attitude.coords[1], self.rocket.attitude.coords[2]]),
        };

        // Step the Flight State Machine
        let control_input_opt = self.fsm.step(&sensor_data);
        let control_input = if let Some(out) = control_input_opt {
            self.held_control = Vector3::new(out[0], out[1], out[2]);
            Vector4::new(out[0], out[1], out[2], out[3])
        } else {
            Vector4::new(self.held_control.x, self.held_control.y, self.held_control.z, 0.0)
        };

        let outside_forces = Vector3::new(0.0, 0.0, 0.0);
        let outside_torques = Vector3::new(0.0, 0.0, 0.0);

        if !self.rocket.step(control_input, outside_forces, outside_torques, self.dt) && self.current_time > self.min_time {
            return false;
        }
        
        if self.debug {
            println!("Rocket State: {:?}", self.rocket.get_state());
            println!("Rocket Mass: {}", mass);
            println!("Control Input: {:?}", control_input);
            println!("Total Force: {:?}", self.rocket.debug_info.total_force);
            println!("Thrust Vector: {:?}", self.rocket.thrust_vector);

            if let Some(rec) = &self.rec {
                let rocket_color = if self.has_exceeded_angle {
                    Color::from_rgb(255, 0, 0)
                } else {
                    Color::from_rgb(0, 255, 0)
                };
                let rotated_offset = self.rocket.attitude.transform_vector(&self.rocket.com_to_ground);
                let rocket_forward = self.rocket.attitude.transform_vector(&Vector3::new(1.0, 0.0, 0.0));

                let _ = rec.log(
                    "world/rocket",
                    &Arrows3D::from_vectors([((-2.0 * rotated_offset.x) as f32, (-2.0 * rotated_offset.y) as f32, (-2.0 * rotated_offset.z) as f32)])
                        .with_origins([[(self.rocket.position.x+rotated_offset.x) as f32, (self.rocket.position.y+rotated_offset.y) as f32, (self.rocket.position.z+rotated_offset.z) as f32]])
                        .with_colors([rocket_color]) 
                );

                let normalized_thrust_vector = self.rocket.thrust_vector / 1000.0;
                let _ = rec.log(
                    "world/thrust_vector",
                    &Arrows3D::from_vectors([((normalized_thrust_vector.x) as f32, (normalized_thrust_vector.y) as f32, (normalized_thrust_vector.z) as f32)])
                        .with_origins([[(self.rocket.position.x+rotated_offset.x) as f32, (self.rocket.position.y+rotated_offset.y) as f32, (self.rocket.position.z+rotated_offset.z) as f32]])
                        .with_colors([Color::from_rgb(0, 0, 255)]) 
                );

                let _ = rec.log(
                    "world/forward_vector",
                    &Arrows3D::from_vectors([((rocket_forward.x) as f32, (rocket_forward.y) as f32, (rocket_forward.z) as f32)])
                        .with_origins([[(self.rocket.position.x) as f32, (self.rocket.position.y) as f32, (self.rocket.position.z) as f32]])
                        .with_colors([Color::from_rgb(255, 165, 0)]) 
                );

                let _ = rec.log(
                    "world/rocket",
                    &Points3D::new([(self.rocket.position.x as f32, self.rocket.position.y as f32, self.rocket.position.z as f32)])
                        .with_colors([rocket_color])
                        .with_radii([0.1])
                );

                let _ = rec.log(
                    "world/timer",
                    &Points3D::new([(self.current_time as f32, 0.0, 0.0)])
                        .with_colors([Color::from_rgb(0, 255, 0)])
                        .with_radii([0.1])
                );

                if let Some(traj) = &self.fsm.get_state().trajectory_state {
                    let elapsed = self.current_time - self.fsm.get_state().trajectory_generation_time;
                    if let Err(e) = Self::plot_trajectory(rec, traj, elapsed) {
                        eprintln!("Warning: Failed to plot trajectory to Rerun: {}", e);
                    }
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

    pub fn plot_trajectory(
        rec: &RecordingStream, 
        traj: &rust_lossless::TrajectoryResult, 
        elapsed_trajectory_time: f64
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