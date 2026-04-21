use nalgebra::{Matrix3, Vector3, Vector4, UnitQuaternion, Quaternion};

use crate::device_sim::*;
use crate::sloshing_sim::*;
use crate::fluid_dynamics::*;
use crate::wind_sim::*;
use crate::aero_tables::*;
use ndarray::{Array1, Array2, array};
use std::error::Error;
use std::fs::File;

#[derive(Debug, Clone)]
pub struct Rocket {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub accel: Vector3<f64>,
    
    pub attitude: UnitQuaternion<f64>,
    pub ang_vel: Vector3<f64>,
    pub ang_accel: Vector3<f64>,

    pub frame_mass: f64,
    pub nitrogen_tank_empty_mass: f64, // the mass of the empty tank
    pub nitrogen_mass: f64,
    starting_nitrogen_mass: f64,
    pub nitrogen_tank_offset: Vector3<f64>, // Vector offset from frame CoM to the nitrogen tank CoM
    pub nitrous_tank_empty_mass: f64, // the mass of the empty tank
    pub pressurizing_nitrogen_mass: f64,
    starting_pressurizing_nitrogen_mass: f64,
    pub nitrous_mass: f64,
    starting_nitrous_mass: f64,
    pub nitrous_tank_offset: Vector3<f64>, // Vector offset from frame CoM to the nitrous tank CoM
    pub tvc_module_empty_mass: f64, // the mass of the empty tvc module
    pub fuel_grain_mass: f64,
    starting_fuel_grain_mass: f64,
    pub frame_com_to_gimbal: Vector3<f64>, // Vector offset from frame CoM to the gimbal point (thrust plate)
    pub gimbal_to_tvc_com: Vector3<f64>, // Vector offset from gimbal point to the TVC's center of mass

    pub moi: Matrix3<f64>,
    pub frame_moi: Matrix3<f64>,
    pub dry_nitrogen_moi: Matrix3<f64>,
    pub wet_nitrogen_moi: Matrix3<f64>,
    pub nitrous_tank_radius: f64,
    pub nitrous_tank_length: f64,
    pub nitrous_level: f64,
    pub dry_nitrous_moi: Matrix3<f64>,
    pub dry_tvc_moi: Matrix3<f64>,
    pub wet_tvc_moi: Matrix3<f64>,

    pub tvc_range: f64,
    pub tvc: TVC,

    pub rcs: RCS,

    // Propellant Feed System Valves
    pub fill_mv: Valve,  // Fill valve (entry to lander tanks — irrelevant in flight)
    pub r_mv: Valve,     // Regulator isolation valve (N2 storage → N2O run tank)
    pub rcs1_mv: Valve,  // RCS thruster pair 1 (controls one roll direction)
    pub rcs2_mv: Valve,  // RCS thruster pair 2 (controls opposite roll direction)
    pub o_iso: Valve,    // Oxidizer isolation valve (N2O run tank → MTV/engine)
    pub o_vnt: Valve,    // Oxidizer vent valve (relieves run tank pressure)

    pub thrust_vector: Vector3<f64>,

    pub imu: IMU,
    pub gps: GPS,
    pub uwb: UWB,

    pub sloshing_model: SloshModel,

    pub thermo_fluid_solver: ThermoFluidSolver,
    pub nitrous_m_dot: f64,

    pub com_to_ground: Vector3<f64>, // Distance from center of mass to ground (for ground interaction)

    // Optional wind model — None means no wind (clean-air baseline)
    pub wind_model: Option<WindModel>,

    // Optional aerodynamic lookup table — None falls back to the hard-coded drag model.
    pub aero_table: Option<AeroTable>,

    // Distance from the rocket's body-frame origin (frame CoM reference) to the
    // physical nose tip, as a 3D vector.
    // Used to convert the center of pressure to rocket body frame coordinates.
    pub nose_offset: Vector3<f64>,

    system_time: f64,

    pub debug_info: RocketDebugInfo,
}

#[derive(Debug, Clone)]
pub struct RocketDebugInfo{
    pub thrust_torque: Vector3<f64>,
    pub total_force: Vector3<f64>,
    pub total_torque: Vector3<f64>,
    pub times: Vec<f64>,
    pub com_offsets: Vec<Vector3<f64>>,
    pub mois: Vec<Matrix3<f64>>,
    pub thrusts: Vec<Vector3<f64>>,
    pub slosh_forces: Vec<Vector3<f64>>,
    pub nitrous_m_dots: Vec<f64>,
    pub valve_angles: Vec<f64>,
    pub chamber_pressures: Vec<f64>,
    pub of_ratios: Vec<f64>,
    pub isps: Vec<f64>,
    pub cstars: Vec<f64>,
    pub port_ds: Vec<f64>,
    pub fuel_masses: Vec<f64>,
    pub nitrous_masses: Vec<f64>,
    pub nitrogen_n2_tank_masses: Vec<f64>,
    pub nitrogen_n2o_tank_masses: Vec<f64>,
    pub wind_vels: Vec<Vector3<f64>>, // Wind velocity applied this tick [m/s], world frame
    pub aero_drags: Vec<Vector3<f64>>, // Drag force applied this tick, body frame [N]
    pub aero_moments: Vec<Vector3<f64>>, // Aerodynamic moment (r_lever × F_drag), body frame [N·m]
    pub attitudes: Vec<UnitQuaternion<f64>>,
    pub imu_readings: Vec<IMUReading>,
    pub gps_readings: Vec<GPSReading>,
    pub uwb_readings: Vec<UWBReading>,
}

impl RocketDebugInfo {
    pub fn default() -> Self {
        Self {
            thrust_torque: Vector3::zeros(),
            total_force: Vector3::zeros(),
            total_torque: Vector3::zeros(),
            times: Vec::new(),
            com_offsets: Vec::new(),
            mois: Vec::new(),
            thrusts: Vec::new(),
            slosh_forces: Vec::new(),
            nitrous_m_dots: Vec::new(),
            valve_angles: Vec::new(),
            chamber_pressures: Vec::new(),
            of_ratios: Vec::new(),
            isps: Vec::new(),
            cstars: Vec::new(),
            port_ds: Vec::new(),
            fuel_masses: Vec::new(),
            nitrous_masses: Vec::new(),
            nitrogen_n2_tank_masses: Vec::new(),
            nitrogen_n2o_tank_masses: Vec::new(),
            wind_vels: Vec::new(),
            aero_drags: Vec::new(),
            aero_moments: Vec::new(),
            attitudes: Vec::new(),
            imu_readings: Vec::new(),
            gps_readings: Vec::new(),
            uwb_readings: Vec::new(),
        }
    }
}

impl Rocket {
    pub fn new(position: Vector3<f64>, velocity: Vector3<f64>, accel: Vector3<f64>, attitude: UnitQuaternion<f64>, ang_vel: Vector3<f64>, ang_accel: Vector3<f64>, frame_mass: f64, nitrogen_tank_empty_mass: f64, starting_nitrogen_mass: f64, nitrogen_tank_offset: Vector3<f64>, nitrous_tank_empty_mass: f64, starting_pressurizing_nitrogen_mass: f64, starting_nitrous_mass: f64, nitrous_tank_offset: Vector3<f64>, tvc_module_empty_mass: f64, starting_fuel_grain_mass: f64, frame_com_to_gimbal: Vector3<f64>, gimbal_to_tvc_com: Vector3<f64>, frame_moi: Matrix3<f64>, dry_nitrogen_moi: Matrix3<f64>, wet_nitrogen_moi: Matrix3<f64>, nitrous_tank_radius: f64, nitrous_tank_length: f64, nitrous_level: f64, dry_nitrous_moi: Matrix3<f64>, dry_tvc_moi: Matrix3<f64>, wet_tvc_moi: Matrix3<f64>, tvc_range: f64, tvc: TVC, rcs: RCS, fill_mv: Valve, r_mv: Valve, rcs1_mv: Valve, rcs2_mv: Valve, o_iso: Valve, o_vnt: Valve, imu: IMU, gps: GPS, uwb: UWB, sloshing_model: SloshModel, thermo_fluid_solver: ThermoFluidSolver, com_to_ground: Vector3<f64>, wind_model: Option<WindModel>, nose_offset_z: f64, aero_table: Option<AeroTable>) -> Self {
        let mut rocket = Self {
            position,
            velocity,
            accel,
            attitude,
            ang_vel,
            ang_accel,
            frame_mass,
            nitrogen_tank_empty_mass,
            nitrogen_mass: starting_nitrogen_mass,
            starting_nitrogen_mass,
            nitrogen_tank_offset,
            nitrous_tank_empty_mass,
            pressurizing_nitrogen_mass: starting_pressurizing_nitrogen_mass,
            starting_pressurizing_nitrogen_mass,
            nitrous_mass: starting_nitrous_mass,
            starting_nitrous_mass,
            nitrous_tank_offset,
            tvc_module_empty_mass,
            fuel_grain_mass: starting_fuel_grain_mass,
            starting_fuel_grain_mass,
            frame_com_to_gimbal,
            gimbal_to_tvc_com,
            moi: Matrix3::zeros(),
            frame_moi,
            dry_nitrogen_moi,
            wet_nitrogen_moi,
            nitrous_tank_radius,
            nitrous_tank_length,
            nitrous_level,
            dry_nitrous_moi,
            dry_tvc_moi,
            wet_tvc_moi,
            tvc_range,
            tvc,
            rcs,
            fill_mv,
            r_mv,
            rcs1_mv,
            rcs2_mv,
            o_iso,
            o_vnt,
            thrust_vector: Vector3::zeros(),
            imu,
            gps,
            uwb,
            sloshing_model,
            thermo_fluid_solver,
            nitrous_m_dot: 0.0,
            com_to_ground,
            wind_model,
            aero_table,
            nose_offset,
            system_time: 0.0,
            debug_info: RocketDebugInfo::default(),
        };

        rocket.imu.update(rocket.accel, rocket.ang_vel, rocket.attitude, rocket.system_time);
        rocket.gps.update(rocket.position, rocket.system_time);
        rocket.uwb.update(rocket.position, rocket.system_time);

        let com_offset = rocket.get_com_offset(Vector3::z());
        rocket.moi = rocket.get_moi(Vector3::z(), com_offset);

        rocket
    }

    pub fn default() -> Self {
        let position = Vector3::new(0.0, 0.0, 0.0);
        let velocity = Vector3::new(0.0, 0.0, 0.0);
        let acceleration = Vector3::new(0.0, 0.0, 0.0);
        let attitude = UnitQuaternion::identity();
        let angular_velocity = Vector3::new(0.0, 0.0, 0.0);
        let angular_acceleration = Vector3::new(0.0, 0.0, 0.0);

        let frame_mass = 15.66;
        let nitrogen_tank_empty_mass = 15.6;
        let starting_nitrogen_mass = 5.0;
        let nitrogen_tank_offset = Vector3::new(-0.02, -0.01, 0.76);
        let nitrous_tank_empty_mass = 15.86;
        let starting_pressurizing_nitrogen_mass = 0.0;
        let starting_nitrous_mass = 16.0;
        let nitrous_tank_offset = Vector3::new(-0.02, -0.01, 0.592);
        let tvc_module_empty_mass = 8.76;
        let starting_fuel_grain_mass = 3.0;
        let frame_com_to_gimbal = Vector3::new(-0.02, -0.01, -0.24);
        let gimbal_to_tvc_com = Vector3::new(0.0, 0.0, -0.25);
        
        let frame_moi = Matrix3::new(6.65, 0.35, -0.02,
                                    0.35, 6.65, -0.02,
                                    -0.02, -0.02, 2.3);
        let dry_nitrogen_moi = Matrix3::new(0.9275, 0.0, 0.0,
                                            0.0, 0.9275, 0.0,
                                            0.0, 0.0, 1.055);
        let wet_nitrogen_moi = Matrix3::new(1.0075, 0.0, 0.0,
                                            0.0, 1.0075, 0.0,
                                            0.0, 0.0, 1.055);
        let nitrous_tank_radius = 0.0;
        let nitrous_tank_length = 0.0;
        let nitrous_level = 0.85 * 0.75;
        let dry_nitrous_moi = Matrix3::new(1.31, 0.0, 0.0,
                                            0.0, 1.31, 0.0,
                                            0.0, 0.0, 0.21);
        let dry_tvc_moi = Matrix3::new(0.64, 0.0, 0.0,
                                        0.0, 0.64, 0.0,
                                        0.0, 0.0, 0.06);
        let wet_tvc_moi = Matrix3::new(0.84, 0.0, 0.0,
                                        0.0, 0.84, 0.0,
                                        0.0, 0.0, 0.07);
        let tvc_range = 15_f64.to_radians();

        
        let tvc = TVC::default(starting_fuel_grain_mass);

        let rcs = RCS::default();

        let imu = IMU::default();
        
        let gps = GPS::default();

        let uwb = UWB::default();

        // Propellant Feed System Valves — default operational states for flight
        let fill_mv = Valve::new(false);   // Closed in flight (only used during ground fill)
        let r_mv = Valve::new(true);       // Open: allows N2 to pressurize the N2O tank
        let rcs1_mv = Valve::new(false);   // Closed by default: responsive to roll commands
        let rcs2_mv = Valve::new(false);   // Closed by default: responsive to roll commands
        let o_iso = Valve::new(true);      // Open: allows N2O to flow to engine
        let o_vnt = Valve::new(false);     // Closed: only opens to relieve excess pressure

        let slosh_model = SloshModel::default();
        let thermo_fluid_solver = ThermoFluidSolver::default();

        let com_to_ground = Vector3::new(0.0, 0.0, -1.5);

        // let wind_model = Some(WindModel::default());
        let wind_model = None;

        let aero_table = AeroTable::default();
      
        let nose_offset = Vector3::new(0.0, 0.0, 1.5); // TODO: change this to a proper value once frame is re-cadded

        Self::new(position, velocity, acceleration, attitude, angular_velocity, angular_acceleration, frame_mass, nitrogen_tank_empty_mass, starting_nitrogen_mass, nitrogen_tank_offset, nitrous_tank_empty_mass, starting_pressurizing_nitrogen_mass, starting_nitrous_mass, nitrous_tank_offset, tvc_module_empty_mass, starting_fuel_grain_mass, frame_com_to_gimbal, gimbal_to_tvc_com, frame_moi, dry_nitrogen_moi, wet_nitrogen_moi, nitrous_tank_radius, nitrous_tank_length, nitrous_level, dry_nitrous_moi, dry_tvc_moi, wet_tvc_moi, tvc_range, tvc, rcs, fill_mv, r_mv, rcs1_mv, rcs2_mv, o_iso, o_vnt, imu, gps, uwb, slosh_model, thermo_fluid_solver, com_to_ground, wind_model, aero_table, nose_offset)


    /// Update state based on applied forces and torques
    /// forces: Force vector in World Frame
    /// torques: Torque vector in Body Frame
    /// Returns true if the step is successful and false if the simulation has ended (hit the ground)
    pub fn step(&mut self, control_input: Vector4<f64>, outside_forces: Vector3<f64>, outside_torques: Vector3<f64>, dt: f64) -> bool {
        let rotated_offset = self.attitude.transform_vector(&self.com_to_ground);
        let rocket_bottom = self.position + rotated_offset;
        if rocket_bottom.z < 0.0 {
            // 1. Shift the rocket UP by the exact amount it went underground.
            // (Because rocket_bottom.z is negative, subtracting it adds positive height)
            self.position.z -= rocket_bottom.z; // Add a small buffer of 1 cm to prevent immediate re-collision on the next frame

            // 2. CRITICAL: Kill the downward velocity!
            // If you don't do this, the math still thinks it's falling at -5m/s, 
            // and it will just instantly clip back underground on the next frame.
            self.velocity.x = 0.0;
            self.velocity.y = 0.0;
            if self.velocity.z < 0.0 { // Assuming your velocity variable is named this
                self.velocity.z = 0.0;
            }
            return false; // Indicate that we've hit the ground
        }
        
        self.debug_info.times.push(self.system_time);

        // Update Sensors
        self.debug_info.attitudes.push(self.attitude);
        let imu_reading = self.imu.update(self.accel, self.ang_vel, self.attitude, self.system_time);
        let gps_reading = self.gps.update(self.position, self.system_time);
        let uwb_reading = self.uwb.update(self.position, self.system_time);
        self.debug_info.imu_readings.push(imu_reading);
        self.debug_info.gps_readings.push(gps_reading);
        self.debug_info.uwb_readings.push(uwb_reading);

        let com_offset = self.get_com_offset(self.thrust_vector);
        self.moi = self.get_moi(self.thrust_vector, com_offset);
        // println!("ROCKET MOI: {:?}", self.moi);
        // println!("TVC_LEVER_ARM: {:?}", self.frame_com_to_gimbal - com_offset);

        // Update actuated devices
        let tvc_effect: TVCEffect = self.tvc.update(Vector3::new(control_input.x, control_input.y, control_input.z), self.frame_com_to_gimbal - com_offset, self.nitrogen_mass, self.pressurizing_nitrogen_mass, self.nitrous_mass, self.fuel_grain_mass, dt, self.system_time);
        self.nitrogen_mass = tvc_effect.nitrogen_mass;
        self.pressurizing_nitrogen_mass = tvc_effect.pressurizing_nitrogen_mass;
        self.nitrous_mass = tvc_effect.nitrous_mass;
        self.fuel_grain_mass = tvc_effect.fuel_grain_mass;

        // TODO: implement throttle controller
        // TODO: Consider expanding control_input to drive valve commands from GNC algorithms.
        //       For now, RCS valves are toggled based on the sign of the roll command.
        let rcs_command = control_input.w; // 4th element is the roll command
        // Map the scalar roll command to individual valve states:
        //   positive → rcs1_mv opens (clockwise roll)
        //   negative → rcs2_mv opens (counter-clockwise roll)
        //   zero     → both closed
        let rcs1_open = self.rcs1_mv.is_open || rcs_command > 0.0;
        let rcs2_open = self.rcs2_mv.is_open || rcs_command < 0.0;
        let rcs_effect = self.rcs.update(rcs1_open, rcs2_open, self.nitrogen_mass, dt, self.system_time);
        self.nitrogen_mass = rcs_effect.nitrogen_mass;

        
        // Fluid dynamics update — pass valve states to control flow paths
        let thrust_command = control_input[2];
        let is_rcs_on = rcs1_open || rcs2_open;
        let n2o_mass = self.nitrous_mass;
        let n2_mass_total = self.nitrogen_mass + self.pressurizing_nitrogen_mass;
        let fluid_dynamics_dt = dt;
        let fluid_dynamics_output = self.thermo_fluid_solver.fluid_dynamics_update(
            thrust_command,
            is_rcs_on,
            self.o_iso.is_open,
            self.r_mv.is_open,
            self.o_vnt.is_open,
            n2o_mass,
            n2_mass_total,
            fluid_dynamics_dt,
        );

        self.fuel_grain_mass = fluid_dynamics_output.new_fuel_mass;
        self.nitrous_mass = fluid_dynamics_output.new_n2o_mass;
        self.nitrous_level = fluid_dynamics_output.new_n2o_level;
        self.nitrogen_mass = fluid_dynamics_output.new_n2_mass_storagetanks;
        self.pressurizing_nitrogen_mass = fluid_dynamics_output.new_n2_mass_runtank;
        self.nitrous_m_dot = fluid_dynamics_output.mdot_ox;

        self.debug_info.com_offsets.push(com_offset);
        self.debug_info.mois.push(self.moi);
        self.debug_info.nitrous_m_dots.push(self.nitrous_m_dot);
        self.debug_info.valve_angles.push(fluid_dynamics_output.valve_angle); // This is in degrees!
        self.debug_info.chamber_pressures.push(fluid_dynamics_output.pc_bar);
        self.debug_info.of_ratios.push(fluid_dynamics_output.of_ratio_realized);
        self.debug_info.isps.push(fluid_dynamics_output.isp_realized);
        self.debug_info.cstars.push(fluid_dynamics_output.cstar_realized);
        self.debug_info.port_ds.push(self.thermo_fluid_solver.parameters.port_d);
        self.debug_info.fuel_masses.push(self.fuel_grain_mass);
        self.debug_info.nitrous_masses.push(self.nitrous_mass);
        self.debug_info.nitrogen_n2_tank_masses.push(self.nitrogen_mass);
        self.debug_info.nitrogen_n2o_tank_masses.push(self.pressurizing_nitrogen_mass);


        self.system_time += dt;
        let mass = self.get_mass();

        let slosh_force = self.sloshing_model.step(dt, self.accel, self.attitude, self.nitrous_m_dot);
        // let slosh_force = Vector3::zeros();

        // Translational Dynamics
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        self.thrust_vector = self.attitude.transform_vector(&tvc_effect.thrust);
        let slosh_force_world = self.attitude.transform_vector(&slosh_force);

        // Step the wind model to get current wind in world frame [m/s].
        // Drag depends on velocity relative to the airmass, not the ground.
        // A headwind feels like faster flight; a tailwind reduces drag.
        let wind_vel_world = self.wind_model
            .as_mut()
            .map(|w| w.step(self.position.z, dt, self.system_time))
            .unwrap_or_default();
        self.debug_info.wind_vels.push(wind_vel_world);

        // Relative velocity: how fast we're moving through the air (not the ground)
        let relative_vel_world = self.velocity - wind_vel_world;
        // Rotate into body frame so we can decompose into axial vs. lateral components.
        let body_vel = self.attitude.inverse_transform_vector(&relative_vel_world);

        // --- Atmospheric Model ---
        // ISA barometric formula: rho(h) = 1.225 * exp(-h / 8500)  [kg/m^3]
        // Clamp altitude to zero so rho doesn't blow up underground.
        let h = self.position.z.max(0.0);
        let rho = 1.225_f64 * (-h / 8500.0_f64).exp();

        let speed = body_vel.norm();  // magnitude of the relative wind in body frame [m/s]

        // --- Drag & Aerodynamic Moment ---
        //
        // Everything below lives in the BODY FRAME until we explicitly rotate.
        // Body frame convention (Z-up sim):
        //   +X = rocket starboard (right when looking from aft)
        //   +Y = rocket "up" in the lateral plane
        //   +Z = nose direction (body axis)
        //
        // We use a single lookup point (alpha, Mach) rather than per-axis Cd*A because
        // real aero data is parameterised this way and it naturally handles cross-coupling.
        let (drag_body, aero_moment_body) = if let Some(table) = &self.aero_table {
            // Compute Mach number — speed of sound is ISA sea-level (343 m/s).
            // A proper ISA model would make this altitude-dependent; left as a TODO.
            let speed_of_sound = 343.0_f64;
            let mach = speed / speed_of_sound;

            // Angle of attack: angle between the relative wind vector and the body Z axis.
            // lateral = crossflow component (sqrt of x^2 + y^2); axial = along-axis component.
            // alpha=0 means the wind is perfectly head-on; alpha=90 means pure crossflow.
            let lateral = (body_vel.x.powi(2) + body_vel.y.powi(2)).sqrt();
            let alpha_deg = lateral.atan2(-body_vel.z).to_degrees();

            // Bilinear lookup — both axes are clamped inside lookup() so no panic here.
            let rec = table.lookup(alpha_deg, mach);

            let q_dyn = 0.5 * rho * speed * speed;   // dynamic pressure [Pa]
            let drag_mag = q_dyn * rec.cd * rec.area_ref; // scalar drag force magnitude [N]

            // Drag force vector opposes the relative wind direction, in body frame.
            let drag_body = if speed > 1e-6 {
                -(body_vel / speed) * drag_mag
            } else {
                Vector3::zeros()
            };

            // Lever arm from current CoM to the centre of pressure, body frame.
            // r_lever = r_cp_from_nose - r_com_from_nose (per the design spec).
            // This sim uses Z-UP in world; body +Z = nose direction.
            // cp_z_from_nose is measured aft from nose along body Z, so is a positive number.
            // r_com_from_nose is the nose-to-CoM distance (also positive, nose is forward).
            // Subtracting gives a signed offset: positive = CP is aft of CoM (stable config).
            let r_com_to_nose = self.nose_offset - com_offset;
            let r_lever = r_com_to_nose - Vector3::new(0.0, 0.0, rec.cp_z_from_nose); // TODO: in actual mode, make cp_from_nose a 3d vector

            // M_aero = r_lever × F_drag  (torque about CoM in body frame [N·m])
            // This stays in body frame — Euler's equations operate there directly.
            let aero_moment = r_lever.cross(&drag_body);

            (drag_body, aero_moment)
        } else {
            // Fallback: hard-coded Cd*A per axis (original behaviour).
            // We still use the altitude-corrected rho to improve accuracy vs. before.
            let drag_fallback = Vector3::new(
                -body_vel.x.signum() * 0.5 * rho * 2.0  * 1.2  * body_vel.x.powi(2),
                -body_vel.y.signum() * 0.5 * rho * 2.0  * 1.2  * body_vel.y.powi(2),
                -body_vel.z.signum() * 0.5 * rho * 0.85 * 0.25 * body_vel.z.powi(2),
            );
            (drag_fallback, Vector3::zeros())  // no moment term in the simple model
        };

        // Rotate the body-frame drag into world frame for Newton's second law.
        let drag_world = self.attitude.transform_vector(&drag_body);

        let total_force = outside_forces + (gravity * mass) + self.thrust_vector + slosh_force_world + drag_world;
        self.debug_info.total_force = total_force;
        self.debug_info.thrusts.push(tvc_effect.thrust);
        self.debug_info.slosh_forces.push(slosh_force);
        self.debug_info.aero_drags.push(drag_body);
        self.debug_info.aero_moments.push(aero_moment_body);

        self.accel = total_force / mass;
        self.velocity += self.accel * dt;
        self.position += self.velocity * dt;

        // 2. Rotational Dynamics (Euler's rotation equations)
        // Torque = I * alpha + omega x (I * omega)
        // alpha = I_inv * (Torque - omega x (I * omega))
        
        // We calculate I * omega manually since inertia is a diagonal Vector3 here
        let i_omega = self.moi * self.ang_vel;

        let gyro_torque = self.ang_vel.cross(&i_omega);
        let slosh_lever_arm = self.nitrous_tank_offset - com_offset;
        let slosh_torque = slosh_lever_arm.cross(&slosh_force);
        // aero_moment_body is in body frame — add it directly here alongside TVC and slosh.
        // DO NOT rotate to world frame first; Euler's equations already operate in body frame.
        let net_torque = outside_torques - gyro_torque + tvc_effect.torque + rcs_effect.torque + slosh_torque + aero_moment_body;
        self.debug_info.thrust_torque = tvc_effect.torque;
        self.debug_info.total_torque = net_torque;

        // Angular acceleration (alpha)
        let moi_inv = self.moi.try_inverse().expect("MOI matrix must be invertible");
        let alpha = moi_inv * net_torque;

        // Update Angular Velocity
        self.ang_vel += alpha * dt;

        // 3. Update Attitude (Quaternion Integration)
        // q_dot = 0.5 * quaternion(0, omega) * q
        let omega_quat = Quaternion::new(
            0.0, 
            self.ang_vel.x, 
            self.ang_vel.y, 
            self.ang_vel.z
        );
        
        // Standard approach: q_new = q_old + (0.5 * omega * q_old) * dt
        // nalgebra handles the multiplication logic for us
        let delta_q = UnitQuaternion::from_quaternion(omega_quat); 
        
        // Note: For small timesteps, we approximate the integration.
        // A common robust method is integrating the angle axis directly:
        let angle = self.ang_vel.norm() * dt;
        if angle > 1e-6 {
            let axis = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(self.ang_vel), angle);
            // Apply rotation: new_attitude = old_attitude * delta_rotation
            self.attitude = self.attitude * axis;
        }

        return true; // Indicate successful step
    }

    pub fn get_mass(&self) -> f64 {
        self.get_dry_mass() + self.nitrogen_mass + self.pressurizing_nitrogen_mass + self.nitrous_mass + self.fuel_grain_mass
    }

    pub fn get_dry_mass(&self) -> f64 {
        self.frame_mass + self.nitrogen_tank_empty_mass + self.nitrous_tank_empty_mass + self.tvc_module_empty_mass
    }

    // This is expressed as a vector offset from the frame CoM
    pub fn get_com_offset(&self, thrust_vector: Vector3<f64>) -> Vector3<f64> {
        let total_mass = self.get_mass();

        let nitrogen_com = self.nitrogen_tank_offset;

        let liquid_nitrous_tank_com_offset = Vector3::new(0.0, 0.0, (self.nitrous_level - self.nitrous_tank_length) / 2.0);
        let pressurizing_nitrogen_tank_com_offset = Vector3::new(0.0, 0.0, (self.nitrous_tank_length - self.nitrous_level) / 2.0);
        let nitrous_tank_mass = self.nitrous_tank_empty_mass + self.pressurizing_nitrogen_mass + self.nitrous_mass;
        let nitrous_com = self.nitrous_tank_offset + (self.nitrous_mass / (nitrous_tank_mass)) * liquid_nitrous_tank_com_offset + (self.pressurizing_nitrogen_mass / (nitrous_tank_mass)) * pressurizing_nitrogen_tank_com_offset;
        
        let thrust_direction = thrust_vector.normalize();
        let vertical = Vector3::new(0.0, 0.0, 1.0);
        let rotation = UnitQuaternion::rotation_between(&vertical, &thrust_direction).unwrap_or_else(|| UnitQuaternion::identity());
        let tvc_com = self.frame_com_to_gimbal + rotation * self.gimbal_to_tvc_com;


        let combined_com = (nitrogen_com * (self.nitrogen_tank_empty_mass + self.nitrogen_mass) / total_mass
                            + nitrous_com * (nitrous_tank_mass) / total_mass
                            + tvc_com * (self.tvc_module_empty_mass + self.fuel_grain_mass) / total_mass);
        
        combined_com
    }

    pub fn get_moi(&self, thrust_vector: Vector3<f64>, com_offset: Vector3<f64>) -> Matrix3<f64> {
        let nitrogen_moi = self.transform_moi(self.weight_matrices(self.nitrogen_mass / self.starting_nitrogen_mass, self.wet_nitrogen_moi, self.dry_nitrogen_moi), self.nitrogen_tank_offset + com_offset, self.nitrogen_tank_empty_mass + self.nitrogen_mass);
        
        let liquid_nitrous_tank_com_offset = Vector3::new(0.0, 0.0, (self.nitrous_level - self.nitrous_tank_length) / 2.0);
        let pressurizing_nitrogen_tank_com_offset = Vector3::new(0.0, 0.0, (self.nitrous_tank_length - self.nitrous_level) / 2.0);
        let liquid_nitrous_moi = self.transform_moi(self.get_cylinder_moi(self.nitrous_tank_radius, self.nitrous_tank_length, self.nitrous_mass), -liquid_nitrous_tank_com_offset, self.nitrous_mass);
        let pressurizing_nitrogen_moi = self.transform_moi(self.get_cylinder_moi(self.nitrous_tank_radius, self.nitrous_tank_length, self.pressurizing_nitrogen_mass), -pressurizing_nitrogen_tank_com_offset, self.pressurizing_nitrogen_mass);
        let nitrous_moi = self.transform_moi(self.dry_nitrous_moi, self.nitrous_tank_offset + com_offset, self.nitrous_tank_empty_mass + self.pressurizing_nitrogen_mass + self.nitrous_mass);
        
        let thrust_direction = thrust_vector.normalize();
        let vertical = Vector3::new(0.0, 0.0, 1.0);
        let rotation = UnitQuaternion::rotation_between(&vertical, &thrust_direction).unwrap_or_else(|| UnitQuaternion::identity());
        let rotated_dry_tvc_moi = rotation * self.gimbal_to_tvc_com;
        let tvc_moi = self.transform_moi(self.weight_matrices(self.fuel_grain_mass / self.starting_fuel_grain_mass, self.wet_tvc_moi, self.dry_tvc_moi), com_offset + self.frame_com_to_gimbal + rotated_dry_tvc_moi, self.tvc_module_empty_mass + self.fuel_grain_mass);
        
        
        let moi = self.frame_moi
            + nitrogen_moi
            + nitrous_moi
            + tvc_moi;
        
        moi
    }

    pub fn get_cylinder_moi(&self, radius: f64, height: f64, mass: f64) -> Matrix3<f64> {
        Matrix3::new((mass * (3.0 * radius.powi(2) + height.powi(2))) / 12.0, 0.0, 0.0,
                    0.0, (mass * (3.0 * radius.powi(2) + height.powi(2))) / 12.0, 0.0,
                    0.0, 0.0, 0.5 * mass * radius.powi(2))
    }

    // d is a vector from old center of rotation to new center of rotation
    pub fn transform_moi(&self, moi: Matrix3<f64>, d: Vector3<f64>, mass: f64) -> Matrix3<f64> {
        let y2_z2 = mass * (d.y * d.y + d.z * d.z);
        let x2_z2 = mass * (d.x * d.x + d.z * d.z);
        let x2_y2 = mass * (d.x * d.x + d.y * d.y);
        let neg_xy = -mass * d.x * d.y;
        let neg_xz = -mass * d.x * d.z;
        let neg_yz = -mass * d.y * d.z;

        Matrix3::new(
            moi[(0, 0)] + y2_z2,
            moi[(0, 1)] + neg_xy,
            moi[(0, 2)] + neg_xz,
            moi[(1, 0)] + neg_xy,
            moi[(1, 1)] + x2_z2,
            moi[(1, 2)] + neg_yz,
            moi[(2, 0)] + neg_xz,
            moi[(2, 1)] + neg_yz,
            moi[(2, 2)] + x2_y2,
        )
    }

    pub fn weight_matrices(&self, weight: f64, matrix1: Matrix3<f64>, matrix2: Matrix3<f64>) -> Matrix3<f64> {
        weight * matrix1 + (1.0 - weight) * matrix2
    }

    pub fn weight_vectors(&self, weight: f64, vector1: Vector3<f64>, vector2: Vector3<f64>) -> Vector3<f64> {
        weight * vector1 + (1.0 - weight) * vector2
    }

    // State vector: [x, y, z, qx, qy, qz, qw, x_dot, y_dot, z_dot, wx, wy, wz]
    // This is formatted for use in the MPC controller
    pub fn get_state(&self) -> Array1<f64> {
        let mut state = Array1::zeros(13);
        state[0] = self.position.x;
        state[1] = self.position.y;
        state[2] = self.position.z;
        state[3] = self.attitude.i;
        state[4] = self.attitude.j;
        state[5] = self.attitude.k;
        state[6] = self.attitude.w;
        state[7] = self.velocity.x;
        state[8] = self.velocity.y;
        state[9] = self.velocity.z;
        state[10] = self.ang_vel.x;
        state[11] = self.ang_vel.y;
        state[12] = self.ang_vel.z;

        state
    }

    pub fn get_moi_mpc(&self) -> Array2<f64> {
        array![
            [self.moi[(0, 0)], self.moi[(0, 1)], self.moi[(0, 2)]],
            [self.moi[(1, 0)], self.moi[(1, 1)], self.moi[(1, 2)]],
            [self.moi[(2, 0)], self.moi[(2, 1)], self.moi[(2, 2)]],
        ]
    }

    pub fn save_debug_to_csv(&self, file_path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::create(file_path)?;
        let mut wtr = csv::Writer::from_writer(file);

        // 1. Write the flattened CSV Header
        wtr.write_record(&[
            "time",
            // COM Offset (Vector3)
            "com_x", "com_y", "com_z",
            // MOI Tensor (Matrix3 - Symmetric 6 values)
            "i_xx", "i_yy", "i_zz", "i_xy", "i_xz", "i_yz",
            // Thrust (Vector3)
            "thrust_x", "thrust_y", "thrust_z",
            // Slosh Force (Vector3)
            "slosh_x", "slosh_y", "slosh_z",
            // Thermodynamic & Mass Scalars
            "nitrous_m_dot", "valve_angle", "chamber_pressure",
            "of_ratio", "isp", "cstar", "port_d",
            "fuel_mass", "nitrous_mass", "n2_tank_mass", "n2o_tank_mass",
            // Wind velocity [m/s], world frame
            "wind_x", "wind_y", "wind_z",
            // Aero Drag [N], body frame
            "aero_drag_x", "aero_drag_y", "aero_drag_z",
            // Aero Moment [N*m], body frame
            "aero_moment_x", "aero_moment_y", "aero_moment_z"
        ])?;

        let num_records = self.debug_info.times.len();

        // 2. Loop through every time step and write the row
        for i in 0..num_records {
            // Safety check: Prevents a crash if one array forgot to push a value during a tick!
            if i >= self.debug_info.com_offsets.len() || i >= self.debug_info.mois.len() || i >= self.debug_info.thrusts.len() {
                eprintln!("Warning: Data arrays out of sync at row {}. Stopping export early.", i);
                break;
            }

            let com = self.debug_info.com_offsets[i];
            let moi = self.debug_info.mois[i];
            let thrust = self.debug_info.thrusts[i];
            let slosh = self.debug_info.slosh_forces[i];
            let aero_drag = self.debug_info.aero_drags[i];
            let aero_moment = self.debug_info.aero_moments[i];

            // 3. Write the row data
            wtr.write_record(&[
                self.debug_info.times[i].to_string(),
                
                com.x.to_string(), com.y.to_string(), com.z.to_string(),
                
                moi[(0,0)].to_string(), moi[(1,1)].to_string(), moi[(2,2)].to_string(),
                moi[(0,1)].to_string(), moi[(0,2)].to_string(), moi[(1,2)].to_string(),
                
                thrust.x.to_string(), thrust.y.to_string(), thrust.z.to_string(),
                
                slosh.x.to_string(), slosh.y.to_string(), slosh.z.to_string(),
                
                self.debug_info.nitrous_m_dots[i].to_string(),
                self.debug_info.valve_angles[i].to_string(),
                self.debug_info.chamber_pressures[i].to_string(),
                self.debug_info.of_ratios[i].to_string(),
                self.debug_info.isps[i].to_string(),
                self.debug_info.cstars[i].to_string(),
                self.debug_info.port_ds[i].to_string(),
                self.debug_info.fuel_masses[i].to_string(),
                self.debug_info.nitrous_masses[i].to_string(),
                self.debug_info.nitrogen_n2_tank_masses[i].to_string(),
                self.debug_info.nitrogen_n2o_tank_masses[i].to_string(),
                self.debug_info.wind_vels[i].x.to_string(),
                self.debug_info.wind_vels[i].y.to_string(),
                self.debug_info.wind_vels[i].z.to_string(),
                aero_drag.x.to_string(),
                aero_drag.y.to_string(),
                aero_drag.z.to_string(),
                aero_moment.x.to_string(),
                aero_moment.y.to_string(),
                aero_moment.z.to_string(),
            ])?;
        }

        wtr.flush()?;
        println!("Successfully exported {} rows of telemetry to '{}'", num_records, file_path);
        
        Ok(())
    }
}