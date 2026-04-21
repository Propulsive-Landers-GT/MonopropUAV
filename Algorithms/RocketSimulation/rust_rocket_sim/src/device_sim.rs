use nalgebra::{Vector3, UnitQuaternion};
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

#[derive(Debug, Clone)]
pub struct RefreshUpdater {
    pub overhead: f64, // overhead on startup of process, in seconds
    pub overhead_sigma: f64, // standard deviation of overhead, in seconds
    pub iteration_time: f64, // time each iteration takes, in seconds
    pub iteration_sigma: f64, // standard deviation of iteration time, in seconds
    pub iterations: f64,
    pub start_time: f64,
    current_overhead: f64,
    current_total_iteration_time: f64,
}

impl RefreshUpdater {
    pub fn new(overhead: f64, overhead_sigma: f64, iteration_time: f64, iteration_sigma: f64) -> Self {
        let mut refresh_updater = Self {
            overhead,
            overhead_sigma,
            iteration_time,
            iteration_sigma,
            iterations: 1.0,
            start_time: 0.0,
            current_overhead: overhead,
            current_total_iteration_time: iteration_time,
        };

        refresh_updater.reset(1.0, 0.0);

        refresh_updater
    }

    pub fn reset(&mut self, iterations: f64, system_time: f64) {
        self.iterations = iterations;
        self.start_time = system_time;

        let mut rng = rand::rng();
        self.current_overhead = self.overhead + noise_1_d(self.overhead_sigma, &mut rng);
        self.current_total_iteration_time = self.iteration_time * self.iterations + noise_1_d(self.iteration_sigma * (self.iterations as f64).sqrt(), &mut rng);
    }

    pub fn update(&mut self, system_time: f64) -> bool {
        if system_time - self.start_time > self.current_overhead + self.current_total_iteration_time {
            return true;
        } else {
            return false;
        }
    }
}

/// A simple on/off valve used throughout the propellant feed system.
/// These model actuated valves (solenoid or pneumatic) that are either
/// fully open or fully closed — no partial opening or transition dynamics.
///
/// Valves in the system:
///   - fill_mv : Fill valve (entry to lander tanks, irrelevant in flight)
///   - r_mv    : Regulator isolation valve (N2 storage → N2O run tank)
///   - rcs1_mv : RCS thruster pair 1 (controls one roll direction)
///   - rcs2_mv : RCS thruster pair 2 (controls opposite roll direction)
///   - o_iso   : Oxidizer isolation valve (N2O run tank → MTV/engine)
///   - o_vnt   : Oxidizer vent valve (relieves run tank pressure)
///
/// TODO: Consider integrating valve commands into the control_input vector
///       when we want the GNC algorithms to drive these valves directly.
#[derive(Debug, Clone)]
pub struct Valve {
    pub is_open: bool,
}

impl Valve {
    pub fn new(is_open: bool) -> Self {
        Self { is_open }
    }

    pub fn open(&mut self) {
        self.is_open = true;
    }

    pub fn close(&mut self) {
        self.is_open = false;
    }
}


/*
Acceleration units are in m/s^2
Gyro units are in rad/s
Mag units are in teslas
*/
#[derive(Debug, Clone)]
pub struct IMU {
    pub accel_noise_sigma: Vector3<f64>,
    pub accel_offset: Vector3<f64>,

    pub gyro_noise_sigma: Vector3<f64>,
    pub gyro_drift: Vector3<f64>,

    pub mag_noise_sigma: Vector3<f64>,
    pub mag_offset: Vector3<f64>,
    pub earth_mag: Vector3<f64>,

    pub last_reading: IMUReading,

    pub update_rate: f64, // This is in updates per second
    system_time: f64,
}

// These readings are in the body frame
#[derive(Debug, Clone)]
pub struct IMUReading {
    pub accel: Vector3<f64>,
    pub gyro: Vector3<f64>,
    pub mag: Vector3<f64>,
}

impl IMU {
    pub fn new(accel_noise_sigma: Vector3<f64>, accel_offset: Vector3<f64>, gyro_noise_sigma: Vector3<f64>, gyro_drift: Vector3<f64>, mag_noise_sigma: Vector3<f64>, mag_offset: Vector3<f64>, earth_mag: Vector3<f64>, update_rate: f64) -> Self {
        Self {
            accel_noise_sigma,
            accel_offset,
            gyro_noise_sigma,
            gyro_drift,
            mag_noise_sigma,
            mag_offset,
            earth_mag,
            last_reading: IMUReading {
                accel: Vector3::zeros(),
                gyro: Vector3::zeros(),
                mag: Vector3::zeros(),
            },
            update_rate,
            system_time: 0.0,
        }
    }

    pub fn default() -> Self {
        let imu_accel_noise_sigma = Vector3::new(0.00137, 0.00137, 0.00137);
        let imu_accel_offset = Vector3::new(0.0, 0.0, 0.0);
        let imu_gyro_noise_sigma = Vector3::new(0.000061, 0.000061, 0.000061    );
        let imu_gyro_drift = Vector3::new(0.0, 0.0, 0.0);
        let imu_mag_noise_sigma = Vector3::new(14.0e-9, 14.0e-9, 14.0e-9);
        let imu_mag_offset = Vector3::new(0.0, 0.0, 0.0);
        let imu_earth_magnetic_field = Vector3::new(
                                                    -0.0000020,  // -2.0 microTesla East
                                                    0.0000220,  // +22.0 microTesla North
                                                    -0.0000443); // -44.3 microTesla Up (Pointing Down!)
        let imu_update_rate = 300.0;

        Self::new(imu_accel_noise_sigma, imu_accel_offset, imu_gyro_noise_sigma, imu_gyro_drift, imu_mag_noise_sigma, imu_mag_offset, imu_earth_magnetic_field, imu_update_rate)
    }

    pub fn update(&mut self, accel: Vector3<f64>, ang_vel: Vector3<f64>, attitude: UnitQuaternion<f64>, system_time: f64) -> IMUReading {
        let elapsed_time = system_time - self.system_time;
        if elapsed_time < 1.0 / self.update_rate {
            return self.last_reading.clone();
        }

        let dt_sqrt = elapsed_time.sqrt();

        self.system_time = system_time;

        let mut rng = rand::rng();

        // 1. ACCELEROMETER
        // Accelerometers measure "Proper Acceleration" (Acceleration - Gravity).
        // If the rocket is sitting on the pad, accel is 0, but the sensor feels 1g UP.
        // We assume Gravity is -9.81 in Z.
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let proper_accel = accel - gravity;
        
        // Rotate into Body Frame: q_inverse * vector
        let proper_accel_body = attitude.inverse_transform_vector(&proper_accel);
        
        // Add Bias + Noise
        let measured_accel = proper_accel_body 
            + self.accel_offset 
            + noise_3_d(&(self.accel_noise_sigma / dt_sqrt), &mut rng);


        // 2. GYROSCOPE
        // Gyros measure angular velocity directly in the Body Frame.
        // Usually no rotation needed if input is already body rates.
        let ang_vel_body = attitude.inverse_transform_vector(&ang_vel);

        let measured_gyro = ang_vel_body 
            + self.gyro_drift 
            + noise_3_d(&(self.gyro_noise_sigma / dt_sqrt), &mut rng);


        // 3. MAGNETOMETER
        // Rotates the known Earth magnetic vector (World) into the Body frame.
        let mag_body = attitude.inverse_transform_vector(&self.earth_mag);
        
        let measured_mag = mag_body 
            + self.mag_offset 
            + noise_3_d(&(self.mag_noise_sigma / dt_sqrt), &mut rng);

        self.last_reading = IMUReading {
            accel: measured_accel,
            gyro: measured_gyro,
            mag: measured_mag,
        };

        self.last_reading.clone()
    }
}


#[derive(Debug, Clone)]
pub struct GPS {
    pub pos_noise_sigma: Vector3<f64>,
    pub pos_offset: Vector3<f64>,
    pub last_reading: GPSReading,

    pub update_rate: f64, // This is in updates per second
    system_time: f64,
}

#[derive(Debug, Clone)]
pub struct GPSReading {
    pub position: Vector3<f64>,
}

impl GPS {
    pub fn new(pos_noise_sigma: Vector3<f64>, pos_offset: Vector3<f64>, update_rate: f64) -> Self {
        Self {
            pos_noise_sigma,
            pos_offset,
            last_reading: GPSReading {
                position: Vector3::zeros(),
            },
            update_rate,
            system_time: 0.0,
        }
    }

    pub fn default() -> Self {
        let gps_position_noise_sigma = Vector3::zeros();
        let gps_position_offset = Vector3::zeros();
        let gps_update_rate = 5.0;

        Self::new(gps_position_noise_sigma, gps_position_offset, gps_update_rate)
    }

    pub fn update(&mut self, position: Vector3<f64>, system_time: f64) -> GPSReading {
        let elapsed_time = system_time - self.system_time;
        if elapsed_time < 1.0 / self.update_rate {
            return self.last_reading.clone();
        }
        
        self.system_time = system_time;

        let mut rng = rand::rng();

        let noisy_position = position + self.pos_offset + noise_3_d(&self.pos_noise_sigma, &mut rng);

        self.last_reading = GPSReading {
            position: noisy_position,
        };

        self.last_reading.clone()
    }
}


#[derive(Debug, Clone)]
pub struct UWB {
    pub pos_noise_sigma: Vector3<f64>,
    pub pos_offset: Vector3<f64>,
    pub origin: Vector3<f64>,
    pub range: f64,
    pub last_reading: UWBReading,

    pub update_rate: f64, // This is in updates per second
    system_time: f64,
}

#[derive(Debug, Clone)]
pub struct UWBReading {
    pub position: Vector3<f64>,
}

impl UWB {
    pub fn new(pos_noise_sigma: Vector3<f64>, pos_offset: Vector3<f64>, origin: Vector3<f64>, range: f64, update_rate: f64) -> Self {
        Self {
            pos_noise_sigma,
            pos_offset,
            origin,
            range,
            last_reading: UWBReading {
                position: origin.clone(),
            },
            update_rate,
            system_time: 0.0,
        }
    }

    pub fn default() -> Self {
        let uwb_position_noise_sigma = Vector3::zeros();
        let uwb_position_offset = Vector3::zeros();
        let uwb_origin = Vector3::zeros();
        let uwb_range = 8.0;
        let uwb_update_rate = 5.0;

        Self::new(uwb_position_noise_sigma, uwb_position_offset, uwb_origin, uwb_range, uwb_update_rate)
    }

    pub fn update(&mut self, position: Vector3<f64>, system_time: f64) -> UWBReading {
        let elapsed_time = system_time - self.system_time;
        if elapsed_time < 1.0 / self.update_rate {
            return self.last_reading.clone();
        }
        
        self.system_time = system_time;

        if (position - self.origin).norm() > self.range {
            // Out of range behavior simply returns the last valid read. Check with Avionics for actual behavior
            return self.last_reading.clone();
        } else {
            let mut rng = rand::rng();

            let noisy_position = position + self.pos_offset + noise_3_d(&self.pos_noise_sigma, &mut rng);

            self.last_reading = UWBReading {
                position: noisy_position,
            };

            return self.last_reading.clone();
        }
    }
}


#[derive(Debug, Clone)]
pub struct TVCActuator {
    position: f64,
    velocity: f64,
    accel: f64,

    pub extension_limit: f64,
    pub unloaded_speed: f64,
    pub stall_force: f64,
    pub p_gain: f64,

    pub pos_noise_sigma: f64,

    pub update_rate: f64, // This is in updates per second
    system_time: f64,
}

impl TVCActuator {
    pub fn new (start_position: f64, extension_limit: f64, unloaded_speed: f64, stall_force: f64, p_gain: f64, pos_noise_sigma: f64, update_rate: f64) -> Self {
        Self {
            position: start_position,
            velocity: 0.0,
            accel: 0.0,
            extension_limit,
            unloaded_speed,
            stall_force,
            p_gain,
            pos_noise_sigma,
            update_rate,
            system_time: 0.0,
        }
    }

    pub fn default() -> Self {
        let actuator_start_position = 0.0;
        let actuator_extension_limit = 0.08;
        let acuator_unloaded_speed = 0.14986;
        let actuator_stall_force = 102.06;
        let actuator_p_gain = 10.0;
        let actuator_pos_noise_sigma = 0.0;
        let actuator_update_rate = 200.0;

        Self::new(actuator_start_position, actuator_extension_limit, acuator_unloaded_speed, actuator_stall_force, actuator_p_gain, actuator_pos_noise_sigma, actuator_update_rate)
    }

    pub fn update(&mut self, target_position: f64, load_force: f64, dt: f64, system_time: f64) {
        let elapsed_time = system_time - self.system_time;
        if elapsed_time < 1.0 / self.update_rate {
            // velocity stays constant, therefore acceleration is zero, but position does update
            self.accel = 0.0;
            self.position += self.velocity * dt;
            self.position = self.position.clamp(0.0, self.extension_limit);
            return;
        }
        
        self.system_time = system_time;

        let error = target_position - self.position;
        let command = error * self.p_gain;

        let load_factor = (load_force.abs() / self.stall_force).min(1.0);
        let speed_limit = self.unloaded_speed * (1.0 - load_factor);

        let prev_velocity = self.velocity;
        self.velocity = command.clamp(-speed_limit, speed_limit);
        self.accel = (self.velocity - prev_velocity) / dt;
        self.position += self.velocity * dt;
        self.position = self.position.clamp(0.0, self.extension_limit);
    }

    pub fn get_accel(&self) -> f64 {
        self.accel
    }

    pub fn get_noisy_position(&self) -> f64 {
        let mut rng = rand::rng();
        self.position + noise_1_d(self.pos_noise_sigma, &mut rng)
    }
}

#[derive(Debug, Clone)]
pub struct MTV {
    pub angle: f64,
    pub ang_vel: f64,
    pub ang_accel: f64,

    pub unloaded_speed: f64,
    pub stall_torque: f64,
    pub p_gain: f64,

    pub valve_torque: f64,

    starting_fuel_grain_mass: f64,

    pub update_rate: f64,
    system_time: f64,
}

#[derive(Debug, Clone)]
pub struct MTVEffect {
    pub nitrogen_mass: f64,
    pub pressurizing_nitrogen_mass: f64,
    pub nitrous_mass: f64,
    pub fuel_grain_mass: f64,
    pub thrust: Vector3<f64>,
}

impl MTV {
    pub fn new(angle: f64, ang_vel: f64, ang_accel: f64, unloaded_speed: f64, stall_torque: f64, p_gain: f64, valve_torque: f64, starting_fuel_grain_mass: f64, update_rate: f64) -> Self {
        Self {
            angle,
            ang_vel,
            ang_accel,
            unloaded_speed,
            stall_torque,
            p_gain,
            valve_torque,
            starting_fuel_grain_mass,
            update_rate,
            system_time: 0.0,
        }
    }

    pub fn default(starting_fuel_grain_mass: f64) -> Self {
        let mtv_angle = 60.0;
        let mtv_ang_vel = 0.0;
        let mtv_ang_accel = 0.0;
        let mtv_unloaded_speed = 60.0;
        let mtv_stall_torque = 70.0;
        let mtv_p_gain = 1000.0;
        let mtv_valve_torque = 30.0;
        let mtv_update_rate = 200.0;

        Self::new(mtv_angle, mtv_ang_vel, mtv_ang_accel, mtv_unloaded_speed, mtv_stall_torque, mtv_p_gain, mtv_valve_torque, starting_fuel_grain_mass, mtv_update_rate)
    }

    pub fn update(&mut self, target_thrust: f64, nitrogen_mass: f64, pressurizing_nitrogen_mass: f64, nitrous_mass: f64, fuel_grain_mass: f64, dt: f64, system_time: f64) -> MTVEffect {
        let elapsed_time = system_time - self.system_time;
        if elapsed_time < 1.0 / self.update_rate {
            // angular velocity stays constant, therefore angular acceleration is zero, but angle does update
            self.ang_accel = 0.0;
            self.angle += self.ang_vel * dt;
            self.angle = nalgebra::clamp(self.angle, 0.0, 90.0);
        } else {

            self.system_time = system_time;

            let target_angle = self.get_target_angle(target_thrust);
            self.angle = target_angle; // TODO: For now, we will assume the MTV can achieve the target angle within one time step. Replace with actual dynamics later.

            let error = target_angle - self.angle;
            let command = error * self.p_gain;

            let load_factor = (self.valve_torque / self.stall_torque).min(1.0);
            let speed_limit = self.unloaded_speed * (1.0 - load_factor);

            let prev_ang_vel = self.ang_vel;
            self.ang_vel = command.clamp(-speed_limit, speed_limit);
            self.ang_accel = (self.ang_vel - prev_ang_vel) / dt;
            self.angle += self.ang_vel * dt;
            self.angle = nalgebra::clamp(self.angle, 0.0, 90.0);
        }

        let mtv_effect = self.get_thrust(nitrogen_mass, pressurizing_nitrogen_mass, nitrous_mass, fuel_grain_mass, dt);

        return mtv_effect;
    }

    pub fn get_target_angle(&mut self, target_thrust: f64) -> f64 {
        // TODO: implement target angle calculation based on thrust
        
        // y = 0.1 * e^((x - 300) / 300) - 0.05
        // This is the reversal of the example thrust curve. Replace with actual relation later.
        let target_angle = 90.0 * (0.1 * ((target_thrust - 300.0) / 300.0).exp() - 0.05);
        println!("Target Thrust: {}, Target Angle: {}", target_thrust, target_angle);
        target_angle
    }

    pub fn get_thrust(&mut self, nitrogen_mass:f64, pressurizing_nitrogen_mass: f64, nitrous_mass: f64, fuel_grain_mass: f64, dt: f64) -> MTVEffect {
        // TODO: update the nitrogen, nitrous, and fuel grain masses (nitrogen is used to pressurize the tank, so nitrogen will flow out of the nitrogen tank into the nitrous tank)
        // TODO: use the current throttle angle to find the flow rate, then determine thrust
        // Remember to take into account either mass running out
        // TODO: also model the thrust decay somehow?

        if nitrous_mass <= 0.0 || fuel_grain_mass <= 0.0 {
            return MTVEffect {
                nitrogen_mass,
                pressurizing_nitrogen_mass,
                nitrous_mass,
                fuel_grain_mass,
                thrust: Vector3::zeros(),
            };
        }

        // y = 300 * ln((x + 0.05) / 0.1) + 300 is an example thrust curve. Replace later with true relation
        let thrust = 300.0 * ((self.angle / 90.0 + 0.05) / 0.1).ln() + 300.0;
        println!("MTV Angle: {}, Thrust: {}", self.angle, thrust);
        // let nitrous_alpha = 1.0/(9.81 * 180.0);
        // let nitrogen_alpha = 1.0/(9.81 * 180.0);
        // let fuel_grain_alpha = 1.0/(9.81 * 180.0);
        let nitrous_alpha = 0.0;
        let nitrogen_alpha = 0.0;
        let fuel_grain_alpha = 0.0;

        let new_nitrogen_mass = (nitrogen_mass - nitrogen_alpha * thrust * dt).max(0.0);
        let new_pressurizing_nitrogen_mass = (pressurizing_nitrogen_mass + nitrogen_alpha * thrust * dt).max(0.0);
        let new_nitrous_mass = (nitrous_mass - nitrous_alpha * thrust * dt).max(0.0);
        let new_fuel_grain_mass = (fuel_grain_mass - fuel_grain_alpha * thrust * dt).max(0.0);


        MTVEffect {
            nitrogen_mass: new_nitrogen_mass,
            pressurizing_nitrogen_mass: new_pressurizing_nitrogen_mass,
            nitrous_mass: new_nitrous_mass,
            fuel_grain_mass: new_fuel_grain_mass,
            thrust: Vector3::new(0.0, 0.0, thrust),
        }
    }
}

// TODO: Refine the inertia calculations/usage
#[derive(Debug, Clone)]
pub struct TVC {
    pub mtv: MTV,
    pub x_actuator: TVCActuator,
    pub y_actuator: TVCActuator,
    ang_vel: Vector3<f64>,
    ang_accel: Vector3<f64>,
    tvc_torque: Vector3<f64>,
    pub actuator_lever_arm: f64,
    pub max_fuel_inertia: f64,
    pub min_fuel_inertia: f64,
    starting_fuel_grain_mass: f64,
}

#[derive(Debug, Clone)]
pub struct TVCEffect {
    pub thrust: Vector3<f64>,
    pub torque: Vector3<f64>,
    pub nitrogen_mass: f64,
    pub pressurizing_nitrogen_mass: f64,
    pub nitrous_mass: f64,
    pub fuel_grain_mass: f64,
}

#[derive(Debug, Clone)]
pub struct ActuatorPositions {
    pub x_position: f64,
    pub y_position: f64,
}

impl TVC {
    // Our actuators appear to have a stall force of 400 lbf (need to convert to metric) and an unloaded speed of 4.777173913 in/s (also convert)
    pub fn new(mtv: MTV, x_actuator: TVCActuator, y_actuator: TVCActuator, actuator_lever_arm: f64, max_fuel_inertia: f64, min_fuel_inertia: f64, starting_fuel_grain_mass: f64) -> Self {
        Self {
            mtv,
            x_actuator,
            y_actuator,
            ang_vel: Vector3::zeros(),
            ang_accel: Vector3::zeros(),
            tvc_torque: Vector3::zeros(),
            actuator_lever_arm,
            max_fuel_inertia,
            min_fuel_inertia,
            starting_fuel_grain_mass,
        }
    }

    pub fn default(starting_fuel_grain_mass: f64) -> Self {
        let mtv = MTV::default(starting_fuel_grain_mass);
        let x_actuator = TVCActuator::default();
        let y_actuator = TVCActuator::default();
        let tvc_actuator_lever_arm = 0.1;
        // TODO: Whenever properly implementing TVC, fix these values
        let tvc_max_fuel_inertia = 3.0;
        let tvc_min_fuel_inertia = 8.0;

        Self::new(mtv, x_actuator, y_actuator, tvc_actuator_lever_arm, tvc_max_fuel_inertia, tvc_min_fuel_inertia, starting_fuel_grain_mass)
    }

    // engine is 11 kg with fuel, awaiting empty mass data
    // Returns the reaction torque applied to the rocket body
    // only the first two elements of command are used, the third is thrust
    pub fn update(&mut self, command: Vector3<f64>, tvc_lever_arm: Vector3<f64>, nitrogen_mass: f64, pressurizing_nitrogen_mass: f64, nitrous_mass: f64, fuel_grain_mass: f64, dt: f64, system_time: f64) -> TVCEffect {
        let mtv_effect = self.mtv.update(command[2], nitrogen_mass, pressurizing_nitrogen_mass,nitrous_mass, fuel_grain_mass, dt, system_time);

        let engine_inertia = self.min_fuel_inertia + (self.max_fuel_inertia - self.min_fuel_inertia) * (fuel_grain_mass / self.starting_fuel_grain_mass);
        let x_load = (engine_inertia * self.x_actuator.get_accel()) / self.actuator_lever_arm.powi(2);
        let y_load = (engine_inertia * self.y_actuator.get_accel()) / self.actuator_lever_arm.powi(2);

        // TODO: translate gimbal angle commands to actuator target positions and update ang vels and ang accel
        let x_target = 0.0;
        let y_target = 0.0;
        let x_angle = command[0];
        let y_angle = command[1];

        self.x_actuator.update(x_target, x_load, dt, system_time);
        self.y_actuator.update(y_target, y_load, dt, system_time);

        let reaction_torque = self.ang_accel * engine_inertia * -1.0;
        
        // command[0] = theta (MPC thinks this creates X thrust)
        // To create X thrust, the nozzle must swing along the X axis.
        // To swing along the X axis, we must rotate AROUND the Y axis!
        let y_rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), x_angle);

        // command[1] = phi (MPC thinks this creates Y thrust)
        // To create Y thrust, we must rotate AROUND the X axis!
        // (Note: We use -command[1] because right-hand rule around X positive swings the nozzle -Y)
        let x_rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -y_angle);

        // Combine them (Order matters! Match your physical gimbal design)
        let gimbal_rotation = y_rot * x_rot;

        let thrust_vector = gimbal_rotation.transform_vector(&mtv_effect.thrust);
        let thrust_torque = tvc_lever_arm.cross(&thrust_vector);

        self.tvc_torque = reaction_torque + thrust_torque;

        return TVCEffect {
            thrust: thrust_vector,
            torque: self.tvc_torque.clone(),
            nitrogen_mass: mtv_effect.nitrogen_mass,
            pressurizing_nitrogen_mass: mtv_effect.pressurizing_nitrogen_mass,
            nitrous_mass: mtv_effect.nitrous_mass,
            fuel_grain_mass: mtv_effect.fuel_grain_mass,
        };
    }

    pub fn get_noisy_actuator_positions(&self) -> ActuatorPositions {
        return ActuatorPositions {
            x_position: self.x_actuator.get_noisy_position(),
            y_position: self.y_actuator.get_noisy_position(),
        };
    }

    pub fn get_chamber_pressure(&self) -> f64 {
        // TODO: implement chamber pressure calculation
        0.0
    }
}


#[derive(Debug, Clone)]
pub struct RCS {
    pub thrust: f64, // this is per thruster
    pub lever_arm: f64,
    last_command: f64,

    pub nitrogen_consumption_rate: f64, // this is in kg/s

    pub update_rate: f64, // This is in updates per second
    system_time: f64,
}

#[derive(Debug, Clone)]
pub struct RCSEffect {
    pub torque: Vector3<f64>,
    pub nitrogen_mass: f64,
}

impl RCS {
    pub fn new(thrust: f64, lever_arm: f64, nitrogen_consumption_rate: f64, update_rate: f64) -> Self{
        Self {
            thrust,
            lever_arm,
            last_command: 0.0,
            nitrogen_consumption_rate,
            update_rate,
            system_time: 0.0,
        }
    }

    pub fn default() -> Self {
        let rcs_thrust = 10.0;
        let rcs_lever_arm = 0.15;
        let rcs_nitrogen_consumption_rate = 0.1;
        let rcs_update_rate = 10.0;

        Self::new(rcs_thrust, rcs_lever_arm, rcs_nitrogen_consumption_rate, rcs_update_rate)
    }

    // Each RCS valve controls a pair of thrusters for one roll direction.
    //   rcs1_mv open → clockwise roll  (negative Z torque)
    //   rcs2_mv open → counter-clockwise roll (positive Z torque)
    //
    // Both valves can be open simultaneously (each is independently controlled).
    // When both are open the torques cancel but N2 is consumed by both pairs.
    pub fn update(&mut self, rcs1_open: bool, rcs2_open: bool, nitrogen_mass: f64, dt: f64, system_time: f64) -> RCSEffect {
        let mut remaining_nitrogen_mass = nitrogen_mass;

        let elapsed_time = system_time - self.system_time;
        // Determine the effective valve states — hold previous state between update ticks
        // Encoding: 2.0 = both open, 1.0 = rcs1 only, -1.0 = rcs2 only, 0.0 = neither
        let (active_rcs1, active_rcs2) = if elapsed_time < 1.0 / self.update_rate {
            // Between update ticks: hold previous command
            let prev_rcs1 = self.last_command == 1.0 || self.last_command == 2.0;
            let prev_rcs2 = self.last_command == -1.0 || self.last_command == 2.0;
            (prev_rcs1, prev_rcs2)
        } else {
            self.system_time = system_time;
            // Encode the current valve state into last_command for hold logic
            if rcs1_open && rcs2_open {
                self.last_command = 2.0;
            } else if rcs1_open {
                self.last_command = 1.0;
            } else if rcs2_open {
                self.last_command = -1.0;
            } else {
                self.last_command = 0.0;
            }
            (rcs1_open, rcs2_open)
        };
        
        let torque_per_pair = 2.0 * self.thrust * self.lever_arm;
        let mut torque = Vector3::zeros();

        if active_rcs1 {
            // rcs1_mv open → Negative Z Torque (clockwise from top view)
            torque.z -= torque_per_pair;
            remaining_nitrogen_mass -= self.nitrogen_consumption_rate * dt;
        }
        if active_rcs2 {
            // rcs2_mv open → Positive Z Torque (counter-clockwise from top view)
            torque.z += torque_per_pair;
            remaining_nitrogen_mass -= self.nitrogen_consumption_rate * dt;
        }

        RCSEffect {
            torque,
            nitrogen_mass: remaining_nitrogen_mass,
        }
    }
}

/// A simulated pressure transducer (PT) that measures static pressure at a
/// specific point in the propellant feed system.
///
/// Unlike inertial sensors (IMU) or radio-based sensors (GPS, UWB), pressure
/// transducers are direct-measurement devices and are extremely stable.
/// Aerospace piezoresistive PTs typically achieve ±0.1–0.5% of Full Scale
/// Output (FSO) total accuracy, with the dominant error source being a fixed
/// calibration offset (bias) rather than random noise.  For a 100 bar FS
/// sensor, actual measurement noise is on the order of ±0.005 bar — effectively
/// negligible step-to-step.
///
/// Each PT outputs a scalar pressure reading [bar] with a small Gaussian noise
/// component and an optional fixed bias.  A configurable update rate models
/// the real sensor's sample-and-hold behaviour.
///
/// PTs on this vehicle:
///   - m2_pt  : Downstream of the MTV, before the check valve.
///              Reads the line pressure between the throttle valve and engine.
///   - o_pt   : Immediately downstream of the N2O run tank, before o_vnt.
///              Used by the vent controller to decide whether/how long to
///              open the oxidizer vent valve.
///   - oa_pt  : Upstream of the N2O run tank, downstream of r_mv.
///              Reads the regulated nitrogen supply pressure entering the
///              run tank.
#[derive(Debug, Clone)]
pub struct PressureTransducer {
    /// Human-readable label for logging (e.g. "m2-pt")
    pub label: String,
    /// Standard deviation of the Gaussian measurement noise [bar].
    /// Typical value for a 100 bar FS piezoresistive PT is ~0.005 bar.
    pub noise_sigma: f64,
    /// Fixed measurement bias (calibration offset) [bar]
    pub bias: f64,
    /// Most recent output (held between update ticks)
    pub last_reading: PTReading,
    /// Sensor sample rate [Hz] — piezoresistive PTs commonly run at 1 kHz+
    pub update_rate: f64,
    /// Internal clock tracking when the last update was issued
    system_time: f64,
}

/// A single pressure transducer reading.
#[derive(Debug, Clone)]
pub struct PTReading {
    /// Measured pressure [bar], including noise and bias
    pub pressure_bar: f64,
}

impl PressureTransducer {
    pub fn new(label: &str, noise_sigma: f64, bias: f64, update_rate: f64) -> Self {
        Self {
            label: label.to_string(),
            noise_sigma,
            bias,
            last_reading: PTReading { pressure_bar: 0.0 },
            update_rate,
            system_time: 0.0,
        }
    }

    /// Convenience constructor with realistic defaults for a 0–100 bar
    /// aerospace piezoresistive PT:
    ///   - 0.005 bar noise sigma  (~0.005% FS — essentially negligible)
    ///   - Zero bias
    ///   - 1000 Hz sample rate
    pub fn default_with_label(label: &str) -> Self {
        Self::new(label, 0.005, 0.0, 1000.0)
    }

    /// Feed the sensor the true local pressure.  Returns a (possibly stale)
    /// reading.  Between update ticks the previous value is returned unchanged,
    /// matching sample-and-hold hardware behaviour.
    pub fn update(&mut self, true_pressure_bar: f64, system_time: f64) -> PTReading {
        let elapsed = system_time - self.system_time;
        if elapsed < 1.0 / self.update_rate {
            return self.last_reading.clone();
        }

        self.system_time = system_time;

        let mut rng = rand::rng();
        let noise = noise_1_d(self.noise_sigma, &mut rng);

        self.last_reading = PTReading {
            pressure_bar: true_pressure_bar + self.bias + noise,
        };

        self.last_reading.clone()
    }
}


// Helper to generate 3D noise
pub fn noise_3_d(sigma: &Vector3<f64>, rng: &mut ThreadRng) -> Vector3<f64> {
    let n_x = Normal::new(0.0, sigma.x).unwrap().sample(rng);
    let n_y = Normal::new(0.0, sigma.y).unwrap().sample(rng);
    let n_z = Normal::new(0.0, sigma.z).unwrap().sample(rng);
    Vector3::new(n_x, n_y, n_z)
}

// Helper to generate 3D noise
pub fn noise_1_d(sigma: f64, rng: &mut ThreadRng) -> f64 {
    Normal::new(0.0, sigma).unwrap().sample(rng)
}