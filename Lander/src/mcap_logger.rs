use std::fs::File;
use std::io::BufWriter;
use std::collections::BTreeMap;
use mcap::{Writer, records::MessageHeader};
use crate::state::{SensorData, VehicleState, FlightPhase};
use serde::Serialize;

#[derive(Serialize)]
pub struct VehicleStateLog {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub attitude_euler: [f64; 3], // roll, pitch, yaw
    pub attitude_quat: [f64; 4],  // x, y, z, w
    pub angular_velocity: [f64; 3],
    pub mass: f64,
    pub dry_mass: f64,
}

#[derive(Serialize)]
pub struct ControlOutputLog {
    pub gimbal_theta: f64,
    pub gimbal_phi: f64,
    pub thrust: f64,
}

#[derive(Serialize)]
pub struct FlightPhaseLog {
    pub phase: FlightPhase,
    pub time_elapsed_s: f64,
}

pub struct McapLogger {
    writer: Writer<BufWriter<File>>,
    sensor_channel: u16,
    state_channel: u16,
    control_channel: u16,
    phase_channel: u16,
    sequence: u32,
    start_time: std::time::Instant,
    
    // One-shot warning flags to prevent terminal flooding on logging errors
    warned_sensor: bool,
    warned_state: bool,
    warned_control: bool,
    warned_phase: bool,
    
    // Track previous phase to only log on transitions
    last_logged_phase: Option<FlightPhase>,
}

impl McapLogger {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut writer = Writer::new(BufWriter::new(file))?;
        
        // Document metadata and expected rates for each channel
        let session_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let mut sensor_metadata = BTreeMap::new();
        sensor_metadata.insert("sensor/imu_model".to_string(), "VectorNav VN-200".to_string());
        sensor_metadata.insert("sensor/gps_model".to_string(), "VectorNav VN-200 GNSS".to_string());
        sensor_metadata.insert("rate".to_string(), "500 Hz (raw telemetry)".to_string());
        sensor_metadata.insert("session_id".to_string(), session_id.clone());
        
        let sensor_channel = writer.add_channel(
            0,
            "telemetry/sensor_data",
            "application/json",
            &sensor_metadata,
        )?;
        
        let mut state_metadata = BTreeMap::new();
        state_metadata.insert("rate".to_string(), "500 Hz (fused state estimation)".to_string());
        state_metadata.insert("session_id".to_string(), session_id.clone());
        
        let state_channel = writer.add_channel(
            0,
            "telemetry/vehicle_state",
            "application/json",
            &state_metadata,
        )?;
        
        let mut control_metadata = BTreeMap::new();
        control_metadata.insert("rate".to_string(), "50 Hz / Event-driven".to_string());
        control_metadata.insert("notes".to_string(), "Logged on NMPC solver output or zero-thrust update, not on 500 Hz EKF ticks".to_string());
        control_metadata.insert("session_id".to_string(), session_id.clone());
        
        let control_channel = writer.add_channel(
            0,
            "telemetry/control_output",
            "application/json",
            &control_metadata,
        )?;
        
        let mut phase_metadata = BTreeMap::new();
        phase_metadata.insert("rate".to_string(), "On-transition".to_string());
        phase_metadata.insert("notes".to_string(), "Logged only on flight phase transitions to optimize log file volume".to_string());
        phase_metadata.insert("session_id".to_string(), session_id.clone());
        
        let phase_channel = writer.add_channel(
            0,
            "telemetry/flight_phase",
            "application/json",
            &phase_metadata,
        )?;
        
        Ok(Self {
            writer,
            sensor_channel,
            state_channel,
            control_channel,
            phase_channel,
            sequence: 0,
            start_time: std::time::Instant::now(),
            warned_sensor: false,
            warned_state: false,
            warned_control: false,
            warned_phase: false,
            last_logged_phase: None,
        })
    }
    
    pub fn get_timestamp_ns(&self) -> u64 {
        self.start_time.elapsed().as_nanos() as u64
    }
    
    pub fn log_sensor_data(&mut self, timestamp_ns: u64, data: &SensorData) -> Result<(), Box<dyn std::error::Error>> {
        let payload = serde_json::to_vec(data)?;
        let result = self.writer.write_to_known_channel(
            &MessageHeader {
                channel_id: self.sensor_channel,
                sequence: self.sequence,
                log_time: timestamp_ns,
                publish_time: timestamp_ns,
            },
            &payload,
        );
        
        if let Err(ref e) = result {
            if !self.warned_sensor {
                eprintln!("[MCAP Logger Error] Failed to write sensor data: {}. Logging for this channel might be lost.", e);
                self.warned_sensor = true;
            }
        }
        
        self.sequence += 1;
        result?;
        Ok(())
    }
    
    pub fn log_vehicle_state(&mut self, timestamp_ns: u64, state: &VehicleState) -> Result<(), Box<dyn std::error::Error>> {
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        let q = state.attitude.coords; // [x, y, z, w]
        
        let log_data = VehicleStateLog {
            position: [state.position.x, state.position.y, state.position.z],
            velocity: [state.velocity.x, state.velocity.y, state.velocity.z],
            attitude_euler: [roll, pitch, yaw],
            attitude_quat: [q.x, q.y, q.z, q.w],
            angular_velocity: [state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z],
            mass: state.mass,
            dry_mass: state.dry_mass,
        };
        
        let payload = serde_json::to_vec(&log_data)?;
        let result = self.writer.write_to_known_channel(
            &MessageHeader {
                channel_id: self.state_channel,
                sequence: self.sequence,
                log_time: timestamp_ns,
                publish_time: timestamp_ns,
            },
            &payload,
        );
        
        if let Err(ref e) = result {
            if !self.warned_state {
                eprintln!("[MCAP Logger Error] Failed to write vehicle state: {}. Logging for this channel might be lost.", e);
                self.warned_state = true;
            }
        }
        
        self.sequence += 1;
        result?;
        Ok(())
    }
    
    pub fn log_control_output(
        &mut self, 
        timestamp_ns: u64, 
        gimbal_theta: f64, 
        gimbal_phi: f64, 
        thrust: f64
    ) -> Result<(), Box<dyn std::error::Error>> {
        let log_data = ControlOutputLog {
            gimbal_theta,
            gimbal_phi,
            thrust,
        };
        
        let payload = serde_json::to_vec(&log_data)?;
        let result = self.writer.write_to_known_channel(
            &MessageHeader {
                channel_id: self.control_channel,
                sequence: self.sequence,
                log_time: timestamp_ns,
                publish_time: timestamp_ns,
            },
            &payload,
        );
        
        if let Err(ref e) = result {
            if !self.warned_control {
                eprintln!("[MCAP Logger Error] Failed to write control output: {}. Logging for this channel might be lost.", e);
                self.warned_control = true;
            }
        }
        
        self.sequence += 1;
        result?;
        Ok(())
    }
    
    pub fn log_flight_phase(&mut self, timestamp_ns: u64, phase: FlightPhase, time_elapsed_s: f64) -> Result<(), Box<dyn std::error::Error>> {
        // Only log flight phase changes on transition to prevent spamming
        if Some(phase) != self.last_logged_phase {
            let log_data = FlightPhaseLog {
                phase,
                time_elapsed_s,
            };
            
            let payload = serde_json::to_vec(&log_data)?;
            let result = self.writer.write_to_known_channel(
                &MessageHeader {
                    channel_id: self.phase_channel,
                    sequence: self.sequence,
                    log_time: timestamp_ns,
                    publish_time: timestamp_ns,
                },
                &payload,
            );
            
            if let Err(ref e) = result {
                if !self.warned_phase {
                    eprintln!("[MCAP Logger Error] Failed to write flight phase change: {}. Logging for this channel might be lost.", e);
                    self.warned_phase = true;
                }
            }
            
            self.sequence += 1;
            self.last_logged_phase = Some(phase);
            result?;
            
            println!("Logged phase change to MCAP: {:?}", phase);
        }
        Ok(())
    }
    
    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.writer.finish()?;
        Ok(())
    }
}
