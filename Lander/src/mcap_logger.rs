use std::fs::File;
use std::io::BufWriter;
use std::collections::BTreeMap;
use mcap::{Writer, records::MessageHeader};
use crate::state::{SensorData, VehicleState, FlightPhase};
use serde::Serialize;

#[derive(Serialize, Clone)]
pub struct VehicleStateLog {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub attitude_euler: [f64; 3], // roll, pitch, yaw
    pub attitude_quat: [f64; 4],  // x, y, z, w
    pub angular_velocity: [f64; 3],
    pub mass: f64,
    pub dry_mass: f64,
}

#[derive(Serialize, Clone)]
pub struct ControlOutputLog {
    pub gimbal_theta: f64,
    pub gimbal_phi: f64,
    pub thrust: f64,
}

#[derive(Serialize, Clone)]
pub struct FlightPhaseLog {
    pub phase: FlightPhase,
    pub time_elapsed_s: f64,
}

enum LogMessage {
    Sensor(u64, SensorData),
    VehicleState(u64, VehicleStateLog),
    ControlOutput(u64, ControlOutputLog),
    FlightPhase(u64, FlightPhaseLog),
    Diagnostics(u64, String),
}

pub struct McapLogger {
    tx: std::sync::mpsc::Sender<LogMessage>,
    worker_join_handle: Option<std::thread::JoinHandle<()>>,
    start_time: std::time::Instant,
    last_logged_phase: Option<FlightPhase>,
}

impl McapLogger {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let path_str = path.to_string();
        let (tx, rx) = std::sync::mpsc::channel::<LogMessage>();
        
        let start_time = std::time::Instant::now();
        
        let worker_join_handle = std::thread::spawn(move || {
            let file = File::create(&path_str).expect("Failed to create MCAP file");
            let mut writer = Writer::new(BufWriter::new(file)).expect("Failed to create MCAP writer");
            
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
            ).unwrap();
            
            let mut state_metadata = BTreeMap::new();
            state_metadata.insert("rate".to_string(), "500 Hz (fused state estimation)".to_string());
            state_metadata.insert("session_id".to_string(), session_id.clone());
            
            let state_channel = writer.add_channel(
                0,
                "telemetry/vehicle_state",
                "application/json",
                &state_metadata,
            ).unwrap();
            
            let mut control_metadata = BTreeMap::new();
            control_metadata.insert("rate".to_string(), "50 Hz / Event-driven".to_string());
            control_metadata.insert("notes".to_string(), "Logged on NMPC solver output or zero-thrust update, not on 500 Hz EKF ticks".to_string());
            control_metadata.insert("session_id".to_string(), session_id.clone());
            
            let control_channel = writer.add_channel(
                0,
                "telemetry/control_output",
                "application/json",
                &control_metadata,
            ).unwrap();
            
            let mut phase_metadata = BTreeMap::new();
            phase_metadata.insert("rate".to_string(), "On-transition".to_string());
            phase_metadata.insert("notes".to_string(), "Logged only on flight phase transitions to optimize log file volume".to_string());
            phase_metadata.insert("session_id".to_string(), session_id.clone());
            
            let phase_channel = writer.add_channel(
                0,
                "telemetry/flight_phase",
                "application/json",
                &phase_metadata,
            ).unwrap();
            
            let mut diagnostics_metadata = BTreeMap::new();
            diagnostics_metadata.insert("rate".to_string(), "Event-driven".to_string());
            diagnostics_metadata.insert("notes".to_string(), "Logged on FSM state transitions, trajectory updates, solver diagnostic events, and system errors".to_string());
            diagnostics_metadata.insert("session_id".to_string(), session_id.clone());
            
            let diagnostics_channel = writer.add_channel(
                0,
                "telemetry/diagnostics",
                "application/json",
                &diagnostics_metadata,
            ).unwrap();
            
            let mut sequence = 0u32;
            
            // Loop reading from the queue and writing to disk
            while let Ok(msg) = rx.recv() {
                match msg {
                    LogMessage::Sensor(timestamp_ns, data) => {
                        if let Ok(payload) = serde_json::to_vec(&data) {
                            let _ = writer.write_to_known_channel(
                                &MessageHeader {
                                    channel_id: sensor_channel,
                                    sequence,
                                    log_time: timestamp_ns,
                                    publish_time: timestamp_ns,
                                },
                                &payload,
                            );
                            sequence += 1;
                        }
                    }
                    LogMessage::VehicleState(timestamp_ns, log_data) => {
                        if let Ok(payload) = serde_json::to_vec(&log_data) {
                            let _ = writer.write_to_known_channel(
                                &MessageHeader {
                                    channel_id: state_channel,
                                    sequence,
                                    log_time: timestamp_ns,
                                    publish_time: timestamp_ns,
                                },
                                &payload,
                            );
                            sequence += 1;
                        }
                    }
                    LogMessage::ControlOutput(timestamp_ns, log_data) => {
                        if let Ok(payload) = serde_json::to_vec(&log_data) {
                            let _ = writer.write_to_known_channel(
                                &MessageHeader {
                                    channel_id: control_channel,
                                    sequence,
                                    log_time: timestamp_ns,
                                    publish_time: timestamp_ns,
                                },
                                &payload,
                            );
                            sequence += 1;
                        }
                    }
                    LogMessage::FlightPhase(timestamp_ns, log_data) => {
                        if let Ok(payload) = serde_json::to_vec(&log_data) {
                            let _ = writer.write_to_known_channel(
                                &MessageHeader {
                                    channel_id: phase_channel,
                                    sequence,
                                    log_time: timestamp_ns,
                                    publish_time: timestamp_ns,
                                },
                                &payload,
                            );
                            sequence += 1;
                        }
                    }
                    LogMessage::Diagnostics(timestamp_ns, message) => {
                        #[derive(Serialize)]
                        struct DiagnosticsLog {
                            message: String,
                        }
                        let log_data = DiagnosticsLog { message };
                        if let Ok(payload) = serde_json::to_vec(&log_data) {
                            let _ = writer.write_to_known_channel(
                                &MessageHeader {
                                    channel_id: diagnostics_channel,
                                    sequence,
                                    log_time: timestamp_ns,
                                    publish_time: timestamp_ns,
                                },
                                &payload,
                            );
                            sequence += 1;
                        }
                    }
                }
            }
            
            // Finalize writing bytes to disk
            let _ = writer.finish();
        });
        
        Ok(Self {
            tx,
            worker_join_handle: Some(worker_join_handle),
            start_time,
            last_logged_phase: None,
        })
    }
    
    pub fn get_timestamp_ns(&self) -> u64 {
        self.start_time.elapsed().as_nanos() as u64
    }

    pub fn get_elapsed_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
    
    pub fn log_sensor_data(&mut self, timestamp_ns: u64, data: &SensorData) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.tx.send(LogMessage::Sensor(timestamp_ns, data.clone()));
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
        let _ = self.tx.send(LogMessage::VehicleState(timestamp_ns, log_data));
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
        let _ = self.tx.send(LogMessage::ControlOutput(timestamp_ns, log_data));
        Ok(())
    }
    
    pub fn log_flight_phase(&mut self, timestamp_ns: u64, phase: FlightPhase, time_elapsed_s: f64) -> Result<(), Box<dyn std::error::Error>> {
        if Some(phase) != self.last_logged_phase {
            self.last_logged_phase = Some(phase);
            let log_data = FlightPhaseLog {
                phase,
                time_elapsed_s,
            };
            let _ = self.tx.send(LogMessage::FlightPhase(timestamp_ns, log_data));
            println!("Logged phase change to MCAP: {:?}", phase);
        }
        Ok(())
    }
    
    pub fn log_diagnostics(&mut self, timestamp_ns: u64, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.tx.send(LogMessage::Diagnostics(timestamp_ns, message.to_string()));
        Ok(())
    }
    
    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        drop(self.tx);
        if let Some(handle) = self.worker_join_handle.take() {
            handle.join().map_err(|_| "Failed to join logger thread")?;
        }
        Ok(())
    }
}
