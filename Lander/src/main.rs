mod state;
mod algorithms;
mod fsm;
mod mcap_logger;

use fsm::FlightStateMachine;
use state::SensorData;
use mcap_logger::McapLogger;
use std::sync::mpsc::{self, Receiver};

fn spawn_stdin_channel() -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut line = String::new();
        while stdin.read_line(&mut line).is_ok() {
            let cmd = line.trim().to_lowercase();
            if !cmd.is_empty() {
                let _ = tx.send(cmd);
            }
            line.clear();
        }
    });
    rx
}

fn main() {
    println!("Lander Flight State Machine Starting...");
    println!("Interactive console commands: 'arm', 'disarm', 'launch'");
    
    let mut fsm = FlightStateMachine::new();
    fsm.initialize();
    
    let stdin_rx = spawn_stdin_channel();
    
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let log_filename = format!("flight_log_{}.mcap", timestamp);
    let mut mcap_logger = McapLogger::new(&log_filename).expect("Failed to initialize MCAP logger");
    println!("Logging telemetry to {}", log_filename);
        
    println!("FSM running...");
    
    let mut last_print_second = -1;
    let mut mission_time = 0.0;

    loop {
        let timestamp_ns = mcap_logger.get_timestamp_ns();
        mission_time = mcap_logger.get_elapsed_seconds();

        // Check for stdin commands
        if let Ok(cmd) = stdin_rx.try_recv() {
            match cmd.as_str() {
                "arm" => fsm.arm(mission_time),
                "disarm" => fsm.disarm(mission_time),
                "launch" => fsm.launch(mission_time),
                _ => println!("Unknown command: '{}' (valid: 'arm', 'disarm', 'launch')", cmd),
            }
        }

        // Initialize sensor readings (mock VN-200 INS data packet at current mission_time)
        let sensor_data = SensorData {
            timestamp: mission_time,
            imu_data: Some(state::ImuData {
                accel: [0.0, 0.0, 9.81],
                gyro: [0.0, 0.0, 0.0],
                mag: [-0.04, 0.44, -0.89],
            }),
            gps_data: Some([0.0, 0.0, 0.0]),
            uwb_data: Some([0.0, 0.0, 0.0]),
            chamber_pressure: Some(15.0),
            tank_pressure: Some(300.0),
        };
        
        // Log sensor data
        let _ = mcap_logger.log_sensor_data(timestamp_ns, &sensor_data);
        
        // Step the flight state machine
        let control_output_opt = fsm.step(&sensor_data);
        
        // Drain and log FSM diagnostic messages
        for msg in fsm.get_state_mut().diagnostics_queue.drain(..) {
            let _ = mcap_logger.log_diagnostics(timestamp_ns, &msg);
        }
        
        if fsm.get_state().flight_terminated {
            let reason = fsm.get_state().termination_reason.clone().unwrap_or_else(|| "Unknown".to_string());
            let msg = format!("Flight terminated - zeroing controls. Reason: {}", reason);
            println!("{}", msg);
            let _ = mcap_logger.log_diagnostics(timestamp_ns, &msg);
            
            // Log final vehicle state and flight phase
            let _ = mcap_logger.log_vehicle_state(timestamp_ns, &fsm.get_state().vehicle_state);
            let _ = mcap_logger.log_flight_phase(timestamp_ns, fsm.get_state().flight_phase, mission_time);
            break;
        }
        
        if let Some(control_output) = control_output_opt {
            // Log control output (gimbal_theta, gimbal_phi, thrust)
            let _ = mcap_logger.log_control_output(
                timestamp_ns,
                control_output[0],
                control_output[1],
                control_output[2],
            );
            
            println!("Control (theta, phi, thrust): {:?}", control_output);
        }
        
        // Log vehicle state and flight phase
        let _ = mcap_logger.log_vehicle_state(timestamp_ns, &fsm.get_state().vehicle_state);
        let _ = mcap_logger.log_flight_phase(timestamp_ns, fsm.get_state().flight_phase, mission_time);
                
        // Print status every second
        let current_second = mission_time.floor() as i32;
        if current_second > last_print_second {
            println!("Time: {:.2}s, Alt: {:.2}m, Phase: {:?}", 
                mission_time,
                fsm.get_state().vehicle_state.position.z,
                fsm.get_state().flight_phase
            );
            last_print_second = current_second;
        }
        
        if fsm.get_state().flight_phase == state::FlightPhase::Landed {
            println!("Landed successfully at {:.2}m altitude!", fsm.get_state().vehicle_state.position.z);
            let _ = mcap_logger.log_diagnostics(timestamp_ns, "Landed successfully!");
            break;
        }
        
        // Sleep to maintain EKF / loop rate at 500 Hz
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    
    // Finish logging and flush
    let _ = mcap_logger.finish();
    println!("FSM execution ended");
}