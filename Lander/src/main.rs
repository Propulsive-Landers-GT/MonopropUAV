mod algorithms;
mod control_loop;
use control_loop::ControlLoop;
use algorithms::SensorData;
use std::time::{Duration, Instant};

fn main() {
    println!("Lander Control Loop Starting...");
    
    let mut control_loop = ControlLoop::new();
    control_loop.initialize();
        
    println!("Control loop running...");
    
    // TODO: Timing Controls for Multithreaded Loop
    loop {
        // Initialize sensor readings
        let mut sensor_data = SensorData {
            timestamp: control_loop.get_state().start_time.elapsed().as_secs_f64(),
            imu_data: None,
            gps_data: None,
            uwb_data: None,
            chamber_pressure: None,
            tank_pressure: None,
        };
        
        // TODO: This isn't implemented yet, but we need to get sensor data from the hardware
        update_sensor_data(&mut sensor_data);
        
        // Step the control loop
        if let Some(control_output) = control_loop.step(&sensor_data) {
            if control_loop.get_state().flight_terminated {
                println!("Flight terminated - zeroing controls");
                break;
            }
            
            // Here you would send control_output to actuators
            println!("Control: {:?}", control_output);
        }
                
        // Print status every second
        let elapsed_time = control_loop.get_state().start_time.elapsed().as_secs_f64();
        if (elapsed_time * 100.0) as i64 % 100 == 0 {
            println!("Time: {:.2}s, Alt: {:.2}m, Phase: {:?}", 
                elapsed_time,
                control_loop.get_state().vehicle_state.position.z,
                control_loop.get_state().flight_phase
            );
        }
        if control_loop.get_state().flight_phase == algorithms::FlightPhase::Landed {
            break;
        }
    }
    
    println!("Control loop ended");
}