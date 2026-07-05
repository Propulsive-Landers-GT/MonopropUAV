// SMOKE TEST ONLY: synthetic GPS, not real fusion (no position data in CSV).
//
// This driver exercises the full ES-EKF code path (nominal_prediction,
// error_transition_jacobian, measurement update, error injection) to confirm it
// compiles and runs without NaN/panic. It reads the IMU columns from
// flight_data.csv for the fast predict loop and injects a synthetic zero-position
// GPS measurement on a slow cadence for the update loop. Because the CSV has no
// real position/GPS data, the numerical output is NOT a meaningful trajectory.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use ndarray::{array, Array1, Array2};
use rust_ekf::{RocketESEKF, RocketState};

/// Slow GPS cadence: apply a measurement update once every N IMU predict steps.
const GPS_EVERY_N_STEPS: usize = 100;

fn parse_imu_row(line: &str) -> io::Result<[f64; 7]> {
    // Columns: time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag..., q...
    let cols: Vec<&str> = line.split(',').collect();
    if cols.len() < 7 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected at least 7 columns, found {}", cols.len()),
        ));
    }
    let parse = |i: usize| -> io::Result<f64> {
        cols[i].trim().parse::<f64>().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("column {i}: {e}"))
        })
    };
    Ok([
        parse(0)?, // time
        parse(1)?, // accel_x
        parse(2)?, // accel_y
        parse(3)?, // accel_z
        parse(4)?, // gyro_x
        parse(5)?, // gyro_y
        parse(6)?, // gyro_z
    ])
}

fn run() -> io::Result<()> {
    let input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/testing")
        .join("flight_data.csv");
    let csv = fs::read_to_string(&input_path)?;

    // Nominal state (16D): [p, v, q(w,x,y,z), a_bias, w_bias]. Identity quaternion.
    let mut initial_state = Array1::<f64>::zeros(16);
    initial_state[6] = 1.0; // qw

    // Error covariance P and process noise Q are 15x15 (error-state size).
    let initial_p = Array2::<f64>::eye(15) * 0.01;
    let q = Array2::<f64>::eye(15) * 1e-4;

    let mut ekf = RocketESEKF::new(initial_state, initial_p, q, RocketState);

    // Synthetic GPS: says we are still at the origin, with modest 3D noise.
    let r_matrix = Array2::<f64>::eye(3) * 1.0;
    let synthetic_gps: Array1<f64> = array![0.0, 0.0, 0.0];

    let mut prev_time: Option<f64> = None;
    let mut step: usize = 0;

    // Per-step log: input IMU columns plus the resulting nominal state, so the
    // inputs/outputs can be plotted afterwards.
    let mut log: Vec<[f64; 22]> = Vec::new();

    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let row = parse_imu_row(line)?;
        let time = row[0];
        let dt = match prev_time {
            Some(t) if time - t > 0.0 => time - t,
            _ => 0.01, // fall back to the nominal 100 Hz cadence
        };
        prev_time = Some(time);

        // Fast loop: integrate IMU (accel xyz, gyro xyz).
        let imu = [row[1], row[2], row[3], row[4], row[5], row[6]];
        ekf.predict(&imu, dt);

        // Slow loop: fuse the synthetic GPS position on a fixed cadence.
        if step % GPS_EVERY_N_STEPS == 0 {
            ekf.update(&synthetic_gps, &r_matrix);
        }
        step += 1;

        let s = &ekf.nominal_state;
        log.push([
            time,
            imu[0], imu[1], imu[2], imu[3], imu[4], imu[5], // inputs: accel xyz, gyro xyz
            s[0], s[1], s[2],                                // position
            s[3], s[4], s[5],                                // velocity
            s[6], s[7], s[8], s[9],                          // quaternion w,x,y,z
            s[10], s[11], s[12],                             // accel bias
            s[13], s[14],                                    // gyro bias (x,y logged)
        ]);
    }

    // Write the time series for plotting.
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/testing")
        .join("esekf_output.csv");
    let mut writer = BufWriter::new(File::create(&output_path)?);
    writeln!(
        writer,
        "time,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,px,py,pz,vx,vy,vz,qw,qx,qy,qz,abx,aby,abz,wbx,wby"
    )?;
    for r in &log {
        writeln!(
            writer,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.6},{:.6},{:.6},{:.6},{:.6}",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10],
            r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21]
        )?;
    }
    writer.flush()?;

    let state = &ekf.nominal_state;
    let cov_trace: f64 = (0..ekf.error_covariance.nrows())
        .map(|i| ekf.error_covariance[[i, i]])
        .sum();

    let all_finite = state.iter().all(|v| v.is_finite()) && cov_trace.is_finite();

    println!("ES-EKF smoke test completed over {step} IMU steps.");
    println!("Final nominal state (16D): {state}");
    println!("Position     : [{:.4}, {:.4}, {:.4}]", state[0], state[1], state[2]);
    println!("Velocity     : [{:.4}, {:.4}, {:.4}]", state[3], state[4], state[5]);
    println!(
        "Quaternion   : [w {:.4}, x {:.4}, y {:.4}, z {:.4}]",
        state[6], state[7], state[8], state[9]
    );
    println!("Accel bias   : [{:.4}, {:.4}, {:.4}]", state[10], state[11], state[12]);
    println!("Gyro bias    : [{:.4}, {:.4}, {:.4}]", state[13], state[14], state[15]);
    println!("Covariance trace: {cov_trace:.6}");
    println!("All values finite: {all_finite}");

    if !all_finite {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "ES-EKF produced non-finite values",
        ));
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
