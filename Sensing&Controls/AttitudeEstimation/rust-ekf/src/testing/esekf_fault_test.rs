// FAULT-INJECTION TEST: sensor dropout, wild outliers, and firmware NaNs.
//
// The vehicle flies ONE filter, so that filter has to ride through sensor
// faults on its own instead of switching between per-sensor filters. This
// driver replays flight_data.csv through the ES-EKF under several fault
// scenarios and reports attitude accuracy vs the truth quaternion in the CSV:
//
//   baseline           - all sensors healthy (gated), reference run
//   gnss_dropout       - GPS and UWB both silent from t=8s to t=16s
//   gnss_dropout_nobaro- same dropout with the barometer disabled, to isolate
//                        how much of the dead-reckoning drift the baro removes
//   outliers_gated     - GPS reports +/-300 km, mag reports 1000x field; gate ON
//   outliers_ungated   - the same garbage fused blindly (what the gate prevents)
//   nan_burst          - firmware returns NaN on IMU, mag, and GPS in bursts
//   kitchen_sink       - dropout + outliers + NaNs all in one flight, gate ON
//
// The barometer models the VN-200's onboard pressure sensor (10-1200 mbar,
// +/-1.5 mbar accuracy, 250 Hz sample rate). The +/-1.5 mbar term is an
// absolute error band, zeroed at the pad before flight as usual; the datasheet
// publishes no noise density, so per-sample altitude noise is assumed 0.35 m
// 1-sigma (typical for this sensor class, ~0.042 mbar). It is a pure altitude
// (world-Z position) measurement, fused at 25 Hz, and it does NOT go out
// during the GNSS dropout: pressure needs no radio.
//
// The CSV has no position truth, so a self-consistent position pseudo-truth
// is built by strapdown-integrating the measured accel through the TRUTH
// attitude (drift over 23 s is small, and it is consistent with the IMU data
// the filter sees, unlike an origin-pinned fake). Synthetic sensors sample it:
// GPS at 1 Hz with sigma = 1 m, UWB at 5 Hz with sigma = 0.1 m and an 8 m
// range from the origin, mirroring device_sim.rs. The magnetometer columns
// are real simulated measurements.
//
// Robustness mechanisms under test (es_ekf/filter.rs):
//   - predict() refuses non-finite IMU samples (coasts instead of poisoning P)
//   - update_gated() rejects non-finite measurements
//   - update_gated() chi-square gates the innovation (NIS > 16.27 -> drop)

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use ndarray::{array, Array1, Array2};
use rust_ekf::{ESEKFModel, RocketESEKF, RocketState, UpdateOutcome};

/// Chi-square 99.9% threshold for a 3-DOF measurement: genuine measurements
/// pass 999 times out of 1000; a 300 km outlier scores NIS ~ 1e10.
const GATE_3DOF: f64 = 16.27;
/// Same confidence level for the 1-DOF barometer measurement.
const GATE_1DOF: f64 = 10.83;

const GPS_EVERY_N_STEPS: usize = 100; // 1 Hz at the CSV's 100 Hz cadence
const UWB_EVERY_N_STEPS: usize = 20; // 5 Hz, matching device_sim.rs
const BARO_EVERY_N_STEPS: usize = 4; // 25 Hz fused (sensor itself runs 250 Hz)
const BARO_SIGMA: f64 = 0.35; // m, assumed per-sample noise (see header)

// Fault schedule (seconds into the flight).
const DROPOUT_WINDOW: (f64, f64) = (8.0, 16.0); // GPS + UWB silent
const GPS_OUTLIER_WINDOW: (f64, f64) = (4.0, 6.0); // GPS says +/-300 km
const MAG_OUTLIER_WINDOW: (f64, f64) = (10.0, 10.5); // mag says 1000x field
const IMU_NAN_WINDOW: (f64, f64) = (12.0, 12.2); // firmware NaN on IMU
const MAG_NAN_WINDOW: (f64, f64) = (14.0, 14.2); // firmware NaN on mag
const GPS_NAN_WINDOW: (f64, f64) = (18.0, 19.0); // firmware NaN on GPS

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    gnss_dropout: bool,
    outliers: bool,
    nans: bool,
    gated: bool,
    baro: bool,
}

const SCENARIOS: [Scenario; 7] = [
    Scenario { name: "baseline", gnss_dropout: false, outliers: false, nans: false, gated: true, baro: true },
    Scenario { name: "gnss_dropout", gnss_dropout: true, outliers: false, nans: false, gated: true, baro: true },
    Scenario { name: "gnss_dropout_nobaro", gnss_dropout: true, outliers: false, nans: false, gated: true, baro: false },
    Scenario { name: "outliers_gated", gnss_dropout: false, outliers: true, nans: false, gated: true, baro: true },
    Scenario { name: "outliers_ungated", gnss_dropout: false, outliers: true, nans: false, gated: false, baro: true },
    Scenario { name: "nan_burst", gnss_dropout: false, outliers: false, nans: true, gated: true, baro: true },
    Scenario { name: "kitchen_sink", gnss_dropout: true, outliers: true, nans: true, gated: true, baro: true },
];

struct Row {
    time: f64,
    imu: [f64; 6],
    mag: [f64; 3],
    q_truth: UnitQuaternion<f64>,
    /// Position pseudo-truth (filled in by `build_position_pseudo_truth`).
    pos_truth: Vector3<f64>,
}

/// Strapdown-integrate the measured accel through the truth attitude to get a
/// position trajectory that is consistent with the IMU stream. Accel noise
/// makes it random-walk by a couple of meters over the flight, which is
/// covered by the GPS measurement sigma.
fn build_position_pseudo_truth(rows: &mut [Row]) {
    let gravity = Vector3::new(0.0, 0.0, 9.81);
    let mut pos = Vector3::zeros();
    let mut vel = Vector3::zeros();
    let mut prev_time: Option<f64> = None;
    for row in rows.iter_mut() {
        let dt = match prev_time {
            Some(p) if row.time - p > 0.0 => row.time - p,
            _ => 0.01,
        };
        prev_time = Some(row.time);
        let a_body = Vector3::new(row.imu[0], row.imu[1], row.imu[2]);
        let a_world = row.q_truth.transform_vector(&a_body) - gravity;
        pos += vel * dt + 0.5 * a_world * dt * dt;
        vel += a_world * dt;
        row.pos_truth = pos;
    }
}

/// Small deterministic PRNG (xorshift64*) with Box-Muller, so the synthetic
/// GPS/UWB noise is reproducible without pulling in a rand dependency.
struct Prng(u64);

impl Prng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn gaussian_vec3(&mut self, sigma: f64) -> Vector3<f64> {
        Vector3::new(self.gaussian(), self.gaussian(), self.gaussian()) * sigma
    }
}

fn parse_row(line: &str) -> io::Result<Row> {
    // Columns: time, accel xyz, gyro xyz, mag xyz, q_x, q_y, q_z, q_w
    let cols: Vec<&str> = line.split(',').collect();
    if cols.len() < 14 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected at least 14 columns, found {}", cols.len()),
        ));
    }
    let parse = |i: usize| -> io::Result<f64> {
        cols[i].trim().parse::<f64>().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("column {i}: {e}"))
        })
    };
    Ok(Row {
        time: parse(0)?,
        imu: [parse(1)?, parse(2)?, parse(3)?, parse(4)?, parse(5)?, parse(6)?],
        mag: [parse(7)?, parse(8)?, parse(9)?],
        // CSV order is x,y,z,w; nalgebra wants w,x,y,z.
        q_truth: UnitQuaternion::from_quaternion(Quaternion::new(
            parse(13)?,
            parse(10)?,
            parse(11)?,
            parse(12)?,
        )),
        pos_truth: Vector3::zeros(),
    })
}

fn in_window(t: f64, w: (f64, f64)) -> bool {
    t >= w.0 && t < w.1
}

#[derive(Default)]
struct SensorTally {
    fused: usize,
    gated: usize,
    non_finite: usize,
}

impl SensorTally {
    fn record(&mut self, outcome: UpdateOutcome) {
        match outcome {
            UpdateOutcome::Fused { .. } => self.fused += 1,
            UpdateOutcome::RejectedGate { .. } => self.gated += 1,
            UpdateOutcome::RejectedNonFinite => self.non_finite += 1,
            UpdateOutcome::RejectedSingular => self.non_finite += 1,
        }
    }
    fn rejected(&self) -> usize {
        self.gated + self.non_finite
    }
}

struct StepLog {
    time: f64,
    err_deg: f64,
    /// Attitude error rotation vector in the truth body frame, degrees:
    /// x/y ~ roll/pitch (tilt) error, z ~ yaw error.
    att_err_xyz: [f64; 3],
    pos_err: f64,
    /// Signed per-axis position error (estimate minus pseudo-truth), meters.
    pos_err_xyz: [f64; 3],
    pos_sigma: f64,
    att_sigma_deg: f64,
    rejects_cum: usize,
}

struct ScenarioResult {
    log: Vec<StepLog>,
    gps: SensorTally,
    uwb: SensorTally,
    baro: SensorTally,
    mag: SensorTally,
    imu_skipped: usize,
    all_finite: bool,
}

fn run_scenario(sc: Scenario, rows: &[Row]) -> ScenarioResult {
    let mut initial_state = Array1::<f64>::zeros(16);
    initial_state[6] = 1.0; // qw

    let mut ekf = RocketESEKF::new(
        initial_state,
        RocketState::initial_covariance(),
        RocketState::process_noise(0.01),
        RocketState,
    );

    const GPS_SIGMA: f64 = 1.0;
    const UWB_SIGMA: f64 = 0.1;
    const UWB_RANGE: f64 = 8.0; // device_sim.rs: UWB only works near the pad
    let r_gps = Array2::<f64>::eye(3) * GPS_SIGMA * GPS_SIGMA;
    let r_uwb = Array2::<f64>::eye(3) * UWB_SIGMA * UWB_SIGMA;
    let r_mag = Array2::<f64>::eye(3) * (2.4e-7_f64).powi(2);
    let mag_world = Vector3::new(-2.0e-6, 22.0e-6, -44.3e-6);
    let r_baro = Array2::<f64>::eye(1) * BARO_SIGMA * BARO_SIGMA;
    let h_baro = RocketState::baro_jacobian();
    let gate = if sc.gated { Some(GATE_3DOF) } else { None };
    let baro_gate = if sc.gated { Some(GATE_1DOF) } else { None };
    // Independent noise streams per sensor, so enabling/disabling one sensor
    // (e.g. the no-baro comparison run) does not change the noise the other
    // sensors happen to draw.
    let mut prng_gps = Prng(0x9E3779B97F4A7C15);
    let mut prng_uwb = Prng(0xD1B54A32D192ED03);
    let mut prng_baro = Prng(0x8CB92BA72F3D8DD7);

    let mut result = ScenarioResult {
        log: Vec::with_capacity(rows.len()),
        gps: SensorTally::default(),
        uwb: SensorTally::default(),
        baro: SensorTally::default(),
        mag: SensorTally::default(),
        imu_skipped: 0,
        all_finite: true,
    };

    let mut prev_time: Option<f64> = None;
    for (step, row) in rows.iter().enumerate() {
        let t = row.time;
        let dt = match prev_time {
            Some(p) if t - p > 0.0 => t - p,
            _ => 0.01,
        };
        prev_time = Some(t);

        // --- IMU (fast loop), possibly NaN'd by "firmware" ---
        let mut imu = row.imu;
        if sc.nans && in_window(t, IMU_NAN_WINDOW) {
            imu = [f64::NAN; 6];
        }
        if !ekf.predict(&imu, dt) {
            result.imu_skipped += 1;
        }

        // --- Magnetometer, every step ---
        let mut mag = array![row.mag[0], row.mag[1], row.mag[2]];
        if sc.outliers && in_window(t, MAG_OUTLIER_WINDOW) {
            mag *= 1000.0; // e.g. magnetized structure / bus glitch
        }
        if sc.nans && in_window(t, MAG_NAN_WINDOW) {
            mag.fill(f64::NAN);
        }
        // Skip all-zero rows (simulated sensor not yet ticked), like esekf_test.
        if mag.iter().any(|v| v.abs() > 1e-12) || mag.iter().any(|v| !v.is_finite()) {
            let pred = RocketState::mag_prediction(&ekf.nominal_state, &mag_world);
            let h = RocketState::mag_jacobian(&ekf.nominal_state, &mag_world);
            result.mag.record(ekf.update_gated(&mag, &pred, &h, &r_mag, gate));
        }

        // --- GPS (1 Hz) and UWB (5 Hz), sampling the position pseudo-truth ---
        let gnss_out = sc.gnss_dropout && in_window(t, DROPOUT_WINDOW);
        if !gnss_out {
            if step % GPS_EVERY_N_STEPS == 0 {
                let noisy = row.pos_truth + prng_gps.gaussian_vec3(GPS_SIGMA);
                let mut z = array![noisy.x, noisy.y, noisy.z];
                if sc.outliers && in_window(t, GPS_OUTLIER_WINDOW) {
                    z = array![3.0e5, -3.0e5, 3.0e5]; // "300000" firmware glitch
                }
                if sc.nans && in_window(t, GPS_NAN_WINDOW) {
                    z.fill(f64::NAN);
                }
                let pred = ekf.model.measurement_prediction(&ekf.nominal_state);
                let h = ekf.model.measurement_jacobian(&ekf.nominal_state);
                result.gps.record(ekf.update_gated(&z, &pred, &h, &r_gps, gate));
            }
            // UWB only returns data within range of its pad-mounted anchor.
            if step % UWB_EVERY_N_STEPS == 0 && row.pos_truth.norm() <= UWB_RANGE {
                let noisy = row.pos_truth + prng_uwb.gaussian_vec3(UWB_SIGMA);
                let z = array![noisy.x, noisy.y, noisy.z];
                let pred = ekf.model.measurement_prediction(&ekf.nominal_state);
                let h = ekf.model.measurement_jacobian(&ekf.nominal_state);
                result.uwb.record(ekf.update_gated(&z, &pred, &h, &r_uwb, gate));
            }
        }

        // --- Barometer (25 Hz): altitude keeps working through GNSS loss ---
        if sc.baro && step % BARO_EVERY_N_STEPS == 0 {
            let z = array![row.pos_truth.z + prng_baro.gaussian() * BARO_SIGMA];
            let pred = RocketState::baro_prediction(&ekf.nominal_state);
            result
                .baro
                .record(ekf.update_gated(&z, &pred, &h_baro, &r_baro, baro_gate));
        }

        // --- Log ---
        let s = &ekf.nominal_state;
        if s.iter().any(|v| !v.is_finite()) {
            result.all_finite = false;
        }
        let q_est =
            UnitQuaternion::from_quaternion(Quaternion::new(s[6], s[7], s[8], s[9]));
        let err_deg = q_est.angle_to(&row.q_truth).to_degrees();
        // Error rotation vector in the truth body frame: for small errors its
        // components read directly as roll/pitch/yaw error.
        let rot_err = (row.q_truth.inverse() * q_est).scaled_axis();
        let att_err_xyz = [
            rot_err.x.to_degrees(),
            rot_err.y.to_degrees(),
            rot_err.z.to_degrees(),
        ];
        let pos_err_vec = Vector3::new(s[0], s[1], s[2]) - row.pos_truth;
        let pos_err = pos_err_vec.norm();
        let pos_err_xyz = [pos_err_vec.x, pos_err_vec.y, pos_err_vec.z];
        let p = &ekf.error_covariance;
        let pos_sigma = (p[[0, 0]] + p[[1, 1]] + p[[2, 2]]).max(0.0).sqrt();
        let att_sigma_deg =
            (p[[6, 6]] + p[[7, 7]] + p[[8, 8]]).max(0.0).sqrt().to_degrees();
        let rejects_cum = result.gps.rejected()
            + result.uwb.rejected()
            + result.baro.rejected()
            + result.mag.rejected();
        result.log.push(StepLog {
            time: t,
            err_deg,
            att_err_xyz,
            pos_err,
            pos_err_xyz,
            pos_sigma,
            att_sigma_deg,
            rejects_cum,
        });
    }
    result
}

fn run() -> io::Result<()> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/testing");
    let csv = fs::read_to_string(base.join("flight_data.csv"))?;
    let mut rows: Vec<Row> = csv
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(parse_row)
        .collect::<io::Result<_>>()?;
    build_position_pseudo_truth(&mut rows);
    let max_range = rows.iter().map(|r| r.pos_truth.norm()).fold(0.0, f64::max);
    println!("position pseudo-truth: max distance from pad = {max_range:.1} m\n");

    let mut writer = BufWriter::new(File::create(base.join("esekf_fault_output.csv"))?);
    writeln!(
        writer,
        "scenario,time,att_err_deg,att_err_x_deg,att_err_y_deg,att_err_z_deg,\
         pos_err_m,pos_err_x_m,pos_err_y_m,pos_err_z_m,pos_sigma_m,att_sigma_deg,rejects_cum"
    )?;

    println!(
        "{:<20} {:>8} {:>9} {:>10} {:>10} {:>13} {:>6} {:>6} {:>7} {:>6} {:>6}",
        "scenario", "rms(deg)", "peak(deg)", "peakPos(m)", "finalPos(m)", "peakPosXYZ(m)", "gpsRj", "uwbRj", "baroRj", "magRj", "finite"
    );

    for sc in SCENARIOS {
        let res = run_scenario(sc, &rows);
        for l in &res.log {
            writeln!(
                writer,
                "{},{:.6},{:.9},{:.9},{:.9},{:.9},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                sc.name,
                l.time,
                l.err_deg,
                l.att_err_xyz[0],
                l.att_err_xyz[1],
                l.att_err_xyz[2],
                l.pos_err,
                l.pos_err_xyz[0],
                l.pos_err_xyz[1],
                l.pos_err_xyz[2],
                l.pos_sigma,
                l.att_sigma_deg,
                l.rejects_cum
            )?;
        }
        let n = res.log.len() as f64;
        let rms = (res.log.iter().map(|l| l.err_deg * l.err_deg).sum::<f64>() / n).sqrt();
        let peak = res.log.iter().map(|l| l.err_deg).fold(0.0, f64::max);
        let peak_pos = res.log.iter().map(|l| l.pos_err).fold(0.0, f64::max);
        let last_pos = res.log.last().map(|l| l.pos_err).unwrap_or(f64::NAN);
        let peak_axis = |i: usize| -> f64 {
            res.log.iter().map(|l| l.pos_err_xyz[i].abs()).fold(0.0, f64::max)
        };
        println!(
            "{:<20} {:>8.3} {:>9.3} {:>10.2} {:>10.2} {:>4.2}/{:>4.2}/{:>4.2} {:>6} {:>6} {:>7} {:>6} {:>6}",
            sc.name,
            rms,
            peak,
            peak_pos,
            last_pos,
            peak_axis(0),
            peak_axis(1),
            peak_axis(2),
            res.gps.rejected(),
            res.uwb.rejected(),
            res.baro.rejected(),
            res.mag.rejected(),
            res.all_finite,
        );
        if res.imu_skipped > 0 {
            println!(
                "{:<20} coasted through {} non-finite IMU samples",
                "", res.imu_skipped
            );
        }
    }
    writer.flush()?;
    println!("\nwrote {}", base.join("esekf_fault_output.csv").display());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
