use rust_lossless::{LosslessSolver, TrajectoryResult};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn main() {
    let mut solver = LosslessSolver {
        ..base_solver()
    };
    let solve_run = solver.solve();
    let Some(trajectory) = solve_run.trajectory else {
        eprintln!(
            "ZOH solve failed: coarse={:?}, fine={:?}",
            solve_run.coarse_metrics.last_status,
            solve_run.fine_metrics.last_status
        );
        return;
    };

    println!("Time of flight: {:.3} s", trajectory.time_of_flight_s);
    println!("Final position: {:?}", trajectory.positions.last().unwrap());
    println!("Final velocity: {:?}", trajectory.velocities.last().unwrap());
    println!("Final mass: {:?}", trajectory.masses.last().unwrap());

    let generated_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("generated");
    std::fs::create_dir_all(&generated_dir).expect("Failed to create generated output directory");
    let trajectory_path = generated_dir.join("trajectory.csv");
    write_trajectory_to_csv(&trajectory_path, &trajectory).expect("Failed to write trajectory.csv");
    println!("Wrote {}", trajectory_path.display());
}

fn base_solver() -> LosslessSolver {
    LosslessSolver {
        // landing_point: [0.0, 0.0, 0.0],
        landing_point: [0.0, 5.0, 0.0],
        // initial_position: [1.0, 1.0, 1.0],
        initial_position: [0.0, 0.0, -0.1],
        initial_velocity: [0.0, 0.0, 0.0],
        use_initial_glide_slope: true,
        use_final_glide_slope: false,
        glide_slope: 0.0000001_f64.to_radians(),
        max_velocity: 5.0,
        dry_mass: 50.0,
        fuel_mass: 30.0,
        alpha: 1.0 / (9.81 * 180.0),
        lower_thrust_bound: 1000.0 * 0.4,
        upper_thrust_bound: 1000.0,
        tvc_range_rad: 15_f64.to_radians(),
        coarse_line_search_delta_t: 0.1,
        fine_line_search_delta_t: 0.01,
        use_terminal_lateral_hard_tube: true,
        terminal_lateral_hard_tube_time_s: 0.5,
        terminal_lateral_hard_tube_radius_m: 0.05,
        coarse_delta_t: 0.05,
        fine_delta_t: 0.025,
        timeout: 30_f64,
        N: 20,
        ..Default::default()
    }
}


fn write_trajectory_to_csv(path: impl AsRef<Path>, traj: &TrajectoryResult) -> std::io::Result<()> {
    let path = path.as_ref();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "t,x,y,z,vx,vy,vz,mass,ux,uy,uz,sigma,thrust_force")?;

    for k in 0..traj.positions.len() {
        let x = &traj.positions[k];
        let v = &traj.velocities[k];
        let m = traj.masses[k];
        let sigma = traj.sigmas.get(k).copied().unwrap_or(0.0);
        let u = traj.thrusts.get(k).copied().unwrap_or([0.0, 0.0, 0.0]);
        let u_mag = (u[0].powi(2) + u[1].powi(2) + u[2].powi(2)).sqrt();
        let thrust_force = m * u_mag;

        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            k as f64,
            x[0],
            x[1],
            x[2],
            v[0],
            v[1],
            v[2],
            m,
            u[0],
            u[1],
            u[2],
            sigma,
            thrust_force
        )?;
    }

    writer.flush()?;
    Ok(())
}
