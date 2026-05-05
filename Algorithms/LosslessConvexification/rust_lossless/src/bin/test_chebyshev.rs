#[path = "../lossless.rs"]
mod lossless;
#[path = "../chebyshev_lossless.rs"]
mod chebyshev_lossless;

use crate::chebyshev_lossless::ChebyshevLosslessSolver;

fn main() {
    let mut solver = ChebyshevLosslessSolver {
        landing_point: [0.0, 0.0, 0.0],
        initial_position: [0.0, 0.0, 50.0],
        initial_velocity: [0.0, 0.0, 0.0],
        max_velocity: 500.0,
        dry_mass: 50.0,
        fuel_mass: 30.0,
        alpha: 1.0 / (9.81 * 180.0),
        lower_thrust_bound: 1000.0 * 0.4,
        upper_thrust_bound: 1000.0,
        tvc_range_rad: 15_f64.to_radians(),
        coarse_nodes: 8,
        fine_nodes: 16,
        use_glide_slope: false,
        glide_slope: 5_f64.to_radians(),
        N: 8, // modest size for testing
        ..Default::default()
    };

    println!("Running test_chebyshev...\n");

    let solve_run = solver.solve();
    println!(
        "Metrics: coarse attempts={} fine attempts={} total wall={:.4}s",
        solve_run.coarse_metrics.attempts,
        solve_run.fine_metrics.attempts,
        solve_run.total_metrics.wall_time_s
    );

    match solve_run.trajectory {
        Some(traj) => {
            println!("Sample count: positions={} velocities={} thrusts={} sigmas={}",
                traj.positions.len(), traj.velocities.len(), traj.thrusts.len(), traj.sigmas.len());

            if let Some(p0) = traj.positions.first() {
                println!("first position: {:?}", p0);
            }
            if let Some(pm) = traj.positions.get(traj.positions.len() / 2) {
                println!("mid position: {:?}", pm);
            }
            if let Some(pl) = traj.positions.last() {
                println!("last position: {:?}", pl);
            }

            if let Some(u0) = traj.thrusts.first() {
                println!("first thrust: {:?}", u0);
            }
            if let Some(s0) = traj.sigmas.first() {
                println!("first sigma: {:?}", s0);
            }

            println!("Final mass: {:?}", traj.masses.last());
        }
        None => {
            eprintln!("Solver failed in test_chebyshev");
        }
    }
}
