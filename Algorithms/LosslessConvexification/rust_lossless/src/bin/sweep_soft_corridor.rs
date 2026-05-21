#[path = "../lossless.rs"]
mod lossless;

use lossless::{LosslessSolver, TrajectoryResult};

#[derive(Debug, Clone, Copy)]
struct CorridorStats {
    points: usize,
    sat_count: usize,
    sat_pct: f64,
    max_residual: f64,
    mean_residual: f64,
    max_residual_z1: f64,
    max_residual_z05: f64,
}

fn compute_stats(traj: &TrajectoryResult, kappa: f64, z_f: f64, x_f: f64, y_f: f64) -> CorridorStats {
    let mut sat_count = 0usize;
    let mut max_residual = f64::NEG_INFINITY;
    let mut residual_sum = 0.0;
    let mut max_residual_z1 = f64::NEG_INFINITY;
    let mut max_residual_z05 = f64::NEG_INFINITY;
    let tol = 1e-6;

    for p in &traj.positions {
        let dx = p[0] - x_f;
        let dy = p[1] - y_f;
        let z = p[2];
        let r = (dx * dx + dy * dy).sqrt();
        let residual = r - kappa * (z - z_f);

        if residual <= tol {
            sat_count += 1;
        }
        if residual > max_residual {
            max_residual = residual;
        }
        if z <= 1.0 && residual > max_residual_z1 {
            max_residual_z1 = residual;
        }
        if z <= 0.5 && residual > max_residual_z05 {
            max_residual_z05 = residual;
        }
        residual_sum += residual;
    }

    let points = traj.positions.len();
    let mean_residual = if points > 0 {
        residual_sum / points as f64
    } else {
        0.0
    };

    if max_residual_z1 == f64::NEG_INFINITY {
        max_residual_z1 = f64::NAN;
    }
    if max_residual_z05 == f64::NEG_INFINITY {
        max_residual_z05 = f64::NAN;
    }

    CorridorStats {
        points,
        sat_count,
        sat_pct: if points > 0 {
            100.0 * sat_count as f64 / points as f64
        } else {
            0.0
        },
        max_residual,
        mean_residual,
        max_residual_z1,
        max_residual_z05,
    }
}

fn make_solver(kappa: f64) -> LosslessSolver {
    LosslessSolver {
        landing_point: [0.0, 0.0, 0.0],
        initial_position: [10.0, 20.0, 50.0],
        initial_velocity: [0.0, 0.0, 0.0],
        max_velocity: 5.0,
        dry_mass: 50.0,
        fuel_mass: 30.0,
        alpha: 1.0 / (9.81 * 180.0),
        lower_thrust_bound: 1000.0 * 0.4,
        upper_thrust_bound: 1000.0,
        tvc_range_rad: 15_f64.to_radians(),
        min_time_s: 11.0,
        coarse_line_search_delta_t: 0.1,
        fine_line_search_delta_t: 0.01,
        coarse_delta_t: 0.05,
        fine_delta_t: 0.025,
        delta_t: 0.025,
        use_glide_slope: true,
        glide_slope: 45_f64.to_radians(),
        use_terminal_lateral_soft_penalty: true,
        terminal_lateral_soft_penalty_ratio: kappa,
        terminal_lateral_soft_penalty_weight: 100000.0,
        N: 480,
        ..Default::default()
    }
}

fn main() {
    let kappas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];

    println!(
        "kappa,solved,tof_s,points,sat_pct,max_res,mean_res,max_res_z<=1m,max_res_z<=0.5m"
    );

    for kappa in kappas {
        let mut solver = make_solver(kappa);
        let attempt = solver.solve_at_current_time();

        if let Some(traj) = attempt.trajectory {
            let stats = compute_stats(&traj, kappa, 0.0, 0.0, 0.0);
            println!(
                "{:.6},true,{:.3},{},{:.3},{:.6},{:.6},{:.6},{:.6}",
                kappa,
                traj.time_of_flight_s,
                stats.points,
                stats.sat_pct,
                stats.max_residual,
                stats.mean_residual,
                stats.max_residual_z1,
                stats.max_residual_z05
            );
        } else {
            println!("{:.6},false,NaN,0,0,NaN,NaN,NaN,NaN", kappa);
        }
    }
}
