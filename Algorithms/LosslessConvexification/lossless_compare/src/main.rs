use lossless_compare::chebyshev_lossless::ChebyshevLosslessSolver;
use rust_lossless::{
    LosslessSolver, SolveAttemptResult, SolveMetrics, SolveRunResult, TrajectoryResult,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

struct FineSolveRecord {
    group_name: String,
    run_name: String,
    method: String,
    fine_clarabel_solve_time_s: f64,
    time_of_flight_s: f64,
}

struct SimpleMetricsContext {
    zoh_coarse_dt_s: f64,
    zoh_fine_dt_s: f64,
    cgl_fine_search_delta_t_s: f64,
    cgl_fine_nodes: usize,
}

fn main() {
    let initial_position = [10.0, 20.0, 50.0];
    let max_velocity = 5.0;
    let mut zoh_solver = LosslessSolver {
        landing_point: [0.0, 0.0, 0.0],
        initial_position,
        initial_velocity: [0.0, 0.0, 0.0],
        max_velocity,
        dry_mass: 50.0,
        fuel_mass: 30.0,
        alpha: 1.0 / (9.81 * 180.0),
        lower_thrust_bound: 1000.0 * 0.4,
        upper_thrust_bound: 1000.0,
        tvc_range_rad: 15_f64.to_radians(),
        coarse_line_search_delta_t: 0.1,
        fine_line_search_delta_t: 0.01,
        coarse_delta_t: 0.025,
        fine_delta_t: 0.01,
        use_terminal_lateral_hard_tube: false,
        terminal_lateral_hard_tube_time_s: 0.5,
        terminal_lateral_hard_tube_radius_m: 0.05,
        N: 20,
        ..Default::default()
    };

    let mut chebyshev_solver = ChebyshevLosslessSolver {
        landing_point: [0.0, 0.0, 0.0],
        initial_position,
        initial_velocity: [0.0, 0.0, 0.0],
        max_velocity,
        dry_mass: 50.0,
        fuel_mass: 30.0,
        alpha: 1.0 / (9.81 * 180.0),
        lower_thrust_bound: 1000.0 * 0.4,
        upper_thrust_bound: 1000.0,
        tvc_range_rad: 15_f64.to_radians(),
        coarse_line_search_delta_t: 0.1,
        fine_line_search_delta_t: 0.01,
        coarse_nodes: 15,
        fine_nodes: 6,
        use_glide_slope: true,
        glide_slope: 10_f64.to_radians(),
        use_terminal_lateral_hard_tube: false,
        terminal_lateral_hard_tube_time_s: 0.5,
        terminal_lateral_hard_tube_radius_m: 0.05,
        ..Default::default()
    };

    let group_name = "offset_limited_descent_moreglide";
    let truth_name = "trajectory_zoh_truth_5_vel_limit_offset.csv";
    let run_type = "long";
    let solver_groups_to_run: Vec<&str> = vec!["zoh"];
    let fine_timing_samples_per_group = 1;
    let comparison_nodes = 100;

    let run_label = format!("{}_{}", group_name, run_type);
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_archive_root = project_root.join("output");
    let output_root = project_root.join(group_name).join(run_type);
    std::fs::create_dir_all(&output_root).expect("Failed to create output root directory");

    let simple_metrics_context = SimpleMetricsContext {
        zoh_coarse_dt_s: zoh_solver.coarse_delta_t,
        zoh_fine_dt_s: zoh_solver.fine_delta_t,
        cgl_fine_search_delta_t_s: chebyshev_solver.fine_line_search_delta_t,
        cgl_fine_nodes: chebyshev_solver.fine_nodes,
    };

    let mut fine_records: Vec<FineSolveRecord> = Vec::new();

    for solver_group in &solver_groups_to_run {
        let solver_group = *solver_group;
        if solver_group != "zoh" && solver_group != "cgl" {
            eprintln!(
                "Unsupported solver group '{}'. Expected 'zoh' and/or 'cgl'.",
                solver_group
            );
            continue;
        }

        println!("=== Solving {} ===", solver_group);
        let run_dir = output_root.join(solver_group);
        std::fs::create_dir_all(&run_dir).expect("Failed to create run output directory");

        let trajectory_run_name = format!("{}_{}_1", solver_group, run_label);
        let solve_run = if solver_group == "zoh" {
            zoh_solver.solve()
        } else {
            chebyshev_solver.solve()
        };

        let solve_metrics_path = run_dir.join("solve_metrics.csv");
        if solve_metrics_path.exists() {
            std::fs::remove_file(&solve_metrics_path)
                .expect("Failed to clear existing solve_metrics.csv");
        }
        append_solve_metrics_to_csv(&solve_metrics_path, &trajectory_run_name, &solve_run)
            .expect("Failed to write solve metrics CSV");

        let trajectory = match solve_run.trajectory {
            Some(result) => result,
            None => {
                eprintln!("{} initial solve failed, skipping timing samples.", solver_group);
                continue;
            }
        };

        println!("Final position: {:?}", trajectory.positions.last().unwrap());
        println!("Final velocity: {:?}", trajectory.velocities.last().unwrap());
        println!("Final mass: {:?}", trajectory.masses.last().unwrap());
        println!("Final thrust: {:?}", trajectory.thrusts.last().unwrap());

        let trajectory_path = run_dir.join(format!("trajectory_{}.csv", trajectory_run_name));
        write_trajectory_to_csv(&trajectory_path, &trajectory).expect("Failed to write CSV");

        let time_of_flight_s = trajectory.time_of_flight_s;

        for sample_idx in 1..=fine_timing_samples_per_group {
            let sample_run_name = format!("{}_{}_{}", solver_group, run_label, sample_idx);
            let fine_attempt = if solver_group == "zoh" {
                rerun_zoh_fine_attempt(&mut zoh_solver, time_of_flight_s)
            } else {
                rerun_cgl_fine_attempt(&mut chebyshev_solver, time_of_flight_s)
            };

            if fine_attempt.trajectory.is_none() {
                eprintln!(
                    "{} fine timing sample {} failed; not included in simple metrics.",
                    solver_group, sample_idx
                );
                continue;
            }

            fine_records.push(FineSolveRecord {
                group_name: run_group_name(&sample_run_name),
                run_name: sample_run_name,
                method: solver_group.to_string(),
                fine_clarabel_solve_time_s: fine_attempt.metrics.clarabel_solve_time_s,
                time_of_flight_s,
            });
        }
    }

    let simple_metrics_path = output_root.join("simple_solve_metrics.csv");
    merge_simple_solve_metrics_csv(
        &simple_metrics_path,
        &fine_records,
        fine_timing_samples_per_group,
        &simple_metrics_context,
        &solver_groups_to_run,
        group_name,
        run_type,
    )
    .expect("Failed to write simple solve metrics CSV");

    let truth_path = output_archive_root.join(truth_name);
    run_batch_compare(&output_root, &truth_path, comparison_nodes)
        .expect("Failed to run batch comparison against truth CSV");
}

fn rerun_zoh_fine_attempt(solver: &mut LosslessSolver, time_of_flight_s: f64) -> SolveAttemptResult {
    let previous_delta_t = solver.delta_t;
    let previous_n = solver.N;

    solver.delta_t = solver.fine_delta_t;
    let target_n = (time_of_flight_s / solver.fine_delta_t).round() as i64;
    solver.N = target_n.max(1);

    let attempt = solver.solve_at_current_time();

    solver.delta_t = previous_delta_t;
    solver.N = previous_n;
    attempt
}

fn rerun_cgl_fine_attempt(
    solver: &mut ChebyshevLosslessSolver,
    time_of_flight_s: f64,
) -> SolveAttemptResult {
    let previous_n = solver.N;
    solver.N = solver.fine_nodes;
    let attempt = solver.solve_at_current_time(time_of_flight_s);
    solver.N = previous_n;
    attempt
}

fn run_batch_compare(output_root: &Path, truth_csv: &Path, nodes: usize) -> std::io::Result<()> {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script_path = project_root.join("scripts").join("batch_compare_rust_lossless.py");

    let status = Command::new("python")
        .arg(script_path)
        .arg(output_root)
        .arg(truth_csv)
        .arg("--nodes")
        .arg(nodes.to_string())
        .current_dir(project_root)
        .status()?;

    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "batch_compare_rust_lossless.py exited with non-zero status",
        ));
    }
    Ok(())
}

fn write_trajectory_to_csv(path: &Path, traj: &TrajectoryResult) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "t,x,y,z,vx,vy,vz,mass,ux,uy,uz,sigma,thrust_force")?;

    for k in 0..traj.positions.len() {
        let x = &traj.positions[k];
        let v = &traj.velocities[k];
        let m = traj.masses[k];
        let sigma = if k < traj.sigmas.len() { traj.sigmas[k] } else { 0.0 };
        let u = if k < traj.thrusts.len() {
            &traj.thrusts[k]
        } else {
            &[0.0, 0.0, 0.0]
        };

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

fn append_solve_metrics_to_csv(
    path: &Path,
    run_name: &str,
    solve_run: &SolveRunResult,
) -> std::io::Result<()> {
    const SOLVE_METRICS_HEADER_V1: &str =
        "run_name,pass,attempts,successes,iterations,clarabel_solve_time_s,wall_time_s,last_status";
    const SOLVE_METRICS_HEADER_V2: &str =
        "run_name,pass,attempts,successes,iterations,clarabel_solve_time_s,wall_time_s,last_status,time_of_flight_s";

    ensure_solve_metrics_header(path, SOLVE_METRICS_HEADER_V1, SOLVE_METRICS_HEADER_V2)?;

    let file_exists = path.exists();
    let file = OpenOptions::new().append(true).create(true).open(path)?;
    let write_header = !file_exists || file.metadata()?.len() == 0;

    let mut writer = BufWriter::new(file);
    let total_time_of_flight_s = solve_run
        .trajectory
        .as_ref()
        .map(|trajectory| trajectory.time_of_flight_s);

    if write_header {
        writeln!(writer, "{}", SOLVE_METRICS_HEADER_V2)?;
    }

    write_solve_metrics_row(
        &mut writer,
        run_name,
        "coarse",
        &solve_run.coarse_metrics,
        solve_run.coarse_time_of_flight_s,
    )?;
    write_solve_metrics_row(
        &mut writer,
        run_name,
        "fine",
        &solve_run.fine_metrics,
        solve_run.fine_time_of_flight_s,
    )?;
    write_solve_metrics_row(
        &mut writer,
        run_name,
        "total",
        &solve_run.total_metrics,
        total_time_of_flight_s,
    )?;

    writer.flush()?;
    Ok(())
}

fn ensure_solve_metrics_header(path: &Path, old_header: &str, new_header: &str) -> std::io::Result<()> {
    if !path.exists() || path.metadata()?.len() == 0 {
        return Ok(());
    }

    let file_contents = std::fs::read_to_string(path)?;
    let mut lines = file_contents.lines();
    let existing_header = lines.next().unwrap_or_default().trim();

    if existing_header == new_header {
        return Ok(());
    }

    if existing_header == old_header {
        let mut migrated = String::new();
        migrated.push_str(new_header);
        migrated.push('\n');

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            migrated.push_str(line);
            migrated.push(',');
            migrated.push('\n');
        }

        std::fs::write(path, migrated)?;
        return Ok(());
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "Unsupported solve metrics CSV header in {}: {}",
            path.display(),
            existing_header
        ),
    ))
}

fn write_solve_metrics_row<W: Write>(
    writer: &mut W,
    run_name: &str,
    pass_name: &str,
    metrics: &SolveMetrics,
    time_of_flight_s: Option<f64>,
) -> std::io::Result<()> {
    let time_of_flight_str = time_of_flight_s
        .map(|time| format!("{:.6}", time))
        .unwrap_or_default();

    writeln!(
        writer,
        "{},{},{},{},{},{:.6},{:.6},{:?},{}",
        run_name,
        pass_name,
        metrics.attempts,
        metrics.successes,
        metrics.iterations,
        metrics.clarabel_solve_time_s,
        metrics.wall_time_s,
        metrics.last_status,
        time_of_flight_str
    )
}

fn run_group_name(run_name: &str) -> String {
    if let Some((prefix, suffix)) = run_name.rsplit_once('_') {
        if suffix.parse::<usize>().is_ok() {
            return prefix.to_string();
        }
    }
    run_name.to_string()
}

fn merge_simple_solve_metrics_csv(
    path: &Path,
    records: &[FineSolveRecord],
    timing_samples_per_group: usize,
    context: &SimpleMetricsContext,
    methods_to_replace: &[&str],
    group_name: &str,
    run_type: &str,
) -> std::io::Result<()> {
    let methods_set: HashSet<&str> = methods_to_replace.iter().copied().collect();
    let target_groups: HashSet<String> = methods_set
        .iter()
        .map(|method| format!("{}_{}_{}", method, group_name, run_type))
        .collect();

    let mut preserved_rows: Vec<Vec<String>> = Vec::new();
    if path.exists() && path.metadata()?.len() > 0 {
        let contents = std::fs::read_to_string(path)?;
        let mut lines = contents.lines();
        if let Some(header_line) = lines.next() {
            let existing_headers = split_simple_csv_line(header_line);
            for line in lines {
                if line.trim().is_empty() {
                    continue;
                }
                let fields = split_simple_csv_line(line);
                if let Some(normalized_row) =
                    normalize_simple_metrics_row(&existing_headers, &fields)
                {
                    let existing_group_name = normalized_row.get(0).cloned().unwrap_or_default();
                    let existing_method = normalized_row.get(2).cloned().unwrap_or_default();
                    let should_replace = methods_set.contains(existing_method.as_str())
                        && target_groups.contains(&existing_group_name);
                    if !should_replace {
                        preserved_rows.push(normalized_row);
                    }
                }
            }
        }
    }

    let new_rows = build_simple_solve_metrics_rows(records, timing_samples_per_group, context)?;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", simple_solve_metrics_header())?;

    for row in preserved_rows {
        writeln!(writer, "{}", row.join(","))?;
    }
    for row in new_rows {
        writeln!(writer, "{}", row.join(","))?;
    }

    writer.flush()?;
    Ok(())
}

fn build_simple_solve_metrics_rows(
    records: &[FineSolveRecord],
    timing_samples_per_group: usize,
    context: &SimpleMetricsContext,
) -> std::io::Result<Vec<Vec<String>>> {
    if timing_samples_per_group == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "timing_samples_per_group must be greater than 0",
        ));
    }

    let mut grouped: BTreeMap<String, Vec<&FineSolveRecord>> = BTreeMap::new();
    for record in records {
        grouped
            .entry(record.group_name.clone())
            .or_default()
            .push(record);
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    for (group_name, group_records) in grouped {
        for record in &group_records {
            rows.push(vec![
                group_name.clone(),
                record.run_name.clone(),
                record.method.clone(),
                format!("{:.6}", record.fine_clarabel_solve_time_s),
                format!("{:.6}", record.time_of_flight_s),
                format!("{:.6}", context.zoh_coarse_dt_s),
                format!("{:.6}", context.zoh_fine_dt_s),
                format!("{:.6}", context.cgl_fine_search_delta_t_s),
                context.cgl_fine_nodes.to_string(),
                String::new(),
                String::new(),
            ]);
        }

        for (set_idx, chunk) in group_records.chunks(timing_samples_per_group).enumerate() {
            if chunk.is_empty() {
                continue;
            }

            let avg = chunk
                .iter()
                .map(|record| record.fine_clarabel_solve_time_s)
                .sum::<f64>()
                / (chunk.len() as f64);

            let std_error = if chunk.len() > 1 {
                let variance = chunk
                    .iter()
                    .map(|record| {
                        let delta = record.fine_clarabel_solve_time_s - avg;
                        delta * delta
                    })
                    .sum::<f64>()
                    / ((chunk.len() - 1) as f64);
                variance.sqrt() / (chunk.len() as f64).sqrt()
            } else {
                0.0
            };

            let method = &chunk[0].method;
            let time_of_flight_s = chunk[0].time_of_flight_s;

            rows.push(vec![
                group_name.clone(),
                format!("set_{}_avg", set_idx + 1),
                method.clone(),
                String::new(),
                format!("{:.6}", time_of_flight_s),
                format!("{:.6}", context.zoh_coarse_dt_s),
                format!("{:.6}", context.zoh_fine_dt_s),
                format!("{:.6}", context.cgl_fine_search_delta_t_s),
                context.cgl_fine_nodes.to_string(),
                format!("{:.6}", avg),
                format!("{:.6}", std_error),
            ]);
        }
    }

    Ok(rows)
}

fn simple_solve_metrics_columns() -> &'static [&'static str] {
    &[
        "group_name",
        "run_name",
        "method",
        "fine_clarabel_solve_time_s",
        "time_of_flight_s",
        "zoh_coarse_dt_s",
        "zoh_fine_dt_s",
        "cgl_fine_search_delta_t_s",
        "cgl_fine_nodes",
        "set_avg_fine_clarabel_solve_time_s",
        "set_std_error_fine_clarabel_solve_time_s",
    ]
}

fn simple_solve_metrics_header() -> String {
    simple_solve_metrics_columns().join(",")
}

fn split_simple_csv_line(line: &str) -> Vec<String> {
    line.split(',').map(|value| value.to_string()).collect()
}

fn normalize_simple_metrics_row(headers: &[String], fields: &[String]) -> Option<Vec<String>> {
    if headers.is_empty() {
        return None;
    }

    let mut by_name: HashMap<&str, &str> = HashMap::new();
    for (idx, header) in headers.iter().enumerate() {
        let value = fields.get(idx).map(|value| value.as_str()).unwrap_or("");
        by_name.insert(header.as_str(), value);
    }

    let group_name = by_name.get("group_name").copied().unwrap_or("");
    let run_name = by_name.get("run_name").copied().unwrap_or("");
    let method = by_name.get("method").copied().unwrap_or("");
    if group_name.is_empty() || run_name.is_empty() || method.is_empty() {
        return None;
    }

    Some(
        simple_solve_metrics_columns()
            .iter()
            .map(|column| by_name.get(*column).copied().unwrap_or("").to_string())
            .collect(),
    )
}
