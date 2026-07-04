# `lossless_compare`

This crate owns the multi-method comparison workflow.

It depends on `../rust_lossless` for the ZOH solver and keeps the Chebyshev solver, comparison runner, sweep binaries, and Python reporting scripts in one place.

Archived run outputs, figures, and baseline truth/reference CSVs live under `output/`.

## What This Crate Does

- Runs one or more solver methods on the same scenario.
- Writes trajectory CSVs and solve-metrics CSVs for each method.
- Compares those trajectories against a baseline truth CSV.
- Aggregates the results into group-level CSVs and figures.

## Layout

- `src/main.rs`: the top-level comparison runner.
- `src/chebyshev_lossless.rs`: the Chebyshev-based comparison solver.
- `src/bin/test_chebyshev.rs`: quick Chebyshev smoke test.
- `src/bin/sweep_soft_corridor.rs`: parameter sweep for the soft terminal corridor.
- `scripts/`: post-processing and plotting scripts.

## End-To-End Process

1. Configure the scenario and enabled methods in `src/main.rs`.
2. Run the comparison executable.
3. The executable writes method-specific outputs under a run directory such as `offset_limited_descent_moreglide/long/`.
4. The executable calls `scripts/batch_compare_rust_lossless.py` to compare those outputs against a baseline truth CSV from `output/`.
5. Use the scripts in `scripts/` to merge, summarize, and plot results across groups and run types.

## Running The Comparison

From this directory:

```powershell
cargo run
```

`src/main.rs` currently:

- writes outputs under this crate directory
- looks up the baseline truth CSV under `output/`
- writes per-method `trajectory_*.csv`
- writes per-method `solve_metrics.csv`
- writes `simple_solve_metrics.csv`
- writes `comparison_results.csv`

## Common Commands

Run the main comparison:

```powershell
cargo run
```

Run the Chebyshev smoke test:

```powershell
cargo run --bin test_chebyshev
```

Run the soft-corridor sweep:

```powershell
cargo run --bin sweep_soft_corridor
```

Refresh derived figures and summary tables:

```powershell
python scripts/refresh_clean_group_data_and_tables.py
```

## Python Scripts

See [scripts/README.md](scripts/README.md) for a script-by-script explanation.
