# Comparison Script Reference

These scripts are the post-processing layer for `lossless_compare`.

They are intentionally separate from the Rust executables: the Rust side generates trajectory and metrics CSVs, and the Python side compares, aggregates, and plots them.

## Workflow Order

1. Run `cargo run` in `lossless_compare` to generate raw outputs.
2. Use `batch_compare_rust_lossless.py` or `compare_rust_lossless.py` to compare trajectories.
3. Use `prepare_group_comparison_graph_data.py` to normalize group comparison outputs into one plotting dataset.
4. Use the plotting scripts to generate figures.
5. Use `render_run_config_tables.py` to render summary tables.
6. Use `refresh_clean_group_data_and_tables.py` to run the main post-processing steps in sequence.

## Script Index

### `batch_compare_rust_lossless.py`

Recursively scans a run directory, compares every compatible trajectory CSV against a baseline truth CSV, and writes `comparison_results.csv`.

Use this when you already have one baseline CSV and many generated trajectory CSVs beneath a directory tree.

### `compare_rust_lossless.py`

Contains the lower-level comparison logic used by the batch script.

It resolves CSV paths, samples comparable nodes, computes metrics such as RMSE-style trajectory differences, and writes comparison rows.

### `group_compare_rust_lossless.py`

Aggregates comparison results across scenario groups so you can compare solver behavior over many runs rather than one directory at a time.

### `prepare_group_comparison_graph_data.py`

Takes scattered comparison outputs and assembles a clean consolidated CSV used by the plotting and table-generation scripts.

This is the main “data cleaning” step between raw CSV outputs and publication-style figures.

### `plot_rmse_split_by_method.py`

Builds RMSE plots split by solver method from the cleaned aggregate dataset.

### `plot_avg_solve_time_grouped.py`

Builds grouped average solve-time plots from the cleaned aggregate dataset.

### `plot_max_thrust_rate_change_split_by_method.py`

Plots thrust-rate-change metrics split by solver method.

### `compute_max_thrust_rate_change.py`

Computes max thrust-rate-change values from trajectory/control outputs so they can be merged into the comparison dataset.

### `render_run_config_tables.py`

Renders summary tables, including run configuration tables, from the cleaned dataset.

### `refresh_clean_group_data_and_tables.py`

Convenience orchestrator for the common reporting pipeline.

It runs the core cleaning and plotting steps in order so you do not have to call each script manually.

### `backfill_simple_solve_metrics_std_error.py`

Fills in standard-error fields for older or partially populated `simple_solve_metrics.csv` files.

Use this when historical solve-metrics CSVs predate the current summary format.

### `lossless_venv.sh`

Shell helper for setting up the Python environment used by the reporting scripts.

Use it when you want a repeatable local environment for the plotting/reporting pipeline.
