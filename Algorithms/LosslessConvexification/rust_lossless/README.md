# `rust_lossless`

This crate contains the ZOH lossless-convexification solver and a minimal runner for generating a single trajectory.

## What Stays In This Crate

- `src/lossless.rs`: the ZOH solver and result types.
- `src/lib.rs`: library export for other Rust crates.
- `src/main.rs`: a simple executable that runs one ZOH solve and writes `trajectory.csv`.
- `src/inspect_rust_lossless.py`: plots the 2D/side-view trajectory from a trajectory CSV.
- `src/inspect_rust_lossless_3d.py`: plots the 3D trajectory from a trajectory CSV.
- `src/inspect_rust_lossless_control.py`: plots thrust and control history from a trajectory CSV.

Everything related to comparing multiple methods, aggregating runs, or generating comparison figures now lives in `../lossless_compare`.

Legacy comparison outputs and archived truth/reference files were moved to `../lossless_compare/output`.

## Quick Start

Run a single solve:

```powershell
cargo run --bin rust_lossless
```

That writes:

- `trajectory.csv`: the latest trajectory sample from `src/main.rs`

Generate plots from that output:

```powershell
python src/inspect_rust_lossless.py trajectory.csv
python src/inspect_rust_lossless_3d.py trajectory.csv
python src/inspect_rust_lossless_control.py trajectory.csv
```

Those scripts produce PNGs next to the CSV unless you edit their arguments or defaults.

## Single-Run Workflow

1. Edit the solver setup in `src/main.rs`.
2. Run `cargo run --bin rust_lossless`.
3. Inspect `trajectory.csv`.
4. Run one of the plotting scripts to visualize the trajectory or controls.

## Python Script Reference

### `src/inspect_rust_lossless.py`

Reads a trajectory CSV and makes a standard 2D inspection plot for position and trajectory shape.

### `src/inspect_rust_lossless_3d.py`

Reads the same CSV format and renders the trajectory in 3D.

### `src/inspect_rust_lossless_control.py`

Reads the trajectory CSV and plots thrust/control quantities over the trajectory.
