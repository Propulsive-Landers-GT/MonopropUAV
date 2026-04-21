# Lossless Convexification Folder

A folder containing all files related to our implementation of programatically solving the soft landing problem with lossless convexification.

## Features

- **Rust Naive Lossless Implementation**: A Rust-based implementation of the naive method of time-based discretization of the lossless convexification problem. This is implemented as a struct, so it can be instantiated and easily used in other Rust files. This will be our baseline Rust implementation. This handles the entire solving of the landing problem and returns the entire computed trajectory.
- **Python Naive Lossless Implementation**: This is a rougher implemntation that is done in Python to serve as a sort of ground truth for how the lossless implementations should work. The file is known to be correct, so setting the same initial conditions between two implementations will allow us to compare if the new implementation is correct.
- **Trajectory Visualization**: Can plot the position, velocity, mass, and thrust of a trajectory from a CSV file. This is useful in determining the validity of the solver, as well as in predicting the fuel usage over the course of a trajectory for mass estimation and propellant purposes.


## Project Structure

Files such as image and CSV files are excluded here for the sake of brevity.

```
LosslessConvexification/
├── rust_lossless/src/
│   ├── chebyshevlossless.rs       # Template for the creation of the Chebyshev polynomial implemtation of lossless
│   ├── inspect_rust_lossless.py   # Plots and stores position, velocity, mass, and thrust throughout a trajectory CSV
│   ├── lossless.rs                # Naive Rust lossless implementation
│   ├── main.rs                    # Rust file to run the Rust lossless and create a CSV of trajectory
│   └── lossless_venv.sh           # Creates a virtual environment with all packages installed for running inspect_rust_lossless.py
├── lossless_hover_estimator.py    # Estimates the propellant consumption of hovering
├── lossless_venv.sh               # Creates a virtual environment with all packages installed for running the Python lossless
├── lossless.py                    # The Python implementation of lossless convexification
└── parse_trajectory.py            # Prints out the total impulse from the Python lossless
```


## File Usage

### Naive Rust Lossless (lossless.rs)

This calculates the trajectory by first running a rough pass with a high `dt` to find the time frame of the trajectory, then a second calculation with a lower `dt` to produce a higher resolution solution. It returns the solution in the form of a `TrajectoryResult`, which contains the position, velocity, mass, thrust vector, and thrust bound in the form of `Vec<f64>` and `Vec<[f64; 3]>` arrays. To use, simply instantiate the solver with all the parameters filled in, call `solve()` on the solver struct, and store/use the returned `TrajectoryResult` as necessary.

**Example Usage**
```rust
mod lossless; // Adjust these imports as needed depending on where in the directory you are. This is used in a file in the same folder as lossless.rs.
use crate::lossless::LosslessSolver;

fn main() {
    let mut solver = LosslessSolver {
        landing_point: [0.0, 0.0, 0.0], // This is the point where you want to end up.
        initial_position: [0.0, 0.0, 50.0], // This is the point where you start from.
        initial_velocity: [0.0, 0.0, 0.0], // This is the velocity you start with.
        max_velocity: 500.0, // This is the maximum velocity the vehicle can/should achieve in flight.
        dry_mass: 50.0, // This is the mass of the vehicle, without any fuel/propellant on board.
        fuel_mass: 30.0, // This is the mass of the fuel/propellant.
        alpha: 1.0/(9.81 * 180.0), // This is the conversion ratio from thrust to delta mass.
        lower_thrust_bound: 1000.0 * 0.4, // This is the minimum thrust the vehicle must keep.
        upper_thrust_bound: 1000.0, // This is the maximmum thrust the vehicle can attain.
        tvc_range_rad: 15_f64.to_radians(), // This is the range from the vertical axis that the thrust vector control can deviate.
        coarse_delta_t: 0.5, // This is the dt used to solve for the time frame of the trajectory.
        fine_delta_t: 0.05, // This is the dt used to solve for the higher resolution trajectory.
        use_glide_slope: false, // Enables the glide-slope corridor constraint.
        glide_slope: 5_f64.to_radians(), // Half-angle used to define the allowable lateral-vs-vertical excursion from each endpoint.
        N: 20, // This is the number of time steps the solver uses. It is set here, but is recalculated internally solve() is called. This is simply exposed so that the number of time steps can be accessed externally, if necessary.
        ..Default::default()
    };
    
    let mut traj_result = solver.solve();
    

    match traj_result {
        Some(result) => {
            println!("Final position: {:?}", result.positions.last().unwrap());
            println!("Final velocity: {:?}", result.velocities.last().unwrap());
            println!("Final mass: {:?}", result.masses.last().unwrap());
            println!("Final thrust: {:?}", result.thrusts.last().unwrap());
            
            write_trajectory_to_csv("trajectory.csv", &result)
                .expect("Failed to write CSV");
        }
        None => {
            eprintln!("Solve failed!");
        }
    }
}
```

**Glide Slope Constraint**

The Rust ZOH lossless solver does not use the older one-sided "stay above the landing cone" interpretation anymore. Instead, it applies a two-sided corridor between the start point and end point.

Let the state position at step `k` be `p_k = [x_k, y_k, z_k]^T`, and let `p_s` and `p_f` be the start and finish positions. Define the lateral distance from an anchor point `a` as:

```text
r_a(p_k) = || [x_k - a_x, y_k - a_y] ||_2
```

For a glide-slope angle `gamma`, the solver enforces:

```text
r_a(p_k) <= |z_k - a_z| / tan(gamma)
```

using second-order-cone constraints. The solver applies those cones from both endpoints, but with the direction chosen so the trajectory stays between the endpoint altitudes:

- If `z_start >= z_end`, then every node must stay above the landing-point cone and below the start-point cone.
- If `z_start < z_end`, then every node must stay above the start-point cone and below the landing-point cone.

That means the trajectory is kept inside a corridor that widens laterally away from each endpoint while also preventing the rocket from going far above the higher endpoint or far below the lower endpoint.

Mathematically, each bound is written in SOC form as either:

```text
|| [x_k - a_x, y_k - a_y] ||_2 <= (z_k - a_z) / tan(gamma)
```

or

```text
|| [x_k - a_x, y_k - a_y] ||_2 <= (a_z - z_k) / tan(gamma)
```

depending on whether that anchor provides the lower or upper vertical bound.

### Chebyshev Rust Lossless (chebyshev_lossless.rs)
This is currently incomplete and mostly a copy of `lossless.rs` with some TODO comments attached to areas that need to be changed for the Chebyshev parametization version of the lossless problem. However, in the future, it should function similarly to `lossless.rs` and contain a solver struct that returns a `TrajectoryResult` struct.

### Python Lossless (lossless.py)
This is fairly straightforward to use. Since this is mostly just a rough draft, it is not intended to be instantiated into other Python classes. TO change parameters, adjust lines `123-141` for the initial constraints and rough `dt` and see line `247` for the fine `dt`. The Python file still contains the older commented one-sided glide-slope cone near the landing point; the Rust ZOH solver is the more up-to-date reference for the current two-sided corridor implementation.
To actually run it, simply activate the venv then run `python3 lossless.py` from the main `LosslessConvexification` folder. It should automatically save images and plots of the trajectory, as well as produce a CSV file containing the trajectory information.

### Python Venvs (lossless_venv.sh)
This applies for both of the `lossless_venv.sh`. To run, simply navigate to the directory of the `lossless_venv.sh` and run `./lossless_venv.sh`. Then, after the script is done running, run `source ./venv/bin/activate` to activate and enter the virtual environment. Now, you'll have whatever packages you need to run whatever Python files are in that directory. You can also run Rust files in this as well, making debugging Rust implementations easier.

## Python Visualization and Helper Files (inspect_rust_lossless.py, lossless_hover_estimator.py, parse_trajectory.py)
For `inspect_rust_lossless.py` and `parse_trajectory.py`, run the lossless implemetation in their directories, then directly run the python files with `python3 [filename]`. These will display and save additional information about the trajectories that can be useful for debugging or propellant estimation.
For `lossless_hover_estimator.py`, adjust the parameters at the top of the file and run it with `python3 lossless_hover_estimator.py`. It will print out the total impulse over the hover period, remaining fuel mass after the hover, and remaining vehicle mass after the hover.
