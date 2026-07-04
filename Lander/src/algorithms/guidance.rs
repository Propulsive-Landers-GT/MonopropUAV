use super::GuidancePlanner;

#[allow(dead_code)]
pub struct Lossless {
    pub max_velocity: f64,
    pub dry_mass: f64,
    pub alpha: f64,
    pub lower_thrust_bound: f64,
    pub upper_thrust_bound: f64,
    pub tvc_range_rad: f64,
    pub coarse_delta_t: f64,
    pub fine_delta_t: f64,
    pub glide_slope: f64,
    pub use_glide_slope: bool,
    pub flip_glide_slope: bool,
}

impl Lossless {
    pub fn new() -> Self {
        let max_velocity = 5.0;
        let dry_mass = 10.0;
        let alpha = 1.0 / (9.81 * 180.0);
        let lower_thrust_bound = 0.0;
        let upper_thrust_bound = 50.0;
        let tvc_range_rad = 15_f64.to_radians();
        let coarse_delta_t = 0.25;
        let fine_delta_t = 0.1;
        let glide_slope = 0.05_f64.to_radians();
        let use_glide_slope = true;
        let flip_glide_slope = true;

        Self {
            max_velocity,
            dry_mass,
            alpha,
            lower_thrust_bound,
            upper_thrust_bound,
            tvc_range_rad,
            coarse_delta_t,
            fine_delta_t,
            glide_slope,
            use_glide_slope,
            flip_glide_slope,
        }
    }

    pub fn solve(&mut self, current_position: [f64; 3], current_velocity: [f64; 3], target_position: [f64; 3], propellant_mass: f64) -> Option<rust_lossless::TrajectoryResult> {
        let mut solver = rust_lossless::LosslessSolver {
            landing_point: target_position,
            initial_position: current_position,
            initial_velocity: current_velocity,
            max_velocity: self.max_velocity,
            dry_mass: self.dry_mass,
            fuel_mass: propellant_mass,
            alpha: self.alpha,
            lower_thrust_bound: self.lower_thrust_bound,
            upper_thrust_bound: self.upper_thrust_bound,
            tvc_range_rad: self.tvc_range_rad,
            coarse_delta_t: self.coarse_delta_t,
            fine_delta_t: self.fine_delta_t,
            use_bottom_glide_slope: self.use_glide_slope,
            use_top_glide_slope: self.use_glide_slope,
            glide_slope: self.glide_slope,
            N: 20,
            ..Default::default()
        };

        let result = solver.solve();
        result.trajectory
    }
}

impl GuidancePlanner for Lossless {
    fn solve(&mut self, current_position: [f64; 3], current_velocity: [f64; 3], target_position: [f64; 3], propellant_mass: f64) -> Option<rust_lossless::TrajectoryResult> {
        self.solve(current_position, current_velocity, target_position, propellant_mass)
    }
}
