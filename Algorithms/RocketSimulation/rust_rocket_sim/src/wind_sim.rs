use nalgebra::Vector3;
use rand::rng;
use rand_distr::{Normal, Distribution};

// Mean wind velocity at a given altitude, world frame (x=east, y=north, z=up)
#[derive(Debug, Clone)]
pub struct WindBand {
    pub altitude_m: f64,
    pub wind_mps: Vector3<f64>,
}

// Altitude-keyed wind profile, interpolated between bands
#[derive(Debug, Clone)]
pub struct WindProfile {
    pub bands: Vec<WindBand>,
}

impl WindProfile {
    pub fn new(bands: Vec<WindBand>) -> Self {
        let mut bands = bands;
        bands.sort_by(|a, b| a.altitude_m.partial_cmp(&b.altitude_m).unwrap());
        Self { bands }
    }

    pub fn default() -> Self {
        // Altitude-keyed mean wind profile [m/s], world frame (x=east, y=north, z=up).
        // These are rough placeholder values — replace with site-specific data before flight.
        Self::new(vec![
            WindBand { altitude_m:  0.0, wind_mps: Vector3::new(1.0, 0.0, 0.0) },   // near-ground, light crosswind
            WindBand { altitude_m: 15.0, wind_mps: Vector3::new(2.5, 0.5, 0.0) },   // mid-range, picking up
            WindBand { altitude_m: 35.0, wind_mps: Vector3::new(4.0, 1.5, 0.0) },   // upper range, higher shear
            WindBand { altitude_m: 60.0, wind_mps: Vector3::new(5.0, 2.0, 0.0) },  // near apogee
        ])
    }


    // Linear interpolation between the two nearest altitude bands
    pub fn sample(&self, altitude_m: f64) -> Vector3<f64> {
        if self.bands.is_empty() {
            return Vector3::zeros();
        }
        let n = self.bands.len();

        if altitude_m <= self.bands[0].altitude_m {
            return self.bands[0].wind_mps;
        }
        if altitude_m >= self.bands[n - 1].altitude_m {
            return self.bands[n - 1].wind_mps;
        }

        for i in 0..n - 1 {
            let lo = &self.bands[i];
            let hi = &self.bands[i + 1];
            if altitude_m >= lo.altitude_m && altitude_m <= hi.altitude_m {
                let t = (altitude_m - lo.altitude_m) / (hi.altitude_m - lo.altitude_m);
                return lo.wind_mps + t * (hi.wind_mps - lo.wind_mps);
            }
        }

        Vector3::zeros()
    }
}

// Gauss-Markov gust model (Ornstein-Uhlenbeck process)
// Real gusts are correlated in time, not white noise. tau_s controls how long
// a gust holds its value; sigma_mps controls intensity.
// A separate one-shot peak gust can be scheduled to stress-test the controller.
#[derive(Debug, Clone)]
pub struct GustModel {
    pub sigma_mps: Vector3<f64>,   // turbulence intensity per axis [m/s]
    pub tau_s: f64,                 // correlation time [s]
    pub gust_magnitude_mps: Vector3<f64>, // peak gust vector [m/s]
    pub gust_start_s: f64,
    pub gust_duration_s: f64,
    state: Vector3<f64>,            // internal Gauss-Markov state
}

impl GustModel {
    pub fn new(
        sigma_mps: Vector3<f64>,
        tau_s: f64,
        gust_magnitude_mps: Vector3<f64>,
        gust_start_s: f64,
        gust_duration_s: f64,
    ) -> Self {
        Self {
            sigma_mps,
            tau_s,
            gust_magnitude_mps,
            gust_start_s,
            gust_duration_s,
            state: Vector3::zeros(),
        }
    }

    pub fn default() -> Self {
        // Gauss-Markov turbulence: sigma is how intense, tau is how "smooth" gusts are.
        // One-shot peak gust fires at t=5s for 0.5s to stress-test the controller early.
        Self::new(
            Vector3::new(0.75, 0.5, 0.15), // sigma [m/s] per axis
            3.0,                          // tau [s] — 3 seconds is realistic for low-altitude turbulence
            Vector3::new(2.0, 0.0, 0.0), // peak gust direction + magnitude [m/s]
            5.0,                          // gust starts at t=5s
            0.5,                          // gust lasts 0.5s
        )
    }

    // Steps the gust model and returns stochastic wind velocity [m/s]
    pub fn step(&mut self, dt: f64, system_time: f64) -> Vector3<f64> {
        let mut rng = rng();

        // Exact discrete-time solution to the OU SDE
        let decay = (-dt / self.tau_s).exp();
        let noise_scale = (1.0 - (-2.0 * dt / self.tau_s).exp()).max(0.0).sqrt();

        let noise = Vector3::new(
            Normal::new(0.0, self.sigma_mps.x).unwrap().sample(&mut rng),
            Normal::new(0.0, self.sigma_mps.y).unwrap().sample(&mut rng),
            Normal::new(0.0, self.sigma_mps.z).unwrap().sample(&mut rng),
        );

        self.state = decay * self.state + noise_scale * noise;

        // One-shot peak gust: full magnitude at start, decays linearly to zero
        let peak_gust = if system_time >= self.gust_start_s
            && system_time < self.gust_start_s + self.gust_duration_s
        {
            let progress = (system_time - self.gust_start_s) / self.gust_duration_s;
            self.gust_magnitude_mps * (1.0 - progress)
        } else {
            Vector3::zeros()
        };

        self.state + peak_gust
    }
}

// Combines a steady altitude profile with stochastic gusts
#[derive(Debug, Clone)]
pub struct WindModel {
    pub profile: WindProfile,
    pub gusts: GustModel,
}

impl WindModel {
    pub fn new(profile: WindProfile, gusts: GustModel) -> Self {
        Self { profile, gusts }
    }

    pub fn default() -> Self {
        // These are rough placeholder values — replace with site-specific data before flight.
        let profile = WindProfile::default();

        let gusts = GustModel::default();

        Self::new(profile, gusts)
    }

    // Returns total wind velocity [m/s] in world frame at the given altitude.
    // Subtract from rocket velocity to get velocity relative to the airmass.
    pub fn step(&mut self, altitude_m: f64, dt: f64, system_time: f64) -> Vector3<f64> {
        let mean = self.profile.sample(altitude_m);
        let gust = self.gusts.step(dt, system_time);
        mean + gust
    }
}
