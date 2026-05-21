use std::f64::consts::PI;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use ndarray::{array, Array1};

/// Struct to neatly package the 8 outputs from the solver
#[derive(Debug, Clone, Copy)]
pub struct FlowrateSolution {
    pub thrust_realized: f64,
    pub new_port_d: f64,
    pub fuel_mass: f64,
    pub mdot_ox: f64,
    pub pc_bar: f64,
    pub of_ratio_realized: f64,
    pub isp_realized: f64,
    pub cstar_realized: f64,
}

/// The output of the Angle Solver
#[derive(Debug, Clone, Copy)]
pub struct AngleSolution {
    pub theta_deg: f64,
    pub a_ev: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct FluidDynamicsOutput {
    pub thrust_realized: f64,
    pub valve_angle: f64,
    pub mdot_ox: f64,
    pub new_fuel_mass: f64,
    pub new_n2o_mass: f64,
    pub new_n2o_level: f64,
    pub new_n2_mass_runtank: f64,
    pub new_n2_mass_storagetanks: f64,
    pub new_port_d: f64,
    pub of_ratio_realized: f64,
    pub isp_realized: f64,
    pub cstar_realized: f64,
    pub pc_bar: f64,

    // --- Pressure transducer tap points ---
    // TODO: p_downstream_mtv_bar currently aliases pc_bar (chamber pressure).
    //       Once the check valve is modeled, this should account for the
    //       pressure drop across the check valve between the MTV and engine.
    /// Pressure downstream of the MTV, before the check valve [bar]
    pub p_downstream_mtv_bar: f64,
    /// N2O run tank pressure (N2 ullage via ideal gas law) [bar]
    pub p_runtank_bar: f64,
    /// Pressure upstream of the run tank, downstream of r_mv [bar]
    pub p_upstream_runtank_bar: f64,
}

#[derive(Debug, Clone)]
pub struct ThermoFluidParameters {
    pub regression_a: f64,
    pub regression_n: f64,
    pub fuel_length: f64, // m
    pub fuel_od: f64, // m
    pub rho_f: f64, // solid fuel density (kg/m^3)
    pub throat_diameter: f64,
    pub cd_throat: f64, // throat discharge coefficient
    pub port_d: f64,
    pub isp_coeff: Array1<f64>,
    pub cstar_coeff: Array1<f64>,
    pub flowrate_solver_num_steps: i32,
    pub flowrate_solver_step_size: f64,
    pub nitrogen_iso: IsoData,
    pub nitrous_iso: IsoData,
    pub nist_data: NistData,
    pub p1_mpa: f64,
    pub t1_k: f64,
    pub a_inj: f64,
    pub cd_inj: f64,
    pub cd_valve: f64,
    pub runtank_vol: f64,
    pub tank_d: f64,
    pub temp: f64,
    pub n2_mass_flowrate_rcs: f64,

    /// Volume of the plumbing line segment between r_mv and the run tank [m³].
    /// When r_mv closes, nitrogen is trapped in this volume and its pressure
    /// decays as it slowly equalises with the run tank via leakage/diffusion.
    /// This is a placeholder value — refine with actual CAD measurements of
    /// the line length and inner diameter.
    pub line_vol_r_mv: f64,
}

impl ThermoFluidParameters {
    pub fn default() -> Self {
        Self {
            regression_a: 0.0000722,
            regression_n: 0.67,
            fuel_length: 0.1559,
            fuel_od: 0.177,
            rho_f: 856.0,
            throat_diameter: 14.56 / 1000.0,
            cd_throat: 0.95,
            port_d: 0.05,
            isp_coeff: array![1040.6, 595.11, -250.95, 60.909, -7.3421, 0.4167, -0.009],
            cstar_coeff: array![693.7, 383.09, -159.09, 38.88, -4.7368, 0.2715, -0.0059],
            flowrate_solver_num_steps: 560, // 0.001 to 0.560 kg/s
            flowrate_solver_step_size: 0.001,
            nitrogen_iso: IsoData::new("isobaric_nitrogen.csv"),
            nitrous_iso: IsoData::new("isobaric_liquid_nitrous_oxide.csv"),
            nist_data: NistData::new(),
            p1_mpa: 6.5,
            t1_k: 294.26,
            a_inj: 3.9528e-5,
            cd_inj: 0.8,
            cd_valve: 0.8,
            runtank_vol: 0.032,
            tank_d: 0.254,
            temp: 301.15,
            n2_mass_flowrate_rcs: 0.0085 * 2.0,
            line_vol_r_mv: 0.001,  // ~1 litre placeholder — refine with CAD
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThermoFluidSolver {
    pub parameters: ThermoFluidParameters,

    /// Tracked mass of nitrogen trapped in the line between r_mv and the
    /// run tank [kg].  When r_mv is open this is kept at the set-point
    /// density × line volume.  When r_mv closes, no more nitrogen enters
    /// and this mass is held constant while the trapped pressure decays
    /// via the ideal gas law as the line slowly equalises with the tank.
    pub n2_mass_line: f64,
}

impl ThermoFluidSolver {
    pub fn new(parameters: ThermoFluidParameters) -> Self {
        // Initialise the trapped line mass at the regulated set-point pressure.
        // P = m * R * T / V  =>  m = P * V / (R * T)
        let r_specific_n2 = 296.8; // J/(kg·K)
        let p_set_pa = parameters.p1_mpa * 1e6;
        let n2_mass_line = (p_set_pa * parameters.line_vol_r_mv)
            / (r_specific_n2 * parameters.temp);
        Self {
            parameters,
            n2_mass_line,
        }
    }

    pub fn default() -> Self {
        let params = ThermoFluidParameters::default();
        Self::new(params)
    }

    pub fn flowrate_solver(
        &mut self,
        thrust_commanded: f64,
        dt: f64,
    ) -> Option<FlowrateSolution> {
        // Fuel and engine parameters
        let a_throat: f64 = PI * (self.parameters.throat_diameter / 2.0).powi(2); // Throat Area in m^2
        
        // Track the closest match to the commanded thrust
        let mut best_diff = f64::INFINITY;
        let mut best_solution = None;

        // Equivalent to mdot_ox_arr = [0.001:0.001:0.56]
        for i in 1..=self.parameters.flowrate_solver_num_steps {
            let mdot_ox = (i as f64) * self.parameters.flowrate_solver_step_size;

            // 1. Solve for total port change (Dp_t)
            let power = 2.0 * self.parameters.regression_n + 1.0;
            let term1 = self.parameters.port_d.powf(power);
            let term2 = power * 2.0_f64.powf(power) * self.parameters.regression_a * mdot_ox.powf(self.parameters.regression_n) * dt;
            let term3 = PI.powf(self.parameters.regression_n);
            let dp_t = (term1 + (term2 / term3)).powf(1.0 / power);

            // 2. Solve for Oxidizer Mass Flux (Gox)
            let gox = (4.0 * mdot_ox) / (PI * dp_t.powi(2));

            // 3. Solve for Regression Rate
            let r_dot_t = self.parameters.regression_a * gox.powf(self.parameters.regression_n);

            // 4. Solve for Fuel Mass Flow Rate
            let mdot_f = self.parameters.rho_f * PI * dp_t * self.parameters.fuel_length * r_dot_t;

            // 5. O/F ratio and Total Mass Flow
            let of_rat = mdot_ox / mdot_f;
            let mdot_total = mdot_ox + mdot_f;

            // 6. Isp and C* Polynomials
            let mut isp = 0.0;
            for i in 0..self.parameters.isp_coeff.len() {
                isp += self.parameters.isp_coeff[i] * of_rat.powi(i as i32);
            }

            // Gemini says this also works, but im keeping the code readable for now:
            // let isp = self.parameters.isp_coeff.iter().fold(0.0, |acc, &c| acc * of_rat + c);

            // let isp = self.parameters.isp_coeff[6] * of_rat.powi(6)
            //     + self.parameters.isp_coeff[5] * of_rat.powi(5)
            //     + self.parameters.isp_coeff[4] * of_rat.powi(4)
            //     + self.parameters.isp_coeff[3] * of_rat.powi(3)
            //     + self.parameters.isp_coeff[2] * of_rat.powi(2)
            //     + self.parameters.isp_coeff[1] * of_rat
            //     + self.parameters.isp_coeff[0]; // m/s

            
            let mut cstar = 0.0;
            for i in 0..self.parameters.cstar_coeff.len() {
                cstar += self.parameters.cstar_coeff[i] * of_rat.powi(i as i32);
            }
            
            // Gemini says this also works, but im keeping the code readable for now:
            // let cstar = self.parameters.cstar_coeff.iter().fold(0.0, |acc, &c| acc * of_rat + c);

            // let cstar = self.parameters.cstar_coeff[6] * of_rat.powi(6)
            //     + self.parameters.cstar_coeff[5] * of_rat.powi(5)
            //     + self.parameters.cstar_coeff[4] * of_rat.powi(4)
            //     + self.parameters.cstar_coeff[3] * of_rat.powi(3)
            //     + self.parameters.cstar_coeff[2] * of_rat.powi(2)
            //     + self.parameters.cstar_coeff[1] * of_rat
            //     + self.parameters.cstar_coeff[0]; // m/s

            // 7. Calculate thrust
            let thrust = isp * mdot_total;

            // 8. Find thrust closest to desired thrust
            let diff = (thrust - thrust_commanded).abs();
            
            if diff < best_diff {
                best_diff = diff;
                
                // Calculate final geometry constraints for this specific flow rate
                let fuelm = self.parameters.rho_f * (PI / 4.0) * (self.parameters.fuel_od.powi(2) - dp_t.powi(2)) * self.parameters.fuel_length;
                let pc = (mdot_total * cstar * 0.85 / (a_throat * self.parameters.cd_throat)) / 100_000.0;

                // Save this as the current winning solution
                best_solution = Some(FlowrateSolution {
                    thrust_realized: thrust,
                    new_port_d: dp_t,
                    fuel_mass: fuelm,
                    mdot_ox,
                    pc_bar: pc,
                    of_ratio_realized: of_rat,
                    isp_realized: isp,
                    cstar_realized: cstar,
                });
            }
        }

        // Return the outputs for the closest matched thrust
        best_solution
    }

    /// The Main Angle Solver
    pub fn angle_solver(
        &mut self,
        m_dot_target: f64,
        pc_bar: f64,
    ) -> Result<AngleSolution, &'static str> {
        // 1. SYSTEM INPUTS
        let p_ch_mpa: f64 = pc_bar / 10.0;

        let p1: f64 = self.parameters.p1_mpa * 1e6;
        let p0: f64 = p_ch_mpa * 1e6;

        // 2. NIST DATA & INTERPOLATION
        let h_l_iso_j: Vec<f64> = self.parameters.nitrous_iso.h_l_iso_kj.iter().map(|&h| h * 1000.0).collect();
        let p_sat_pa: Vec<f64> = self.parameters.nist_data.p_sat_mpa.iter().map(|&p| p * 1e6).collect();

        let h1: f64 = Self::interp1(&self.parameters.nitrous_iso.t_iso, &h_l_iso_j, self.parameters.t1_k);
        let p_sat_at_t1: f64 = Self::interp1(&self.parameters.nist_data.t_sat, &p_sat_pa, self.parameters.t1_k);

        let pmin_sat: f64 = p_sat_pa.first().cloned().unwrap_or(0.0);
        let pmax_sat: f64 = p_sat_pa.last().cloned().unwrap_or(f64::MAX);

        // Safety Bounds
        let p_lo_lim: f64 = (p0 * 1.001).max(pmin_sat * 1.001);
        let p_hi_lim: f64 = (p1 * 0.999).min(pmax_sat * 0.999);

        if p_lo_lim >= p_hi_lim {
            // Chamber pressure is higher than tank pressure! Flow is choked/reversed.
            // panic!("Valid pressure range collapsed. Check P0/P1 vs saturation table bounds.");
            return Err("TooHigh");
        }

        // 3. PART A: Bracket search for a sign change (MATLAB's 60-point grid)
        let num_points = 60;
        let step = (p_hi_lim - p_lo_lim) / (num_points - 1) as f64;
        
        let mut p_br_lo = 0.0;
        let mut p_br_hi = 0.0;
        let mut found = false;
        let mut prev_f = self.dyer_model(p_lo_lim, p0, h1, p_sat_at_t1, self.parameters.a_inj, self.parameters.cd_inj) - m_dot_target;

        for i in 1..num_points {
            let p_guess = p_lo_lim + (i as f64) * step;
            let f_val = self.dyer_model(p_guess, p0, h1, p_sat_at_t1, self.parameters.a_inj, self.parameters.cd_inj) - m_dot_target;

            if prev_f.signum() != f_val.signum() {
                p_br_lo = p_guess - step; // The previous guess
                p_br_hi = p_guess;        // The current guess
                found = true;
                break;
            }
            prev_f = f_val;
        }

        // if !found {
        //     // panic!("No sign change found for f(P)=mdot-mdot_target. Requested mdot is likely impossible.");
        //     return None;
        // }
        if !found {
            if prev_f < 0.0 {
                // Dyer flow is LESS than target flow. The injector is maxed out.
                return Err("TooHigh");
            } else {
                // Dyer flow is MORE than target flow, even at the bottom of the NIST table.
                return Err("TooLow");
            }
        }

        // 4. BISECTION ROOT FINDER (Replaces MATLAB's fzero)
        let mut p2_solution = 0.0;
        for _ in 0..50 { // 50 iterations guarantees extreme sub-pascal precision
            p2_solution = (p_br_lo + p_br_hi) / 2.0;
            let f_mid = self.dyer_model(p2_solution, p0, h1, p_sat_at_t1, self.parameters.a_inj, self.parameters.cd_inj) - m_dot_target;

            if f_mid.abs() < 1e-9 { break; }

            let f_lo = self.dyer_model(p_br_lo, p0, h1, p_sat_at_t1, self.parameters.a_inj, self.parameters.cd_inj) - m_dot_target;
            if f_mid.signum() == f_lo.signum() {
                p_br_lo = p2_solution;
            } else {
                p_br_hi = p2_solution;
            }
        }

        // 5. PART B: Size the valve area A_ev
        let unit_flow_rate = self.dyer_model(p1, p2_solution, h1, p_sat_at_t1, 1.0, 1.0);

        if unit_flow_rate <= 0.0 {
            // panic!("Unit_Flow_Rate <= 0. Valve sizing cannot proceed.");
            return Err("TooHigh");
        }

        let a_ev = m_dot_target / (self.parameters.cd_valve * unit_flow_rate);

        // 6. COUPLING: Convert A_ev to valve mechanical angle
        let theta = -1e14 * a_ev.powi(3) + 1e10 * a_ev.powi(2) + 959373.0 * a_ev - 1.4629;

        Ok(AngleSolution {
            theta_deg: theta,
            a_ev,
        })
    }

    pub fn valve_angle_to_thrust(
        &mut self,
        target_angle_deg: f64,
        max_thrust: f64,
        dt: f64,
    ) -> FlowrateSolution {
        let mut thrust_lo = 0.0;
        let mut thrust_hi = max_thrust;

        // Default initialization using the minimum thrust
        let mut best_flow_solution = match self.flowrate_solver(thrust_lo, dt) {
            Some(flow_sol) => flow_sol,
            None => FlowrateSolution {
                thrust_realized: 0.0,
                new_port_d: self.parameters.port_d, // Safe fallback
                fuel_mass: 0.0,
                mdot_ox: 0.0,
                pc_bar: 0.0,
                of_ratio_realized: 0.0,
                isp_realized: 0.0,
                cstar_realized: 0.0,
            },
        };

        // 15 iterations provides ~0.01 Newton precision across a 600N range.
        // (Going higher is unnecessary since flowrate_solver discretizes mdot into 560 fixed steps anyway!)
        for _ in 0..15 { 
            let thrust_mid = (thrust_lo + thrust_hi) / 2.0;

            let flow_sol = match self.flowrate_solver(thrust_mid, dt) {
                Some(sol) => sol,
                None => FlowrateSolution {
                    thrust_realized: 0.0,
                    new_port_d: self.parameters.port_d, // Safe fallback
                    fuel_mass: 0.0,
                    mdot_ox: 0.0,
                    pc_bar: 0.0,
                    of_ratio_realized: 0.0,
                    isp_realized: 0.0,
                    cstar_realized: 0.0,
                },
            };
            
            match self.angle_solver(flow_sol.mdot_ox, flow_sol.pc_bar) {
                Ok(angle_sol) => {
                    best_flow_solution = flow_sol.clone();

                    if (angle_sol.theta_deg - target_angle_deg).abs() < 0.1 {
                        break;
                    }

                    if angle_sol.theta_deg < target_angle_deg {
                        thrust_lo = thrust_mid; 
                    } else {
                        thrust_hi = thrust_mid; 
                    }
                },
                Err(e) => {
                    // THE FIX: Route the binary search based on the physical cliff we hit!
                    if e == "TooHigh" {
                        thrust_hi = thrust_mid;
                    } else if e == "TooLow" {
                        thrust_lo = thrust_mid;
                    }
                }
            }
        }

        best_flow_solution
    }
    
    pub fn fluid_dynamics_update(
        &mut self,
        thrust_commanded: f64,
        o_iso_open: bool,
        r_mv_open: bool,
        o_vnt_open: bool,
        n2o_mass: f64,
        n2_mass_runtank: f64,
        n2_mass_total: f64,
        dt: f64,
    ) -> FluidDynamicsOutput {
        // Interpolate N2O Density
        let rho_n2o = Self::interp1(&self.parameters.nitrous_iso.t_iso, &self.parameters.nitrous_iso.rho_iso, self.parameters.temp);
        // Interpolate N2 Density
        let rho_n2 = Self::interp1(&self.parameters.nitrogen_iso.t_iso, &self.parameters.nitrogen_iso.rho_iso, self.parameters.temp);
        
        // 1. Calculate the required mass flow rate and resulting change in fuel grain
        //    o_iso must be open for oxidizer to flow to the engine
        let fr_sol = if o_iso_open {
            match self.flowrate_solver(thrust_commanded, dt) {
                Some(sol) => sol,
                None => FlowrateSolution {
                        thrust_realized: 0.0,
                        new_port_d: self.parameters.port_d,
                        fuel_mass: 0.0,
                        mdot_ox: 0.0,
                        pc_bar: 0.0,
                        of_ratio_realized: 0.0,
                        isp_realized: 0.0,
                        cstar_realized: 0.0,
                },
            }
        } else {
            // o_iso is closed — no oxidizer flows, no thrust
            FlowrateSolution {
                thrust_realized: 0.0,
                new_port_d: self.parameters.port_d,
                fuel_mass: 0.0,
                mdot_ox: 0.0,
                pc_bar: 0.0,
                of_ratio_realized: 0.0,
                isp_realized: 0.0,
                cstar_realized: 0.0,
            }
        };
        
        // 2. Safely calculate the MTV angle with a Wide-Open-Throttle failsafe
        let ang_sol = if o_iso_open && fr_sol.mdot_ox > 0.0 {
            match self.angle_solver(fr_sol.mdot_ox, fr_sol.pc_bar) {
                Ok(sol) => sol,
                Err(_) => AngleSolution {
                    theta_deg: 0.0,
                    a_ev: 0.0,
                },
            }
        } else {
            AngleSolution {
                theta_deg: 0.0,
                a_ev: 0.0,
            }
        };

        // 3. Mass tracking — oxidizer consumed
        let new_n2o_mass = n2o_mass - fr_sol.mdot_ox * dt;
        
        // 4. Nitrogen mass balance
        //    r_mv controls whether nitrogen flows from the storage tanks into the run tank
        //    to maintain ~65 bar pressure as nitrous is consumed.
        //
        //    The ullage volume grows as N2O is consumed. Nitrogen must fill that volume
        //    to hold the set-point pressure:
        //      delta_m_n2 = (P_set * delta_V_ullage) / (R_specific_n2 * T)
        //
        //    R_specific for N2 = 296.8 J/(kg·K)
        //    TODO: Update with actual regulator behavior / flow rate limits
        let r_specific_n2 = 296.8; // J/(kg·K)
        let p_set = self.parameters.p1_mpa * 1e6; // Convert MPa to Pa (65 bar = 6.5 MPa)

        let new_n2_mass_runtank;
        let mut new_n2_mass_storagetanks;

        if r_mv_open {
            // Nitrogen flows in to maintain constant pressure as nitrous depletes
            let v_ullage_old = self.parameters.runtank_vol - n2o_mass / rho_n2o;
            let v_ullage_new = self.parameters.runtank_vol - new_n2o_mass / rho_n2o;
            let delta_v_ullage = (v_ullage_new - v_ullage_old).max(0.0);

            // Mass of N2 needed to fill the new ullage at set pressure
            let delta_m_n2 = (p_set * delta_v_ullage) / (r_specific_n2 * self.parameters.temp);

            new_n2_mass_runtank = (self.parameters.runtank_vol - new_n2o_mass / rho_n2o) * rho_n2;
            new_n2_mass_storagetanks = (n2_mass_total - new_n2_mass_runtank - delta_m_n2).max(0.0);
        } else {
            // r_mv closed — masses are conserved; pressure drops naturally as ullage expands.
            // The ideal gas law at the PT tap point (below) captures the pressure change.
            // RCS N2 consumption is already applied upstream via rcs.update(), so
            // n2_mass_total already reflects it — no separate subtraction needed here.
            new_n2_mass_runtank = n2_mass_runtank;
            new_n2_mass_storagetanks = (n2_mass_total - n2_mass_runtank).max(0.0);
        }

        // 5. Vent valve — relieves run tank pressure by venting nitrogen
        //    TODO: Update with actual vent mass flow rate from hardware specs
        let o_vnt_mass_flowrate = 0.05; // kg/s placeholder
        let mut n2_mass_runtank_after_vent = new_n2_mass_runtank;
        if o_vnt_open {
            n2_mass_runtank_after_vent = (new_n2_mass_runtank - o_vnt_mass_flowrate * dt).max(0.0);
        }

        // 6. Calculate Nitrous fluid level
        let tank_cross_sectional_area = std::f64::consts::PI * (self.parameters.tank_d / 2.0).powi(2);
        let new_n2o_level = (new_n2o_mass / rho_n2o) / tank_cross_sectional_area;

        // ── Pressure transducer tap-point calculations ──────────────────────
        //
        // These compute the *true* physical pressures at the three PT
        // locations.  The PT sensors in device_sim then add their own
        // (very small) noise on top of these values.

        // m2-pt: pressure downstream of the MTV, before the check valve.
        // For now this is simply the chamber pressure from the flowrate
        // solver.  See the TODO on FluidDynamicsOutput for the future
        // check-valve pressure-drop extension.
        let p_downstream_mtv_bar = fr_sol.pc_bar;

        // o-pt: run tank pressure.  Total pressure = N2 partial pressure + N2O vapor pressure.
        //   P_N2  = m_N2 * R_N2 * T / V_ullage  (ideal gas law on the nitrogen)
        //   P_N2O = saturation pressure at tank temperature (from NIST table)
        // While liquid N2O is present the tank is two-phase and N2O vapor pressure
        // dominates (~52 bar at 28 °C), so omitting it would severely underestimate
        // the true tank pressure.
        let v_ullage = (self.parameters.runtank_vol - new_n2o_mass / rho_n2o).max(1e-9);
        let p_n2_pa = n2_mass_runtank_after_vent * r_specific_n2 * self.parameters.temp / v_ullage;
        let p_sat_pa: Vec<f64> = self.parameters.nist_data.p_sat_mpa.iter().map(|&p| p * 1e6).collect();
        let p_n2o_vapor_pa = Self::interp1(&self.parameters.nist_data.t_sat, &p_sat_pa, self.parameters.temp);
        let p_runtank_pa = p_n2_pa + p_n2o_vapor_pa;
        let p_runtank_bar = p_runtank_pa / 1e5;

        // oa-pt: pressure upstream of the run tank, downstream of r_mv.
        //
        // We model the short plumbing line between r_mv and the run tank
        // inlet as a small fixed volume (`line_vol_r_mv`).  This captures
        // the physical behaviour that:
        //   • When r_mv is OPEN the regulator maintains set-point pressure
        //     in the line, so we refill the trapped mass to match.
        //   • When r_mv is CLOSED no new nitrogen enters.  The trapped
        //     mass stays constant and the line pressure is simply the
        //     ideal-gas pressure of that trapped mass in the fixed volume.
        //     Over time, if the run tank were to draw from this volume
        //     (not yet modeled), the pressure would drop.
        //
        // This is an isothermal ideal-gas approximation — good enough for
        // GNC-level simulation fidelity.  Replace with a more detailed
        // model if transient accuracy during valve transitions matters.
        if r_mv_open {
            // Regulator keeps the line at set-point.  Recompute trapped mass
            // so that P_line = P_set.
            self.n2_mass_line = (p_set * self.parameters.line_vol_r_mv)
                / (r_specific_n2 * self.parameters.temp);
        }
        // else: r_mv is closed — n2_mass_line stays at whatever it was
        //       when the valve last closed, holding a fixed mass of gas.

        let p_upstream_runtank_pa = self.n2_mass_line * r_specific_n2
            * self.parameters.temp / self.parameters.line_vol_r_mv;
        let p_upstream_runtank_bar = p_upstream_runtank_pa / 1e5;

        // 7. Return outputs neatly packaged
        FluidDynamicsOutput {
            thrust_realized: fr_sol.thrust_realized,
            valve_angle: ang_sol.theta_deg,
            mdot_ox: fr_sol.mdot_ox,
            new_fuel_mass: fr_sol.fuel_mass,
            new_n2o_mass,
            new_n2o_level,
            new_n2_mass_runtank: n2_mass_runtank_after_vent,
            new_n2_mass_storagetanks,
            new_port_d: fr_sol.new_port_d,
            of_ratio_realized: fr_sol.of_ratio_realized,
            isp_realized: fr_sol.isp_realized,
            cstar_realized: fr_sol.cstar_realized,
            pc_bar: fr_sol.pc_bar,
            p_downstream_mtv_bar,
            p_runtank_bar,
            p_upstream_runtank_bar,
        }
    }
        
    /// Fast 1D Linear Interpolator (equivalent to MATLAB's `interp1` with 'extrap')
    pub fn interp1(x: &[f64], y: &[f64], x_query: f64) -> f64 {
        let n = x.len();

        // --- THE FIX: Guard against empty or 1-element arrays ---
        if n == 0 {
            return 0.0; // Return a default value if the array is empty
        }
        if n == 1 {
            return y[0]; // If there's only one data point, return it
        }
        // Clamp to minimum boundary
        if x_query <= x[0] { return y[0]; }
        // Clamp to maximum boundary
        if x_query >= x[x.len() - 1] { return y[y.len() - 1]; }
        // --------------------------------------------------------

        if x_query <= x[0] {
            // Extrapolate left
            let slope = (y[1] - y[0]) / (x[1] - x[0]);
            return y[0] + slope * (x_query - x[0]);
        }
        if x_query >= x[n - 1] {
            // Extrapolate right
            let slope = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
            return y[n - 1] + slope * (x_query - x[n - 1]);
        }
        // Interpolate
        for i in 0..n - 1 {
            if x_query >= x[i] && x_query <= x[i + 1] {
                let slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
                return y[i] + slope * (x_query - x[i]);
            }
        }
        0.0
    }

    /// The Dyer Model (Computes mass flow rate through an orifice)
    fn dyer_model(
        &mut self,
        p_in: f64, p_out: f64, h_in: f64, p_sat_in: f64, 
        area: f64, cd: f64
    ) -> f64 {
        if p_in <= p_out {
            return 0.0;
        }
        // Convert NIST MPa array to Pa on the fly for lookups
        let p_sat_pa: Vec<f64> = self.parameters.nist_data.p_sat_mpa.iter().map(|&p| p * 1e6).collect();
        let h_l_j: Vec<f64> = self.parameters.nist_data.h_l_kj.iter().map(|&h| h * 1000.0).collect();
        let h_v_j: Vec<f64> = self.parameters.nist_data.h_v_kj.iter().map(|&h| h * 1000.0).collect();

        let hl_in = Self::interp1(&p_sat_pa, &h_l_j, p_in);
        let hv_in = Self::interp1(&p_sat_pa, &h_v_j, p_in);
        let hl_out = Self::interp1(&p_sat_pa, &h_l_j, p_out);
        let hv_out = Self::interp1(&p_sat_pa, &h_v_j, p_out);
        let rhol_in = Self::interp1(&p_sat_pa, &self.parameters.nist_data.rho_l, p_in);
        let rhol_out = Self::interp1(&p_sat_pa, &self.parameters.nist_data.rho_l, p_out);
        let rhov_out = Self::interp1(&p_sat_pa, &self.parameters.nist_data.rho_v, p_out);

        let denom_in = hv_in - hl_in;
        let x_in = if denom_in.abs() < 1e-9 { 0.0 } else { ((h_in - hl_in) / denom_in).clamp(0.0, 1.0) };

        let m_dot_spi = cd * area * (2.0 * rhol_in * (p_in - p_out)).max(0.0).sqrt();

        let denom_out = hv_out - hl_out;
        let x_out = if denom_out.abs() < 1e-9 { 0.0 } else { ((h_in - hl_out) / denom_out).clamp(0.0, 1.0) };

        let rho_mix_out = 1.0 / ((x_out / rhov_out) + ((1.0 - x_out) / rhol_out));
        let h_mix_out = hl_out + x_out * (hv_out - hl_out);

        let m_dot_hem = if h_in <= h_mix_out {
            0.0
        } else {
            cd * area * rho_mix_out * (2.0 * (h_in - h_mix_out)).max(0.0).sqrt()
        };

        let k = if x_in > 0.0 {
            0.0
        } else {
            let delta_p_subcool = p_in - p_out;
            if delta_p_subcool <= 1e-5 {
                0.0
            } else {
                (delta_p_subcool / (p_sat_in - p_out).max(1e-6)).sqrt()
            }
        };

        (1.0 - 1.0 / (1.0 + k)) * m_dot_spi + (1.0 / (1.0 + k)) * m_dot_hem
    }
}


/// Holds the Isobaric Fluid Data (State 1 Tank Properties)
#[derive(Clone, Debug, Default)]
pub struct IsoData {
    pub t_iso: Vec<f64>,       // [K]
    pub rho_iso: Vec<f64>,     // [kg/m^3]  <-- ADD THIS
    pub h_l_iso_kj: Vec<f64>,  // [kJ/kg]
}

impl IsoData {
    pub fn new(file_name : &str) -> Self {
        if file_name == "isobaric_liquid_nitrous_oxide.csv" {
            return Self::parse_csv(include_str!("isobaric_liquid_nitrous_oxide.csv"));
        } else {
            return Self::parse_csv(include_str!("isobaric_nitrogen.csv"));
        }
    }

    pub fn parse_csv(csv_content: &str) -> Self {
        // Bakes the CSV directly into the executable at compile time
        // (Adjust path to "../N2O_NIST_sat_table.csv" if the file is in your root folder)        
        let mut data = IsoData::default();
        let mut lines = csv_content.lines();

        // 1. Skip the header row
        lines.next();

        // 2. Parse the data rows
        for line in lines {
            let text: &str = line;
            if text.trim().is_empty() {
                continue;
            }

            let cols: Vec<&str> = text.split(',').collect();

            // 0: Temperature, 1: Pressure, 2: Density, ..., 5: Enthalpy
            if cols.len() >= 6 {
                data.t_iso.push(cols[0].trim().parse::<f64>().expect("Failed to parse t_iso"));
                data.rho_iso.push(cols[2].trim().parse::<f64>().expect("Failed to parse rho_iso"));
                data.h_l_iso_kj.push(cols[5].trim().parse::<f64>().expect("Failed to parse h_l_iso_kj"));
            }
        }
        
        data
    }
}

/// Holds the Saturated Fluid Data (Expansion/Flashing Properties)
#[derive(Clone, Debug, Default)]
pub struct NistData {
    pub t_sat: Vec<f64>,       // [K]
    pub p_sat_mpa: Vec<f64>,   // [MPa]
    pub rho_l: Vec<f64>,       // [kg/m^3]
    pub rho_v: Vec<f64>,       // [kg/m^3]
    pub h_l_kj: Vec<f64>,      // [kJ/kg]
    pub h_v_kj: Vec<f64>,      // [kJ/kg]
    pub s_l_kj: Vec<f64>, 
    pub s_v_kj: Vec<f64>,
}

impl NistData {
    pub fn new() -> Self {
        Self::parse_csv(include_str!("N2O_NIST_sat_table.csv"))
    }

    pub fn parse_csv(csv_content: &str) -> Self {
        // Bakes the CSV directly into the executable at compile time
        // (Adjust path to "../N2O_NIST_sat_table.csv" if the file is in your root folder)        
        let mut data = NistData::default();
        let mut lines = csv_content.lines();

        // 1. Skip the header row
        lines.next();

        // 2. Parse the data rows
        for line in lines {
            let text: &str = line;
            if text.trim().is_empty() {
                continue;
            }

            let cols: Vec<&str> = text.split(',').collect();

            // 0: Temperature_K, 1: Pressure_MPa, 2: rho_L, 3: h_L, 4: s_L, 5: rho_V, 6: h_V
            if cols.len() >= 7 {
                data.t_sat.push(cols[0].trim().parse::<f64>().expect("Failed to parse t_sat"));
                data.p_sat_mpa.push(cols[1].trim().parse::<f64>().expect("Failed to parse p_sat_mpa"));
                data.rho_l.push(cols[2].trim().parse::<f64>().expect("Failed to parse rho_l"));
                data.h_l_kj.push(cols[3].trim().parse::<f64>().expect("Failed to parse h_l_kj"));
                data.s_l_kj.push(cols[4].trim().parse::<f64>().expect("Parse error S_L"));
                data.rho_v.push(cols[5].trim().parse::<f64>().expect("Failed to parse rho_v"));
                data.h_v_kj.push(cols[6].trim().parse::<f64>().expect("Failed to parse h_v_kj"));
                data.s_v_kj.push(cols[7].trim().parse::<f64>().expect("Parse error S_V"));
            }
        }
        
        data
    }
}