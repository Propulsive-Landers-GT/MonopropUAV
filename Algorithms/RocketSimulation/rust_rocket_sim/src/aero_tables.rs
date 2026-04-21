use serde::Deserialize;
use std::error::Error;

// One row from the CSV. All angles in degrees, distances in meters.
// We only track cp_z_from_nose because this is an axisymmetric rocket —
// the lateral CP components are zero by geometry.
#[derive(Debug, Clone, Deserialize)]
pub struct AeroRecord {
    pub alpha_deg:      f64,  // angle of attack [deg]
    pub mach:           f64,  // freestream Mach number [-]
    pub cd:             f64,  // drag coefficient [-]
    pub area_ref:       f64,  // reference area [m^2]
    pub cp_z_from_nose: f64,  // CP offset from nose tip, along body +Z axis [m]
                               // (body Z runs nose-to-tail; positive = aft of nose)
}

// ─── AeroTable ───────────────────────────────────────────────────────────────

/// Holds the full aero dataset as a structured 2-D grid indexed by
/// [alpha_deg, mach]. Once built it's read-only, so Clone is cheap to derive.
#[derive(Debug, Clone)]
pub struct AeroTable {
    /// Sorted, unique AoA breakpoints [deg]
    pub alphas: Vec<f64>,
    /// Sorted, unique Mach breakpoints [-]
    pub machs: Vec<f64>,
    /// grid[i_alpha][i_mach] — the raw record at each (alpha, mach) corner
    grid: Vec<Vec<AeroRecord>>,
}

impl AeroTable {
    /// Load a CSV with columns: alpha_deg, mach, cd, area_ref, cp_z_from_nose.
    /// Rows can be in any order; we sort and grid them here.
    /// Returns Err if the file is missing or malformed — caller decides the fallback.
    pub fn from_csv(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut rdr = csv::Reader::from_path(path)?;

        // Collect all rows first — we need to sort before building the grid.
        let mut records: Vec<AeroRecord> = rdr
            .deserialize()
            .collect::<Result<Vec<AeroRecord>, _>>()?;

        // Sort by alpha first, then Mach.  This means rows with the same alpha
        // end up contiguous, which is exactly the order we want to slice into columns.
        records.sort_by(|a, b| {
            a.alpha_deg
                .partial_cmp(&b.alpha_deg)
                .unwrap()
                .then(a.mach.partial_cmp(&b.mach).unwrap())
        });

        // Extract unique breakpoints (Vec is already sorted, just dedup).
        let mut alphas: Vec<f64> = records.iter().map(|r| r.alpha_deg).collect();
        alphas.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        let mut machs: Vec<f64> = records.iter().map(|r| r.mach).collect();
        machs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        let n_alpha = alphas.len();
        let n_mach  = machs.len();

        if records.len() != n_alpha * n_mach {
            return Err(format!(
                "aero_table.csv: expected {} rows ({}α × {}M), got {}",
                n_alpha * n_mach, n_alpha, n_mach, records.len()
            ).into());
        }

        // Build the 2-D grid. Records sorted alpha-major means we can chunk them
        // directly: records[i_alpha*n_mach .. (i_alpha+1)*n_mach] = one alpha slice.
        let grid: Vec<Vec<AeroRecord>> = records
            .chunks(n_mach)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(Self { alphas, machs, grid })
    }

    pub fn default() -> Option<Self> {
        // Load aero table from root. Fail gracefully if missing.
        match crate::aero_tables::AeroTable::from_csv("aero_table.csv") {
            Ok(t) => {
                // println!("Successfully loaded aero_table.csv");
                Some(t)
            },
            Err(e) => {
                eprintln!("Warning: Failed to load aero lookup table ({}); falling back to static drag.", e);
                None
            }
        }
    }
}

// ─── Lookup (public API) ──────────────────────────────────────────────────────

impl AeroTable {
    /// Bilinearly interpolate the table at (alpha_deg, mach).
    ///
    /// Both axes are CLAMPED to the table's known range, so Mach 3.0 in a table
    /// that only goes to 2.0 returns the extrapolated (last-row) record rather
    /// than panicking. That's the correct conservative behaviour: past the table
    /// edge we assume the last known aerodynamic state, not zeros.
    pub fn lookup(&self, alpha_deg: f64, mach: f64) -> AeroRecord {
        // 1. Clamp inputs to the table range.
        let alpha_c = alpha_deg.clamp(
            *self.alphas.first().unwrap(),
            *self.alphas.last().unwrap(),
        );
        let mach_c = mach.clamp(
            *self.machs.first().unwrap(),
            *self.machs.last().unwrap(),
        );

        // 2. Find the lower bracket index for each axis.
        //    windows(2) gives us overlapping pairs [a0,a1], [a1,a2], ...
        //    We pick the first pair that contains the clamped query point.
        let i_a = self.alphas
            .windows(2)
            .position(|w| alpha_c >= w[0] && alpha_c <= w[1])
            .unwrap_or(self.alphas.len() - 1)          // exact-max edge case
            .min(self.alphas.len().saturating_sub(2));  // keep i_a+1 in bounds

        let i_m = self.machs
            .windows(2)
            .position(|w| mach_c >= w[0] && mach_c <= w[1])
            .unwrap_or(self.machs.len() - 1)
            .min(self.machs.len().saturating_sub(2));

        // 3. Normalised fractional position within the cell — 0.0 = low corner, 1.0 = high.
        let d_alpha = self.alphas[i_a + 1] - self.alphas[i_a];
        let d_mach  = self.machs[i_m + 1]  - self.machs[i_m];

        let t_alpha = if d_alpha > 1e-12 { (alpha_c - self.alphas[i_a]) / d_alpha } else { 0.0 };
        let t_mach  = if d_mach  > 1e-12 { (mach_c  - self.machs[i_m])  / d_mach  } else { 0.0 };

        // 4. Pull the four surrounding corners out of the grid.
        //    grid is indexed [i_alpha][i_mach].
        let q00 = &self.grid[i_a    ][i_m    ];  // low-alpha,  low-mach
        let q10 = &self.grid[i_a + 1][i_m    ];  // high-alpha, low-mach
        let q01 = &self.grid[i_a    ][i_m + 1];  // low-alpha,  high-mach
        let q11 = &self.grid[i_a + 1][i_m + 1];  // high-alpha, high-mach

        bilinear_interp(q00, q10, q01, q11, t_alpha, t_mach, alpha_c, mach_c)
    }
}
// ─── Bilinear Interpolation ───────────────────────────────────────────────────

/// Interpolate between four corner AeroRecords using normalised coordinates
/// t_alpha ∈ [0,1] and t_mach ∈ [0,1], where 0 = low corner, 1 = high corner.
///
/// Standard bilinear weights:
///   (1-ta)(1-tm)·Q00 + ta(1-tm)·Q10 + (1-ta)tm·Q01 + ta·tm·Q11
///
/// Only the three physical scalars are mixed; alpha_deg and mach on the
/// returned record are set to the queried (clamped) values.
fn bilinear_interp(
    q00: &AeroRecord, q10: &AeroRecord,  // low-mach:  (low-alpha, high-alpha)
    q01: &AeroRecord, q11: &AeroRecord,  // high-mach: (low-alpha, high-alpha)
    t_alpha: f64,
    t_mach:  f64,
    alpha_query: f64,
    mach_query:  f64,
) -> AeroRecord {
    let w00 = (1.0 - t_alpha) * (1.0 - t_mach);
    let w10 =        t_alpha  * (1.0 - t_mach);
    let w01 = (1.0 - t_alpha) *        t_mach;
    let w11 =        t_alpha  *        t_mach;

    AeroRecord {
        alpha_deg:      alpha_query,
        mach:           mach_query,
        cd:             w00*q00.cd             + w10*q10.cd             + w01*q01.cd             + w11*q11.cd,
        area_ref:       w00*q00.area_ref       + w10*q10.area_ref       + w01*q01.area_ref       + w11*q11.area_ref,
        cp_z_from_nose: w00*q00.cp_z_from_nose + w10*q10.cp_z_from_nose + w01*q01.cp_z_from_nose + w11*q11.cp_z_from_nose,
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory 2×2 table so tests are hermetic (no file I/O).
    fn tiny_table() -> AeroTable {
        let alphas = vec![0.0, 10.0];
        let machs  = vec![0.5, 1.0];
        let grid = vec![
            vec![
                AeroRecord { alpha_deg: 0.0,  mach: 0.5, cd: 0.30, area_ref: 0.05, cp_z_from_nose: 1.50 },
                AeroRecord { alpha_deg: 0.0,  mach: 1.0, cd: 0.55, area_ref: 0.05, cp_z_from_nose: 1.45 },
            ],
            vec![
                AeroRecord { alpha_deg: 10.0, mach: 0.5, cd: 0.45, area_ref: 0.08, cp_z_from_nose: 1.40 },
                AeroRecord { alpha_deg: 10.0, mach: 1.0, cd: 0.70, area_ref: 0.08, cp_z_from_nose: 1.35 },
            ],
        ];
        AeroTable { alphas, machs, grid }
    }

    #[test]
    fn exact_corner_lookup() {
        let t = tiny_table();
        let rec = t.lookup(0.0, 0.5);
        assert!((rec.cd - 0.30).abs() < 1e-10, "cd mismatch at corner (0°, M0.5)");
        assert!((rec.cp_z_from_nose - 1.50).abs() < 1e-10);

        let rec2 = t.lookup(10.0, 1.0);
        assert!((rec2.cd - 0.70).abs() < 1e-10, "cd mismatch at corner (10°, M1.0)");
    }

    #[test]
    fn mach_clamping_no_panic() {
        let t = tiny_table();
        // Mach way beyond the table — should silently clamp to M=1.0.
        let rec = t.lookup(0.0, 999.0);
        assert!((rec.cd - 0.55).abs() < 1e-10, "should clamp to last Mach column");
    }

    #[test]
    fn alpha_clamping_no_panic() {
        let t = tiny_table();
        let rec = t.lookup(-5.0, 0.5);
        assert!((rec.cd - 0.30).abs() < 1e-10, "should clamp to first alpha row");
    }

    #[test]
    fn bilinear_midpoint_is_average() {
        let t = tiny_table();
        // At the exact cell midpoint the result must equal the simple average of all 4 corners.
        let rec = t.lookup(5.0, 0.75);
        let expected_cd = (0.30 + 0.55 + 0.45 + 0.70) / 4.0;
        assert!(
            (rec.cd - expected_cd).abs() < 1e-10,
            "midpoint cd = {:.6}, expected {:.6}", rec.cd, expected_cd
        );
    }
}
