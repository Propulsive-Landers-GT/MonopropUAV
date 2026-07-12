#[derive(Debug, Clone)]
pub struct RcsController {
    pub kp: f64,
    pub kd: f64,
    pub dead_theta_enter: f64,
    pub dead_omega_enter: f64,
    pub dead_theta_exit: f64,
    pub dead_omega_exit: f64,
    pub fire_threshold: f64,
    pub in_deadband: bool,
    pub last_update: f64,
    pub update_period: f64,
    pub last_command: f64,
}

impl RcsController {
    pub fn new() -> Self {
        Self {
            kp: 15.3526,
            kd: 9.6944,
            dead_theta_enter: 0.025,
            dead_omega_enter: 0.08,
            dead_theta_exit: 0.055,
            dead_omega_exit: 0.35,
            fire_threshold: 1.5,
            in_deadband: false,
            last_update: 0.0,
            update_period: 0.1, // 10 Hz
            last_command: 0.0,
        }
    }

    pub fn update(&mut self, roll_angle: f64, roll_rate: f64, now: f64) -> f64 {
        if now - self.last_update >= self.update_period - 1e-6 {
            self.last_update = now;
            
            if self.in_deadband {
                if roll_angle.abs() > self.dead_theta_exit || roll_rate.abs() > self.dead_omega_exit {
                    self.in_deadband = false;
                }
            } else {
                if roll_angle.abs() < self.dead_theta_enter && roll_rate.abs() < self.dead_omega_enter {
                    self.in_deadband = true;
                }
            }

            if self.in_deadband {
                self.last_command = 0.0;
            } else {
                let sv = -self.kp * roll_angle - self.kd * roll_rate;
                if sv > self.fire_threshold {
                    self.last_command = -1.0;
                } else if sv < -self.fire_threshold {
                    self.last_command = 1.0;
                } else {
                    self.last_command = 0.0;
                }
            }
        }
        self.last_command
    }
}
