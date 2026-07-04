pub struct RcsCommand {
    pub firing_positive: bool,
    pub firing_negative: bool,
}

pub fn rcs_controller(roll_angle: f64, roll_rate: f64) -> RcsCommand {

    let dead_omega = 0.08;
    let dead_theta = 0.025;

    if roll_angle.abs() < dead_theta && roll_rate.abs() < dead_omega {
        return RcsCommand {
            firing_positive: false,
            firing_negative: false,
        };
    }

    // Gains from discrete LQR (MATLAB dlqr)
    let kp = 59.9687;
    let kd = 11.7118;

    let torque = -kp * roll_angle - kd * roll_rate;

    let threshold = 0.8;

    if torque > threshold {
        RcsCommand { firing_positive: true,  firing_negative: false }
    } else if torque < -threshold {
        RcsCommand { firing_positive: false, firing_negative: true  }
    } else {
        RcsCommand { firing_positive: false, firing_negative: false }
    }
}