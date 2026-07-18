pub struct Scheduler {
    sensor_fusion_rate: f64,
    navigation_rate: f64,
    mpc_rate: f64,
}

impl Scheduler {
    pub fn new(sensor_fusion_rate: f64, navigation_rate: f64, mpc_rate: f64) -> Self {
        Self {
            sensor_fusion_rate,
            navigation_rate,
            mpc_rate,
        }
    }
    
    #[inline]
    fn due(last: f64, now: f64, rate: f64) -> bool {
        now - last >= (1.0 / rate) - 1e-6
    }
    
    pub fn is_sensor_fusion_due(&self, last: f64, now: f64) -> bool {
        Self::due(last, now, self.sensor_fusion_rate)
    }
    
    pub fn is_navigation_due(&self, last: f64, now: f64) -> bool {
        Self::due(last, now, self.navigation_rate)
    }
    
    pub fn is_mpc_due(&self, last: f64, now: f64) -> bool {
        Self::due(last, now, self.mpc_rate)
    }
    
    pub fn sensor_dt(&self, last: f64, now: f64) -> f64 {
        if last > 0.0 {
            now - last
        } else {
            1.0 / self.sensor_fusion_rate
        }
    }
}
