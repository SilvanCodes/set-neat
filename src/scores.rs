use serde::{Deserialize, Serialize};
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct Score {
    pub raw: f64,
    pub shifted: f64,
    pub normalized: f64,
}

impl Score {
    pub fn new(raw: f64, baseline: f64, with: f64) -> Self {
        Self {
            raw,
            shifted: raw - baseline,
            normalized: (raw - baseline) / with,
        }
    }
}
