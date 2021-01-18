use rand::{prelude::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

#[derive(Debug)]
pub struct NeatRng {
    pub small: SmallRng,
    pub weight_distribution: Normal<f64>,
}

impl NeatRng {
    pub fn new(seed: u64, std_dev: f64) -> Self {
        Self {
            small: SmallRng::seed_from_u64(seed),
            weight_distribution: Normal::new(0.0, std_dev)
                .expect("could not create weight distribution"),
        }
    }

    pub fn gamble(&mut self, chance: f64) -> bool {
        self.small.gen::<f64>() < chance
    }

    pub fn weight_perturbation(&mut self) -> f64 {
        self.weight_distribution.sample(&mut self.small)
    }
}
