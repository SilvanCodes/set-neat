use crate::context::Context;
use rand::random;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Weight(pub f64);

impl Default for Weight {
    fn default() -> Self {
        Weight(random::<f64>() * 2.0 - 1.0)
    }
}

impl Weight {
    pub fn abs(&self) -> f64 {
        self.0.abs()
    }

    pub fn difference(&self, other: &Weight) -> f64 {
        // (self.0 - other.0).abs() / (self.0.abs() + other.0.abs())
        (self.0 - other.0).abs()
    }

    #[inline]
    pub fn perturbate(&mut self, context: &mut Context) {
        self.0 += context.sample()
    }

    #[inline]
    pub fn random(&mut self, context: &mut Context) {
        self.0 = context.sample()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum WeightDistribution {
    Uniform,
    Normal,
}

impl Default for WeightDistribution {
    fn default() -> Self {
        WeightDistribution::Uniform
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WeightInitialization {
    Fixed(f64),
    Strategy(String),
}

impl WeightInitialization {
    pub fn init(&self) -> Weight {
        match self {
            WeightInitialization::Fixed(value) => Weight(*value),
            WeightInitialization::Strategy(strategy) if strategy == "Random" => {
                Weight(random::<f64>() * 2.0 - 1.0)
            }
            WeightInitialization::Strategy(_) => Weight(random::<f64>() * 2.0 - 1.0),
        }
    }
}

impl Default for WeightInitialization {
    fn default() -> Self {
        WeightInitialization::Strategy("Random".into())
    }
}

pub enum WeightPerturbator {
    Uniform(Uniform<f64>),
    Normal(Normal<f64>),
}

impl WeightPerturbator {
    pub fn new(kind: &WeightDistribution, range: f64) -> Self {
        match kind {
            WeightDistribution::Uniform => WeightPerturbator::Uniform(Uniform::new(-range, range)),
            WeightDistribution::Normal => {
                WeightPerturbator::Normal(Normal::new(0.0, range).unwrap())
            }
        }
    }

    pub fn sample(&mut self, rng: &mut SmallRng) -> f64 {
        match self {
            WeightPerturbator::Uniform(dist) => dist.sample(rng),
            WeightPerturbator::Normal(dist) => dist.sample(rng),
        }
    }
}
