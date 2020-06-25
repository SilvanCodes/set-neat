use crate::context::Context;
use serde::{Deserialize, Serialize};
use rand::rngs::SmallRng;
use rand::random;
use rand_distr::{Uniform, Normal, Distribution};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Weight(pub f64);

impl Default for Weight {
    fn default() -> Self {
        Weight(random::<f64>() * 2.0 - 1.0)
    }
}

impl Weight {
    pub fn difference(&self, other: &Weight) -> f64 {
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
    Normal
}

impl Default for WeightDistribution {
    fn default() -> Self {
        WeightDistribution::Uniform
    }
}

pub enum Perturbator {
    Uniform(Uniform<f64>),
    Normal(Normal<f64>)
}

impl Perturbator {
    pub fn new(kind: &WeightDistribution, range: f64) -> Self {
        match kind {
            WeightDistribution::Uniform => Perturbator::Uniform(Uniform::new(-range, range)),
            WeightDistribution::Normal => Perturbator::Normal(Normal::new(0.0, range).unwrap())
        }
    }

    pub fn sample(&mut self, rng: &mut SmallRng) -> f64 {
        match self {
            Perturbator::Uniform(dist) => dist.sample(rng),
            Perturbator::Normal(dist) => dist.sample(rng),
        }
    }
}