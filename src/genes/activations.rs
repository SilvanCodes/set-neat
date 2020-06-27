use rand::random;
use rand::Rng;
use rand_distr::{Distribution, Standard};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub enum ActivationStrategy {
    FixedLinear,
    FixedSigmoid,
    FixedTanh,
    FixedGaussian,
    Random,
}

impl Default for ActivationStrategy {
    fn default() -> Self {
        ActivationStrategy::FixedTanh
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Gaussian,
}

impl Activation {
    pub fn new(strategy: &ActivationStrategy) -> Self {
        match strategy {
            ActivationStrategy::FixedLinear => Activation::Linear,
            ActivationStrategy::FixedSigmoid => Activation::Sigmoid,
            ActivationStrategy::FixedTanh => Activation::Tanh,
            ActivationStrategy::FixedGaussian => Activation::Gaussian,
            ActivationStrategy::Random => random(),
        }
    }
}

impl Default for Activation {
    fn default() -> Self {
        Activation::Tanh
    }
}

impl Distribution<Activation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Activation {
        match rng.gen_range(0, 4) {
            0 => Activation::Linear,
            1 => Activation::Sigmoid,
            2 => Activation::Tanh,
            _ => Activation::Gaussian,
        }
    }
}

pub const LINEAR: fn(f64) -> f64 = |val| val;
// pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-1.0 * val).exp());
pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-4.9 * val).exp());
pub const TANH: fn(f64) -> f64 = |val| 2.0 * SIGMOID(2.0 * val) - 1.0;
pub const GAUSSIAN: fn(f64) -> f64 = |val| (val * val / -2.0).exp(); // a = 1, b = 0, c = 1
