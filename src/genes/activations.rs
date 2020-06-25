use serde::{Deserialize, Serialize};
use rand::Rng;
use rand_distr::{Distribution, Standard};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Gaussian,
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
            _ => Activation::Gaussian
        }
    }
}

pub const LINEAR: fn(f64) -> f64 = |val| val;
// pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-1.0 * val).exp());
pub const SIGMOID: fn(f64) -> f64 =|val| 1.0 / (1.0 + (-4.9 * val).exp());
pub const TANH: fn(f64) -> f64 = |val| 2.0 * SIGMOID(2.0 * val) - 1.0;
pub const GAUSSIAN: fn(f64) -> f64 = |val| (val * val / -2.0).exp(); // a = 1, b = 0, c = 1