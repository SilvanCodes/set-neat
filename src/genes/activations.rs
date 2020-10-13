use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Gaussian,
    Step,
    Sine,
    Cosine,
    Inverse,
    Absolute,
    Relu,
    Squared,
}

impl Default for Activation {
    fn default() -> Self {
        Activation::Tanh
    }
}

pub const LINEAR: fn(f64) -> f64 = |val| val;
// pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-1.0 * val).exp());
pub const SIGMOID: fn(f64) -> f64 = |val| 1.0 / (1.0 + (-4.9 * val).exp());
pub const TANH: fn(f64) -> f64 = |val| 2.0 * SIGMOID(2.0 * val) - 1.0;
pub const GAUSSIAN: fn(f64) -> f64 = |val| (val * val / -2.0).exp(); // a = 1, b = 0, c = 1
pub const STEP: fn(f64) -> f64 = |val| if val > 0.0 { 1.0 } else { 0.0 };
pub const SINE: fn(f64) -> f64 = |val| (val * std::f64::consts::PI).sin();
pub const COSINE: fn(f64) -> f64 = |val| (val * std::f64::consts::PI).cos();
pub const INVERSE: fn(f64) -> f64 = |val| -val;
pub const ABSOLUTE: fn(f64) -> f64 = |val| val.abs();
pub const RELU: fn(f64) -> f64 = |val| 0f64.max(val);
pub const SQUARED: fn(f64) -> f64 = |val| val * val;

#[cfg(test)]
mod tests {
    use super::{
        ABSOLUTE, COSINE, GAUSSIAN, INVERSE, LINEAR, RELU, SIGMOID, SINE, SQUARED, STEP, TANH,
    };
    #[test]
    fn test_linear() {
        // input 1.0
        assert!(!LINEAR(1.0).is_nan());
        assert!(!LINEAR(1.0).is_infinite());
        // input 0.0
        assert!(!LINEAR(0.0).is_nan());
        assert!(!LINEAR(0.0).is_infinite());
    }
    #[test]
    fn test_sigmoid() {
        // input 1.0
        assert!(!SIGMOID(1.0).is_nan());
        assert!(!SIGMOID(1.0).is_infinite());
        // input 0.0
        assert!(!SIGMOID(0.0).is_nan());
        assert!(!SIGMOID(0.0).is_infinite());
    }
    #[test]
    fn test_tanh() {
        // input 1.0
        assert!(!TANH(1.0).is_nan());
        assert!(!TANH(1.0).is_infinite());
        // input 0.0
        assert!(!TANH(0.0).is_nan());
        assert!(!TANH(0.0).is_infinite());
    }
    #[test]
    fn test_gaussian() {
        // input 1.0
        assert!(!GAUSSIAN(1.0).is_nan());
        assert!(!GAUSSIAN(1.0).is_infinite());
        // input 0.0
        assert!(!GAUSSIAN(0.0).is_nan());
        assert!(!GAUSSIAN(0.0).is_infinite());
    }
    #[test]
    fn test_step() {
        // input 1.0
        assert!(!STEP(1.0).is_nan());
        assert!(!STEP(1.0).is_infinite());
        // input 0.0
        assert!(!STEP(0.0).is_nan());
        assert!(!STEP(0.0).is_infinite());
    }
    #[test]
    fn test_sine() {
        // input 1.0
        assert!(!SINE(1.0).is_nan());
        assert!(!SINE(1.0).is_infinite());
        // input 0.0
        assert!(!SINE(0.0).is_nan());
        assert!(!SINE(0.0).is_infinite());
    }
    #[test]
    fn test_cosine() {
        // input 1.0
        assert!(!COSINE(1.0).is_nan());
        assert!(!COSINE(1.0).is_infinite());
        // input 0.0
        assert!(!COSINE(0.0).is_nan());
        assert!(!COSINE(0.0).is_infinite());
    }
    #[test]
    fn test_inverse() {
        // input 1.0
        assert!(!INVERSE(1.0).is_nan());
        assert!(!INVERSE(1.0).is_infinite());
        // input 0.0
        assert!(!INVERSE(0.0).is_nan());
        assert!(!INVERSE(0.0).is_infinite());
    }
    #[test]
    fn test_absolute() {
        // input 1.0
        assert!(!ABSOLUTE(1.0).is_nan());
        assert!(!ABSOLUTE(1.0).is_infinite());
        // input 0.0
        assert!(!ABSOLUTE(0.0).is_nan());
        assert!(!ABSOLUTE(0.0).is_infinite());
    }
    #[test]
    fn test_relu() {
        // input 1.0
        assert!(!RELU(1.0).is_nan());
        assert!(!RELU(1.0).is_infinite());
        // input 0.0
        assert!(!RELU(0.0).is_nan());
        assert!(!RELU(0.0).is_infinite());
    }
    #[test]
    fn test_squared() {
        // input 1.0
        assert!(!SQUARED(1.0).is_nan());
        assert!(!SQUARED(1.0).is_infinite());
        // input 0.0
        assert!(!SQUARED(0.0).is_nan());
        assert!(!SQUARED(0.0).is_infinite());
    }
}
