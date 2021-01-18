mod favannat_impl;
mod genes;
mod genomee;
mod parameters;
mod population;
mod runtime;
mod species;
// remove ASAP
pub mod utility;

mod individual;
mod rng;
mod statistics;

use parameters::Parameters;

// re-exports
pub use individual::Individual;
pub use runtime::{Evaluation, Progress, Runtime};

pub struct Neat {
    pub parameters: Parameters,
    progress_function: Box<dyn Fn(&Individual) -> Progress + Send + Sync>,
}

// public API
impl Neat {
    pub fn new(
        path: &str,
        progress_function: Box<dyn Fn(&Individual) -> Progress + Send + Sync>,
    ) -> Self {
        Neat {
            parameters: Parameters::new(path).unwrap(),
            progress_function,
        }
    }

    pub fn run(&self) -> Runtime {
        Runtime::new(&self)
    }
}
