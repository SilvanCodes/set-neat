mod context;
mod genes;
mod genome;
mod parameters;
mod runtime;
mod species;

// re-exports
pub use crate::context::Context;
pub use crate::genes::activations;
pub use crate::genome::Genome;
pub use crate::parameters::Parameters;
pub use crate::runtime::Runtime;
pub use crate::runtime::{Evaluation, Progress};

pub struct Neat {
    pub parameters: Parameters,
    progress_function: fn(&Genome) -> Progress,
}

// public API
impl Neat {
    pub fn new(path: &str, progress_function: fn(&Genome) -> Progress) -> Self {
        Neat {
            parameters: Parameters::new(path).unwrap(),
            progress_function,
        }
    }

    pub fn run(&self) -> Runtime {
        Runtime::new(&self)
    }
}
