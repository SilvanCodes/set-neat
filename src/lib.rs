pub mod context;
mod genes;
pub mod genome;
pub mod parameters;
mod runtime;
mod species;

use crate::parameters::Parameters;
use crate::runtime::Runtime;
// re-exports
pub use crate::genome::Genome;
pub use crate::runtime::Evaluation::{Progress, Solution};

pub struct Neat {
    parameters: Parameters,
    fitness_function: fn(&Genome) -> f64,
    required_fitness: f64,
}

// public API
impl Neat {
    pub fn new(path: &str, fitness_function: fn(&Genome) -> f64, required_fitness: f64) -> Self {
        Neat {
            parameters: Parameters::new(path).unwrap(),
            fitness_function,
            required_fitness,
        }
    }

    pub fn run(&self) -> Runtime {
        Runtime::new(&self)
    }

    pub fn run_with(&self, genome: Genome) -> Runtime {
        Runtime::load(&self, genome)
    }
}
