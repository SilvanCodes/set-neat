pub mod genome;
mod genes;
pub mod context;
mod species;
pub mod parameters;
mod runtime;

use crate::runtime::Runtime;
use crate::parameters::Parameters;
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
