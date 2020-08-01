mod context;
mod genes;
mod genome;
mod parameters;
mod runtime;
mod species;

// re-exports
pub use crate::genes::activations;
pub use crate::context::Context;
pub use crate::genome::Genome;
pub use crate::parameters::Parameters;
pub use crate::runtime::Evaluation::{Progress, Solution};
pub use crate::runtime::Runtime;

pub struct Neat {
    pub parameters: Parameters,
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
