extern crate rand;
extern crate config;
extern crate rayon;
// std imports
use crate::runtime::Runtime;
use crate::parameters::Parameters;
use crate::genome::Genome;
// external imports
// sub modules
pub mod genome;
pub mod genes;
pub mod context;
mod species;
pub mod parameters;
pub mod runtime;


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
}
