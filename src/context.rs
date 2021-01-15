use crate::{genes::IdGenerator, parameters::Parameters};
use crate::{genes::WeightPerturbator, runtime::Report};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Debug)]
pub struct Context {
    pub id_gen: IdGenerator,
    pub statistics: Report,
    pub archive_threshold: f64,
    // pub novelty_ratio: f64,
    pub added_to_archive: usize,
    pub consecutive_ineffective_generations: usize,
    // peak_fitness_buffer: Vec<f64>,
    pub peak_average_fitness: f64,
    pub small_rng: SmallRng,
    pub weight_pertubator: WeightPerturbator,
}

impl Context {
    pub fn new(parameters: &Parameters) -> Self {
        Context {
            archive_threshold: 0.0,
            // novelty_ratio: 0.0,
            statistics: Report::default(),
            added_to_archive: 0,
            consecutive_ineffective_generations: 0,
            // peak_fitness_buffer: Vec::new(),
            peak_average_fitness: f64::NEG_INFINITY,
            small_rng: SmallRng::seed_from_u64(parameters.seed),
            weight_pertubator: WeightPerturbator::new(
                &parameters.mutation.weights.distribution,
                parameters.mutation.weights.perturbation_range,
            ),
            id_gen: Default::default(),
        }
    }

    pub fn gamble(&mut self, chance: f64) -> bool {
        self.small_rng.gen::<f64>() < chance
    }

    pub fn sample(&mut self) -> f64 {
        self.weight_pertubator.sample(&mut self.small_rng)
    }
}
