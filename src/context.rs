use crate::{genes::IdGenerator, parameters::Parameters};
use crate::{genes::WeightPerturbator, runtime::Report};
use rand::rngs::{SmallRng, ThreadRng};
use rand::{Rng, SeedableRng};

pub struct Context {
    pub id_gen: IdGenerator,
    pub statistics: Report,
    pub archive_threshold: f64,
    pub score_ratio: f64,
    pub added_to_archive: usize,
    pub small_rng: SmallRng,
    pub weight_pertubator: WeightPerturbator,
}

impl Context {
    pub fn new(parameters: &Parameters) -> Self {
        Context {
            archive_threshold: 0.0,
            score_ratio: 1.0,
            statistics: Report::default(),
            added_to_archive: 0,
            small_rng: SmallRng::from_rng(&mut ThreadRng::default()).unwrap(),
            weight_pertubator: WeightPerturbator::new(
                &parameters.mutation.weight_distribution,
                parameters.mutation.weight_perturbation,
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
