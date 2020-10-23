use std::iter::once;

use crate::{genes::IdGenerator, parameters::Parameters};
use crate::{genes::WeightPerturbator, runtime::Report};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub struct Context {
    pub id_gen: IdGenerator,
    pub statistics: Report,
    pub archive_threshold: f64,
    pub novelty_ratio: f64,
    pub added_to_archive: usize,
    pub consecutive_ineffective_generations: usize,
    peak_fitness_buffer: Vec<f64>,
    pub peak_average_fitness: f64,

    pub small_rng: SmallRng,
    pub weight_pertubator: WeightPerturbator,
}

impl Context {
    pub fn new(parameters: &Parameters) -> Self {
        Context {
            archive_threshold: 0.0,
            novelty_ratio: 0.0,
            statistics: Report::default(),
            added_to_archive: 0,
            consecutive_ineffective_generations: 0,
            peak_fitness_buffer: Vec::new(),
            peak_average_fitness: f64::NEG_INFINITY,
            small_rng: SmallRng::seed_from_u64(parameters.seed),
            weight_pertubator: WeightPerturbator::new(
                &parameters.mutation.weights.distribution,
                parameters.mutation.weights.perturbation_range,
            ),
            id_gen: Default::default(),
        }
    }

    pub fn compare_to_peak_fitness_mean(&mut self, peak_fitness: f64) -> f64 {
        if self.peak_fitness_buffer.is_empty() {
            return 1.0;
        }

        let shift = self
            .peak_fitness_buffer
            .iter()
            .chain(once(&peak_fitness))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - 1.0;

        let average = self
            .peak_fitness_buffer
            .iter()
            .map(|v| v - shift)
            .sum::<f64>()
            / self.peak_fitness_buffer.len() as f64;

        let percent_change = (peak_fitness - shift) / average - 1.0;

        self.statistics.peak_fitness_average =
            self.peak_fitness_buffer.iter().sum::<f64>() / self.peak_fitness_buffer.len() as f64;

        percent_change
    }

    pub fn put_peak_fitness(&mut self, peak_fitness: f64) {
        self.peak_fitness_buffer.push(peak_fitness);
        if self.peak_fitness_buffer.len() > 10 {
            self.peak_fitness_buffer.remove(0);
        }
        /* self.peak_fitness_buffer[self.peak_fitness_buffer_index] = peak_fitness;
        self.peak_fitness_buffer_index += 1;
        self.peak_fitness_buffer_index %= self.peak_fitness_buffer.len(); */
    }

    pub fn gamble(&mut self, chance: f64) -> bool {
        self.small_rng.gen::<f64>() < chance
    }

    pub fn sample(&mut self) -> f64 {
        self.weight_pertubator.sample(&mut self.small_rng)
    }
}

#[cfg(test)]
mod test {
    use crate::{Context, Parameters};

    #[test]
    fn calculate_percent_diff() {
        let mut parameters = Parameters::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        let mut context = Context::new(&parameters);

        let result = context.compare_to_peak_fitness_mean(1.0);

        assert!((result - 1.0).abs() < f64::EPSILON);

        context.put_peak_fitness(1.0);
        context.put_peak_fitness(3.0);

        // 1.0 is -50% of 2.0
        let result = context.compare_to_peak_fitness_mean(1.0);
        dbg!(result);
        assert!((-0.5 - result).abs() < f64::EPSILON);

        // 4.0 is +100% of 2.0
        let result = context.compare_to_peak_fitness_mean(4.0);
        dbg!(result);
        assert!((1.0 - result).abs() < f64::EPSILON);

        // -2.0 is -200% of 2.0
        let result = context.compare_to_peak_fitness_mean(-2.0);
        dbg!(result);
        assert!((-2.0 - result).abs() < f64::EPSILON);
    }
}
