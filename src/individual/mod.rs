use std::ops::{Deref, DerefMut};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::parameters::Parameters;

use self::{behavior::Behavior, genes::IdGenerator, genome::Genome, scores::Score};

pub mod behavior;
pub mod genes;
pub mod genome;
pub mod scores;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Individual {
    pub genome: Genome,
    pub behavior: Behavior,
    pub fitness: Score,
    pub novelty: Score,
}

impl Deref for Individual {
    type Target = Genome;

    fn deref(&self) -> &Self::Target {
        &self.genome
    }
}

impl DerefMut for Individual {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.genome
    }
}

impl Individual {
    pub fn initial(id_gen: &mut IdGenerator, parameters: &Parameters) -> Self {
        Self {
            genome: Genome::new(id_gen, parameters),
            ..Default::default()
        }
    }

    // score is combination of fitness & novelty
    pub fn score(&self) -> f64 {
        let novelty = self.novelty.normalized;
        let fitness = self.fitness.normalized;

        novelty.max(fitness)

        // (novelty + fitness) / 2.0

        // novelty * 0.7 + fitness * 0.3

        /* if novelty == 0.0 && fitness == 0.0 {
            return 0.0;
        }

        let (min, max) = if novelty < fitness {
            (novelty, fitness)
        } else {
            (fitness, novelty)
        }; */

        // ratio tells us what score is dominant in this genome
        // let ratio = min / max / 2.0;

        // we weight the scores by their ratio, i.e. a genome that has a good fitness value is primarily weighted by that
        // min * ratio + max * (1.0 - ratio)
    }

    // self is fitter if it has higher score or in case of equal score has fewer genes, i.e. less complexity
    pub fn is_fitter_than(&self, other: &Self) -> bool {
        let score_self = self.score();
        let score_other = other.score();

        score_self > score_other
            || ((score_self - score_other).abs() < f64::EPSILON
                && self.genome.len() < other.genome.len())
    }

    pub fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let (fitter, weaker) = if self.is_fitter_than(other) {
            (&self.genome, &other.genome)
        } else {
            (&other.genome, &self.genome)
        };

        Individual {
            genome: fitter.cross_in(weaker, rng),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn feature() {}

    /* #[test]
    fn crossover_different_fitness_by_fitter() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let mut parameters: Parameters = Default::default();

        parameters.setup.input_dimension = 2;

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut genome_0 = Genome::new(&mut id_gen, &parameters);

        genome_0.init(&mut rng, &parameters);

        let mut genome_1 = genome_0.clone();

        genome_1.fitness = Score::new(1.0, 0.0, 1.0);

        // mutate genome_0
        genome_0.add_node(&mut rng, &mut id_gen, &parameters);

        // mutate genome_1
        genome_1.add_node(&mut rng, &mut id_gen, &parameters);
        genome_1.add_connection(&mut rng, &parameters).unwrap();

        let offspring = genome_0.cross_in(&genome_1, &mut rng.small);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 5);
    }

    #[test]
    fn crossover_equal_fittnes_different_len() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let mut parameters: Parameters = Default::default();

        parameters.setup.input_dimension = 2;

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut genome_0 = Genome::new(&mut id_gen, &parameters);

        genome_0.init(&mut rng, &parameters);

        let mut genome_1 = genome_0.clone();
        // mutate genome_0
        genome_0.add_node(&mut rng, &mut id_gen, &parameters);

        // mutate genome_1
        genome_1.add_node(&mut rng, &mut id_gen, &parameters);
        genome_1.add_connection(&mut rng, &parameters).unwrap();

        let offspring = genome_0.cross_in(&genome_1, &mut rng.small);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 4);
    }
    */
}
