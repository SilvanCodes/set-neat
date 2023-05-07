use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};
use set_genome::Genome;

use self::{behavior::Behavior, scores::Score};

pub mod behavior;
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
    pub fn from_genome(genome: Genome) -> Self {
        Self {
            genome,
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

    pub fn crossover(&self, other: &Self) -> Self {
        let (fitter, weaker) = if self.is_fitter_than(other) {
            (&self.genome, &other.genome)
        } else {
            (&other.genome, &self.genome)
        };

        Individual {
            genome: fitter.cross_in(weaker),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use set_genome::{Genome, Parameters, Structure};

    use crate::Individual;

    use super::scores::Score;

    // #[test]
    // fn crossover_different_fitness() {
    //     let parameters = Parameters {
    //         structure: Structure {
    //             number_of_inputs: 2,
    //             ..Default::default()
    //         },
    //         ..Default::default()
    //     };

    //     // create randomn source
    //     let mut individual_0 = Individual::from_genome(Genome::initialized(&parameters));

    //     let mut individual_1 = individual_0.clone();

    //     individual_1.fitness = Score::new(1.0, 0.0, 1.0);

    //     // mutate individual_0
    //     individual_0.add_node_with_context(&mut gc);

    //     // mutate individual_1
    //     individual_1.add_node_with_context(&mut gc);
    //     individual_1.add_connection_with_context(&mut gc);

    //     let offspring = individual_0.crossover(&individual_1, &mut gc.rng);

    //     assert_eq!(offspring.hidden.len(), 1);
    //     assert_eq!(offspring.feed_forward.len(), 5);
    // }

    // #[test]
    // fn crossover_equal_fittnes_different_len() {
    //     let parameters = Parameters {
    //         seed: None,
    //         structure: Structure {
    //             inputs: 2,
    //             ..Default::default()
    //         },
    //         ..Default::default()
    //     };

    //     let mut gc = GenomeContext::new(parameters);

    //     // create randomn source
    //     let mut individual_0 = Individual::from_genome(gc.initialized_genome());

    //     let mut individual_1 = individual_0.clone();

    //     // mutate genome_0
    //     individual_0.add_node_with_context(&mut gc);

    //     // mutate genome_1
    //     individual_1.add_node_with_context(&mut gc);
    //     individual_1.add_connection_with_context(&mut gc);

    //     let offspring = individual_0.crossover(&individual_1, &mut gc.rng);

    //     assert_eq!(offspring.hidden.len(), 1);
    //     assert_eq!(offspring.feed_forward.len(), 4);
    // }
}
