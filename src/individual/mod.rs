use std::ops::{Deref, DerefMut};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{genes::IdGenerator, parameters::Parameters};

use self::scores::Score;
use self::{behavior::Behavior, genome::Genome};

pub mod behavior;
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

    pub fn compatability_distance(
        individual_0: &Self,
        individual_1: &Self,
        factor_genes: f64,
        factor_weights: f64,
        factor_activations: f64,
    ) -> f64 {
        let genome_0 = &individual_0.genome;
        let genome_1 = &individual_1.genome;

        let mut weight_difference_total = 0.0;
        let mut activation_difference = 0.0;

        let matching_genes_count_total = (genome_0
            .feed_forward
            .iterate_matches(&genome_1.feed_forward)
            .inspect(|(connection_0, connection_1)| {
                weight_difference_total += (connection_0.weight - connection_1.weight).abs();
            })
            .count()
            + genome_0
                .recurrent
                .iterate_matches(&genome_1.recurrent)
                .inspect(|(connection_0, connection_1)| {
                    weight_difference_total += (connection_0.weight - connection_1.weight).abs();
                })
                .count()) as f64;

        let different_genes_count_total = (genome_0
            .feed_forward
            .iterate_unmatches(&genome_1.feed_forward)
            .count()
            + genome_0
                .recurrent
                .iterate_unmatches(&genome_1.recurrent)
                .count()) as f64;

        let matching_nodes_count = genome_0
            .hidden
            .iterate_matches(&genome_1.hidden)
            .inspect(|(node_0, node_1)| {
                if node_0.activation != node_1.activation {
                    activation_difference += 1.0;
                }
            })
            .count() as f64;

        // percent of different genes, considering unique genes
        let difference = factor_genes * different_genes_count_total / (matching_genes_count_total + different_genes_count_total)
        // average of weight differences
        + factor_weights * if matching_genes_count_total > 0.0 { weight_difference_total / matching_genes_count_total } else { 0.0 }
        // percent of different activation functions, considering matching nodes genes
        + factor_activations * if matching_nodes_count > 0.0 { activation_difference / matching_nodes_count } else { 0.0 };

        if difference.is_nan() {
            dbg!(factor_genes);
            dbg!(different_genes_count_total);
            dbg!(matching_genes_count_total);
            dbg!(different_genes_count_total);
            dbg!(factor_weights);
            dbg!(weight_difference_total);
            dbg!(matching_genes_count_total);
            dbg!(factor_activations);
            dbg!(activation_difference);
            dbg!(matching_nodes_count);
            panic!("difference is nan");
        } else {
            difference
        }

        // neat python function
        //(activation_difference + c1 * different_nodes_count) / genome_0.node_genes.len().max(genome_1.node_genes.len()) as f64
        // + (weight_difference_total + c1 * different_genes_count_total) / (genome_0.connection_genes.len() + genome_0.recurrent_connection_genes.len()).max(genome_1.connection_genes.len() + genome_1.recurrent_connection_genes.len()) as f64
    }
}
