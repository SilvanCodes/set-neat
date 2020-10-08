use std::cmp::Ordering;

use crate::context::Context;
use crate::genome::Genome;
use crate::parameters::Parameters;

#[derive(Debug)]
pub struct Species {
    pub representative: Genome,
    pub members: Vec<Genome>,
    pub score: f64,
    pub stale: usize,
}

// public API
impl Species {
    pub fn new(first_member: Genome) -> Self {
        Species {
            representative: first_member.clone(),
            members: vec![first_member],
            score: 0.0,
            stale: 0,
        }
    }

    pub fn adjust_fitness(&mut self, context: &Context, parameters: &Parameters) {
        let old_score = self.score;
        let factor = self.members.len() as f64;

        /* for genome in &mut self.members {
            // TODO: consider scaling and score factor interaction
            genome.fitness /= factor;
            genome.novelty /= factor;
        } */

        // sort members by descending fitness, i.e. fittest first
        self.members.sort_by(|genome_0, genome_1| {
            genome_1
                .score(context)
                .partial_cmp(&genome_0.score(context))
                .unwrap_or(Ordering::Equal)
        });

        // we set the species fitness as the average of the reproducing members
        self.score = self
            .members
            .iter()
            .take((factor * parameters.reproduction.surviving).ceil() as usize)
            .map(|member| member.score(context) / factor)
            .sum();

        // did fitness increase ?
        if self.score > old_score {
            self.stale = 0;
        } else {
            self.stale += 1;
        }
    }

    // TODO: move to genome namespace
    pub fn compatability_distance(
        genome_0: &Genome,
        genome_1: &Genome,
        c1: f64,
        c2: f64,
        c3: f64,
    ) -> f64 {
        let mut weight_difference_total = 0.0;
        let mut activation_difference = 0.0;

        let matching_genes_count_total = (genome_0
            .feed_forward
            .iterate_matches(&genome_1.feed_forward)
            .inspect(|(connection_0, connection_1)| {
                weight_difference_total += connection_0.1.difference(&connection_1.1);
            })
            .count()
            + genome_0
                .recurrent
                .iterate_matches(&genome_1.recurrent)
                .inspect(|(connection_0, connection_1)| {
                    weight_difference_total += connection_0.1.difference(&connection_1.1);
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
                if node_0.1 != node_1.1 {
                    activation_difference += 1.0;
                }
            })
            .count() as f64;

        /* let different_nodes_count = genome_0
        .node_genes
        .symmetric_difference(&genome_1.node_genes)
        .count() as f64; */

        /* let n = genome_0
            .connection_genes
            .len()
            .min(genome_1.connection_genes.len()) as f64;
        */
        // let matching_genes_count_total = matching_genes_count + recurrent_matching_genes_count;
        // let different_genes_count_total = different_genes_count + recurrent_different_genes_count;

        // percent of different genes, considering unique genes
        let difference = c1 * different_genes_count_total / (matching_genes_count_total + different_genes_count_total)
        // average of weight differences
        + (c2 * weight_difference_total / matching_genes_count_total)
        // + if weight_difference_total > 0.0 { c2 * weight_difference_total / matching_genes_count_total } else { 0.0 }
        // average of activation differences
        + c3 * if matching_nodes_count > 0.0 { activation_difference / matching_nodes_count } else { 0.0 };

        if difference.is_nan() {
            dbg!(c1);
            dbg!(different_genes_count_total);
            dbg!(matching_genes_count_total);
            dbg!(different_genes_count_total);
            dbg!(c2);
            dbg!(weight_difference_total);
            dbg!(matching_genes_count_total);
            dbg!(c3);
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

#[cfg(test)]
mod tests {
    use super::Species;
    use crate::genome::Genome;
    use crate::{
        activations::Activation,
        genes::{
            connections::{Connection, FeedForward},
            nodes::{Input, Node, Output},
            Genes, Id, Weight,
        },
    };

    #[test]
    fn compatability_distance_same_genome() {
        let genome_0 = Genome {
            inputs: Genes(
                vec![Input(Node(Id(0), Activation::Linear))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Output(Node(Id(1), Activation::Linear))]
                    .iter()
                    .cloned()
                    .collect(),
            ),

            feed_forward: Genes(
                vec![FeedForward(Connection(Id(0), Weight(1.0), Id(1)))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        let genome_1 = genome_0.clone();

        let delta = Species::compatability_distance(&genome_0, &genome_1, 1.0, 0.4, 0.0);

        assert!(delta < f64::EPSILON);
    }

    #[test]
    fn compatability_distance_different_weight_genome() {
        let genome_0 = Genome {
            inputs: Genes(
                vec![Input(Node(Id(0), Activation::Linear))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Output(Node(Id(1), Activation::Linear))]
                    .iter()
                    .cloned()
                    .collect(),
            ),

            feed_forward: Genes(
                vec![FeedForward(Connection(Id(0), Weight(1.0), Id(1)))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        let mut genome_1 = genome_0.clone();

        genome_1
            .feed_forward
            .replace(FeedForward(Connection(Id(0), Weight(2.0), Id(1))));

        println!("genome_0: {:?}", genome_0);
        println!("genome_1: {:?}", genome_1);

        let delta = Species::compatability_distance(&genome_0, &genome_1, 0.0, 2.0, 0.0);

        assert!(delta - 2.0 < f64::EPSILON);
    }

    #[test]
    fn compatability_distance_different_connection_genome() {
        let genome_0 = Genome {
            inputs: Genes(
                vec![Input(Node(Id(0), Activation::Linear))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Output(Node(Id(1), Activation::Linear))]
                    .iter()
                    .cloned()
                    .collect(),
            ),

            feed_forward: Genes(
                vec![FeedForward(Connection(Id(0), Weight(1.0), Id(1)))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        let mut genome_1 = genome_0.clone();

        genome_1
            .feed_forward
            .replace(FeedForward(Connection(Id(0), Weight(1.0), Id(2))));
        genome_1
            .feed_forward
            .replace(FeedForward(Connection(Id(2), Weight(2.0), Id(1))));

        println!("genome_0: {:?}", genome_0);
        println!("genome_1: {:?}", genome_1);

        let delta = Species::compatability_distance(&genome_0, &genome_1, 2.0, 0.0, 0.0);

        // factor 2 times 2 different genes
        assert!(delta - 2.0 * 2.0 < f64::EPSILON);
    }
}
