use crate::context::Context;
use crate::genome::Genome;
use crate::parameters::Parameters;

#[derive(Debug)]
pub struct Species {
    pub representative: Genome,
    pub members: Vec<Genome>,
    pub fitness: f64,
    pub stale: usize,
}

// public API
impl Species {
    pub fn new(first_member: Genome) -> Self {
        Species {
            representative: first_member.clone(),
            members: vec![first_member],
            fitness: 0.0,
            stale: 0,
        }
    }

    pub fn compatible(&self, genome: &Genome, parameters: &Parameters, context: &Context) -> bool {
        Species::compatability_distance(
            &genome,
            &self.representative,
            parameters.compatability.factor_genes,
            parameters.compatability.factor_weights,
            parameters.compatability.factor_activations,
        ) < context.compatability_threshold
    }

    pub fn adjust_fitness(&mut self, parameters: &Parameters) {
        let old_fitness = self.fitness;
        let factor = self.members.len() as f64;

        for genome in &mut self.members {
            genome.fitness /= factor;
        }

        // sort members by descending fitness, i.e. fittest first
        self.members
            .sort_by(|genome_0, genome_1| genome_1.fitness.partial_cmp(&genome_0.fitness).unwrap());

        // we set the species fitness as the average of the reproducing members
        self.fitness = self
            .members
            .iter()
            .take((factor * parameters.reproduction.surviving).ceil() as usize)
            .map(|member| member.fitness)
            .sum();

        // did fitness increase ?
        if self.fitness > old_fitness {
            self.stale = 0;
        } else {
            self.stale += 1;
        }
    }

    pub fn compatability_distance(
        genome_0: &Genome,
        genome_1: &Genome,
        c1: f64,
        c2: f64,
        c3: f64,
    ) -> f64 {
        let mut weight_difference = 0.0;
        let mut activation_difference = 0.0;

        let matching_genes_count = genome_0
            .connection_genes
            .intersection(&genome_1.connection_genes)
            .inspect(|connection_gene| {
                weight_difference += connection_gene.weight.difference(
                    &genome_1
                        .connection_genes
                        .get(connection_gene)
                        .unwrap()
                        .weight,
                );
            })
            .count();

        let different_genes_count = genome_0
            .connection_genes
            .symmetric_difference(&genome_1.connection_genes)
            .count();

        // TODO: add term for activation function difference

        let matching_nodes_count = genome_0
            .node_genes
            .intersection(&genome_1.node_genes)
            .inspect(|node_gene| {
                if node_gene.activation != genome_1.node_genes.get(node_gene).unwrap().activation {
                    activation_difference += 1.0;
                }
            })
            .count();

        let n = genome_0
            .connection_genes
            .len()
            .min(genome_1.connection_genes.len()) as f64;

        // distance formula from paper (modified)
        c1 * different_genes_count as f64 / n
            + c2 * weight_difference / matching_genes_count as f64
            + c3 * activation_difference / matching_nodes_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::Species;
    use crate::genes::{ConnectionGene, Id, NodeGene, Weight};
    use crate::genome::Genome;

    #[test]
    fn compatability_distance_same_genome() {
        let genome_0 = Genome {
            node_genes: vec![NodeGene::input(Id(0)), NodeGene::output(Id(1), None)]
                .iter()
                .cloned()
                .collect(),
            connection_genes: vec![ConnectionGene::new(Id(0), Id(1), None)]
                .iter()
                .cloned()
                .collect(),
            fitness: 0.0,
        };

        let genome_1 = Genome::from(&genome_0);

        let delta = Species::compatability_distance(&genome_0, &genome_1, 1.0, 0.4, 0.0);

        assert_eq!(delta, 0.0);
    }

    #[test]
    fn compatability_distance_different_weight_genome() {
        let genome_0 = Genome {
            node_genes: vec![NodeGene::input(Id(0)), NodeGene::output(Id(1), None)]
                .iter()
                .cloned()
                .collect(),
            connection_genes: vec![ConnectionGene::new(Id(0), Id(1), Some(Weight(1.0)))]
                .iter()
                .cloned()
                .collect(),
            fitness: 0.0,
        };

        let mut genome_1 = Genome::from(&genome_0);

        genome_1
            .connection_genes
            .replace(ConnectionGene::new(Id(0), Id(1), Some(Weight(2.0))));

        println!("genome_0: {:?}", genome_0);
        println!("genome_1: {:?}", genome_1);

        let delta = Species::compatability_distance(&genome_0, &genome_1, 0.0, 2.0, 0.0);

        assert_eq!(delta, 2.0);
    }

    #[test]
    fn compatability_distance_different_connection_genome() {
        let genome_0 = Genome {
            node_genes: vec![NodeGene::input(Id(0)), NodeGene::output(Id(1), None)]
                .iter()
                .cloned()
                .collect(),
            connection_genes: vec![ConnectionGene::new(Id(0), Id(1), Some(Weight(1.0)))]
                .iter()
                .cloned()
                .collect(),
            fitness: 0.0,
        };

        let mut genome_1 = Genome::from(&genome_0);

        genome_1
            .connection_genes
            .insert(ConnectionGene::new(Id(0), Id(2), Some(Weight(1.0))));
        genome_1
            .connection_genes
            .insert(ConnectionGene::new(Id(2), Id(1), Some(Weight(2.0))));

        println!("genome_0: {:?}", genome_0);
        println!("genome_1: {:?}", genome_1);

        let delta = Species::compatability_distance(&genome_0, &genome_1, 2.0, 0.0, 0.0);

        // factor 2 times 2 different genes
        assert_eq!(delta, 2.0 * 2.0);
    }
}
