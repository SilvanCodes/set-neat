// crate imports
use crate::context::Context;
use crate::parameters::Parameters;
use crate::genome::Genome;

#[derive(Debug)]
pub struct Species {
    pub members: Vec<Genome>,
    pub representative: Option<Genome>,
    pub fitness: f64,
    pub stale: usize
}

// public API
impl Species {
    pub fn new(first_member: Genome) -> Self {
        Species {
            members: vec![first_member.clone()],
            representative: Some(first_member),
            fitness: 0.0,
            stale: 0
        }
    }

    pub fn compatible(&self, genome: &Genome, parameters: &Parameters, context: &Context) -> bool {
        Species::compatability_distance(
            &genome,
            &self.representative.as_ref().unwrap(),
                parameters.compatability.factor_genes,
                parameters.compatability.factor_weights,
            ) < context.compatability_threshold
    }

    pub fn represent(&mut self) {
        self.representative = self.members.first().map(|genome| genome.clone());
        /* if self.representative.is_some() {
            println!(
                "species representative: connections {:?}, nodes {:?}, fitness: {:?}",
                self.representative.as_ref().unwrap().connection_genes.len(),
                self.representative.as_ref().unwrap().node_genes.len(),
                self.representative.as_ref().unwrap().fitness
            );
        } */
    }

    pub fn adjust_fitness(&mut self, parameters: &Parameters) {
        let old_fitness = self.fitness;
        let factor = self.members.len() as f64;

        self.fitness = 0.0;

        for genome in &mut self.members {
            // println!("plain fitness: {}", genome.fitness);
            genome.fitness = genome.fitness / factor;
            self.fitness += genome.fitness;
        }
        // did fitness increase ?
        if self.fitness > old_fitness + parameters.staleness.epsilon {
            self.stale = 0;
        } else {
            self.stale += 1;
        }
        // println!("staleness: {}", self.stale);
    }

    // compatability distance mechanism assumes sorted connection genes!
    pub fn compatability_distance(genome_0: &Genome, genome_1: &Genome, c1: f64, c2: f64) -> f64 {
        let mut weight_difference = 0.0;
        
        let matching_genes_count = genome_0.connection_genes
            .intersection(&genome_1.connection_genes)
            .inspect(|connection_gene| weight_difference += connection_gene.weight.difference(&genome_1.connection_genes.get(connection_gene).unwrap().weight))
            .count();

        let different_genes_count = genome_0.connection_genes.symmetric_difference(&genome_1.connection_genes).count();

        // N is gene count of larger genome
        // let n = genome_0.connection_genes.len().max(genome_1.connection_genes.len()) as f64;

        // println!("different_genes_count: {:?}", different_genes_count);
        // println!("matching_genes_count: {:?}", matching_genes_count);
        // println!("weight_difference: {:?}", weight_difference);
        // println!("n: {:?}", n);

        // distance formula from paper
        c1 * different_genes_count as f64 + c2 * weight_difference / matching_genes_count as f64
    }
}


#[cfg(test)]
mod tests {
    use super::Species;
    use crate::genome::Genome;
    use crate::genes::{Id, node::{NodeGene, NodeKind}, connection::{ConnectionGene, Weight}};

    #[test]
    fn compatability_distance_same_genome() {
        let genome_0 = Genome {
            node_genes: vec![
                NodeGene::new(Id(0), Some(NodeKind::Input)),
                NodeGene::new(Id(1), Some(NodeKind::Output)),
            ],
            connection_genes: vec![
                ConnectionGene::new(Id(0), Id(1), None)
            ].iter().cloned().collect(),
            fitness: 0.0
        };

        let genome_1 = Genome::from(&genome_0);

        let delta = Species::compatability_distance(&genome_0, &genome_1, 1.0, 0.4);

        assert_eq!(delta, 0.0);
    }

    #[test]
    fn compatability_distance_different_weight_genome() {
        let genome_0 = Genome {
            node_genes: vec![
                NodeGene::new(Id(0), Some(NodeKind::Input)),
                NodeGene::new(Id(1), Some(NodeKind::Output)),
            ],
            connection_genes: vec![
                ConnectionGene::new(Id(0), Id(1), Some(Weight(1.0))),
            ].iter().cloned().collect(),
            fitness: 0.0
        };

        let mut genome_1 = Genome::from(&genome_0);

        genome_1.connection_genes.replace(ConnectionGene::new(Id(0), Id(1), Some(Weight(2.0))));

        println!("genome_0: {:?}", genome_0);
        println!("genome_1: {:?}", genome_1);

        let delta = Species::compatability_distance(&genome_0, &genome_1, 0.0, 2.0);

        assert_eq!(delta, 2.0);
    }

    #[test]
    fn compatability_distance_different_connection_genome() {
        let genome_0 = Genome {
            node_genes: vec![
                NodeGene::new(Id(0), Some(NodeKind::Input)),
                NodeGene::new(Id(1), Some(NodeKind::Output)),
            ],
            connection_genes: vec![
                ConnectionGene::new(Id(0), Id(1), Some(Weight(1.0))),
            ].iter().cloned().collect(),
            fitness: 0.0
        };

        let mut genome_1 = Genome::from(&genome_0);

        genome_1.connection_genes.insert(ConnectionGene::new(Id(0), Id(2), Some(Weight(1.0))));
        genome_1.connection_genes.insert(ConnectionGene::new(Id(2), Id(1), Some(Weight(2.0))));

        println!("genome_0: {:?}", genome_0);
        println!("genome_1: {:?}", genome_1);

        let delta = Species::compatability_distance(&genome_0, &genome_1, 2.0, 0.0);

        // factor 2 times 2 different genes
        assert_eq!(delta, 2.0 * 2.0);
    }
}
