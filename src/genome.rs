// std imports
use crate::parameters::Parameters;
use std::collections::HashSet;
// external imports
use rand::{Rng};
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
// crate imports
use crate::context::Context;
// sub-modules
use crate::genes::{node::{NodeGene, NodeKind}, connection::{ConnectionGene, Weight}};


#[derive(Debug, Clone)]
pub struct Genome {
    pub node_genes: Vec<NodeGene>,
    pub connection_genes: HashSet<ConnectionGene>,
    pub fitness: f64
}

// public API
impl Genome {
    pub fn new(context: &mut Context, parameters: &Parameters) -> Self {
        let mut node_genes = Vec::new();
        for _ in 0..parameters.setup.dimension.input {
            node_genes.push(NodeGene::new(context.get_id(None), Some(NodeKind::Input)))
        }
        for _ in 0..parameters.setup.dimension.output {
            node_genes.push(NodeGene::new(context.get_id(None), Some(NodeKind::Output)))
        }

        Genome {
            node_genes,
            connection_genes: HashSet::new(),
            fitness: 0.0
        }
    }

    pub fn from(genome: &Genome) -> Self {
        Genome {
            node_genes: genome.node_genes.clone(),
            connection_genes: genome.connection_genes.clone(),
            fitness: 0.0
        }
    }

    pub fn init(&mut self) {
        // fully connects inputs and outputs
        for input in self.node_genes.iter().filter(|node| node.kind == NodeKind::Input) {
            for output in self.node_genes.iter().filter(|node| node.kind == NodeKind::Output) {
                self.connection_genes.insert(
                    ConnectionGene::new(
                        input.id,
                        output.id,
                        None
                    )
                );
            }
        }
    }

    pub fn mutate(&mut self, context: &mut Context, parameters: &Parameters) {
        // mutate weigths
        if context.small_rng.gen::<f64>() < parameters.mutation.weight {
            self.connection_genes = self.connection_genes.drain().map(|mut connection_gene| {
                if context.small_rng.gen::<f64>() < parameters.mutation.weight_random {
                    connection_gene.weight.random();
                } else {
                    connection_gene.weight.perturbate(parameters.mutation.weight_perturbation);
                }
                connection_gene
            }).collect();
        }

        // mutate connection gene
        if context.small_rng.gen::<f64>() < parameters.mutation.gene_connection {
            self.add_connection(context).unwrap_or_default();
        }

        // mutate node gene
        if context.small_rng.gen::<f64>() < parameters.mutation.gene_node {
            self.add_node(context);
        }
    }

    pub fn crossover(&self, partner: &Genome, context: &mut Context) -> Self {
        // dbg!(self.connection_genes.difference(&partner.connection_genes).count());
        // dbg!(partner.connection_genes.difference(&self.connection_genes).count());
        // dbg!(partner.connection_genes.symmetric_difference(&self.connection_genes).count());

        // self is fitter if it has higher fitness or in case of equal fitness has fewer genes
        let self_is_fitter = self.fitness > partner.fitness
            || (
                self.fitness == partner.fitness
                && self.connection_genes.len() < partner.connection_genes.len()
            );

        // gamble for matching genes
        let mut offspring_connection_genes: HashSet<ConnectionGene> = self.connection_genes
            .intersection(&partner.connection_genes)
            .map(|gene_self| if context.small_rng.gen::<f64>() < 0.5 { gene_self.clone() } else { partner.connection_genes.get(&gene_self).unwrap().clone() })
            .collect();

        // select different genes
        let different_genes = if self_is_fitter {
            self.connection_genes.difference(&partner.connection_genes).cloned()
        } else {
            partner.connection_genes.difference(&self.connection_genes).cloned()
        };
        // add different genes
        offspring_connection_genes.extend(different_genes);

        // select required nodes
        let offspring_node_genes: Vec<NodeGene> = if self_is_fitter {
            self.node_genes.clone()
        } else {
            partner.node_genes.clone()
        };

        Genome {
            node_genes: offspring_node_genes,
            connection_genes: offspring_connection_genes,
            fitness: 0.0
        }
    }
}

// private API
impl Genome {
    // TODO: reject recurrent connections if set in settings
    pub fn add_connection(&mut self, context: &mut Context) -> Result<(), &'static str> {
        // println!("add_connection called");
        // shuffle node genes for randomly picking some
        self.node_genes.as_mut_slice().shuffle(&mut context.small_rng);

        for possible_start_node_gene in self.node_genes.iter().filter(|node_gene| node_gene.kind != NodeKind::Output) {
            if let Some(possible_end_node_gene) = self.node_genes.iter()
                .find(|&node_gene| node_gene != possible_start_node_gene
                                  && node_gene.kind != NodeKind::Input
                                  && !self.are_connected(&possible_start_node_gene, node_gene)
                                  && !self.would_form_cycle(possible_start_node_gene, node_gene)
            ) {
                // add new connection
                let new_connection_gene = ConnectionGene::new(
                    possible_start_node_gene.id,
                    possible_end_node_gene.id,
                    None
                );

                self.connection_genes.insert(new_connection_gene);

                return Ok(())
            }
            // no possible connection end present
        }
        Err("no connection possible")
    }

    pub fn add_node(&mut self, context: &mut Context) {
        // println!("add_node called");
        let between = Uniform::from(0..self.connection_genes.len());

        loop {
            // select an connection gene and split
            let mut random_connection_gene = self.connection_genes.iter().nth(between.sample(&mut context.small_rng)).unwrap().clone();

            // construct new node gene
            let new_node_gene_0 = NodeGene::new(
                context.get_id(Some((random_connection_gene.input, random_connection_gene.output))),
                None
            );
            // construct connection pointing to new node
            let new_connection_gene_0 = ConnectionGene::new(
                random_connection_gene.input,
                new_node_gene_0.id,
                Some(Weight(1.0))
            );
            // construct connection pointing from new node
            let new_connection_gene_1 = ConnectionGene::new(
                new_node_gene_0.id,
                random_connection_gene.output,
                Some(random_connection_gene.weight)
            );

            // set weight to zero to 'deactivate' connnection
            random_connection_gene.weight = Weight(0.0);

            // insert new structure into genome
            if self.connection_genes.insert(new_connection_gene_0) && self.connection_genes.insert(new_connection_gene_1) {
                self.connection_genes.replace(random_connection_gene);
                self.node_genes.push(new_node_gene_0);
                // println!("success adding node: {:?}", self);
                break;
            };
        }
    }

    // check if to nodes are connected
    fn are_connected(&self, node_gene_start: &NodeGene, node_gene_end: &NodeGene) -> bool {
        for connection_gene in &self.connection_genes {
            if connection_gene.input == node_gene_start.id && connection_gene.output == node_gene_end.id {
                return true;
            }
        }
        false
    }

    // can only operate when no cycles present yet, which is assumed
    fn would_form_cycle(&self, node_gene_start: &NodeGene, node_gene_end: &NodeGene) -> bool {
        // needs to detect if there is a path from end to start
        let mut possible_paths = self.connection_genes.iter().filter(|connection_gene| connection_gene.input == node_gene_end.id).collect::<Vec<&ConnectionGene>>();
        let mut next_possible_path = Vec::new();

        while !possible_paths.is_empty() {
            for path in &possible_paths {
                // we have a cycle if path leads to start_node_gene
                if path.output == node_gene_start.id {
                    return true;
                }
                // collect further paths
                else {
                    next_possible_path.append(&mut self.connection_genes.iter().filter(|connection_gene| connection_gene.input == path.output).collect::<Vec<&ConnectionGene>>());
                }
            }
            possible_paths = next_possible_path;
            next_possible_path = Vec::new();
        }
        false
    }
}



#[cfg(test)]
mod tests {
    use crate::context::Context;
    use crate::genes::{node::{NodeGene, NodeKind}, connection::ConnectionGene, Id};
    use crate::Parameters;
    use super::Genome;

    #[test]
    fn add_random_connection() {
        let mut parameters: Parameters = Default::default(); 
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome = Genome::new(&mut context, &parameters);

        let result = genome.add_connection(&mut context).is_ok();

        println!("{:?}", genome);

        assert_eq!(result, true);
        assert_eq!(genome.connection_genes.len(), 1);
    }

    #[test]
    fn dont_add_same_connection_twice() {
        let mut parameters: Parameters = Default::default(); 
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome = Genome::new(&mut context, &parameters);

        let result_0 = genome.add_connection(&mut context).is_ok();
        if let Err(message) = genome.add_connection(&mut context) {
            assert_eq!(message, "no connection possible");
        } else {
            assert!(false);
        }

        println!("{:?}", genome);

        assert_eq!(result_0, true);
        assert_eq!(genome.connection_genes.len(), 1);
    }

    #[test]
    fn add_random_node() {
        let mut parameters: Parameters = Default::default(); 
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome = Genome::new(&mut context, &parameters);

        genome.init();
        genome.add_node(&mut context);

        println!("{:?}", genome);

        assert_eq!(genome.connection_genes.len(), 3);
    }

    #[test]
    fn crossover_same_fitness() {
        let mut parameters: Parameters = Default::default(); 
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        let mut genome_1 = Genome::from(&genome_0);

        // mutate genome_0
        genome_0.add_node(&mut context);

        // mutate genome_1
        genome_1.add_node(&mut context);
        genome_1.add_node(&mut context);

        println!("genome_0 {:?}", genome_0);
        println!("genome_1 {:?}", genome_1);

        let offspring = genome_0.crossover(&genome_1, &mut context);

        println!("offspring {:?}", offspring);

        assert_eq!(offspring.node_genes.len(), 3);
        assert_eq!(offspring.connection_genes.len(), 3);
    }

    #[test]
    fn crossover_different_fitness() {
        let mut parameters: Parameters = Default::default(); 
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 2;
        parameters.setup.dimension.output = 1;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        let mut genome_1 = Genome::from(&genome_0);

        genome_1.fitness = 1.0;

        // mutate genome_0
        genome_0.add_node(&mut context);

        // mutate genome_1
        genome_1.add_node(&mut context);
        genome_1.add_connection(&mut context).unwrap();

        println!("genome_0 {:?}", genome_0);
        println!("genome_1 {:?}", genome_1);

        let offspring = genome_0.crossover(&genome_1, &mut context);

        assert_eq!(offspring.node_genes.len(), 4);
        assert_eq!(offspring.connection_genes.len(), 5);
    }

    #[test]
    fn detect_no_cycle() {
        let mut parameters: Parameters = Default::default(); 
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        let result = genome_0.would_form_cycle(&genome_0.node_genes[0], &genome_0.node_genes[1]);

        assert!(!result);
    }

    #[test]
    fn detect_cycle() {
        let mut parameters: Parameters = Default::default(); 
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        // mutate genome_0
        genome_0.add_node(&mut context);

        let result = genome_0.would_form_cycle(&genome_0.node_genes[1], &genome_0.node_genes[0]);

        println!("{:?}", genome_0);

        assert!(result);
    }

    #[test]
    fn crossover_no_cycle() {
        let parameters: Parameters = Default::default();
        let mut context = Context::new(&parameters);

        // assumption:
        // crossover of equal fitness genomes should not produce cycles
        // prerequisits:
        // genomes with equal fitness (0.0 in this case)
        // "mirrored" structure as simplest example

        let mut genome0 = Genome {
            node_genes: vec![
                NodeGene::new(Id(0), Some(NodeKind::Input)),
                NodeGene::new(Id(1), Some(NodeKind::Output)),
                NodeGene::new(Id(2), Some(NodeKind::Hidden)),
                NodeGene::new(Id(3), Some(NodeKind::Hidden))
            ],
            connection_genes: vec![
                ConnectionGene::new(Id(0), Id(2), None),
                ConnectionGene::new(Id(2), Id(1), None),
                ConnectionGene::new(Id(0), Id(3), None),
                ConnectionGene::new(Id(3), Id(1), None)
            ].iter().cloned().collect(),
            fitness: 0.0
        };

        let mut genome1 = Genome::from(&genome0);

        // insert connectio one way in genome0
        genome0.connection_genes.insert(ConnectionGene::new(Id(2), Id(3), None));

        // insert connection the other way in genome1
        genome1.connection_genes.insert(ConnectionGene::new(Id(3), Id(2), None));

        let offspring = genome0.crossover(&genome1, &mut context);

        println!("offspring {:?}", offspring);

        for connection0 in &offspring.connection_genes {
            for connection1 in &offspring.connection_genes {
                println!("{:?}->{:?}, {:?}->{:?}", connection0.input, connection0.output, connection1.input, connection1.output);
                assert!(!(connection0.input == connection1.output && connection0.output == connection1.input))
            }
        }
    }
}
