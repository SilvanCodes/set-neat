use std::collections::HashSet;
use favannat::network::NetLike;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::genes::{ConnectionGene, NodeGene, Weight};
use crate::{Context, Parameters};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub node_genes: HashSet<NodeGene>,
    pub connection_genes: HashSet<ConnectionGene>,
    pub fitness: f64,
}

impl NetLike<NodeGene, ConnectionGene> for Genome {
    fn nodes(&self) -> Vec<&NodeGene> {
        self.node_genes.iter().collect()
    }
    fn edges(&self) -> Vec<&ConnectionGene> {
        self.connection_genes.iter().collect()
    }
    fn inputs(&self) -> Vec<&NodeGene> {
        self.node_genes
            .iter()
            .filter(|node_gene| node_gene.is_input())
            .collect()
    }
    fn outputs(&self) -> Vec<&NodeGene> {
        self.node_genes
            .iter()
            .filter(|node_gene| node_gene.is_output())
            .collect()
    }
}

// public API
impl Genome {
    pub fn new(context: &mut Context, parameters: &Parameters) -> Self {
        let mut node_genes = HashSet::new();
        for _ in 0..parameters.setup.dimension.input {
            node_genes.insert(NodeGene::input(context.get_id()));
        }
        for _ in 0..parameters.setup.dimension.output {
            node_genes.insert(NodeGene::output(
                context.get_id(),
                Some(parameters.initialization.output),
            ));
        }

        Genome {
            node_genes,
            connection_genes: HashSet::new(),
            fitness: 0.0,
        }
    }

    pub fn from(genome: &Genome) -> Self {
        Genome {
            node_genes: genome.node_genes.clone(),
            connection_genes: genome.connection_genes.clone(),
            fitness: 0.0,
        }
    }

    pub fn init(&mut self) {
        // fully connects inputs and outputs
        for input in self
            .node_genes
            .iter()
            .filter(|node_gene| node_gene.is_input())
        // .take(1)
        {
            for output in self
                .node_genes
                .iter()
                .filter(|node_gene| node_gene.is_output())
            {
                self.connection_genes
                    .insert(ConnectionGene::new(input.id, output.id, None));
            }
        }
    }

    pub fn init_with(&mut self, context: &mut Context, parameters: &Parameters) {
        // fully connects inputs and outputs
        for input in self
            .node_genes
            .iter()
            .filter(|node_gene| node_gene.is_input())
            // make iterator wrap
            .cycle()
            // randomly offest into the iterator
            .skip((context.small_rng.gen::<f64>() * self.node_genes.len() as f64).floor() as usize)
            // connect configured percent of inputs to outputs
            .take(
                (parameters.initialization.connections * parameters.setup.dimension.input as f64)
                    .ceil() as usize,
            )
        {
            for output in self
                .node_genes
                .iter()
                .filter(|node_gene| node_gene.is_output())
            {
                assert!(self
                    .connection_genes
                    .insert(ConnectionGene::new(input.id, output.id, None)));
            }
        }
    }

    pub fn mutate(&mut self, context: &mut Context, parameters: &Parameters) {
        // mutate weigths
        if context.gamble(parameters.mutation.weight) {
            self.connection_genes = self
                .connection_genes
                .drain()
                .map(|mut connection_gene| {
                    if context.gamble(parameters.mutation.weight_random) {
                        connection_gene.weight.random(context);
                    } else {
                        connection_gene.weight.perturbate(context);
                    }
                    connection_gene
                })
                .collect();
        }

        // mutate connection gene
        if context.gamble(parameters.mutation.gene_connection) {
            self.add_connection(context, parameters).unwrap_or_default();
        }

        // mutate node gene
        if context.gamble(parameters.mutation.gene_node) {
            self.add_node(context, parameters);
        }

        if parameters.initialization.activations.len() > 1 && context.gamble(parameters.mutation.activation_change) {
            self.alter_activation(context, parameters);
        }
    }

    pub fn crossover(&self, partner: &Genome, context: &mut Context) -> Self {
        // self is fitter if it has higher fitness or in case of equal fitness has fewer genes
        let self_is_fitter = self.fitness > partner.fitness
            || ((self.fitness - partner.fitness).abs() < f64::EPSILON
                && self.connection_genes.len() < partner.connection_genes.len());

        // gamble for matching genes
        let mut offspring_connection_genes: HashSet<ConnectionGene> = self
            .connection_genes
            .intersection(&partner.connection_genes)
            .map(|gene_self| {
                if context.gamble(0.5) {
                    gene_self.clone()
                } else {
                    partner.connection_genes.get(&gene_self).unwrap().clone()
                }
            })
            .collect();

        // add different genes
        offspring_connection_genes.extend(if self_is_fitter {
            self.connection_genes
                .difference(&partner.connection_genes)
                .cloned()
        } else {
            partner
                .connection_genes
                .difference(&self.connection_genes)
                .cloned()
        });

        // select required nodes
        let offspring_node_genes: HashSet<NodeGene> = if self_is_fitter {
            self.node_genes.clone()
        } else {
            partner.node_genes.clone()
        };

        Genome {
            node_genes: offspring_node_genes,
            connection_genes: offspring_connection_genes,
            fitness: 0.0,
        }
    }

    pub fn unroll(&self) -> Genome {
        Genome::from(self)
    }
}

// private API
impl Genome {
    pub fn alter_activation(&mut self, context: &mut Context, parameters: &Parameters) {
        if let Some(node) = self
            .node_genes
            .iter()
            .cycle()
            .skip((context.small_rng.gen::<f64>() * self.node_genes.len() as f64).floor() as usize)
            .find(|node_gene| !node_gene.is_output())
        {
            let mut updated = node.clone();

            updated.update_activation(
                parameters
                    .initialization
                    .activations
                    .iter()
                    .filter(|&&activation| activation != updated.activation)
                    .choose(&mut context.small_rng)
                    .cloned(),
            );

            self.node_genes.replace(updated);
        }
    }

    // TODO: reject recurrent connections if set in settings
    pub fn add_connection(
        &mut self,
        context: &mut Context,
        parameters: &Parameters,
    ) -> Result<(), &'static str> {
        for possible_start_node_gene in self
            .node_genes
            .iter()
            .filter(|node_gene| !node_gene.is_output())
            // make iterator wrap
            .cycle()
            // randomly offset into the iterator to choose any node
            .skip((context.small_rng.gen::<f64>() * self.node_genes.len() as f64).floor() as usize)
            // just loop every value once
            .take(self.node_genes.len())
        {
            if let Some(possible_end_node_gene) = self.node_genes.iter().find(|&node_gene| {
                node_gene != possible_start_node_gene
                    && !node_gene.is_input()
                    && !self.are_connected(&possible_start_node_gene, node_gene)
                    && !self.would_form_cycle(possible_start_node_gene, node_gene)
            }) {
                // add new connection
                assert!(self.connection_genes.insert(ConnectionGene::new(
                    possible_start_node_gene.id,
                    possible_end_node_gene.id,
                    Some(parameters.initialization.weights.init()),
                )));

                return Ok(());
            }
            // no possible connection end present
        }
        Err("no connection possible")
    }

    pub fn add_node(&mut self, context: &mut Context, parameters: &Parameters) {
        // select an connection gene and split
        let mut random_connection_gene = self
            .connection_genes
            .iter()
            .choose(&mut context.small_rng)
            .cloned()
            .unwrap();

        let id = context
            .get_id_iter(random_connection_gene.id())
            .find(|&id| self.node_genes.get(&NodeGene::input(id)).is_none())
            .unwrap();

        // construct new node gene
        let new_node_gene_0 = NodeGene::new(
            id,
            None,
            parameters
                .initialization
                .activations
                .choose(&mut context.small_rng)
                .cloned(),
        );

        // insert new connection pointing to new node
        assert!(self.connection_genes.insert(ConnectionGene::new(
            random_connection_gene.input,
            new_node_gene_0.id,
            Some(Weight(1.0)),
        )));
        // insert new connection pointing from new node
        assert!(self.connection_genes.insert(ConnectionGene::new(
            new_node_gene_0.id,
            random_connection_gene.output,
            Some(random_connection_gene.weight),
        )));
        // insert new node into genome
        assert!(self.node_genes.insert(new_node_gene_0));

        // update weight to zero to 'deactivate' connnection
        random_connection_gene.weight = Weight(0.0);
        self.connection_genes.replace(random_connection_gene);
    }

    // check if to nodes are connected
    fn are_connected(&self, node_gene_start: &NodeGene, node_gene_end: &NodeGene) -> bool {
        self.connection_genes.iter().any(|connection_gene| {
            connection_gene.input == node_gene_start.id
                && connection_gene.output == node_gene_end.id
        })
    }

    // can only operate when no cycles present yet, which is assumed
    fn would_form_cycle(&self, node_gene_start: &NodeGene, node_gene_end: &NodeGene) -> bool {
        // needs to detect if there is a path from end to start
        let mut possible_paths: Vec<&ConnectionGene> = self
            .connection_genes
            .iter()
            .filter(|connection_gene| connection_gene.input == node_gene_end.id)
            .collect();
        let mut next_possible_path = Vec::new();

        while !possible_paths.is_empty() {
            for path in &possible_paths {
                // we have a cycle if path leads to start_node_gene
                if path.output == node_gene_start.id {
                    return true;
                }
                // collect further paths
                else {
                    next_possible_path.extend(
                        self.connection_genes
                            .iter()
                            .filter(|connection_gene| connection_gene.input == path.output),
                    );
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
    use super::Genome;
    use crate::context::Context;
    use crate::genes::{ConnectionGene, Id, NodeGene, Activation};
    use crate::Parameters;

    #[test]
    fn alter_activation() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 0;
        parameters.initialization.activations = vec![ Activation::Absolute, Activation::Cosine ];

        let mut genome = Genome::new(&mut context, &parameters);

        let old_activation = genome.node_genes.iter().nth(0).unwrap().activation.clone();

        genome.alter_activation(&mut context, &parameters);

        assert_ne!(
            genome.node_genes.iter().nth(0).unwrap().activation,
            old_activation
        );
    }

    #[test]
    fn add_random_connection() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome = Genome::new(&mut context, &parameters);

        let result = genome.add_connection(&mut context, &parameters).is_ok();

        println!("{:?}", genome);

        assert_eq!(result, true);
        assert_eq!(genome.connection_genes.len(), 1);
    }

    #[test]
    fn dont_add_same_connection_twice() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome = Genome::new(&mut context, &parameters);

        let result_0 = genome.add_connection(&mut context, &parameters).is_ok();
        if let Err(message) = genome.add_connection(&mut context, &parameters) {
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
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome = Genome::new(&mut context, &parameters);

        genome.init();
        genome.add_node(&mut context, &parameters);

        println!("{:?}", genome);

        assert_eq!(genome.connection_genes.len(), 3);
    }

    #[test]
    fn crossover_same_fitness() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        let mut genome_1 = Genome::from(&genome_0);

        // mutate genome_0
        genome_0.add_node(&mut context, &parameters);

        // mutate genome_1
        genome_1.add_node(&mut context, &parameters);
        genome_1.add_node(&mut context, &parameters);

        println!("genome_0 {:?}", genome_0);
        println!("genome_1 {:?}", genome_1);

        // shorter genome is fitter genome
        let offspring = genome_0.crossover(&genome_1, &mut context);

        println!("offspring {:?}", offspring);

        assert_eq!(offspring.node_genes.len(), 3);
        assert_eq!(offspring.connection_genes.len(), 3);
    }

    #[test]
    fn crossover_different_fitness() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 2;
        parameters.setup.dimension.output = 1;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        let mut genome_1 = Genome::from(&genome_0);

        genome_1.fitness = 1.0;

        // mutate genome_0
        genome_0.add_node(&mut context, &parameters);

        // mutate genome_1
        genome_1.add_node(&mut context, &parameters);
        genome_1.add_connection(&mut context, &parameters).unwrap();

        println!("genome_0 {:?}", genome_0);
        println!("genome_1 {:?}", genome_1);

        let offspring = genome_0.crossover(&genome_1, &mut context);

        assert_eq!(offspring.node_genes.len(), 4);
        assert_eq!(offspring.connection_genes.len(), 5);
    }

    #[test]
    fn detect_no_cycle() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        let input = genome_0
            .node_genes
            .iter()
            .find(|node_gene| node_gene.is_input())
            .unwrap();
        let output = genome_0
            .node_genes
            .iter()
            .find(|node_gene| node_gene.is_output())
            .unwrap();

        let result = genome_0.would_form_cycle(&input, &output);

        assert!(!result);
    }

    #[test]
    fn detect_cycle() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init();

        // mutate genome_0
        genome_0.add_node(&mut context, &parameters);

        let input = genome_0
            .node_genes
            .iter()
            .find(|node_gene| node_gene.is_input())
            .unwrap();
        let output = genome_0
            .node_genes
            .iter()
            .find(|node_gene| node_gene.is_output())
            .unwrap();

        let result = genome_0.would_form_cycle(&output, &input);

        println!("{:?}", genome_0);

        assert!(result);
    }

    #[test]
    fn crossover_no_cycle() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        // assumption:
        // crossover of equal fitness genomes should not produce cycles
        // prerequisits:
        // genomes with equal fitness (0.0 in this case)
        // "mirrored" structure as simplest example

        let mut genome0 = Genome {
            node_genes: vec![
                NodeGene::input(Id(0)),
                NodeGene::output(Id(1), None),
                NodeGene::new(Id(2), None, None),
                NodeGene::new(Id(3), None, None),
            ]
            .iter()
            .cloned()
            .collect(),
            connection_genes: vec![
                ConnectionGene::new(Id(0), Id(2), None),
                ConnectionGene::new(Id(2), Id(1), None),
                ConnectionGene::new(Id(0), Id(3), None),
                ConnectionGene::new(Id(3), Id(1), None),
            ]
            .iter()
            .cloned()
            .collect(),
            fitness: 0.0,
        };

        let mut genome1 = Genome::from(&genome0);

        // insert connectio one way in genome0
        genome0
            .connection_genes
            .insert(ConnectionGene::new(Id(2), Id(3), None));

        // insert connection the other way in genome1
        genome1
            .connection_genes
            .insert(ConnectionGene::new(Id(3), Id(2), None));

        let offspring = genome0.crossover(&genome1, &mut context);

        println!("offspring {:?}", offspring);

        for connection0 in &offspring.connection_genes {
            for connection1 in &offspring.connection_genes {
                println!(
                    "{:?}->{:?}, {:?}->{:?}",
                    connection0.input, connection0.output, connection1.input, connection1.output
                );
                assert!(
                    !(connection0.input == connection1.output
                        && connection0.output == connection1.input)
                )
            }
        }
    }
}
