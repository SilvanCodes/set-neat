use crate::{
    individual::genes::{connections::Connection, nodes::Node, Activation, Genes, IdGenerator},
    parameters::Parameters,
    rng::NeatRng,
};

use rand::{
    prelude::{IteratorRandom, SliceRandom},
    Rng,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub inputs: Genes<Node>,
    pub hidden: Genes<Node>,
    pub outputs: Genes<Node>,
    pub feed_forward: Genes<Connection>,
    pub recurrent: Genes<Connection>,
}

impl Genome {
    pub fn new(id_gen: &mut IdGenerator, parameters: &Parameters) -> Self {
        Genome {
            inputs: (0..parameters.setup.input_dimension)
                .map(|_| Node::new(id_gen.next_id(), Activation::Linear))
                .collect(),
            outputs: (0..parameters.setup.output_dimension)
                .map(|_| Node::new(id_gen.next_id(), parameters.activations.output_nodes))
                .collect(),
            ..Default::default()
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.inputs
            .iter()
            // .iterate_unwrapped()
            .chain(self.hidden.iter() /* .iterate_unwrapped() */)
            .chain(self.outputs.iter() /* .iterate_unwrapped() */)
    }

    pub fn init(&mut self, rng: &mut NeatRng, parameters: &Parameters) {
        for input in self.inputs.iterate_with_random_offset(&mut rng.small).take(
            (rng.small.gen::<f64>() * parameters.setup.input_dimension as f64).ceil() as usize,
        ) {
            // connect to every output
            for output in self.outputs.iter() {
                assert!(self.feed_forward.insert(Connection::new(
                    input.id,
                    rng.weight_perturbation(),
                    output.id
                )));
            }
        }
    }

    pub fn len(&self) -> usize {
        self.feed_forward.len() + self.recurrent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.feed_forward.is_empty() && self.recurrent.is_empty()
    }

    pub fn cross_in(&self, other: &Self, rng: &mut impl Rng) -> Self {
        let feed_forward = self.feed_forward.cross_in(&other.feed_forward, rng);

        let recurrent = self.recurrent.cross_in(&other.recurrent, rng);

        let hidden = self.hidden.cross_in(&other.hidden, rng);

        Genome {
            feed_forward,
            recurrent,
            hidden,
            // use input and outputs from fitter, but they should be identical with weaker
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }

    pub fn mutate(&mut self, rng: &mut NeatRng, id_gen: &mut IdGenerator, parameters: &Parameters) {
        // mutate weigths
        // if context.gamble(parameters.mutation.weight) {
        self.change_weights(rng);
        // }

        // mutate connection gene
        if rng.gamble(parameters.mutation.new_connection_chance) {
            self.add_connection(rng, parameters).unwrap_or_default();
        }

        // mutate node gene
        if rng.gamble(parameters.mutation.new_node_chance) {
            self.add_node(rng, id_gen, parameters);
        }

        // change some activation
        if rng.gamble(parameters.mutation.change_activation_function_chance) {
            self.alter_activation(rng, parameters);
        }
    }

    pub fn change_weights(&mut self, rng: &mut NeatRng) {
        self.feed_forward = self
            .feed_forward
            .drain_into_random(&mut rng.small)
            .map(|mut connection| {
                connection.weight += rng.weight_perturbation();
                connection
            })
            .collect();

        self.recurrent = self
            .recurrent
            .drain_into_random(&mut rng.small)
            .map(|mut connection| {
                connection.weight += rng.weight_perturbation();
                connection
            })
            .collect();
    }

    pub fn alter_activation(&mut self, rng: &mut NeatRng, parameters: &Parameters) {
        if let Some(node) = self.hidden.random(&mut rng.small) {
            let updated = Node::new(
                node.id,
                parameters
                    .activations
                    .hidden_nodes
                    .iter()
                    .filter(|&&activation| activation != node.activation)
                    .choose(&mut rng.small)
                    .cloned()
                    .unwrap_or(node.activation),
            );

            self.hidden.replace(updated);
        }
    }

    pub fn add_node(
        &mut self,
        rng: &mut NeatRng,
        id_gen: &mut IdGenerator,
        parameters: &Parameters,
    ) {
        // select an connection gene and split
        let mut random_connection = self.feed_forward.random(&mut rng.small).cloned().unwrap();

        let id = id_gen
            .cached_id_iter(random_connection.id())
            .find(|&id| {
                self.hidden
                    .get(&Node::new(id, Activation::Linear))
                    .is_none()
            })
            .unwrap();

        // construct new node gene
        let new_node = Node::new(
            id,
            parameters
                .activations
                .hidden_nodes
                .choose(&mut rng.small)
                .cloned()
                .unwrap(),
        );

        // insert new connection pointing to new node
        assert!(self.feed_forward.insert(Connection::new(
            random_connection.input,
            1.0,
            new_node.id,
        )));
        // insert new connection pointing from new node
        assert!(self.feed_forward.insert(Connection::new(
            new_node.id,
            random_connection.weight,
            random_connection.output,
        )));
        // insert new node into genome
        assert!(self.hidden.insert(new_node));

        // update weight to zero to 'deactivate' connnection
        random_connection.weight = 0.0;
        self.feed_forward.replace(random_connection);
    }

    pub fn add_connection(
        &mut self,
        rng: &mut NeatRng,
        parameters: &Parameters,
    ) -> Result<(), &'static str> {
        let is_recurrent = rng.gamble(parameters.mutation.connection_is_recurrent_chance);

        let start_node_iterator = self
            .inputs
            .iter()
            // .iterate_unwrapped()
            .chain(self.hidden.iter() /* .iterate_unwrapped() */);

        let end_node_iterator = self
            .hidden
            .iter()
            // .iterate_unwrapped()
            .chain(self.outputs.iter() /* .iterate_unwrapped() */);

        for start_node in start_node_iterator
            // make iterator wrap
            .cycle()
            // randomly offset into the iterator to choose any node
            .skip(
                (rng.small.gen::<f64>() * (self.inputs.len() + self.hidden.len()) as f64).floor()
                    as usize,
            )
            // just loop every value once
            .take(self.inputs.len() + self.hidden.len())
        {
            if let Some(end_node) = end_node_iterator.clone().find(|&end_node| {
                end_node != start_node
                    && !self.are_connected(&start_node, end_node, is_recurrent)
                    && (is_recurrent || !self.would_form_cycle(start_node, end_node))
            }) {
                if is_recurrent {
                    assert!(self.recurrent.insert(Connection::new(
                        start_node.id,
                        rng.weight_perturbation(),
                        end_node.id,
                    )));
                } else {
                    // add new feed-forward connection
                    assert!(self.feed_forward.insert(Connection::new(
                        start_node.id,
                        rng.weight_perturbation(),
                        end_node.id,
                    )));
                }
                return Ok(());
            }
            // no possible connection end present
        }
        Err("no connection possible")
    }

    // check if to nodes are connected
    fn are_connected(&self, start_node: &Node, end_node: &Node, recurrent: bool) -> bool {
        if recurrent {
            self.recurrent
                .contains(&Connection::new(start_node.id, 0.0, end_node.id))
        } else {
            self.feed_forward
                .contains(&Connection::new(start_node.id, 0.0, end_node.id))
        }
    }

    // can only operate when no cycles present yet, which is assumed
    fn would_form_cycle(&self, start_node: &Node, end_node: &Node) -> bool {
        // needs to detect if there is a path from end to start
        let mut possible_paths: Vec<&Connection> = self
            .feed_forward
            .iter()
            .filter(|connection| connection.input == end_node.id)
            .collect();
        let mut next_possible_path = Vec::new();

        while !possible_paths.is_empty() {
            for path in possible_paths {
                // we have a cycle if path leads to start_node_gene
                if path.output == start_node.id {
                    return true;
                }
                // collect further paths
                else {
                    next_possible_path.extend(
                        self.feed_forward
                            .iter()
                            .filter(|connection| connection.input == path.output),
                    );
                }
            }
            possible_paths = next_possible_path;
            next_possible_path = Vec::new();
        }
        false
    }

    pub fn compatability_distance(
        genome_0: &Self,
        genome_1: &Self,
        factor_genes: f64,
        factor_weights: f64,
        factor_activations: f64,
    ) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::Genome;
    use crate::{
        individual::genes::{
            connections::Connection, nodes::Node, Activation, Genes, Id, IdGenerator,
        },
        parameters::Parameters,
        rng::NeatRng,
    };

    #[test]
    fn alter_activation() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let parameters: Parameters = Default::default();

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut genome = Genome::new(&mut id_gen, &parameters);

        genome.init(&mut rng, &parameters);

        genome.add_node(&mut rng, &mut id_gen, &parameters);

        let old_activation = genome.hidden.iter().next().unwrap().activation;

        genome.alter_activation(&mut rng, &parameters);

        assert_ne!(
            genome.hidden.iter().next().unwrap().activation,
            old_activation
        );
    }

    #[test]
    fn add_random_connection() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let parameters: Parameters = Default::default();

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut genome = Genome::new(&mut id_gen, &parameters);

        let result = genome.add_connection(&mut rng, &parameters).is_ok();

        println!("{:?}", genome);

        assert_eq!(result, true);
        assert_eq!(genome.feed_forward.len(), 1);
    }

    #[test]
    fn dont_add_same_connection_twice() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let parameters: Parameters = Default::default();

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut genome = Genome::new(&mut id_gen, &parameters);

        let result_0 = genome.add_connection(&mut rng, &parameters).is_ok();
        if let Err(message) = genome.add_connection(&mut rng, &parameters) {
            assert_eq!(message, "no connection possible");
        } else {
            // assert!(false);
            unreachable!()
        }

        println!("{:?}", genome);

        assert_eq!(result_0, true);
        assert_eq!(genome.feed_forward.len(), 1);
    }

    #[test]
    fn add_random_node() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let parameters: Parameters = Default::default();

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut genome = Genome::new(&mut id_gen, &parameters);

        genome.init(&mut rng, &parameters);
        genome.add_node(&mut rng, &mut id_gen, &parameters);

        println!("{:?}", genome);

        assert_eq!(genome.feed_forward.len(), 3);
    }

    #[test]
    fn crossover() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let parameters: Parameters = Default::default();

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
        genome_1.add_node(&mut rng, &mut id_gen, &parameters);

        println!("genome_0 {:?}", genome_0);
        println!("genome_1 {:?}", genome_1);

        // shorter genome is fitter genome
        let offspring = genome_0.cross_in(&genome_1, &mut rng.small);

        println!("offspring {:?}", offspring);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 3);
    }

    #[test]
    fn detect_no_cycle() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let parameters: Parameters = Default::default();

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut genome_0 = Genome::new(&mut id_gen, &parameters);

        genome_0.init(&mut rng, &parameters);

        let input = genome_0.inputs.iter().next().unwrap();
        let output = genome_0.outputs.iter().next().unwrap();

        let result = genome_0.would_form_cycle(&input, &output);

        assert!(!result);
    }

    #[test]
    fn detect_cycle() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        let parameters: Parameters = Default::default();

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut genome_0 = Genome::new(&mut id_gen, &parameters);

        genome_0.init(&mut rng, &parameters);

        // mutate genome_0
        genome_0.add_node(&mut rng, &mut id_gen, &parameters);

        let input = genome_0.inputs.iter().next().unwrap();
        let output = genome_0.outputs.iter().next().unwrap();

        let result = genome_0.would_form_cycle(&output, &input);

        println!("{:?}", genome_0);

        assert!(result);
    }

    #[test]
    fn crossover_no_cycle() {
        let parameters: Parameters = Default::default();

        // create random source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        // assumption:
        // crossover of equal fitness genomes should not produce cycles
        // prerequisits:
        // genomes with equal fitness (0.0 in this case)
        // "mirrored" structure as simplest example

        let mut genome_0 = Genome {
            inputs: Genes(
                vec![Node::new(Id(0), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(1), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            hidden: Genes(
                vec![
                    Node::new(Id(2), Activation::Tanh),
                    Node::new(Id(3), Activation::Tanh),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            feed_forward: Genes(
                vec![
                    Connection::new(Id(0), 1.0, Id(2)),
                    Connection::new(Id(2), 1.0, Id(1)),
                    Connection::new(Id(0), 1.0, Id(3)),
                    Connection::new(Id(3), 1.0, Id(1)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            ..Default::default()
        };

        let mut genome_1 = genome_0.clone();

        // insert connectio one way in genome0
        genome_0
            .feed_forward
            .insert(Connection::new(Id(2), 1.0, Id(3)));

        // insert connection the other way in genome1
        genome_1
            .feed_forward
            .insert(Connection::new(Id(3), 1.0, Id(2)));

        let offspring = genome_0.cross_in(&genome_1, &mut rng.small);

        println!("offspring {:?}", offspring);

        for connection0 in offspring.feed_forward.iter() {
            for connection1 in offspring.feed_forward.iter() {
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

    #[test]
    fn compatability_distance_same_genome() {
        let genome_0 = Genome {
            inputs: Genes(
                vec![Node::new(Id(0), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(1), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),

            feed_forward: Genes(
                vec![Connection::new(Id(0), 1.0, Id(1))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        let genome_1 = genome_0.clone();

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 1.0, 0.4, 0.0);

        assert!(delta < f64::EPSILON);
    }

    #[test]
    fn compatability_distance_different_weight_genome() {
        let genome_0 = Genome {
            inputs: Genes(
                vec![Node::new(Id(0), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(1), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),

            feed_forward: Genes(
                vec![Connection::new(Id(0), 1.0, Id(1))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        let mut genome_1 = genome_0.clone();

        genome_1
            .feed_forward
            .replace(Connection::new(Id(0), 2.0, Id(1)));

        println!("genome_0: {:?}", genome_0);
        println!("genome_1: {:?}", genome_1);

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 0.0, 2.0, 0.0);

        assert!((delta - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compatability_distance_different_connection_genome() {
        let genome_0 = Genome {
            inputs: Genes(
                vec![Node::new(Id(0), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            outputs: Genes(
                vec![Node::new(Id(1), Activation::Linear)]
                    .iter()
                    .cloned()
                    .collect(),
            ),

            feed_forward: Genes(
                vec![Connection::new(Id(0), 1.0, Id(1))]
                    .iter()
                    .cloned()
                    .collect(),
            ),
            ..Default::default()
        };

        let mut genome_1 = genome_0.clone();

        genome_1
            .feed_forward
            .insert(Connection::new(Id(0), 1.0, Id(2)));
        genome_1
            .feed_forward
            .insert(Connection::new(Id(2), 2.0, Id(1)));

        println!("genome_0: {:?}", genome_0);
        println!("genome_1: {:?}", genome_1);

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 2.0, 0.0, 0.0);

        // factor 2 times 2 different genes over 3 total genes
        assert!((delta - 2.0 * 2.0 / 3.0).abs() < f64::EPSILON);
    }
}
