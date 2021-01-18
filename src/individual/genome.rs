use crate::{
    genes::{connections::Connection, nodes::Node, Activation, Genes, IdGenerator},
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
}

#[cfg(test)]
mod tests {
    /* use super::Genome;
    use crate::{
        context::{rng::NeatRng, Context},
        genes::{
            connections::{Connection, FeedForward},
            nodes::{Hidden, Input, Node, Output},
            Genes,
        },
    };
    use crate::{
        genes::{Activation, Id, Weight},
        parameters::Parameters,
    };

    #[test]
    fn alter_activation() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;
        parameters.initialization.connections = 1.0;
        parameters.initialization.activations = vec![Activation::Absolute, Activation::Cosine];

        let mut genome = Genome::new(&mut context, &parameters);

        genome.init(&mut context, &parameters);

        genome.add_node(&mut context, &parameters);

        let old_activation = genome.hidden.iter().next().unwrap().1;

        genome.alter_activation(&mut context, &parameters);

        assert_ne!(genome.hidden.iter().next().unwrap().1, old_activation);
    }

    #[test]
    fn add_random_connection() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome = Genome::new(&mut context, &parameters);

        let result = genome.add_connection(&mut context, &parameters).is_ok();

        println!("{:?}", genome);

        assert_eq!(result, true);
        assert_eq!(genome.feed_forward.len(), 1);
    }

    #[test]
    fn dont_add_same_connection_twice() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;

        let mut genome = Genome::new(&mut context, &parameters);

        let result_0 = genome.add_connection(&mut context, &parameters).is_ok();
        if let Err(message) = genome.add_connection(&mut context, &parameters) {
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
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        parameters.initialization.activations = vec![Activation::Tanh];
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;
        parameters.initialization.connections = 1.0;

        let mut genome = Genome::new(&mut context, &parameters);

        genome.init(&mut context, &parameters);
        genome.add_node(&mut context, &parameters);

        println!("{:?}", genome);

        assert_eq!(genome.feed_forward.len(), 3);
    }

    #[test]
    fn crossover_same_fitness() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        parameters.initialization.activations = vec![Activation::Tanh];
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;
        parameters.initialization.connections = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init(&mut context, &parameters);

        let mut genome_1 = genome_0.clone();

        // mutate genome_0
        genome_0.add_node(&mut context, &parameters);

        // mutate genome_1
        genome_1.add_node(&mut context, &parameters);
        genome_1.add_node(&mut context, &parameters);

        println!("genome_0 {:?}", genome_0);
        println!("genome_1 {:?}", genome_1);

        // shorter genome is fitter genome
        let offspring = genome_0.cross_in(&genome_1, &mut context);

        println!("offspring {:?}", offspring);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 3);
    }

    #[test]
    fn crossover_different_fitness_by_fitter() {
        todo!("move to individual/mod.rs");

        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        parameters.initialization.activations = vec![Activation::Tanh];
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 2;
        parameters.setup.dimension.output = 1;
        parameters.initialization.connections = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init(&mut context, &parameters);

        let mut genome_1 = genome_0.clone();

        /* genome_1.fitness.raw = Raw::fitness(1.0);
        genome_1.fitness.shifted = genome_1.fitness.raw.shift(0.0);
        genome_1.fitness.normalized = genome_1.fitness.shifted.normalize(1.0); */

        // mutate genome_0
        genome_0.add_node(&mut context, &parameters);

        // mutate genome_1
        genome_1.add_node(&mut context, &parameters);
        genome_1.add_connection(&mut context, &parameters).unwrap();

        let offspring = genome_0.cross_in(&genome_1, &mut context);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 5);
    }

    #[test]
    fn crossover_different_fitness_by_equal_fittnes_different_len() {
        todo!("move to individual/mod.rs");

        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        parameters.initialization.activations = vec![Activation::Tanh];
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 2;
        parameters.setup.dimension.output = 1;
        parameters.initialization.connections = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init(&mut context, &parameters);

        let mut genome_1 = genome_0.clone();
        // mutate genome_0
        genome_0.add_node(&mut context, &parameters);

        // mutate genome_1
        genome_1.add_node(&mut context, &parameters);
        genome_1.add_connection(&mut context, &parameters).unwrap();

        let offspring = genome_0.cross_in(&genome_1, &mut context);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 4);
    }

    #[test]
    fn detect_no_cycle() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;
        parameters.initialization.connections = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init(&mut context, &parameters);

        let input = genome_0.inputs.iter().next().unwrap();
        let output = genome_0.outputs.iter().next().unwrap();

        let result = genome_0.would_form_cycle(&input, &output);

        assert!(!result);
    }

    #[test]
    fn detect_cycle() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_range = 1.0;
        parameters.initialization.activations = vec![Activation::Tanh];
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 1;
        parameters.setup.dimension.output = 1;
        parameters.initialization.connections = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init(&mut context, &parameters);

        // mutate genome_0
        genome_0.add_node(&mut context, &parameters);

        let input = genome_0.inputs.iter().next().unwrap();
        let output = genome_0.outputs.iter().next().unwrap();

        let result = genome_0.would_form_cycle(&output, &input);

        println!("{:?}", genome_0);

        assert!(result);
    }

    #[test]
    fn crossover_no_cycle() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weights.perturbation_std_dev = 1.0;
        let mut context = Context::new(&parameters);

        let mut rng = NeatRng::new(
            parameters.seed,
            parameters.mutation.weights.perturbation_std_dev,
        );

        // assumption:
        // crossover of equal fitness genomes should not produce cycles
        // prerequisits:
        // genomes with equal fitness (0.0 in this case)
        // "mirrored" structure as simplest example

        let mut genome_0 = Genome {
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
            hidden: Genes(
                vec![
                    Hidden(Node(Id(2), Activation::Tanh)),
                    Hidden(Node(Id(3), Activation::Tanh)),
                ]
                .iter()
                .cloned()
                .collect(),
            ),
            feed_forward: Genes(
                vec![
                    FeedForward(Connection(Id(0), Weight::default(), Id(2))),
                    FeedForward(Connection(Id(2), Weight::default(), Id(1))),
                    FeedForward(Connection(Id(0), Weight::default(), Id(3))),
                    FeedForward(Connection(Id(3), Weight::default(), Id(1))),
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
            .insert(FeedForward(Connection(Id(2), Weight::default(), Id(3))));

        // insert connection the other way in genome1
        genome_1
            .feed_forward
            .insert(FeedForward(Connection(Id(3), Weight::default(), Id(2))));

        let offspring = genome_0.cross_in(&genome_1, &mut rng.small);

        println!("offspring {:?}", offspring);

        for connection0 in offspring.feed_forward.iter() {
            for connection1 in offspring.feed_forward.iter() {
                println!(
                    "{:?}->{:?}, {:?}->{:?}",
                    connection0.input(),
                    connection0.output(),
                    connection1.input(),
                    connection1.output()
                );
                assert!(
                    !(connection0.input() == connection1.output()
                        && connection0.output() == connection1.input())
                )
            }
        }
    }

    /* #[test]
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

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 1.0, 0.4, 0.0);

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

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 0.0, 2.0, 0.0);

        assert!((delta - 2.0).abs() < f64::EPSILON);
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

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 2.0, 0.0, 0.0);

        // factor 2 times 2 different genes
        assert!((delta - 2.0 * 2.0).abs() < f64::EPSILON);
    } */ */
}
