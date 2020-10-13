use crate::{
    activations::Activation,
    genes::{
        connections::{Connection, FeedForward, Recurrent},
        nodes::{Hidden, Input, Node, Output},
        Genes,
    },
    scores::FitnessScore,
    scores::NoveltyScore,
};
use crate::{genes::Weight, scores::ScoreValue};
use crate::{Context, Parameters};
use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub inputs: Genes<Input<Node>>,
    pub hidden: Genes<Hidden<Node>>,
    pub outputs: Genes<Output<Node>>,
    pub feed_forward: Genes<FeedForward<Connection>>,
    pub recurrent: Genes<Recurrent<Connection>>,
    pub fitness: FitnessScore,
    pub novelty: NoveltyScore,
}

impl Genome {
    pub fn new(context: &mut Context, parameters: &Parameters) -> Self {
        Genome {
            inputs: (0..parameters.setup.dimension.input)
                .map(|_| Input(Node(context.id_gen.next_id(), Activation::Linear)))
                .collect(),
            outputs: (0..parameters.setup.dimension.output)
                .map(|_| {
                    Output(Node(
                        context.id_gen.next_id(),
                        parameters.initialization.output,
                    ))
                })
                .collect(),
            ..Default::default()
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.inputs
            .iterate_unwrapped()
            .chain(self.hidden.iterate_unwrapped())
            .chain(self.outputs.iterate_unwrapped())
    }

    pub fn init(&mut self, context: &mut Context, parameters: &Parameters) {
        for input in self
            .inputs
            .iterate_with_random_offset(&mut context.small_rng)
            // connect configured percent of inputs to outputs, ceil for at least one
            .take(
                (parameters.initialization.connections * parameters.setup.dimension.input as f64)
                    .ceil() as usize,
            )
        {
            // connect to every output
            for output in self.outputs.iter() {
                assert!(self.feed_forward.insert(FeedForward(Connection(
                    input.id(),
                    Weight::default(),
                    output.id()
                ))));
            }
        }
    }

    pub fn len(&self) -> usize {
        self.feed_forward.len() + self.recurrent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.feed_forward.is_empty() && self.recurrent.is_empty()
    }

    // score is combination of fitness & novelty
    pub fn score(&self, context: &Context) -> f64 {
        // self.fitness.normalized.value()
        self.fitness.normalized.value() * (1.0 - context.novelty_ratio)
            + self.novelty.normalized.value() * context.novelty_ratio
    }

    // self is fitter if it has higher score or in case of equal score has fewer genes, i.e. less complexity
    pub fn is_fitter_than(&self, other: &Self, context: &Context) -> bool {
        let score_self = self.score(context);
        let score_other = other.score(context);

        score_self > score_other
            || ((score_self - score_other).abs() < f64::EPSILON && self.len() < other.len())
    }

    pub fn crossover(&self, other: &Self, context: &mut Context) -> Self {
        let (fitter, weaker) = if self.is_fitter_than(other, context) {
            (self, other)
        } else {
            (other, self)
        };

        // gamble for matching feedforward genes
        let mut feed_forward: Genes<FeedForward<Connection>> = self
            .feed_forward
            .iterate_matches(&other.feed_forward)
            .map(|(gene_self, gene_other)| {
                if context.gamble(0.5) {
                    gene_self.clone()
                } else {
                    gene_other.clone()
                }
            })
            .collect();

        // gamble for matching recurrent genes
        let mut recurrent: Genes<Recurrent<Connection>> = self
            .recurrent
            .iterate_matches(&other.recurrent)
            .map(|(gene_self, gene_other)| {
                if context.gamble(0.5) {
                    gene_self.clone()
                } else {
                    gene_other.clone()
                }
            })
            .collect();

        // add different feedforward genes
        feed_forward.extend(
            fitter
                .feed_forward
                .difference(&weaker.feed_forward)
                .cloned(),
        );

        // add different recurrent genes
        recurrent.extend(fitter.recurrent.difference(&weaker.recurrent).cloned());

        Genome {
            feed_forward,
            recurrent,
            ..fitter.clone()
        }
    }

    pub fn mutate(&mut self, context: &mut Context, parameters: &Parameters) {
        // mutate weigths
        // if context.gamble(parameters.mutation.weight) {
        self.change_weights(context, parameters);
        // }

        // mutate connection gene
        if context.gamble(parameters.mutation.gene_connection) {
            self.add_connection(context, parameters).unwrap_or_default();
        }

        // mutate node gene
        if context.gamble(parameters.mutation.gene_node) {
            self.add_node(context, parameters);
        }

        // change some activation
        if context.gamble(parameters.mutation.activation_change) {
            self.alter_activation(context, parameters);
        }
    }

    pub fn change_weights(&mut self, context: &mut Context, parameters: &Parameters) {
        // generate percent of changing connections
        let change_percent = context.small_rng.gen::<f64>();
        let num_feed_forward = (change_percent * self.feed_forward.len() as f64).floor() as usize;
        let num_recurrent = (change_percent * self.recurrent.len() as f64).floor() as usize;

        self.feed_forward = self
            .feed_forward
            .drain()
            .enumerate()
            .map(|(num, mut connection)| {
                if num < num_feed_forward {
                    if context.gamble(parameters.mutation.weight_random) {
                        connection.1.random(context);
                    } else {
                        connection.1.perturbate(context);
                    }
                }

                connection
            })
            .collect();

        self.recurrent = self
            .recurrent
            .drain()
            .enumerate()
            .map(|(num, mut connection)| {
                if num < num_recurrent {
                    if context.gamble(parameters.mutation.weight_random) {
                        connection.1.random(context);
                    } else {
                        connection.1.perturbate(context);
                    }
                }

                connection
            })
            .collect();
    }

    pub fn alter_activation(&mut self, context: &mut Context, parameters: &Parameters) {
        if let Some(node) = self
            .hidden
            .iter()
            .nth((context.small_rng.gen::<f64>() * self.hidden.len() as f64).floor() as usize)
        {
            let updated = Hidden(Node(
                node.id(),
                parameters
                    .initialization
                    .activations
                    .iter()
                    .filter(|&&activation| activation != node.1)
                    .choose(&mut context.small_rng)
                    .cloned()
                    .unwrap_or(node.1),
            ));

            self.hidden.replace(updated);
        }
    }

    pub fn add_node(&mut self, context: &mut Context, parameters: &Parameters) {
        // select an connection gene and split
        let mut random_connection = self
            .feed_forward
            .iter()
            .choose(&mut context.small_rng)
            .cloned()
            .unwrap();

        let id = context
            .id_gen
            .cached_id_iter(random_connection.id())
            .find(|&id| {
                self.hidden
                    .get(&Hidden(Node(id, Activation::Linear)))
                    .is_none()
            })
            .unwrap();

        // construct new node gene
        let new_node = Hidden(Node(
            id,
            parameters
                .initialization
                .activations
                .choose(&mut context.small_rng)
                .cloned()
                .unwrap(),
        ));

        // insert new connection pointing to new node
        assert!(self.feed_forward.insert(FeedForward(Connection(
            random_connection.input(),
            Weight(1.0),
            new_node.id(),
        ))));
        // insert new connection pointing from new node
        assert!(self.feed_forward.insert(FeedForward(Connection(
            new_node.id(),
            random_connection.1,
            random_connection.output(),
        ))));
        // insert new node into genome
        assert!(self.hidden.insert(new_node));

        // update weight to zero to 'deactivate' connnection
        random_connection.1 = Weight(0.0);
        self.feed_forward.replace(random_connection);
    }

    pub fn add_connection(
        &mut self,
        context: &mut Context,
        parameters: &Parameters,
    ) -> Result<(), &'static str> {
        let is_recurrent = context.gamble(parameters.mutation.recurrent);

        let start_node_iterator = self
            .inputs
            .iterate_unwrapped()
            .chain(self.hidden.iterate_unwrapped());

        let end_node_iterator = self
            .hidden
            .iterate_unwrapped()
            .chain(self.outputs.iterate_unwrapped());

        for start_node in start_node_iterator
            // make iterator wrap
            .cycle()
            // randomly offset into the iterator to choose any node
            .skip(
                (context.small_rng.gen::<f64>() * (self.inputs.len() + self.hidden.len()) as f64)
                    .floor() as usize,
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
                    assert!(self.recurrent.insert(Recurrent(Connection(
                        start_node.id(),
                        parameters.initialization.weights.init(),
                        end_node.id(),
                    ))));
                } else {
                    // add new feed-forward connection
                    assert!(self.feed_forward.insert(FeedForward(Connection(
                        start_node.id(),
                        parameters.initialization.weights.init(),
                        end_node.id(),
                    ))));
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
            self.recurrent.contains(&Recurrent(Connection(
                start_node.id(),
                Weight::default(),
                end_node.id(),
            )))
        } else {
            self.feed_forward.contains(&FeedForward(Connection(
                start_node.id(),
                Weight::default(),
                end_node.id(),
            )))
        }
    }

    // can only operate when no cycles present yet, which is assumed
    fn would_form_cycle(&self, start_node: &Node, end_node: &Node) -> bool {
        // needs to detect if there is a path from end to start
        let mut possible_paths: Vec<&FeedForward<Connection>> = self
            .feed_forward
            .iter()
            .filter(|connection| connection.input() == end_node.id())
            .collect();
        let mut next_possible_path = Vec::new();

        while !possible_paths.is_empty() {
            for path in possible_paths {
                // we have a cycle if path leads to start_node_gene
                if path.output() == start_node.id() {
                    return true;
                }
                // collect further paths
                else {
                    next_possible_path.extend(
                        self.feed_forward
                            .iter()
                            .filter(|connection| connection.input() == path.output()),
                    );
                }
            }
            possible_paths = next_possible_path;
            next_possible_path = Vec::new();
        }
        false
    }

    pub fn compatability_distance(
        genome_0: &Genome,
        genome_1: &Genome,
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

        // percent of different genes, considering unique genes
        let difference = factor_genes * different_genes_count_total / (matching_genes_count_total + different_genes_count_total)
        // average of weight differences
        + factor_weights * if matching_genes_count_total > 0.0 { weight_difference_total / matching_genes_count_total } else { 0.0 }
        // average of activation differences
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
    use crate::genes::{Activation, Id, Weight};
    use crate::Parameters;
    use crate::{
        context::Context,
        genes::{
            connections::{Connection, FeedForward},
            nodes::{Hidden, Input, Node, Output},
            Genes,
        },
        scores::Raw,
    };

    #[test]
    fn alter_activation() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
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
        parameters.mutation.weight_perturbation = 1.0;
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
        parameters.mutation.weight_perturbation = 1.0;
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
        parameters.mutation.weight_perturbation = 1.0;
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
        parameters.mutation.weight_perturbation = 1.0;
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
        let offspring = genome_0.crossover(&genome_1, &mut context);

        println!("offspring {:?}", offspring);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 3);
    }

    #[test]
    fn crossover_different_fitness_by_fitter() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        parameters.initialization.activations = vec![Activation::Tanh];
        let mut context = Context::new(&parameters);

        parameters.setup.dimension.input = 2;
        parameters.setup.dimension.output = 1;
        parameters.initialization.connections = 1.0;

        let mut genome_0 = Genome::new(&mut context, &parameters);

        genome_0.init(&mut context, &parameters);

        let mut genome_1 = genome_0.clone();

        genome_1.fitness.raw = Raw::fitness(1.0);
        genome_1.fitness.shifted = genome_1.fitness.raw.shift(0.0);
        genome_1.fitness.normalized = genome_1.fitness.shifted.normalize(1.0);

        // mutate genome_0
        genome_0.add_node(&mut context, &parameters);

        // mutate genome_1
        genome_1.add_node(&mut context, &parameters);
        genome_1.add_connection(&mut context, &parameters).unwrap();

        let offspring = genome_0.crossover(&genome_1, &mut context);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 5);
    }

    #[test]
    fn crossover_different_fitness_by_equal_fittnes_different_len() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
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

        let offspring = genome_0.crossover(&genome_1, &mut context);

        assert_eq!(offspring.hidden.len(), 1);
        assert_eq!(offspring.feed_forward.len(), 4);
    }

    #[test]
    fn detect_no_cycle() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
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
        parameters.mutation.weight_perturbation = 1.0;
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
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

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

        let offspring = genome_0.crossover(&genome_1, &mut context);

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

        let delta = Genome::compatability_distance(&genome_0, &genome_1, 2.0, 0.0, 0.0);

        // factor 2 times 2 different genes
        assert!(delta - 2.0 * 2.0 < f64::EPSILON);
    }
}
