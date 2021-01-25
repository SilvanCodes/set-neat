use crate::individual::genes::Activation;
use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Default, Debug, Clone)]
pub struct Parameters {
    pub setup: Setup,
    pub reproduction: Reproduction,
    pub mutation: Mutation,
    pub activations: Activations,
    pub speciation: Speciation,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Setup {
    pub seed: u64,
    pub population_size: usize,
    pub input_dimension: usize,
    pub output_dimension: usize,
    pub connected_input_percent: f64,
    pub add_to_archive_chance: f64,
    pub novelty_nearest_neighbors: usize,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            seed: 42,
            population_size: 250,
            input_dimension: 1,
            output_dimension: 1,
            connected_input_percent: 1.0,
            add_to_archive_chance: 0.5,
            novelty_nearest_neighbors: 16,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Activations {
    pub output_nodes: Activation,
    pub hidden_nodes: Vec<Activation>,
}

impl Default for Activations {
    fn default() -> Self {
        Self {
            output_nodes: Activation::Tanh,
            hidden_nodes: vec![
                Activation::Linear,
                Activation::Sigmoid,
                Activation::Tanh,
                Activation::Gaussian,
                Activation::Step,
                Activation::Sine,
                Activation::Cosine,
                Activation::Inverse,
                Activation::Absolute,
                Activation::Relu,
            ],
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Mutation {
    pub new_node_chance: f64,
    pub new_connection_chance: f64,
    pub connection_is_recurrent_chance: f64,
    pub change_activation_function_chance: f64,
    pub weight_perturbation_std_dev: f64,
}

impl Default for Mutation {
    fn default() -> Self {
        Self {
            new_node_chance: 0.05,
            new_connection_chance: 0.1,
            connection_is_recurrent_chance: 0.0,
            change_activation_function_chance: 0.05,
            weight_perturbation_std_dev: 1.0,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Reproduction {
    pub survival_rate: f64,
    pub generations_until_stale: usize,
    pub elitism_species: usize,
    pub elitism_individuals: usize,
}

impl Default for Reproduction {
    fn default() -> Self {
        Self {
            survival_rate: 0.2,
            generations_until_stale: 15,
            elitism_species: 1,
            elitism_individuals: 0,
        }
    }
}

#[derive(Deserialize, Serialize, Default, Debug, Clone)]
pub struct Speciation {
    pub target_species_count: usize,
    pub factor_weights: f64,
    pub factor_genes: f64,
    pub factor_activations: f64,
}

impl Parameters {
    pub fn new(path: &str) -> Result<Self, ConfigError> {
        let mut s = Config::new();

        // Start off by merging in the "default" configuration file
        s.merge(File::with_name(path))?;

        // You can deserialize (and thus freeze) the entire configuration as
        s.try_into()
    }
}
