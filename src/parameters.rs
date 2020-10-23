use crate::genes::{Activation, WeightDistribution, WeightInitialization};
use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Parameters {
    pub seed: u64,
    pub setup: Setup,
    pub initialization: Initialization,
    pub reproduction: Reproduction,
    pub mutation: Mutation,
    pub compatability: Compatability,
    pub novelty: Novelty,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Setup {
    pub population: usize,
    pub dimension: Dimension,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Initialization {
    pub output: Activation,
    pub activations: Vec<Activation>,
    pub connections: f64,
    pub weights: WeightInitialization,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Reproduction {
    pub surviving: f64,
    pub stale_after: usize,
    pub elitism_species: usize,
    pub elitism_individuals: usize,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Dimension {
    pub input: usize,
    pub output: usize,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Mutation {
    // pub weight: f64,
    // pub weight_random: f64,
    // pub weight_perturbation: f64,
    // pub weight_distribution: WeightDistribution,
    pub gene_node: f64,
    pub gene_connection: f64,
    pub recurrent: f64,
    pub activation_change: f64,
    pub weights: WeightsMutation,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct WeightsMutation {
    // minimum percent of weights mutated per individual
    pub percent_min: f64,
    // maximum percent of weights mutated per individual
    pub percent_max: f64,
    // chance to mutate to a new random sample of the distribution
    pub random: f64,
    // range of weight perutrbation, i.e. hard upper/lower cap for uniform distribution, stdandard deviation for normal distribution
    pub perturbation_range: f64,
    // type of distribution to sample from, normal or uniform
    pub distribution: WeightDistribution,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Compatability {
    pub target_species: usize,
    pub threshold: f64,
    pub threshold_delta: f64,
    pub factor_weights: f64,
    pub factor_genes: f64,
    pub factor_activations: f64,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Novelty {
    pub cap: f64,
    pub nearest_neighbors: usize,
    pub impatience: usize,
    pub demanded_increase_percent: f64,
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

#[cfg(test)]
mod tests {
    use super::Parameters;

    #[test]
    fn read_parameters() {
        let parameters = Parameters::new("src/Config.toml").unwrap();

        assert_eq!(parameters.reproduction.stale_after, 15)
    }
}
