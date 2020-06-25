use crate::genes::weights::WeightDistribution;
use config::{ConfigError, Config, File};
use serde::Deserialize;
use crate::genes::activations::Activation;

#[derive(Deserialize, Default)]
pub struct Parameters {
    pub setup: Setup,
    pub reproduction: Reproduction,
    pub mutation: Mutation,
    pub compatability: Compatability,
}

#[derive(Deserialize, Default)]
pub struct Setup {
    pub population: usize,
    pub dimension: Dimension,
    pub activation: Activation
}

#[derive(Deserialize, Default)]
pub struct Reproduction {
    pub offspring_from_crossover: f64,
    pub offspring_from_crossover_interspecies: f64,
    pub surviving: f64,
    pub stale_after: usize
}

#[derive(Deserialize, Default)]
pub struct Dimension {
    pub input: usize,
    pub output: usize,
}

#[derive(Deserialize, Default)]
pub struct Mutation {
    pub weight: f64,
    pub weight_random: f64,
    pub weight_perturbation: f64,
    pub weight_distribution: WeightDistribution,
    pub gene_node: f64,
    pub gene_connection: f64,
}

#[derive(Deserialize, Default)]
pub struct Compatability {
    pub target_species: usize,
    pub threshold: f64,
    pub threshold_delta: f64,
    pub factor_weights: f64,
    pub factor_genes: f64,
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

        assert_eq!(parameters.reproduction.offspring_from_crossover_interspecies, 0.001)
    }
}