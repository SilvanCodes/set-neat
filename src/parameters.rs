use crate::genes::{Activation, WeightDistribution, WeightInitialization};
use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Parameters {
    pub setup: Setup,
    pub initialization: Initialization,
    pub reproduction: Reproduction,
    pub mutation: Mutation,
    pub compatability: Compatability,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Setup {
    pub population: usize,
    pub dimension: Dimension,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Initialization {
    #[serde(default)]
    pub output: Activation,
    #[serde(default)]
    pub activations: Vec<Activation>,
    pub connections: f64,
    #[serde(default)]
    pub weights: WeightInitialization,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Reproduction {
    pub offspring_from_crossover: f64,
    pub offspring_from_crossover_interspecies: f64,
    pub surviving: f64,
    pub stale_after: usize,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Dimension {
    pub input: usize,
    pub output: usize,
}

#[derive(Deserialize, Serialize, Default, Debug)]
pub struct Mutation {
    pub weight: f64,
    pub weight_random: f64,
    pub weight_perturbation: f64,
    #[serde(default)]
    pub weight_distribution: WeightDistribution,
    pub gene_node: f64,
    pub gene_connection: f64,
    pub recurrent: f64,
    #[serde(default)]
    pub activation_change: f64,
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

        assert_eq!(
            parameters
                .reproduction
                .offspring_from_crossover_interspecies,
            0.001
        )
    }
}
