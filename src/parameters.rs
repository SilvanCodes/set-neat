use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};
use set_genome::Parameters as GenomeParameters;

#[derive(Deserialize, Serialize, Default, Debug, Clone)]
pub struct Parameters {
    pub neat: NeatParameters,
    pub genome: GenomeParameters,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NeatParameters {
    pub population_size: usize,
    #[serde(default)]
    pub add_to_archive_chance: f64,
    #[serde(default)]
    pub novelty_nearest_neighbors: usize,
    #[serde(default)]
    pub reproduction: Reproduction,
    #[serde(default)]
    pub speciation: Speciation,
}

impl Default for NeatParameters {
    fn default() -> Self {
        Self {
            population_size: 100,
            add_to_archive_chance: 0.0,
            novelty_nearest_neighbors: 0,
            reproduction: Default::default(),
            speciation: Default::default(),
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
            generations_until_stale: 10,
            elitism_species: 1,
            elitism_individuals: 0,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Speciation {
    pub target_species_count: usize,
    pub factor_weights: f64,
    pub factor_genes: f64,
    pub factor_activations: f64,
}

impl Default for Speciation {
    fn default() -> Self {
        Self {
            target_species_count: 10,
            factor_weights: 1.0,
            factor_genes: 1.0,
            factor_activations: 1.0,
        }
    }
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
