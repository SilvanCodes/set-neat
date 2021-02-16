use serde::Serialize;

use crate::individual::Individual;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Statistics {
    pub population: PopulationStatistics,
    pub milliseconds_elapsed_evaluation: u128,
    pub time_stamp: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FitnessStatisitcs {
    pub raw_maximum: f64,
    pub raw_minimum: f64,
    pub raw_average: f64,
    pub raw_std_dev: f64,
    pub shifted_maximum: f64,
    pub shifted_minimum: f64,
    pub shifted_average: f64,
    pub normalized_maximum: f64,
    pub normalized_minimum: f64,
    pub normalized_average: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct NoveltyStatisitcs {
    pub raw_maximum: f64,
    pub raw_minimum: f64,
    pub raw_average: f64,
    pub shifted_maximum: f64,
    pub shifted_minimum: f64,
    pub shifted_average: f64,
    pub normalized_maximum: f64,
    pub normalized_minimum: f64,
    pub normalized_average: f64,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct PopulationStatistics {
    pub milliseconds_elapsed_reproducing: u128,
    pub top_performer: Individual,
    pub age_maximum: usize,
    pub age_average: f64,
    pub num_species_stale: usize,
    pub num_generation: usize,
    pub num_offpring: usize,
    pub num_species: usize,
    pub fitness: FitnessStatisitcs,
    pub novelty: NoveltyStatisitcs,
    pub species: SpeciesStatistics,
    pub compatability_threshold: f64,
    pub milliseconds_elapsed_speciation: u128,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SpeciesStatistics {
    pub raw_genes_diff: Vec<f64>,
    pub raw_weights_diff: Vec<f64>,
    pub raw_activations_diff: Vec<f64>,
    pub avg_genes_diff: f64,
    pub avg_weights_diff: f64,
    pub avg_activations_diff: f64,
    pub sizes: Vec<usize>,
}
