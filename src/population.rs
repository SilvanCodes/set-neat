use set_genome::{Genome, GenomeContext};
use std::{mem, time::Instant};

use crate::{
    individual::{
        behavior::{Behavior, Behaviors},
        // genes::IdGenerator,
        // genome::Genome,
        scores::Score,
        Individual,
    },
    parameters::{NeatParameters, Reproduction, Speciation},
    species::Species,
    statistics::PopulationStatistics,
    Parameters, Progress,
};

pub struct Population {
    pub individuals: Vec<Individual>,
    archive: Vec<Individual>,
    species: Vec<Species>,
    genome_context: GenomeContext,
    // rng: GenomeRng,
    // id_gen: IdGenerator,
    compatability_threshold: f64,
    parameters: NeatParameters,
    statistics: PopulationStatistics,
}

impl Population {
    pub fn new(parameters: Parameters) -> Self {
        let mut genome_context = GenomeContext::new(parameters.genome);

        // create id book-keeping
        // let mut id_gen = IdGenerator::default();

        // generate genome with initial ids for structure
        let initial_individual = Individual::from_genome(genome_context.uninitialized_genome());

        // create randomn source
        // let mut rng = GenomeRng::new(
        //     parameters.setup.seed,
        //     parameters.mutation.weight_perturbation_std_dev,
        // );

        let mut individuals = Vec::new();

        // generate initial, mutated individuals
        for _ in 0..parameters.neat.population_size {
            let mut other_genome = initial_individual.clone();
            other_genome.init_with_context(&mut genome_context);
            other_genome.mutate_with_context(&mut genome_context);
            individuals.push(other_genome);
        }

        let mut population = Population {
            species: Vec::new(),
            parameters: parameters.neat,
            archive: Vec::new(),
            individuals,
            genome_context,
            // id_gen,
            // rng,
            statistics: Default::default(),
            compatability_threshold: f64::NAN,
        };

        population.init_threshold();

        population
    }

    pub fn individuals(&self) -> &Vec<Individual> {
        &self.individuals
    }

    fn place_genome_into_species(&mut self, individual: Individual) {
        let Speciation {
            factor_genes,
            factor_weights,
            factor_activations,
            ..
        } = self.parameters.speciation;

        let weight_cap = self.genome_context.parameters.structure.weight_cap;
        let compatability_threshold = self.compatability_threshold;
        let species_statistics = &mut self.statistics.species;

        // place into matching species
        if let Some(species) = self.species.iter_mut().find(|species| {
            let (compatability, gene_diff, weight_diff, activation_diff) =
                Genome::compatability_distance(
                    &individual,
                    &species.representative,
                    factor_genes,
                    factor_weights,
                    factor_activations,
                    weight_cap,
                );

            if compatability < compatability_threshold {
                species_statistics.raw_genes_diff.push(gene_diff);
                species_statistics.raw_weights_diff.push(weight_diff);
                species_statistics
                    .raw_activations_diff
                    .push(activation_diff);
            }

            compatability < compatability_threshold
        }) {
            species.members.push(individual);
        } else {
            self.species.push(Species::new(individual));
        }
    }

    fn remove_stale_species(&mut self) {
        // sort species by fitness in descending order
        self.species
            .sort_by(|species_0, species_1| species_1.score.partial_cmp(&species_0.score).unwrap());

        // clear stale species
        let len_before_threshold = self.species.len();

        let Reproduction {
            elitism_species,
            generations_until_stale,
            ..
        } = self.parameters.reproduction;

        self.species = self
            .species
            .iter()
            // keep at least configured amount of species
            .take(elitism_species)
            .chain(
                self.species
                    .iter()
                    .skip(elitism_species)
                    .filter(|species| species.stale < generations_until_stale),
            )
            .cloned()
            .collect();

        self.statistics.num_species_stale = len_before_threshold - self.species.len();
    }

    pub fn speciate(&mut self) {
        let now = Instant::now();

        // clear population and sort into species
        for genome in mem::replace(&mut self.individuals, Vec::new()) {
            self.place_genome_into_species(genome);
        }

        self.statistics.species.sizes = self.species.iter().map(|s| s.members.len()).collect();

        // sort members of species and compute species score
        for species in &mut self.species {
            species.adjust(self.parameters.reproduction.survival_rate);
        }

        // check on final species count
        // do immediatly after speciation ?
        self.adjust_threshold();

        self.remove_stale_species();

        // collect statistics
        self.statistics.species.avg_genes_diff =
            self.statistics.species.raw_genes_diff.iter().sum::<f64>()
                / self.statistics.species.raw_genes_diff.len() as f64;
        self.statistics.species.avg_weights_diff =
            self.statistics.species.raw_weights_diff.iter().sum::<f64>()
                / self.statistics.species.raw_weights_diff.len() as f64;
        self.statistics.species.avg_activations_diff = self
            .statistics
            .species
            .raw_activations_diff
            .iter()
            .sum::<f64>()
            / self.statistics.species.raw_activations_diff.len() as f64;

        self.statistics.species.raw_genes_diff = Vec::new();
        self.statistics.species.raw_weights_diff = Vec::new();
        self.statistics.species.raw_activations_diff = Vec::new();

        self.statistics.compatability_threshold = self.compatability_threshold;
        self.statistics.milliseconds_elapsed_speciation = now.elapsed().as_millis();
    }

    pub fn init_threshold(&mut self) {
        let mut threshold = Vec::new();

        let average_species_size = (self.parameters.population_size as f64
            / self.parameters.speciation.target_species_count as f64)
            .floor() as usize;

        for individual_0 in &self.individuals {
            let mut distances = Vec::new();

            for individual_1 in &self.individuals {
                distances.push(
                    Genome::compatability_distance(
                        individual_0,
                        individual_1,
                        self.parameters.speciation.factor_genes,
                        self.parameters.speciation.factor_weights,
                        self.parameters.speciation.factor_activations,
                        self.genome_context.parameters.structure.weight_cap,
                    )
                    .0,
                );
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            threshold.push(distances[average_species_size]);
        }

        self.compatability_threshold = threshold.iter().sum::<f64>() / threshold.len() as f64;
    }

    fn adjust_threshold(&mut self) {
        self.compatability_threshold *= (self.species.len() as f64
            / self.parameters.speciation.target_species_count as f64)
            .sqrt();
    }

    pub fn reproduce(&mut self) {
        let now = Instant::now();

        // expresses how much offspring one point of fitness is worth
        let offspring_ratio;

        let total_fitness = self
            .species
            .iter()
            .fold(0.0, |sum, species| sum + species.score);

        // if we can not differentiate species
        if total_fitness == 0.0 {
            // remove species with no members
            self.species.retain(|species| !species.members.is_empty());
            // give everyone equal offspring
            offspring_ratio = self.parameters.population_size as f64 / self.species.len() as f64;
        }
        // if we can differentiate species
        else {
            // give offspring in proportion to archived score
            offspring_ratio = self.parameters.population_size as f64 / total_fitness;
            // remove species that do not qualify for offspring
            self.species
                .retain(|species| species.score * offspring_ratio >= 1.0);
        }

        // iterate species, only ones that qualify for reproduction are left
        for species in &self.species {
            // calculate offspring count for species
            let offspring_count = if total_fitness == 0.0 {
                offspring_ratio
            } else {
                species.score * offspring_ratio
            }
            .round() as usize;

            self.individuals.extend(species.reproduce(
                &mut self.genome_context,
                &self.parameters.reproduction,
                offspring_count,
            ));
        }

        // clear species members
        for species in &mut self.species {
            species.prepare_next_generation();
        }

        // collect statistics
        self.statistics.num_generation += 1;
        self.statistics.num_offpring = self.individuals.len();
        self.statistics.num_species = self.species.len();
        self.statistics.milliseconds_elapsed_reproducing = now.elapsed().as_millis();
    }

    pub fn next_generation(&mut self, progress: &[Progress]) -> PopulationStatistics {
        self.assign_fitness(progress);
        self.assign_behavior(progress);

        self.determine_top_performer();

        self.speciate();
        self.reproduce();

        self.statistics.clone()
    }

    fn determine_top_performer(&mut self) {
        self.statistics.top_performer = self
            .individuals
            .iter()
            .max_by(|a, b| a.score().partial_cmp(&b.score()).unwrap())
            .unwrap()
            .clone();
    }

    fn assign_behavior(&mut self, progress: &[Progress]) {
        let behaviors: Vec<(usize, &Behavior)> = progress
            .iter()
            // enumerate here in case some option is None
            .enumerate()
            .flat_map(|(index, progress)| progress.behavior().map(|raw| (index, raw)))
            .collect();

        if behaviors.is_empty() {
            return;
        }

        for (index, behavior) in behaviors {
            self.individuals[index].behavior = behavior.clone();
        }

        // calculate novelty based on previously assigned behavior
        self.calculate_novelty();
    }

    fn assign_fitness(&mut self, progress: &[Progress]) {
        let fitnesses: Vec<(usize, f64)> = progress
            .iter()
            // enumerate here in case some option is None
            .enumerate()
            .flat_map(|(index, progress)| progress.raw_fitness().map(|raw| (index, raw)))
            .collect();

        if fitnesses.is_empty() {
            return;
        }

        let mut raw_minimum = f64::INFINITY;
        let mut raw_sum = 0.0;
        let mut raw_maximum = f64::NEG_INFINITY;

        // analyse raw fitness values
        for &(_, raw_fitness) in &fitnesses {
            if raw_fitness > raw_maximum {
                raw_maximum = raw_fitness;
            }
            if raw_fitness < raw_minimum {
                raw_minimum = raw_fitness;
            }
            raw_sum += raw_fitness;
        }

        let raw_average = raw_sum / fitnesses.len() as f64;

        let baseline = raw_minimum;

        let shifted_minimum = raw_minimum - baseline;
        let shifted_average = raw_average - baseline;
        let shifted_maximum = raw_maximum - baseline;

        let with = shifted_maximum.max(1.0);

        let normalized_minimum = shifted_minimum / with;
        let normalized_average = shifted_average / with;
        let normalized_maximum = shifted_maximum / with;

        // shift and normalize fitness
        for (index, raw_fitness) in fitnesses {
            self.individuals[index].fitness = Score::new(raw_fitness, baseline, with);
        }

        self.statistics.fitness.raw_maximum = raw_maximum;
        self.statistics.fitness.raw_minimum = raw_minimum;
        self.statistics.fitness.raw_average = raw_average;

        self.statistics.fitness.shifted_maximum = shifted_maximum;
        self.statistics.fitness.shifted_minimum = shifted_minimum;
        self.statistics.fitness.shifted_average = shifted_average;

        self.statistics.fitness.normalized_maximum = normalized_maximum;
        self.statistics.fitness.normalized_minimum = normalized_minimum;
        self.statistics.fitness.normalized_average = normalized_average;
    }

    fn calculate_novelty(&mut self) {
        // TODO: handle non-uniform behavior vectors
        let behaviors: Behaviors = self
            .individuals
            .iter()
            .map(|individual| &individual.behavior)
            .chain(
                self.archive
                    .iter()
                    .map(|archived_individual| &archived_individual.behavior),
            )
            .collect::<Vec<&Behavior>>()
            .into();

        let behavior_count = behaviors.len() as f64;

        let raw_novelties = behaviors.compute_novelty(self.parameters.novelty_nearest_neighbors);

        // add most novel current individual to archive
        if self
            .genome_context
            .rng
            .gamble(self.parameters.add_to_archive_chance)
        {
            let current_generation_most_novel = raw_novelties
                .iter()
                .enumerate()
                .take(self.individuals.len())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("could not compare floats"))
                .map(|(index, _)| index)
                .expect("failed finding most novel");

            self.archive
                .push(self.individuals[current_generation_most_novel].clone());
        }

        let mut raw_minimum = f64::INFINITY;
        let mut raw_sum = 0.0;
        let mut raw_maximum = f64::NEG_INFINITY;

        // analyse raw novelty values
        for &novelty in &raw_novelties {
            if novelty > raw_maximum {
                raw_maximum = novelty;
            }
            if novelty < raw_minimum {
                raw_minimum = novelty;
            }
            raw_sum += novelty;
        }

        let raw_average = raw_sum / behavior_count;

        let baseline = raw_minimum;

        let shifted_minimum = raw_minimum - baseline;
        let shifted_average = raw_average - baseline;
        let shifted_maximum = raw_maximum - baseline;

        let with = shifted_maximum.max(1.0);

        let normalized_minimum = shifted_minimum / with;
        let normalized_average = shifted_average / with;
        let normalized_maximum = shifted_maximum / with;

        // assign computed novelty to current and archived individuals
        for (&novelty, individual) in raw_novelties
            .iter()
            .zip(self.individuals.iter_mut().chain(self.archive.iter_mut()))
        {
            individual.novelty = Score::new(novelty, baseline, with);
        }

        self.statistics.novelty.raw_maximum = raw_maximum;
        self.statistics.novelty.raw_minimum = raw_minimum;
        self.statistics.novelty.raw_average = raw_average;

        self.statistics.novelty.shifted_maximum = shifted_maximum;
        self.statistics.novelty.shifted_minimum = shifted_minimum;
        self.statistics.novelty.shifted_average = shifted_average;

        self.statistics.novelty.normalized_maximum = normalized_maximum;
        self.statistics.novelty.normalized_minimum = normalized_minimum;
        self.statistics.novelty.normalized_average = normalized_average;
    }
}
