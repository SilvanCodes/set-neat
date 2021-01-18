use std::{mem, time::Instant};

use crate::{
    genes::IdGenerator,
    individual::{
        behavior::{Behavior, Behaviors},
        scores::Score,
        Individual,
    },
    parameters::{Reproduction, Speciation},
    rng::NeatRng,
    species::Species,
    statistics::PopulationStatistics,
    Parameters, Progress,
};

pub struct Population {
    pub individuals: Vec<Individual>,
    archive: Vec<Individual>,
    species: Vec<Species>,
    rng: NeatRng,
    id_gen: IdGenerator,
    parameters: Parameters,
    statistics: PopulationStatistics,
}

// uses setup/compatability/reproduction

impl Population {
    pub fn new(parameters: Parameters) -> Self {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();

        // generate genome with initial ids for structure
        let initial_individual = Individual::initial(&mut id_gen, &parameters);

        // create randomn source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let mut individuals = Vec::new();

        // generate initial, mutated individuals
        for _ in 0..parameters.setup.population_size {
            let mut other_genome = initial_individual.clone();
            other_genome.init(&mut rng, &parameters);
            other_genome.mutate(&mut rng, &mut id_gen, &parameters);
            individuals.push(other_genome);
        }

        let mut population = Population {
            species: Vec::new(),
            parameters,
            archive: Vec::new(),
            individuals,
            id_gen,
            rng,
            statistics: Default::default(),
        };

        population.init_threshold();

        population
    }

    pub fn individuals(&self) -> &Vec<Individual> {
        &self.individuals
    }

    fn place_genome_into_species(&mut self, individual: Individual) {
        let context = &self.parameters.speciation;
        // place into matching species
        if let Some(species) = self.species.iter_mut().find(|species| {
            Individual::compatability_distance(
                &individual,
                &species.representative,
                context.factor_genes,
                context.factor_weights,
                context.factor_activations,
            ) < context.compatability_threshold
        }) {
            species.members.push(individual);
        } else {
            self.species.push(Species::new(individual));
        }

        /* if let Some(species_index) = self.find_best_fitting_species(&genome) {
            self.species[species_index].members.push(genome);
        }
        // or open new species
        else {
            self.species.push(Species::new(genome));
        } */
    }

    /* fn find_best_fitting_species(&self, genome: &Genome) -> Option<usize> {
        // return when no species exist
        if self.species.is_empty() {
            return None;
        };

        self.species
            .iter()
            // map to compatability distance
            .map(|species| {
                Genome::compatability_distance(
                    genome,
                    &species.representative,
                    self.context.factor_genes,
                    self.context.factor_weights,
                    self.context.factor_activations,
                )
            })
            .enumerate()
            // select minimum distance
            .min_by(|(_, distance_0), (_, distance_1)| distance_0.partial_cmp(distance_1).unwrap())
            // check against compatability threshold
            .filter(|(_, distance)| distance < &self.context.compatability_threshold)
            // return species index
            .map(|(index, _)| index)
    } */

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
        // MAYBE: update representative by most similar descendant ?

        let now = Instant::now();

        // clear population and sort into species
        for genome in mem::replace(&mut self.individuals, Vec::new()) {
            self.place_genome_into_species(genome);
        }

        // sort members of species and compute species score
        for species in &mut self.species {
            species.adjust(self.parameters.reproduction.survival_rate);
        }

        // check on final species count
        // do immediatly after speciation ?
        self.adjust_threshold();

        self.remove_stale_species();

        // collect statistics
        self.statistics.compatability_threshold =
            self.parameters.speciation.compatability_threshold;
        self.statistics.milliseconds_elapsed_speciation = now.elapsed().as_millis();
    }

    pub fn init_threshold(&mut self) {
        let mut threshold = Vec::new();

        let approximate_species_size = (self.parameters.setup.population_size as f64
            / self.parameters.speciation.target_species_count as f64)
            .floor() as usize;

        for individual_0 in &self.individuals {
            let mut distances = Vec::new();

            for individual_1 in &self.individuals {
                distances.push(Individual::compatability_distance(
                    individual_0,
                    individual_1,
                    self.parameters.speciation.factor_genes,
                    self.parameters.speciation.factor_weights,
                    self.parameters.speciation.factor_activations,
                ));
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            threshold.push(distances[approximate_species_size]);
        }

        // dbg!(&threshold);

        self.parameters.speciation.compatability_threshold =
            threshold.iter().sum::<f64>() / threshold.len() as f64 * 1.0;
    }

    fn adjust_threshold(&mut self) {
        let Speciation {
            target_species_count,
            compatability_threshold_delta,
            ..
        } = self.parameters.speciation;

        let species_count = target_species_count;
        // check if num species near target species
        match self.species.len() {
            length if length > species_count => {
                self.parameters.speciation.compatability_threshold +=
                    compatability_threshold_delta * (length as f64 / species_count as f64);
                //.powi(2);
            }
            length if length < species_count => {
                self.parameters.speciation.compatability_threshold -=
                    compatability_threshold_delta * (species_count as f64 / length as f64);
                //.powi(2);
            }
            _ => {}
        }

        // use threshold_delta as lower cap of compatability_threshold
        self.parameters.speciation.compatability_threshold =
            compatability_threshold_delta.max(self.parameters.speciation.compatability_threshold);
    }

    pub fn reproduce(&mut self) {
        let now = Instant::now();

        // expresses how much offspring one point of fitness is worth
        let offspring_ratio;

        let total_fitness = self
            .species
            .iter()
            .fold(0.0, |sum, species| sum + species.score);

        // dbg!(self.species.len());

        // if we can not differentiate species
        if total_fitness == 0.0 {
            // remove species with no members
            self.species.retain(|species| !species.members.is_empty());
            // give everyone equal offspring
            offspring_ratio =
                self.parameters.setup.population_size as f64 / self.species.len() as f64;
        }
        // if we can differentiate species
        else {
            // give offspring in proportion to archived score
            offspring_ratio = self.parameters.setup.population_size as f64 / total_fitness;
            // remove species that do not qualify for offspring
            self.species
                .retain(|species| species.score * offspring_ratio >= 1.0);
        }

        // dbg!(self.species.len());

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
                &mut self.rng,
                &mut self.id_gen,
                &self.parameters,
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

        self.speciate();
        self.reproduce();

        self.collect_statistics()
    }

    fn collect_statistics(&self) -> PopulationStatistics {
        self.statistics.clone()
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

        let raw_novelties =
            behaviors.compute_novelty(self.parameters.setup.novelty_nearest_neighbors);

        // add most novel current individual to archive
        if self.rng.gamble(self.parameters.setup.add_to_archive_chance) {
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
