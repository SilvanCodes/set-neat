use std::{mem, time::Instant};

use rand::prelude::SliceRandom;

use crate::{parameters::Compatability, species::Species, Context, Genome, Parameters};

pub struct Population {
    pub individuals: Vec<Genome>,
    species: Vec<Species>,
    context: PopulationContext,
}

struct PopulationContext {
    compatability_threshold: f64,
    threshold_delta: f64,
    compatability_distance: Box<dyn Fn(&Genome, &Genome) -> f64>,
}

impl Population {
    pub fn new(context: &mut Context, parameters: &Parameters) -> Self {
        // generate genome with initial ids for structure
        let initial_genome = Genome::new(context, parameters);

        let mut individuals = Vec::new();

        for _ in 0..parameters.setup.population {
            let mut other_genome = initial_genome.clone();
            other_genome.init(context, parameters);
            other_genome.mutate(context, parameters);
            individuals.push(other_genome);
        }

        let Compatability {
            factor_genes,
            factor_weights,
            factor_activations,
            ..
        } = parameters.compatability;

        let compatability_distance = Box::new(move |genome_0: &Genome, genome_1: &Genome| {
            Species::compatability_distance(
                genome_0,
                genome_1,
                factor_genes,
                factor_weights,
                factor_activations,
            )
        });

        /* let mut compatabilities = Vec::new();

        for individual_0 in &individuals {
            for individual_1 in &individuals {
                compatabilities.push((compatability_distance)(individual_0, individual_1))
            }
        }

        let compatability_threshold = compatabilities.iter().sum::<f64>()
            / compatabilities.len() as f64
            / (parameters.compatability.target_species as f64 * 2.0);
        // / 10.0;

        let threshold_delta = compatability_threshold / 500.0; */

        Population {
            individuals,
            species: Vec::new(),
            context: PopulationContext {
                compatability_threshold: parameters.compatability.threshold,
                threshold_delta: parameters.compatability.threshold_delta,
                compatability_distance,
            },
        }
    }

    fn place_genome_into_species(&mut self, genome: Genome) {
        // place into matching species
        if let Some(species_index) = self.find_best_fitting_species(&genome) {
            self.species[species_index].members.push(genome);
        }
        // or open new species
        else {
            self.species.push(Species::new(genome));
        }
    }

    fn find_best_fitting_species(&self, genome: &Genome) -> Option<usize> {
        // return when no species exist
        if self.species.is_empty() {
            return None;
        };

        self.species
            .iter()
            // map to compatability distance
            .map(|species| (self.context.compatability_distance)(genome, &species.representative))
            .enumerate()
            // select minimum distance
            .min_by(|(_, distance_0), (_, distance_1)| distance_0.partial_cmp(distance_1).unwrap())
            // check against compatability threshold
            .filter(|(_, distance)| distance < &self.context.compatability_threshold)
            // return species index
            .map(|(index, _)| index)
    }

    pub fn speciate(&mut self, context: &mut Context, parameters: &Parameters) {
        // MAYBE: update representative by most similar descendant ?

        let now = Instant::now();

        // clear population and sort into species
        for genome in mem::replace(&mut self.individuals, Vec::new()) {
            self.place_genome_into_species(genome);
        }

        // sort members of species by adjusted fitness in descending order
        for species in &mut self.species {
            species.adjust_fitness(context, parameters);
        }

        // clear stale species
        let threshold = parameters.reproduction.stale_after;
        let len_before_threshold = self.species.len();
        // keep at least one species
        let mut i = 0;
        self.species
            .retain(|species| (i == 0 || species.stale < threshold, i += 1).0);

        // sort species by fitness in descending order
        self.species
            .sort_by(|species_0, species_1| species_1.score.partial_cmp(&species_0.score).unwrap());

        // collect statistics
        context.statistics.compatability_threshold = self.context.compatability_threshold;
        // context.statistics.num_species_stale = len_before_threshold - self.species.len();
        context.statistics.milliseconds_elapsed_speciation = now.elapsed().as_millis();
    }

    fn adjust_threshold(&mut self, parameters: &Parameters) {
        // check if num species near target species
        match self.species.len() {
            length if length > parameters.compatability.target_species => {
                self.context.compatability_threshold += self.context.threshold_delta
                    * (length as f64 / parameters.compatability.target_species as f64)
                // .powi(3);
            }
            length if length < parameters.compatability.target_species => {
                self.context.compatability_threshold -= self.context.threshold_delta
                    * (parameters.compatability.target_species as f64 / length as f64)
            }
            _ => {}
        }

        // use threshold_delta as lower cap of compatability_threshold
        self.context.compatability_threshold = self
            .context
            .threshold_delta
            .max(self.context.compatability_threshold);
    }

    pub fn reproduce(&mut self, context: &mut Context, parameters: &Parameters) {
        let now = Instant::now();
        let offspring_ratio; // expresses how much offspring one point of fitness is worth

        let total_fitness = self
            .species
            .iter()
            .fold(0.0, |sum, species| sum + species.score);

        if total_fitness == 0.0 {
            // remove species with no members
            self.species.retain(|species| !species.members.is_empty());
            offspring_ratio = parameters.setup.population as f64 / self.species.len() as f64;
        } else {
            offspring_ratio = parameters.setup.population as f64 / total_fitness;
            // remove species that do not qualify for offspring
            self.species
                .retain(|species| species.score * offspring_ratio >= 1.0);
        }

        let all_species = &self.species;

        for species in &self.species {
            // collect members that are allowed to reproduce
            let allowed_members: Vec<&Genome> = species
                .members
                .iter()
                // need to ceil due to choose + unwrap, i.e. at least one member
                .take(
                    (species.members.len() as f64 * parameters.reproduction.surviving).ceil()
                        as usize,
                )
                .collect();

            // calculate offspring count for species
            let offspring_count = if total_fitness == 0.0 {
                offspring_ratio
            } else {
                species.score * offspring_ratio
            }
            .round() as usize;

            self.individuals
                .extend(
                    allowed_members
                        .iter()
                        .cycle()
                        .take(offspring_count)
                        .map(|member| {
                            member.crossover(
                                allowed_members.choose(&mut context.small_rng).unwrap(),
                                context,
                            )
                        }),
                );
        }

        // mutate the new population
        for genome in &mut self.individuals {
            genome.mutate(context, parameters);
        }

        // always keep top performing member
        self.individuals.extend(
            all_species
                .iter()
                // .filter(|species| species.members.len() >= 5)
                .flat_map(|species| species.members.iter().take(1))
                .cloned(),
        );

        // clear species members
        for species in &mut self.species {
            species.members.clear();
        }

        self.adjust_threshold(parameters);

        // collect statistics
        context.statistics.num_generation += 1;
        context.statistics.num_offpring = self.individuals.len();
        context.statistics.num_species = self.species.len();
        context.statistics.milliseconds_elapsed_reproducing = now.elapsed().as_millis();
    }
}
