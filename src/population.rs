use std::{mem, time::Instant};

use crate::{parameters::Compatability, species::Species, Context, Genome, Parameters};

pub struct Population {
    pub individuals: Vec<Genome>,
    species: Vec<Species>,
    context: PopulationContext,
}

struct PopulationContext {
    compatability_threshold: f64,
    threshold_delta: f64,
    factor_genes: f64,
    factor_weights: f64,
    factor_activations: f64,
}

// uses setup/compatability/reproduction

impl Population {
    pub fn new(context: &mut Context, parameters: &Parameters) -> Self {
        // generate genome with initial ids for structure
        let initial_genome = Genome::new(context, parameters);

        let mut individuals = Vec::new();

        // generate initial, mutated individuals
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

        Population {
            individuals,
            species: Vec::new(),
            context: PopulationContext {
                compatability_threshold: parameters.compatability.threshold,
                threshold_delta: parameters.compatability.threshold_delta,
                factor_genes,
                factor_weights,
                factor_activations,
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
    }

    fn remove_stale_species(&mut self, context: &mut Context, parameters: &Parameters) {
        // sort species by fitness in descending order
        self.species
            .sort_by(|species_0, species_1| species_1.score.partial_cmp(&species_0.score).unwrap());

        // clear stale species
        let threshold = parameters.reproduction.stale_after;
        let len_before_threshold = self.species.len();
        self.species = self
            .species
            .iter()
            // keep at least one species
            .take(1)
            .chain(
                self.species
                    .iter()
                    .skip(1)
                    .filter(|species| species.stale < threshold),
            )
            .cloned()
            .collect();

        context.statistics.num_species_stale = len_before_threshold - self.species.len();
    }

    pub fn speciate(&mut self, context: &mut Context, parameters: &Parameters) {
        // MAYBE: update representative by most similar descendant ?

        let now = Instant::now();

        // clear population and sort into species
        for genome in mem::replace(&mut self.individuals, Vec::new()) {
            self.place_genome_into_species(genome);
        }

        // sort members of species and compute species score
        for species in &mut self.species {
            species.adjust(context, parameters);
        }

        self.remove_stale_species(context, parameters);

        // collect statistics
        context.statistics.compatability_threshold = self.context.compatability_threshold;
        context.statistics.milliseconds_elapsed_speciation = now.elapsed().as_millis();
    }

    fn adjust_threshold(&mut self, parameters: &Parameters) {
        // check if num species near target species
        match self.species.len() {
            length if length > parameters.compatability.target_species => {
                self.context.compatability_threshold += self.context.threshold_delta
                    * (length as f64 / parameters.compatability.target_species as f64).powi(2);
            }
            length if length < parameters.compatability.target_species => {
                self.context.compatability_threshold -= self.context.threshold_delta
                    * (parameters.compatability.target_species as f64 / length as f64).powi(2);
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
            offspring_ratio = parameters.setup.population as f64 / self.species.len() as f64;
        }
        // if we can differentiate species
        else {
            // give offspring in proportion to archived score
            offspring_ratio = parameters.setup.population as f64 / total_fitness;
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

            self.individuals
                .extend(species.reproduce(context, parameters, offspring_count));
        }

        // clear species members
        for species in &mut self.species {
            species.members.clear();
        }

        // check on final species count
        self.adjust_threshold(parameters);

        // collect statistics
        context.statistics.num_generation += 1;
        context.statistics.num_offpring = self.individuals.len();
        context.statistics.num_species = self.species.len();
        context.statistics.milliseconds_elapsed_reproducing = now.elapsed().as_millis();
    }
}
