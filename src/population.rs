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
            species: Vec::new(),
            context: PopulationContext {
                compatability_threshold: dbg!(Population::init_threshold(&individuals, parameters)),
                threshold_delta: parameters.compatability.threshold_delta,
                factor_genes,
                factor_weights,
                factor_activations,
            },
            individuals,
        }
    }

    fn place_genome_into_species(&mut self, genome: Genome) {
        let context = &self.context;
        // place into matching species
        if let Some(species) = self.species.iter_mut().find(|species| {
            Genome::compatability_distance(
                &genome,
                &species.representative,
                context.factor_genes,
                context.factor_weights,
                context.factor_activations,
            ) < context.compatability_threshold
        }) {
            species.members.push(genome);
        } else {
            self.species.push(Species::new(genome));
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
            .take(parameters.reproduction.elitism_species)
            .chain(
                self.species
                    .iter()
                    .skip(parameters.reproduction.elitism_species)
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

        // check on final species count
        // do immediatly after speciation ?
        self.adjust_threshold(parameters);

        self.remove_stale_species(context, parameters);

        // collect statistics
        context.statistics.compatability_threshold = self.context.compatability_threshold;
        context.statistics.milliseconds_elapsed_speciation = now.elapsed().as_millis();
    }

    fn init_threshold(individuals: &[Genome], parameters: &Parameters) -> f64 {
        let mut threshold = Vec::new();

        let approximate_species_size = (parameters.setup.population as f64
            / parameters.compatability.target_species as f64)
            .floor() as usize;

        for individual_0 in individuals {
            let mut distances = Vec::new();

            for individual_1 in individuals {
                distances.push(Genome::compatability_distance(
                    individual_0,
                    individual_1,
                    parameters.compatability.factor_genes,
                    parameters.compatability.factor_weights,
                    parameters.compatability.factor_activations,
                ));
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            threshold.push(distances[approximate_species_size]);
        }

        dbg!(&threshold);

        threshold.iter().sum::<f64>() / threshold.len() as f64 * 1.5
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

        dbg!(self.species.len());

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

        dbg!(self.species.len());

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
            species.prepare_next_generation();
        }

        // collect statistics
        context.statistics.num_generation += 1;
        context.statistics.num_offpring = self.individuals.len();
        context.statistics.num_species = self.species.len();
        context.statistics.milliseconds_elapsed_reproducing = now.elapsed().as_millis();
    }
}
