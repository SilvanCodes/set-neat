use std::mem;
use std::time::{Instant, SystemTime};

use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::Serialize;

use crate::context::Context;
use crate::genome::Genome;
use crate::species::Species;
use crate::Neat;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub top_performer: Genome,
    pub compatability_threshold: f64,
    pub archive_threshold: f64,
    pub fitness_average: f64,
    pub fitness_peak: f64,
    pub fitness_min: f64,
    pub num_generation: usize,
    pub num_offpring: usize,
    pub num_species: usize,
    pub num_species_stale: usize,
    pub milliseconds_elapsed_evaluation: u128,
    pub milliseconds_elapsed_reproducing: u128,
    pub milliseconds_elapsed_speciation: u128,
    pub time_stamp: u64,
}

pub struct Runtime<'a> {
    neat: &'a Neat,
    context: Context,
    population: Vec<Genome>,
    archive: Vec<Vec<f64>>,
    species: Vec<Species>,
    statistics: Report,
}

enum Step {
    Progress(Vec<Progress>),
    Solution(Genome),
}
pub enum Progress {
    // progress in fitness based search
    Fitness(f64),
    // progress in novelty based search
    Novelty(Vec<f64>),
    // progressed to solution
    Solution(Genome),
}

pub enum Evaluation {
    Progress(Report),
    Solution(Genome),
}

impl<'a> Iterator for Runtime<'a> {
    type Item = Evaluation;

    fn next(&mut self) -> Option<Self::Item> {
        self.reset_generational_statistics();
        match self.step() {
            Step::Progress(progress) => match progress.as_slice() {
                values @ [Progress::Fitness(_), ..] => {
                    // do fitness calculations
                    self.assign_fitness(values);
                    self.speciate();
                    self.reproduce();
                    Some(Evaluation::Progress(self.statistics.clone()))
                }
                values @ [Progress::Novelty(_), ..] => {
                    // do novelty calculations
                    self.assign_novelty(values);
                    self.speciate();
                    self.reproduce();
                    Some(Evaluation::Progress(self.statistics.clone()))
                }
                _ => panic!("empty progress"),
            },
            Step::Solution(winner) => Some(Evaluation::Solution(winner)),
        }

        /* if let Some(genome) = self.evaluate() {
            Some(Evaluation::Solution(genome))
        } else {
            self.reset_generational_statistics();
            self.speciate();
            self.reproduce();
            Some(Evaluation::Progress(self.statistics.clone()))
        } */
    }
}

impl<'a> Runtime<'a> {
    pub fn new(neat: &'a Neat) -> Self {
        let mut runtime = Runtime {
            neat,
            context: Context::new(&neat.parameters),
            archive: Vec::new(),
            population: Vec::with_capacity(neat.parameters.setup.population),
            species: Vec::new(),
            statistics: Default::default(),
        };

        // setup fully connected input -> output
        let initial_genome = Genome::new(&mut runtime.context, &neat.parameters);

        runtime.populate(&initial_genome);

        runtime
    }
}

// private API
impl<'a> Runtime<'a> {
    fn populate(&mut self, genome: &Genome) {
        for _ in 0..self.neat.parameters.setup.population {
            let mut other_genome = Genome::from(genome);
            other_genome.init_with(&mut self.context, &self.neat.parameters);
            other_genome.mutate(&mut self.context, &self.neat.parameters);
            self.population.push(other_genome);
        }
    }

    fn place_genome_into_species(&mut self, genome: Genome) {
        // place into matching species
        if let Some(species_index) = self.find_best_fitting_species(&genome) {
            // println!("FOUND MATCHING SPECIES");
            self.species[species_index].members.push(genome);
        }
        // or open new species
        else {
            self.species.push(Species::new(genome));
        }
    }

    fn compatability_distance(&self, genome_0: &Genome, genome_1: &Genome) -> f64 {
        Species::compatability_distance(
            genome_0,
            genome_1,
            self.neat.parameters.compatability.factor_genes,
            self.neat.parameters.compatability.factor_weights,
            self.neat.parameters.compatability.factor_activations,
        )
    }

    fn find_best_fitting_species(&self, genome: &Genome) -> Option<usize> {
        if self.species.is_empty() {
            return None;
        };

        let initial = (
            0,
            self.compatability_distance(genome, &self.species[0].representative),
        );

        let fitting_species = self.species.iter().skip(1).enumerate().fold(
            initial,
            |best_fit, (position, species)| {
                let distance = self.compatability_distance(genome, &species.representative);
                if distance < best_fit.1 {
                    (position, distance)
                } else {
                    best_fit
                }
            },
        );

        Some(fitting_species)
            .filter(|(_, distance)| distance < &self.context.compatability_threshold)
            .map(|(position, _)| position)
    }

    fn speciate(&mut self) {
        // MAYBE: update representative by most similar descendant ?

        let now = Instant::now();

        // clear population and sort into species
        for genome in mem::replace(&mut self.population, Vec::new()) {
            self.place_genome_into_species(genome);
        }

        let parameters = &self.neat.parameters;
        let context = &mut self.context;

        // sort members of species by adjusted fitness in descending order
        for species in &mut self.species {
            species.adjust_fitness(parameters);
        }

        // clear stale species
        let threshold = parameters.reproduction.stale_after;
        let len_before_threshold = self.species.len();
        // keep at least one species
        let mut i = 0;
        self.species
            .retain(|species| (i == 0 || species.stale < threshold, i += 1).0);

        // sort species by fitness in descending order
        self.species.sort_by(|species_0, species_1| {
            species_1.fitness.partial_cmp(&species_0.fitness).unwrap()
        });

        // check if num species near target species
        match self.species.len() {
            len if len > parameters.compatability.target_species => {
                context.compatability_threshold += parameters.compatability.threshold_delta
                    * (self.species.len() as f64 / parameters.compatability.target_species as f64)
                // .powi(3);
            }
            len if len < parameters.compatability.target_species => {
                context.compatability_threshold -= parameters.compatability.threshold_delta
                    * (parameters.compatability.target_species as f64 / self.species.len() as f64)
                // .powi(3);
            }
            _ => {}
        }

        // use threshold_delta as lower cap of compatability_threshold
        context.compatability_threshold = parameters
            .compatability
            .threshold_delta
            .max(context.compatability_threshold);

        // collect statistics
        self.statistics.compatability_threshold = context.compatability_threshold;
        self.statistics.num_species_stale = len_before_threshold - self.species.len();
        self.statistics.milliseconds_elapsed_speciation = now.elapsed().as_millis();
    }

    // allow exact comparison of floats as we got the value we are looking for from the same list we are searching
    #[allow(clippy::float_cmp)]
    fn collect_fitness_statistics(&mut self) {
        self.statistics.fitness_peak = self
            .population
            .iter()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b.fitness));
        self.statistics.fitness_min = self
            .population
            .iter()
            .fold(f64::INFINITY, |a, b| a.min(b.fitness));
        self.statistics.fitness_average =
            self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population.len() as f64;
        self.statistics.top_performer = self
            .population
            .iter()
            .find(|&genome| genome.fitness == self.statistics.fitness_peak)
            .unwrap()
            .clone();
    }

    fn step(&mut self) -> Step {
        let now = Instant::now();

        let progress = self
            .population
            .par_iter()
            .map(self.neat.progress_function)
            .collect::<Vec<Progress>>();

        self.statistics.milliseconds_elapsed_evaluation = now.elapsed().as_millis();

        let solution = progress.iter().find_map(|progress| match progress {
            Progress::Solution(winner) => Some(winner),
            _ => None,
        });

        if let Some(winner) = solution {
            Step::Solution(winner.clone())
        } else {
            Step::Progress(progress)
        }
    }

    fn assign_novelty(&mut self, novelties: &[Progress]) {
        let novelties = novelties
            .iter()
            .map(|progress| match progress {
                Progress::Novelty(novelty) => novelty,
                _ => panic!("non homogenous progress vector"),
            })
            .collect::<Vec<&Vec<f64>>>();

        // map to z-score

        /* let sums = novelties
            .iter()
            .fold(vec![0.0; novelties[0].len()], |acc, val| {
                acc.iter().zip(val.iter()).map(|(a, v)| a + v).collect()
            });

        let means: Vec<f64> = sums.iter().map(|s| s / sums.len() as f64).collect();

        let variances = novelties
            .iter()
            .map(|values| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v - means[i])
                    .map(|v| v * v)
                    .collect::<Vec<f64>>()
            })
            .fold(vec![0.0; novelties[0].len()], |acc, val| {
                acc.iter().zip(val.iter()).map(|(a, v)| a + v).collect()
            }); */

        for (index, novelty) in novelties.iter().enumerate() {
            // calulate every distance
            let mut distances = novelties
                .iter()
                .cloned()
                .chain(self.archive.iter())
                .map(|neighbor| {
                    neighbor
                        .iter()
                        .zip(novelty.iter())
                        .map(|(n, s)| (n - s).powi(2))
                        .sum::<f64>()
                })
                .map(|sum| sum.sqrt())
                .collect::<Vec<f64>>();

            distances.sort_by(|dist_0, dist_1| dist_0.partial_cmp(&dist_1).unwrap());

            // take k nearest neighbors, calculate and assign spareseness
            let sparseness = distances
                .iter()
                .take(self.neat.parameters.novelty.nearest_neighbors)
                .sum::<f64>()
                / self.neat.parameters.novelty.nearest_neighbors as f64;
            self.population[index].fitness = sparseness;

            // add to archive if over threshold
            if sparseness > self.context.archive_threshold {
                self.archive.push((*novelty).clone());
                self.context.added_to_archive += 1;
            }
        }

        if self.statistics.num_generation % 10 == 0 {
            if self.context.added_to_archive > 4 {
                self.context.archive_threshold *= 1.2;
            }

            if self.context.added_to_archive == 0 {
                self.context.archive_threshold *= 0.95;
            }

            self.context.added_to_archive = 0;
            self.statistics.archive_threshold = self.context.archive_threshold;
        }
        self.collect_fitness_statistics();
    }

    fn assign_fitness(&mut self, fitnesses: &[Progress]) {
        let mut fitnesses = fitnesses
            .iter()
            .map(|progress| match progress {
                Progress::Fitness(fitness) => fitness,
                _ => panic!("non homogenous progress vector"),
            })
            .cloned()
            .collect::<Vec<f64>>();

        let maximum = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let average = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        // move all fitness values above zero baseline, when some negative
        let minimum = fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        if minimum < 0.0 {
            for fitness in &mut fitnesses {
                *fitness = (*fitness - minimum).abs();
            }
        }

        // set fitness for every genome
        for (index, &fitness) in fitnesses.iter().enumerate() {
            self.population[index].fitness = fitness;
        }

        self.statistics.fitness_peak = maximum;
        self.statistics.fitness_min = minimum;
        self.statistics.fitness_average = average;
    }

    /* fn evaluate(&mut self) -> Option<Genome> {
        let now = Instant::now();

        // evaluate nets in parallel
        let mut fitnesses: Vec<f64> = self
            .population
            .par_iter()
            .map(self.neat.fitness_function)
            .collect();

        // dbg!(&fitnesses);

        let maximum = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let average = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        // allow exact comparison of floats as we got the value we are looking for from the same list we are searching
        #[allow(clippy::float_cmp)]
        let pos = fitnesses
            .iter()
            .position(|&fitness| fitness == maximum)
            .unwrap();
        let mut top_performer = self.population[pos].clone();

        // check if some net is sufficient
        if maximum > self.neat.required_fitness {
            top_performer.fitness = maximum;
            return Some(top_performer);
        }

        // move all fitness values to zero baseline, when some negative
        let minimum = fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        if minimum < 0.0 {
            for fitness in &mut fitnesses {
                *fitness = (*fitness - minimum).abs();
            }
        }

        // set fitness for every genome
        for (index, &fitness) in fitnesses.iter().enumerate() {
            self.population[index].fitness = fitness;
        }

        self.statistics.milliseconds_elapsed_evaluation = now.elapsed().as_millis();
        self.statistics.fitness_peak = maximum;
        self.statistics.fitness_min = minimum;
        self.statistics.fitness_average = average;
        self.statistics.top_performer = top_performer;
        None
    } */

    fn reproduce(&mut self) {
        let now = Instant::now();
        let context = &mut self.context;
        let parameters = &self.neat.parameters;
        let offspring_ratio; // expresses how much offspring one point of fitness is worth

        let total_fitness = self
            .species
            .iter()
            .fold(0.0, |sum, species| sum + species.fitness);

        if total_fitness == 0.0 {
            // remove species with no members
            self.species.retain(|species| !species.members.is_empty());
            offspring_ratio =
                self.neat.parameters.setup.population as f64 / self.species.len() as f64;
        } else {
            offspring_ratio = self.neat.parameters.setup.population as f64 / total_fitness;
            // remove species that do not qualify for offspring
            self.species
                .retain(|species| species.fitness * offspring_ratio >= 1.0);
        }

        let all_species = &self.species;

        for species in all_species {
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
                species.fitness * offspring_ratio
            }
            .round() as usize;

            self.population
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
        for genome in &mut self.population {
            genome.mutate(context, parameters);
        }

        // always keep top performing member
        self.population.extend(
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

        // collect statistics
        self.statistics.num_generation += 1;
        self.statistics.num_offpring = self.population.len();
        self.statistics.num_species = self.species.len();
        self.statistics.milliseconds_elapsed_reproducing = now.elapsed().as_millis();
    }

    fn reset_generational_statistics(&mut self) {
        self.statistics.time_stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.statistics.num_species_stale = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::Neat;
    use crate::genome::Genome;
    use crate::runtime::{Evaluation, Progress};

    #[test]
    fn place_into_species() {
        let mut neat = Neat::new("src/Config.toml", |_| Progress::Fitness(0.0));

        neat.parameters.compatability.threshold = 3.0;
        neat.parameters.compatability.factor_genes = 10.0;
        neat.parameters.compatability.factor_weights = 10.0;
        neat.parameters.compatability.factor_activations = 10.0;
        neat.parameters.reproduction.surviving = 1.0;

        let mut runtime = neat.run();

        let mut genome_0 = Genome::new(&mut runtime.context, &runtime.neat.parameters);

        genome_0.init();

        let mut genome_1 = Genome::from(&genome_0);

        // manipulate weight for genome distance to exceed threshold
        let mut connection_gene_0 = genome_1.connection_genes.iter().next().unwrap().clone();
        connection_gene_0.weight.0 = 15.0;
        genome_0.connection_genes.replace(connection_gene_0);

        let mut connection_gene_1 = genome_1.connection_genes.iter().next().unwrap().clone();
        connection_gene_1.weight.0 = -15.0;
        genome_1.connection_genes.replace(connection_gene_1);

        // make genome 1 more fit
        genome_1.fitness = 1.0;

        runtime.population = Vec::new();

        runtime.population.push(genome_0);
        runtime.population.push(genome_1);

        assert_eq!(runtime.population.len(), 2);

        runtime.speciate();

        println!("species: {:?}", runtime.species);

        assert_eq!(runtime.species.len(), 2);
        assert!(runtime.species[0].fitness - 1.0 < f64::EPSILON);
        assert_eq!(runtime.population.len(), 0);
    }

    #[test]
    fn species_sorted_descending() {
        let mut neat = Neat::new("src/Config.toml", |_| Progress::Fitness(0.0));

        neat.parameters.reproduction.surviving = 1.0;

        let mut runtime = neat.run();

        let mut genome_0 = Genome::new(&mut runtime.context, &runtime.neat.parameters);

        genome_0.init();

        let mut genome_1 = Genome::from(&genome_0);

        genome_0.fitness = 1.0;
        genome_1.fitness = 2.0;

        runtime.population = Vec::new();

        runtime.population.push(genome_0);
        runtime.population.push(genome_1);

        assert_eq!(runtime.population.len(), 2);

        runtime.speciate();

        println!("species: {:?}", runtime.species);

        assert_eq!(runtime.species.len(), 1);
        // should have adjusted fitness
        assert!(runtime.species[0].members[0].fitness - 1.0 < f64::EPSILON);
        // should have averaged fitness
        assert!(runtime.species[0].fitness - 1.5 < f64::EPSILON);
    }

    #[test]
    fn run_neat_till_10_connections() {
        fn fitness_function(genome: &Genome) -> Progress {
            let fitness = genome.connection_genes.len();
            if fitness > 10 {
                Progress::Solution(genome.clone())
            } else {
                Progress::Fitness(fitness as f64)
            }
        }

        let neat = Neat::new("src/Config.toml", fitness_function);

        if let Some(winner) = neat
            .run()
            .filter_map(|evaluation| match evaluation {
                Evaluation::Progress(_) => None,
                Evaluation::Solution(genome) => Some(genome),
            })
            .next()
        {
            assert!((winner.connection_genes.len() as i64 - 10) >= 0);
        }
    }

    #[test]
    fn run_neat_till_50_connections() {
        fn fitness_function(genome: &Genome) -> Progress {
            let fitness = genome.connection_genes.len();
            if fitness > 10 {
                Progress::Solution(genome.clone())
            } else {
                Progress::Fitness(fitness as f64)
            }
        }

        let neat = Neat::new("src/Config.toml", fitness_function);

        if let Some(winner) = neat
            .run()
            .filter_map(|evaluation| match evaluation {
                Evaluation::Progress(_) => None,
                Evaluation::Solution(genome) => Some(genome),
            })
            .next()
        {
            assert!((winner.connection_genes.len() as i64 - 50) >= 0);
        }
    }

    #[test]
    fn move_negative_fitness_to_zero_baseline() {
        fn fitness_function(_genome: &Genome) -> Progress {
            Progress::Fitness(-1.0)
        }

        let neat = Neat::new("src/Config.toml", fitness_function);

        let mut runtime = neat.run();

        runtime.step();

        assert!(runtime.statistics.fitness_peak - (-1.0) < f64::EPSILON);
        assert!(runtime.population[0].fitness < f64::EPSILON);
        assert!(runtime.population[1].fitness < f64::EPSILON);
    }
}
