use std::time::{Instant, SystemTime};

use ndarray::{Array2, ArrayView1, Axis};
use rayon::prelude::*;
use serde::Serialize;

use crate::{
    context::Context,
    population::Population,
    scores::{Fitness, Raw},
};
use crate::{genome::Genome, scores::ScoreValue};
use crate::{scores::FitnessScore, Neat};

#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub top_performer: Genome,
    pub compatability_threshold: f64,
    pub archive_threshold: f64,
    pub fitness: FitnessReport,
    pub num_generation: usize,
    pub num_offpring: usize,
    pub num_species: usize,
    pub num_species_stale: usize,
    pub milliseconds_elapsed_evaluation: u128,
    pub milliseconds_elapsed_reproducing: u128,
    pub milliseconds_elapsed_speciation: u128,
    pub time_stamp: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FitnessReport {
    raw_maximum: f64,
    raw_minimum: f64,
    raw_average: f64,
    shifted_maximum: f64,
    shifted_minimum: f64,
    shifted_average: f64,
    normalized_maximum: f64,
    normalized_minimum: f64,
    normalized_average: f64,
}

pub struct Runtime<'a> {
    neat: &'a Neat,
    context: Context,
    population: Population,
    archive: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub struct Progress {
    raw_fitness: Raw<Fitness>,
    behavior: Vec<f64>,
}

impl Progress {
    pub fn new(raw_fitness: Raw<Fitness>, behavior: Vec<f64>) -> Self {
        Progress {
            raw_fitness,
            behavior,
        }
    }
}

pub enum Evaluation {
    Progress(Report),
    Solution(Genome),
}

impl<'a> Iterator for Runtime<'a> {
    type Item = Evaluation;

    fn next(&mut self) -> Option<Self::Item> {
        self.reset_generational_statistics();

        let now = Instant::now();

        // generate progress by running progress function for everty individual
        let progress = self.generate_progress();
        // assign fitnesses from progress to individuals
        self.assign_fitness(progress.as_slice());

        self.context.statistics.milliseconds_elapsed_evaluation = now.elapsed().as_millis();

        // set top performer for report
        self.determine_top_performer();

        if let Some(winner) = self.check_for_solution() {
            Some(Evaluation::Solution(winner))
        } else {
            self.population
                .speciate(&mut self.context, &self.neat.parameters);
            self.population
                .reproduce(&mut self.context, &self.neat.parameters);
            Some(Evaluation::Progress(self.context.statistics.clone()))
        }
    }
}

impl<'a> Runtime<'a> {
    pub fn new(neat: &'a Neat) -> Self {
        let mut context = Context::new(&neat.parameters);

        let population = Population::new(&mut context, &neat.parameters);

        let mut runtime = Runtime {
            neat,
            context,
            archive: Vec::new(),
            population,
        };

        // runtime.init_novelty();

        runtime
    }
}

// private API
impl<'a> Runtime<'a> {
    // allow exact comparison of floats as we got the value we are looking for from the same list we are searching
    /* #[allow(clippy::float_cmp)]
    fn collect_fitness_statistics(&mut self) {
        self.context.statistics.fitness_peak = self
            .population
            .individuals
            .iter()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b.score(&self.context)));
        self.context.statistics.fitness_min = self
            .population
            .individuals
            .iter()
            .fold(f64::INFINITY, |a, b| a.min(b.score(&self.context)));
        self.context.statistics.fitness_average = self
            .population
            .individuals
            .iter()
            .map(|i| i.score(&self.context))
            .sum::<f64>()
            / self.population.individuals.len() as f64;
        self.context.statistics.top_performer = self
            .population
            .individuals
            .iter()
            .find(|&genome| genome.score(&self.context) == self.context.statistics.fitness_peak)
            .unwrap()
            .clone();
    } */

    fn generate_progress(&self) -> Vec<Progress> {
        let progress_fn = &self.neat.progress_function;

        // apply progress function to every individual
        self.population
            .individuals
            .par_iter()
            .map(progress_fn)
            .collect::<Vec<Progress>>()
    }

    /* fn step(&mut self) -> Step {
        let now = Instant::now();

        // make closure only capture progress_function field
        let progress_fn = &self.neat.progress_function;

        // apply progress function to every individual
        let progress = self
            .population
            .individuals
            .par_iter()
            .map(progress_fn)
            .collect::<Vec<Progress>>();

        self.context.statistics.milliseconds_elapsed_evaluation = now.elapsed().as_millis();

        // check if sufficient individual evolved
        let possible_solution = progress.iter().position(|progress| {
            progress.raw_fitness.value() > self.neat.parameters.required_fitness
        });

        // either complete evolution with solution or carry on with progress
        if let Some(winner) = possible_solution {
            Step::Solution(self.population.individuals[winner].clone())
        } else {
            Step::Progress(progress)
        }
    } */

    fn check_for_solution(&self) -> Option<Genome> {
        self.population
            .individuals
            .iter()
            .find(|genome| genome.fitness.raw.value() > self.neat.parameters.required_fitness)
            .cloned()
    }

    /* fn init_novelty(&mut self) {
        if let Step::Progress(progress) = self.step() {
            // init archive threshold to initial fitness peak
            self.assign_novelty(progress.as_slice());
            // FIXME: use seperate novelty statistic field, score statistic field
            self.context.archive_threshold = self.context.statistics.fitness_peak;
        }
    } */

    /* fn assign_novelty(&mut self, progress: &[Progress]) {
        let mut behaviors = progress
            .iter()
            .map(|progress| &progress.behavior)
            .chain(self.archive.iter());

        let width = progress[0].behavior.len();
        let height = progress.len() + self.archive.len();

        let mut behavior_arr: Array2<f64> = Array2::zeros((width, height));
        for mut row in behavior_arr.axis_iter_mut(Axis(1)) {
            row += &ArrayView1::from(behaviors.next().unwrap());
        }

        let means = behavior_arr.mean_axis(Axis(1)).unwrap();
        let mut std_dev = behavior_arr.std_axis(Axis(1), 0.0);

        // clean std_dev from zeroes
        std_dev.map_inplace(|v| {
            if *v == 0.0 {
                *v = 1.0
            }
        });

        let mut z_scores_arr: Array2<f64> = Array2::zeros((width, height));

        for (index, row) in behavior_arr.axis_iter(Axis(0)).enumerate() {
            let mut z_row = z_scores_arr.index_axis_mut(Axis(0), index);
            z_row += &((&row - &means) / &std_dev);
        }

        let mut add_to_archive = Vec::new();

        for (index, z_score) in z_scores_arr
            .axis_iter(Axis(0))
            .enumerate()
            .take(progress.len())
        {
            let mut distances = z_scores_arr
                .axis_iter(Axis(1))
                .map(|neighbor| {
                    neighbor
                        .iter()
                        .zip(z_score.iter())
                        .map(|(n, z)| (n - z).powi(2))
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
            self.population.individuals[index].novelty = sparseness;

            // add to archive if over threshold
            if sparseness > self.context.archive_threshold {
                add_to_archive.push(behavior_arr.index_axis(Axis(1), index).clone().to_vec());
            }
        }

        // actually add to archive if over threshold
        self.context.added_to_archive += add_to_archive.len();
        self.archive.extend(add_to_archive.into_iter());

        // adjust archive threshold every 10 generations
        if self.context.statistics.num_generation % 10 == 0 {
            if self.context.added_to_archive > 4 {
                self.context.archive_threshold *= 1.2;
            }

            if self.context.added_to_archive == 0 {
                self.context.archive_threshold *= 0.95;
            }

            self.context.added_to_archive = 0;
            self.context.statistics.archive_threshold = self.context.archive_threshold;
        }
        // self.collect_fitness_statistics();
    } */

    fn determine_top_performer(&mut self) {
        let pos = self
            .population
            .individuals
            .iter()
            .enumerate()
            .map(|(index, individual)| (index, individual.score(&self.context)))
            .fold((0_usize, f64::NEG_INFINITY), |acc, val| {
                if val.1 > acc.1 {
                    val
                } else {
                    acc
                }
            });

        self.context.statistics.top_performer = self.population.individuals[pos.0].clone();
    }

    #[allow(clippy::float_cmp)]
    fn assign_fitness(&mut self, progress: &[Progress]) /* -> FitnessReport */
    {
        let mut raw_minimum = f64::INFINITY;
        let mut raw_sum = 0.0;
        let mut raw_maximum = f64::NEG_INFINITY;

        // analyse and set raw fitness values
        for raw_fitness in progress.iter().map(|p| p.raw_fitness) {
            if raw_fitness.value() > raw_maximum {
                raw_maximum = raw_fitness.value();
            }
            if raw_fitness.value() < raw_minimum {
                raw_minimum = raw_fitness.value();
            }
            raw_sum += raw_fitness.value();
            // self.population.individuals[index].fitness.raw = raw_fitness;
        }

        let raw_minimum = Raw::new(raw_minimum);
        let raw_average = Raw::new(raw_sum / progress.len() as f64);
        let raw_maximum = Raw::new(raw_maximum);

        let baseline = raw_minimum;

        let shifted_minimum = raw_minimum.shift(baseline.value());
        let shifted_average = raw_average.shift(baseline.value());
        let shifted_maximum = raw_maximum.shift(baseline.value());

        let with = shifted_maximum;

        let normalized_minimum = shifted_minimum.normalize(with.value());
        let normalized_average = shifted_average.normalize(with.value());
        let normalized_maximum = shifted_maximum.normalize(with.value());

        // shift and normalize fitness
        for (index, individual) in self.population.individuals.iter_mut().enumerate() {
            individual.fitness = FitnessScore::new(
                progress[index].raw_fitness.value(),
                baseline.value(),
                with.value(),
            );
        }

        self.context.statistics.fitness.raw_maximum = raw_maximum.value();
        self.context.statistics.fitness.raw_minimum = raw_minimum.value();
        self.context.statistics.fitness.raw_average = raw_average.value();

        self.context.statistics.fitness.shifted_maximum = shifted_maximum.value();
        self.context.statistics.fitness.shifted_minimum = shifted_minimum.value();
        self.context.statistics.fitness.shifted_average = shifted_average.value();

        self.context.statistics.fitness.normalized_maximum = normalized_maximum.value();
        self.context.statistics.fitness.normalized_minimum = normalized_minimum.value();
        self.context.statistics.fitness.normalized_average = normalized_average.value();

        /* FitnessReport {
            raw_average,
            raw_maximum,
            raw_minimum,
        } */
    }

    fn reset_generational_statistics(&mut self) {
        self.context.statistics.time_stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.context.statistics.num_species_stale = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::Neat;
    use crate::genome::Genome;
    use crate::runtime::{Evaluation, Progress};
    use crate::scores::Raw;

    #[test]
    fn place_into_species() {
        todo!("move to population test")
        /* let mut neat = Neat::new(
            "src/Config.toml",
            Box::new(|_| Progress {
                raw_fitness: Raw::new(0.0),
                behavior: Vec::new(),
            }),
        );

        neat.parameters.compatability.threshold = 3.0;
        neat.parameters.compatability.factor_genes = 10.0;
        neat.parameters.compatability.factor_weights = 10.0;
        neat.parameters.compatability.factor_activations = 10.0;
        neat.parameters.reproduction.surviving = 1.0;

        let mut runtime = neat.run();

        let mut genome_0 = Genome::new(&mut runtime.context, &runtime.neat.parameters);

        genome_0.init(&mut runtime.context, &runtime.neat.parameters);

        let mut genome_1 = genome_0.clone();

        // manipulate weight for genome distance to exceed threshold
        let mut connection_gene_0 = genome_1.feed_forward.iter().next().unwrap().clone();
        (connection_gene_0.1).0 = 15.0;
        genome_0.feed_forward.replace(connection_gene_0);

        let mut connection_gene_1 = genome_1.feed_forward.iter().next().unwrap().clone();
        (connection_gene_1.1).0 = -15.0;
        genome_1.feed_forward.replace(connection_gene_1);

        // make genome 1 more fit
        genome_1.fitness.raw = Raw::new(1.0);
        genome_1.fitness.shifted = genome_1.fitness.raw.shift(0.0);
        genome_1.fitness.normalized = genome_1.fitness.shifted.normalize(1.0);

        runtime.population = ;

        runtime.population.push(genome_0);
        runtime.population.push(genome_1);

        assert_eq!(runtime.population.len(), 2);

        runtime.speciate();

        println!("species: {:?}", runtime.species);

        assert_eq!(runtime.species.len(), 2);
        assert!(runtime.species[0].fitness - 1.0 < f64::EPSILON);
        assert_eq!(runtime.population.len(), 0); */
    }

    #[test]
    fn species_sorted_descending() {
        todo!("move to population test")
        /* let mut neat = Neat::new(
            "src/Config.toml",
            Box::new(|_| Progress {
                fitness: 0.0,
                behavior: Vec::new(),
            }),
        );

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
        assert!(runtime.species[0].fitness - 1.5 < f64::EPSILON); */
    }

    #[test]
    fn run_neat_till_10_connections() {
        fn fitness_function(genome: &Genome) -> Progress {
            let fitness = genome.feed_forward.len() as f64;
            Progress {
                raw_fitness: Raw::new(fitness),
                behavior: Vec::new(),
            }
        }

        let mut neat = Neat::new("src/Config.toml", Box::new(fitness_function));

        neat.parameters.required_fitness = 10.0;

        if let Some(winner) = neat
            .run()
            .filter_map(|evaluation| match evaluation {
                Evaluation::Progress(_) => None,
                Evaluation::Solution(genome) => Some(genome),
            })
            .next()
        {
            assert!((winner.feed_forward.len() as i64 - 10) >= 0);
        }
    }

    #[test]
    fn run_neat_till_50_connections() {
        fn fitness_function(genome: &Genome) -> Progress {
            let fitness = genome.feed_forward.len() as f64;
            Progress {
                raw_fitness: Raw::new(fitness),
                behavior: Vec::new(),
            }
        }

        let mut neat = Neat::new("src/Config.toml", Box::new(fitness_function));

        neat.parameters.required_fitness = 50.0;

        if let Some(winner) = neat
            .run()
            .filter_map(|evaluation| match evaluation {
                Evaluation::Progress(_) => None,
                Evaluation::Solution(genome) => Some(genome),
            })
            .next()
        {
            assert!((winner.feed_forward.len() as i64 - 50) >= 0);
        }
    }

    #[test]
    fn move_negative_fitness_to_zero_baseline() {
        fn fitness_function(_genome: &Genome) -> Progress {
            Progress {
                raw_fitness: Raw::new(-1.0),
                behavior: Vec::new(),
            }
        }

        let neat = Neat::new("src/Config.toml", Box::new(fitness_function));

        let mut runtime = neat.run();

        if let Some(Evaluation::Progress(report)) = runtime.next() {
            assert!(report.fitness.raw_minimum + 1.0 < f64::EPSILON);
            assert!(report.fitness.shifted_minimum < f64::EPSILON);
        }
    }
}
