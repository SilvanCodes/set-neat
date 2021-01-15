use std::{
    ops::Deref,
    time::{Instant, SystemTime},
};

use ndarray::{arr1, Array2, ArrayView1, Axis};
use rayon::prelude::*;
use serde::Serialize;

use crate::genome::Genome;
use crate::{context::Context, population::Population, utility::gym::StandardScaler};
use crate::{scores::Score, Neat};

#[derive(Debug, Clone, Default, Serialize)]
pub struct Report {
    pub top_performer: Genome,
    pub compatability_threshold: f64,
    pub archive_threshold: f64,
    pub fitness: FitnessReport,
    pub novelty: NoveltyReport,
    // pub novelty_ratio: f64,
    pub peak_fitness_average: f64,
    pub num_generation: usize,
    pub num_offpring: usize,
    pub num_species: usize,
    pub num_species_stale: usize,
    pub num_consecutive_ineffective_generations: usize,
    pub milliseconds_elapsed_evaluation: u128,
    pub milliseconds_elapsed_reproducing: u128,
    pub milliseconds_elapsed_speciation: u128,
    pub time_stamp: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FitnessReport {
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
pub struct NoveltyReport {
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

pub struct Runtime<'a> {
    neat: &'a Neat,
    context: Context,
    population: Population,
    archive: Vec<Behavior>,
}

#[derive(Debug)]
pub enum Progress {
    Empty,
    Fitness(f64),
    Novelty(Behavior),
    Status(f64, Behavior),
    Solution(Option<f64>, Option<Behavior>, Box<Genome>),
}

#[derive(Debug, Default)]
pub struct Behavior(pub Vec<f64>);

impl Deref for Behavior {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Progress {
    pub fn new(fitness: f64, behavior: Vec<f64>) -> Self {
        Progress::Status(fitness, Behavior(behavior))
    }

    pub fn empty() -> Self {
        Progress::Empty
    }

    pub fn solved(self, genome: Genome) -> Self {
        match self {
            Progress::Fitness(fitness) => Progress::Solution(Some(fitness), None, Box::new(genome)),
            Progress::Novelty(behavior) => {
                Progress::Solution(None, Some(behavior), Box::new(genome))
            }
            Progress::Status(fitness, behavior) => {
                Progress::Solution(Some(fitness), Some(behavior), Box::new(genome))
            }
            Progress::Solution(fitness, behavior, _) => {
                Progress::Solution(fitness, behavior, Box::new(genome))
            }
            Progress::Empty => Progress::Solution(None, None, Box::new(genome)),
        }
    }

    pub fn fitness(fitness: f64) -> Self {
        Self::Fitness(fitness)
    }

    pub fn novelty(behavior: Vec<f64>) -> Self {
        Self::Novelty(Behavior(behavior))
    }

    pub fn behavior(&self) -> Option<&Behavior> {
        match self {
            Progress::Status(_, behavior) => Some(behavior),
            Progress::Solution(_, behavior, _) => behavior.as_ref(),
            Progress::Novelty(behavior) => Some(behavior),
            Progress::Fitness(_) => None,
            Progress::Empty => None,
        }
    }

    pub fn raw_fitness(&self) -> Option<f64> {
        match *self {
            Progress::Status(fitness, _) => Some(fitness),
            Progress::Solution(fitness, _, _) => fitness,
            Progress::Fitness(fitness) => Some(fitness),
            Progress::Novelty(_) => None,
            Progress::Empty => None,
        }
    }

    pub fn is_solution(&self) -> Option<&Genome> {
        match self {
            Progress::Solution(_, _, genome) => Some(genome),
            _ => None,
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
        // assign fitnesses from progress fitness to individuals
        self.assign_fitness(&progress);
        // assign novelty from progress behavior to individuals
        self.assign_novelty(&progress);

        self.context.statistics.milliseconds_elapsed_evaluation = now.elapsed().as_millis();

        // set top performer for report
        self.determine_top_performer();

        self.measure_effectiveness(&progress);

        if let Some(winner) = self.check_for_solution(&progress) {
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

        runtime.init_novelty();

        runtime
    }
}

// private API
impl<'a> Runtime<'a> {
    fn measure_effectiveness(&mut self, progress: &[Progress]) {
        let raw_fitnesses_arr = arr1(
            progress
                .iter()
                .flat_map(|p| p.raw_fitness())
                .collect::<Vec<f64>>()
                .as_slice(),
        );

        // determine standard deviation of agent fitness
        let raw_fitness_std_dev = raw_fitnesses_arr.std_axis(Axis(0), 0.0).as_slice().unwrap()[0];

        self.context.statistics.fitness.raw_std_dev = raw_fitness_std_dev;

        if self.context.statistics.fitness.raw_average < self.context.peak_average_fitness {
            self.context.consecutive_ineffective_generations += 1;
        } else {
            self.context.consecutive_ineffective_generations = 0;
            self.context.peak_average_fitness = self.context.statistics.fitness.raw_average;
        }

        self.context
            .statistics
            .num_consecutive_ineffective_generations =
            self.context.consecutive_ineffective_generations;

        // determine if impatience triggers
        /* if self.context.consecutive_ineffective_generations
            > self.neat.parameters.novelty.impatience
        {
            // determine novelty ratio, capped
            self.context.novelty_ratio = ((self.context.consecutive_ineffective_generations
                - self.neat.parameters.novelty.impatience)
                as f64
                / self.neat.parameters.novelty.impatience as f64)
                .min(self.neat.parameters.novelty.cap);
        } else {
            // on progress jump back to full fitness based search
            self.context.novelty_ratio = 0.0;
        }

        self.context.statistics.novelty_ratio = self.context.novelty_ratio; */
    }

    fn generate_progress(&self) -> Vec<Progress> {
        let progress_fn = &self.neat.progress_function;

        // apply progress function to every individual
        self.population
            .individuals
            .par_iter()
            .map(progress_fn)
            .collect::<Vec<Progress>>()
    }

    fn check_for_solution(&self, progress: &[Progress]) -> Option<Genome> {
        progress
            .iter()
            .filter_map(|p| p.is_solution())
            .cloned()
            .next()
    }

    fn init_novelty(&mut self) {
        let progress = self.generate_progress();
        // init archive threshold to initial fitness peak
        self.assign_novelty(progress.as_slice());
        // set initial archive theshold from sample taken
        self.context.archive_threshold = self.context.statistics.novelty.raw_maximum;
    }

    fn assign_novelty(&mut self, progress: &[Progress]) {
        let behaviors: Vec<&Behavior> = progress
            .iter()
            .flat_map(|progress| progress.behavior())
            .chain(self.archive.iter())
            .collect();

        if behaviors.is_empty() {
            return;
        }

        let width = behaviors[0].len();
        let height = behaviors.len();

        let mut behavior_iter = behaviors.iter();

        let mut behavior_arr: Array2<f64> = Array2::zeros((width, height));
        for mut row in behavior_arr.axis_iter_mut(Axis(1)) {
            row += &ArrayView1::from(behavior_iter.next().unwrap().as_slice());
        }

        let standard_scaler = StandardScaler::new(behavior_arr.view().t());

        let mut z_scores_arr: Array2<f64> = Array2::zeros((width, height));

        for (index, row) in behavior_arr.axis_iter(Axis(1)).enumerate() {
            let mut z_row = z_scores_arr.index_axis_mut(Axis(1), index);
            z_row += &standard_scaler.scale(row);
        }

        let mut add_to_archive = Vec::new();

        let mut raw_minimum = f64::INFINITY;
        let mut raw_sum = 0.0;
        let mut raw_maximum = f64::NEG_INFINITY;

        let mut raw_novelties = Vec::new();

        for (index, z_score) in z_scores_arr
            .axis_iter(Axis(1))
            .enumerate()
            .take(behaviors.len() - self.archive.len())
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

            distances.sort_by(|dist_0, dist_1| {
                dist_0
                    .partial_cmp(&dist_1)
                    .unwrap_or_else(|| panic!("failed to compare {} and {}", dist_0, dist_1))
            });

            // take k nearest neighbors, calculate and assign spareseness
            let sparseness = distances
                .iter()
                .take(self.neat.parameters.novelty.nearest_neighbors)
                .sum::<f64>()
                / self.neat.parameters.novelty.nearest_neighbors as f64;

            raw_novelties.push(sparseness);

            // collect sparseness a.k.a novelty analytics
            if sparseness > raw_maximum {
                raw_maximum = sparseness;
            }
            if sparseness < raw_minimum {
                raw_minimum = sparseness;
            }
            raw_sum += sparseness;

            // add to archive if over threshold
            if sparseness > self.context.archive_threshold {
                add_to_archive.push(Behavior(
                    behavior_arr.index_axis(Axis(1), index).clone().to_vec(),
                ));
            }
        }

        let raw_average = raw_sum / behaviors.len() as f64;

        let baseline = raw_minimum;

        let shifted_minimum = raw_minimum - baseline;
        let shifted_average = raw_average - baseline;
        let shifted_maximum = raw_maximum - baseline;

        let with = shifted_maximum.max(1.0);

        let normalized_minimum = shifted_minimum / with;
        let normalized_average = shifted_average / with;
        let normalized_maximum = shifted_maximum / with;

        for (index, individual) in self.population.individuals.iter_mut().enumerate() {
            individual.novelty = Score::new(raw_novelties[index], baseline, with);
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

        self.context.statistics.novelty.raw_maximum = raw_maximum;
        self.context.statistics.novelty.raw_minimum = raw_minimum;
        self.context.statistics.novelty.raw_average = raw_average;

        self.context.statistics.novelty.shifted_maximum = shifted_maximum;
        self.context.statistics.novelty.shifted_minimum = shifted_minimum;
        self.context.statistics.novelty.shifted_average = shifted_average;

        self.context.statistics.novelty.normalized_maximum = normalized_maximum;
        self.context.statistics.novelty.normalized_minimum = normalized_minimum;
        self.context.statistics.novelty.normalized_average = normalized_average;
    }

    // should be top performer with regard to raw fitness ?
    fn determine_top_performer(&mut self) {
        let pos = self
            .population
            .individuals
            .iter()
            .enumerate()
            .map(|(index, individual)| (index, individual.fitness.raw))
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
    fn assign_fitness(&mut self, progress: &[Progress]) {
        let mut raw_minimum = f64::INFINITY;
        let mut raw_sum = 0.0;
        let mut raw_maximum = f64::NEG_INFINITY;

        let fitnesses: Vec<(usize, f64)> = progress
            .iter()
            .enumerate()
            .flat_map(|(index, progress)| progress.raw_fitness().map(|raw| (index, raw)))
            .collect();

        if fitnesses.is_empty() {
            return;
        }

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

        // let raw_minimum = Raw::fitness(raw_minimum);
        let raw_average = raw_sum / fitnesses.len() as f64;
        // let raw_maximum = Raw::fitness(raw_maximum);

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
            self.population.individuals[index].fitness = Score::new(raw_fitness, baseline, with);
        }
        /*
        for (index, individual) in self.population.individuals.iter_mut().enumerate() {
            individual.fitness =
                FitnessScore::new(progress[index].raw_fitness().value(), baseline, with);
        } */

        self.context.statistics.fitness.raw_maximum = raw_maximum;
        self.context.statistics.fitness.raw_minimum = raw_minimum;
        self.context.statistics.fitness.raw_average = raw_average;

        self.context.statistics.fitness.shifted_maximum = shifted_maximum;
        self.context.statistics.fitness.shifted_minimum = shifted_minimum;
        self.context.statistics.fitness.shifted_average = shifted_average;

        self.context.statistics.fitness.normalized_maximum = normalized_maximum;
        self.context.statistics.fitness.normalized_minimum = normalized_minimum;
        self.context.statistics.fitness.normalized_average = normalized_average;
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

    #[test]
    fn place_into_species() {
        todo!("move to population test")
        /* let mut neat = Neat::new(
            "src/Config.toml",
            Box::new(|_| Progress {
                raw_fitness: Raw::fitness(0.0),
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
        genome_1.fitness.raw = Raw::fitness(1.0);
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
            if fitness < 10.0 {
                Progress::fitness(fitness)
            } else {
                Progress::fitness(fitness).solved(genome.clone())
            }
        }

        let mut neat = Neat::new("src/Config.toml", Box::new(fitness_function));

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
            if fitness < 50.0 {
                Progress::fitness(fitness)
            } else {
                Progress::fitness(fitness).solved(genome.clone())
            }
        }

        let mut neat = Neat::new("src/Config.toml", Box::new(fitness_function));

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
            Progress::fitness(-1.0)
        }

        let neat = Neat::new("src/Config.toml", Box::new(fitness_function));

        let mut runtime = neat.run();

        if let Some(Evaluation::Progress(report)) = runtime.next() {
            assert!(report.fitness.raw_minimum + 1.0 < f64::EPSILON);
            assert!(report.fitness.shifted_minimum.abs() < f64::EPSILON);
        }
    }
}
