use std::time::{Instant, SystemTime};

use rayon::prelude::*;

use crate::{individual::Individual, population::Population, statistics::Statistics, Neat};

pub use self::progress::Progress;

mod progress;

pub struct Runtime<'a> {
    neat: &'a Neat,
    population: Population,
    statistics: Statistics,
}

impl<'a> Runtime<'a> {
    pub fn new(neat: &'a Neat) -> Self {
        Self {
            neat,
            population: Population::new(neat.parameters.clone()),
            statistics: Statistics::default(),
        }
    }

    fn generate_progress(&self) -> Vec<Progress> {
        let progress_fn = &self.neat.progress_function;

        // apply progress function to every individual
        self.population
            .individuals()
            .par_iter()
            .map(progress_fn)
            .collect::<Vec<Progress>>()
    }

    fn check_for_solution(&self, progress: &[Progress]) -> Option<Individual> {
        progress
            .iter()
            .filter_map(|p| p.is_solution())
            .cloned()
            .next()
    }
}

impl<'a> Iterator for Runtime<'a> {
    type Item = (Statistics, Option<Individual>);

    fn next(&mut self) -> Option<Self::Item> {
        self.statistics.time_stamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let now = Instant::now();

        // generate progress by running progress function for every individual
        let progress = self.generate_progress();

        self.statistics.milliseconds_elapsed_evaluation = now.elapsed().as_millis();

        // collect statistics and prepare next generation
        self.statistics.population = self.population.next_generation(&progress);

        Some((self.statistics.clone(), self.check_for_solution(&progress)))
    }
}

#[cfg(test)]
mod tests {
    /* use super::Neat;
    use crate::individual::Individual;
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
    } */
}
