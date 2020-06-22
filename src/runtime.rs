use std::mem;
use std::time::Instant;

use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;

use crate::species::Species;
use crate::genome::Genome;
use crate::context::Context;
use crate::Neat;

#[derive(Debug, Clone, Copy, Default)]
pub struct Report {
    num_generation: usize,
    num_species: usize,
    num_species_stale: usize,
    num_offpring: usize,
    num_offspring_from_crossover: usize,
    num_offspring_from_crossover_interspecies: usize,
    milliseconds_elapsed_reproducing: u128,
    milliseconds_elapsed_speciation: u128,
    milliseconds_elapsed_evaluation: u128,
    top_fitness: f64
}

pub struct Runtime<'a> {
    neat: &'a Neat,
    context: Context,
    population: Vec<Genome>,
    species: Vec<Species>,
    statistics: Report
}

pub enum Evaluation {
    Progress(Report),
    Solution(Genome),
}

impl<'a> Iterator for Runtime<'a> {
    type Item = Evaluation;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(genome) = self.evaluate() {
            Some(Evaluation::Solution(genome))
        } else {
            self.reset_generational_statistics();
            self.speciate();
            self.reproduce();
            Some(Evaluation::Progress(self.statistics))
        }
    }
}

impl<'a> Runtime<'a> {
    pub fn new(neat: &'a Neat) -> Self {
        let mut context = Context::new(&neat.parameters);

        let mut initial_genome = Genome::new(&mut context, &neat.parameters);
        // setup fully connected input -> output
        initial_genome.init();

        let mut population = Vec::with_capacity(neat.parameters.setup.population);

        // build initial population
        for _ in 0..neat.parameters.setup.population {
            let mut other_genome = Genome::from(&initial_genome);
            other_genome.mutate(&mut context, &neat.parameters);
            population.push(other_genome);
        }

        context.cached_node_genes.clear();

        Runtime {
            neat,
            context,
            population,
            species: Vec::new(),
            statistics: Default::default()
        }
    }
}

// private API
impl<'a> Runtime<'a> {
    fn place_genome_into_species(&mut self, genome: Genome) {
        let mut matching_species = None;

        for species in &mut self.species {
            // find matching species
            if species.compatible(&genome, &self.neat.parameters, &self.context) {
                matching_species = Some(species);
                break;
            }
        }

        // place into matching species
        if let Some(species) = matching_species {
            species.members.push(genome);
        }
        // or open new species
        else {
            self.species.push(Species::new(genome));
        }
    }

    fn speciate(&mut self) {
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
        self.species.retain(|species| species.stale < threshold);

        // sort species by fitness in descending order
        self.species.sort_by(|species_0, species_1| species_1.fitness.partial_cmp(&species_0.fitness).unwrap());

        // check if num species near target species
        if self.species.len() > parameters.compatability.target_species
            && self.species.len() >= context.last_num_species {
                context.compatability_threshold += parameters.compatability.threshold_delta;
        }
        else if self.species.len() < parameters.compatability.target_species
            && self.species.len() <= context.last_num_species {
                context.compatability_threshold -= parameters.compatability.threshold_delta;
        }

        // remember number of species of last generation
        context.last_num_species = self.species.len();

        // collect statistics
        self.statistics.milliseconds_elapsed_speciation = now.elapsed().as_millis();
    }

    fn evaluate(&mut self) -> Option<Genome> {
        let now = Instant::now();

        // evaluate nets in parallel
        let mut fitnesses: Vec<f64> = self.population.par_iter().map(self.neat.fitness_function).collect();

        dbg!(&fitnesses);

        // check if some net is sufficient
        if let Some((index, &fitness)) = fitnesses.iter().enumerate().find(|(_, &x)| x >= self.neat.required_fitness) {
            let mut winner = self.population[index].clone();
            winner.fitness = fitness;
            return Some(winner);
        }

        let maximum = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

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
        self.statistics.top_fitness = maximum;
        None
    }

    fn reproduce(&mut self) {
        let now = Instant::now();
        let context = &mut self.context;
        let parameters = &self.neat.parameters;
        let statistics = &mut self.statistics;
        let offspring_ratio; // expresses how much offspring one point of fitness is worth

        let total_fitness = self.species.iter().fold(0.0, |sum, species| sum + species.fitness);

        if total_fitness == 0.0 {
            // remove species with no members
            self.species.retain(|species| species.members.len() > 0);
            offspring_ratio = self.neat.parameters.setup.population as f64 / self.species.len() as f64;
        } else {
            offspring_ratio = self.neat.parameters.setup.population as f64 / total_fitness;
            // remove species that do not qualify for offspring
            self.species.retain(|species| species.fitness * offspring_ratio >= 1.0);
        }

        let all_species = &self.species;
        let species_range = Uniform::from(0..all_species.len());

        for species in all_species {
            // collect members that are allowed to reproduce (need to ceil due to use in uniform distribution)
            let allowed_member_count = (species.members.len() as f64 * parameters.reproduction.surviving).ceil() as usize;
            let member_range = Uniform::from(0..allowed_member_count);

            // calculate offspring count for species
            let offspring_count = if total_fitness == 0.0 {
                offspring_ratio
            } else {
                species.fitness * offspring_ratio
            }.round() as usize;

            let offspring_from_crossover_count = (offspring_count as f64 * parameters.reproduction.offspring_from_crossover).round() as usize;

            self.population.extend(
                species.members
                    .iter()
                    .take(allowed_member_count)
                    .cycle()
                    .take(offspring_count)
                    .enumerate()
                    .map(|(count, member)| {
                        if count < offspring_from_crossover_count {
                            if context.gamble(parameters.reproduction.offspring_from_crossover_interspecies) {
                                statistics.num_offspring_from_crossover_interspecies += 1;
                                member.crossover(
                                    &all_species[species_range.sample(&mut context.small_rng)].members[0],
                                    context
                                )
                            } else {
                                statistics.num_offspring_from_crossover += 1;
                                member.crossover(
                                    &species.members[member_range.sample(&mut context.small_rng)],
                                    context
                                )
                            }
                        } else {
                            member.clone()
                        }
                    })
            );
        }

        // mutate the new population
        for genome in &mut self.population {
            genome.mutate(context, parameters);
        }

        // clear species members
        for species in &mut self.species {
            species.members.clear();
        }

        // reset cache to not limit node mutations
        self.context.cached_node_genes.clear();

        // collect statistics
        self.statistics.num_generation += 1;
        self.statistics.num_offpring = self.population.len();
        self.statistics.num_species = self.species.len();
        self.statistics.milliseconds_elapsed_reproducing = now.elapsed().as_millis();
    }

    fn reset_generational_statistics(&mut self) {
        self.statistics.num_offpring = 0;
        self.statistics.num_offspring_from_crossover = 0;
        self.statistics.num_offspring_from_crossover_interspecies = 0;
        self.statistics.num_species_stale = 0;
    }
}


#[cfg(test)]
mod tests {
    use crate::runtime::Evaluation::{Progress, Solution};
    use super::Neat;
    use crate::genome::Genome;

    #[test]
    fn place_into_species() {
        let mut neat = Neat::new("src/Config.toml", |_| 0.0, 0.0);

        neat.parameters.compatability.threshold = 3.0;
        neat.parameters.reproduction.surviving = 1.0;

        let mut runtime = neat.run();

        let mut genome_0 = Genome::new(&mut runtime.context, &runtime.neat.parameters);

        genome_0.init();

        let mut genome_1 = Genome::from(&genome_0);

        // manipulate weight for genome distance to exceed threshold
        let mut connection_gene_0 = genome_1.connection_genes.iter().nth(0).unwrap().clone();
        connection_gene_0.weight.0 = 15.0;
        genome_0.connection_genes.replace(connection_gene_0);

        let mut connection_gene_1 = genome_1.connection_genes.iter().nth(0).unwrap().clone();
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
        assert_eq!(runtime.species[0].fitness, 1.0);
        assert_eq!(runtime.population.len(), 0);
    }

    #[test]
    fn species_sorted_descending() {
        let mut neat = Neat::new("src/Config.toml", |_| 0.0, 0.0);

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
        assert_eq!(runtime.species[0].members[0].fitness, 1.0);
        // should have averaged fitness
        assert_eq!(runtime.species[0].fitness, 1.5);
    }

    #[test]
    fn run_neat_till_10_connections() {

        fn fitness_function(genome: &Genome) -> f64 {
            genome.connection_genes.len() as f64
        }

        let neat = Neat::new("src/Config.toml", fitness_function, 10.0);

        if let Some(winner) = neat.run().filter_map(|evaluation| {
            match evaluation {
                Progress(_) => None,
                Solution(genome) => Some(genome)
            }
        }).next() {
            assert!((winner.connection_genes.len() as i64 - 10) >= 0);
        }
    }

    #[test]
    fn run_neat_till_50_connections() {
        fn fitness_function(genome: &Genome) -> f64 {
            genome.connection_genes.len() as f64
        }

        let neat = Neat::new("src/Config.toml", fitness_function, 50.0);

        if let Some(winner) = neat.run().filter_map(|evaluation| {
            match evaluation {
                Progress(_) => None,
                Solution(genome) => Some(genome)
            }
        }).next() {
            assert!((winner.connection_genes.len() as i64 - 50) >= 0);
        }
    }
}
