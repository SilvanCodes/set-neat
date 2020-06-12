use std::mem;
use std::time::Instant;

use rand::distributions::{Distribution, Uniform};
use rand::{Rng};
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
    top_shared_fitness: f64
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
        // println!("placing genome with {} connection genes", genome.connection_genes.len());

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
        // println!("speciate");`   

        // clear population and sort into species
        for genome in mem::replace(&mut self.population, Vec::new()) {
            self.place_genome_into_species(genome);
        }

        // sort members of species by adjusted fitness in descending order
        for species in &mut self.species {
            species.adjust_fitness(&self.neat.parameters);
            species.members.sort_by(|genome_0, genome_1| genome_1.fitness.partial_cmp(&genome_0.fitness).unwrap());
            // dbg!(species.members.iter().map(|m| m.fitness).collect::<Vec<f64>>());
            // place top performing net as representative
            species.represent();
        }

        // check if num species near target species
        if self.species.len() > self.neat.parameters.compatability.target_species {
            self.context.compatability_threshold += self.neat.parameters.compatability.threshold_delta;
        } else if self.species.len() < self.neat.parameters.compatability.target_species {
            self.context.compatability_threshold -= self.neat.parameters.compatability.threshold_delta;
        }

        // sort species by fitness in descending order
        self.species.sort_by(|species_0, species_1| species_1.fitness.partial_cmp(&species_0.fitness).unwrap());

        // dbg!(self.species.iter().map(|s| s.fitness).collect::<Vec<f64>>());

        self.statistics.top_shared_fitness = self.species[0].representative.as_ref().unwrap().fitness;
    }

    fn evaluate(&mut self) -> Option<Genome> {
        let now = Instant::now();

        // evaluate nets in parallel
        let fitnesses: Vec<f64> = self.population.par_iter().map(self.neat.fitness_function).collect();

        // check if some net is sufficient
        if let Some(sufficient) = fitnesses.iter().enumerate().find_map(|(i, &x)| if x > self.neat.required_fitness { Some(i) } else { None }) {
            return Some(self.population[sufficient].clone());
        }

        // set fitnes for every genome
        for (index, &fitness) in fitnesses.iter().enumerate() {
            self.population[index].fitness = fitness;
        }

        self.statistics.milliseconds_elapsed_evaluation = now.elapsed().as_millis();
        None
    }

    fn reproduce(&mut self) {
        // println!("reproduce new generation");
        self.statistics.num_generation += 1;

        let now = Instant::now();

        // place genomes in species
        self.speciate();

        // collect statistics
        self.statistics.milliseconds_elapsed_speciation = now.elapsed().as_millis();
        self.statistics.num_species = self.species.len();

        // clear stale species
        let threshold = self.neat.parameters.staleness.after;
        self.species.retain(|species| species.stale < threshold);

        // calculate offspring factor
        let total_fitness = self.species.iter().fold(0.0, |sum, species| sum + species.fitness);
        // expresses how much offspring one point of fitness is worth
        let offspring_ratio = self.neat.parameters.setup.population as f64 / total_fitness;
        // remove species that do not qualify for offspring
        self.species.retain(|species| species.fitness * offspring_ratio >= 1.0);
        

        // println!("reproducing species: {}", self.species.len());

        let species_range = Uniform::from(0..self.species.len());

        let now = Instant::now();

        for species in &self.species {
            // collect members that are allowed to reproduce
            let allowed_member_count = (species.members.len() as f64 * self.neat.parameters.reproduction.surviving).ceil() as usize;

            // stale species only top member shall reproduce
            /* if species.stale > self.neat.parameters.staleness.after {
                println!("##################### species gone stale! #####################");
                allowed_member_count = 1;
                self.statistics.num_species_stale += 1;
            } */

            let allowed_members: Vec<&Genome> = species.members.iter().take(allowed_member_count).collect();

            let member_range = Uniform::from(0..allowed_members.len());
            // calculate offspring count for species
            let offspring_count = (species.fitness * offspring_ratio).round() as usize;
            let offspring_from_crossover_count = (offspring_count as f64 * self.neat.parameters.reproduction.offspring_from_crossover).round() as usize;

            let mut offspring = Vec::new();

            // keep top performing members unchanged when qualified
            /* if species.members.len() > self.neat.parameters.reproduction.champions_minimal_species_size {
                offspring.extend(species.members.iter().take(self.neat.parameters.reproduction.champions).cloned());
            } */

            // cycle members to produce offspring
            for member in allowed_members.iter().cycle() {
                if offspring.len() < offspring_from_crossover_count {
                    if self.context.small_rng.gen::<f64>() < self.neat.parameters.reproduction.offspring_from_crossover_interspecies {
                        // produce offspring from crossover with other species
                        let mut new_offspring = member.crossover(
                            &self.species[species_range.sample(&mut self.context.small_rng)].representative.as_ref().unwrap(),
                            &mut self.context
                        );

                        // collect statistics
                        self.statistics.num_offspring_from_crossover_interspecies += 1;

                        new_offspring.mutate(&mut self.context, &self.neat.parameters);
                        offspring.push(new_offspring);
                    } else {
                        // produce offspring from crossover
                        let mut new_offspring = member.crossover(
                            &species.members[member_range.sample(&mut self.context.small_rng)],
                            &mut self.context
                        );

                        // collect statistics
                        self.statistics.num_offspring_from_crossover += 1;

                        new_offspring.mutate(&mut self.context, &self.neat.parameters);
                        offspring.push(new_offspring);
                    }
                } else if offspring.len() < offspring_count {
                    // create offspring just by mutating
                    let mut new_offspring = (*member).clone();

                    new_offspring.mutate(&mut self.context, &self.neat.parameters);
                    offspring.push(new_offspring);
                } else {
                    // all offspring created
                    break;
                }
                // collect statistics
                self.statistics.num_offpring += 1;
            }
            // place offspring in population
            self.population.append(&mut offspring);
        }

        // collect statistics
        self.statistics.milliseconds_elapsed_reproducing = now.elapsed().as_millis();

        // println!("next population size: {:?}", self.population.len());

        // clear old species members
        for species in &mut self.species {
            species.members = Vec::new();
        }
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
        let neat = Neat::new("src/Config.toml", |_| 0.0, 0.0);

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
        assert_eq!(runtime.species[0].representative.as_ref().unwrap().fitness, 1.0);
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
