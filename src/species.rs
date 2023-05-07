use rand::{prelude::SliceRandom, Rng};

use crate::{individual::Individual, parameters::Reproduction};
use set_genome::Parameters as GenomeParameters;

#[derive(Debug, Clone)]
pub struct Species {
    pub representative: Individual,
    pub members: Vec<Individual>,
    pub score: f64,
    pub stale: usize,
}

// public API
impl Species {
    pub fn new(first_member: Individual) -> Self {
        Species {
            representative: first_member.clone(),
            members: vec![first_member],
            score: 0.0,
            stale: 0,
        }
    }

    fn order_surviving_members(&mut self, survival_rate: f64) {
        // sort members by descending score, i.e. fittest first
        self.members
            .sort_by(|genome_0, genome_1| genome_1.score().partial_cmp(&genome_0.score()).unwrap());
        // reduce to surviving members
        self.members
            .truncate((self.members.len() as f64 * survival_rate).ceil() as usize);
    }

    fn compute_score(&mut self) {
        let factor = self.members.len() as f64;
        let old_score = self.score;

        // we set the species fitness as the average of the members
        self.score = self
            .members
            .iter()
            .map(|member| member.score() / factor)
            .sum();

        // did score increase ?
        if self.score > old_score {
            self.stale = 0;
        } else {
            self.stale += 1;
        }
    }

    pub fn adjust(&mut self, survival_rate: f64) {
        self.order_surviving_members(survival_rate);
        self.compute_score();
    }

    pub fn prepare_next_generation(&mut self) {
        // remove all members
        self.members.clear();
    }

    pub fn reproduce<'a>(
        &'a self,
        rng: &'a mut impl Rng,
        genome_parameters: &'a GenomeParameters,
        reproduction: &'a Reproduction,
        offspring: usize,
    ) -> impl Iterator<Item = Individual> + 'a {
        self.members
            .iter()
            .cycle()
            // produce as many offspring from crossover and mutation as individual elitism taking into account actual species length allows
            .take((offspring - reproduction.elitism_individuals.min(self.members.len())).max(0))
            .map(move |member| {
                let mut offspring = member.crossover(self.members.choose(rng).unwrap());
                offspring.mutate(genome_parameters);
                offspring
            })
            // add as many members to offspring as specified individual elitism taking into account actual species length
            .chain(
                self.members
                    .iter()
                    .take(
                        reproduction
                            .elitism_individuals
                            .min(self.members.len())
                            .min(offspring),
                    )
                    .cloned(),
            )
    }
}

#[cfg(test)]
mod tests {

    use crate::{individual::scores::Score, parameters::Reproduction, Individual};
    use rand::thread_rng;
    use set_genome::{Genome, Parameters as GenomeParameters};

    use super::Species;

    #[test]
    fn order_and_truncate_members() {
        let parameters = GenomeParameters::default();

        let individual_0 = Individual::from_genome(Genome::initialized(&parameters));

        let mut individual_1 = individual_0.clone();
        individual_1.fitness = Score::new(5.0, 0.0, 1.0);

        let mut species = Species::new(individual_0);
        species.members.extend(vec![individual_1]);
        species.order_surviving_members(0.3);

        assert!((species.members[0].fitness.raw - 5.0).abs() < f64::EPSILON);
        assert!(species.members.len() == 1);
    }

    #[test]
    fn check_score_and_staleness() {
        let parameters = GenomeParameters::default();

        let mut individual_0 = Individual::from_genome(Genome::initialized(&parameters));

        individual_0.fitness = Score::new(5.0, 0.0, 1.0);

        let mut species = Species::new(individual_0.clone());

        species.members.push(individual_0.clone());

        species.compute_score();
        assert!((species.score - 5.0).abs() < f64::EPSILON);

        species.compute_score();
        assert!(species.stale == 1);

        species.compute_score();
        assert!(species.stale == 2);

        individual_0.fitness = Score::new(15.0, 0.0, 1.0);

        species.members.push(individual_0);

        species.compute_score();
        assert!(species.stale == 0);
    }

    #[test]
    fn check_reproduction() {
        let parameters = GenomeParameters::default();

        let reproduction = Reproduction {
            survival_rate: 0.2,
            generations_until_stale: 10,
            elitism_species: 1,
            elitism_individuals: 0,
        };

        let individual = Individual::from_genome(Genome::initialized(&parameters));

        let species = Species::new(individual);

        let expected_offspring = 5;

        let offspring: Vec<Individual> = species
            .reproduce(
                &mut thread_rng(),
                &parameters,
                &reproduction,
                expected_offspring,
            )
            .collect();

        assert!(offspring.len() == expected_offspring);
    }

    #[test]
    fn check_reproduction_with_elitism() {
        let parameters = GenomeParameters::default();

        let reproduction = Reproduction {
            survival_rate: 0.2,
            generations_until_stale: 10,
            elitism_species: 1,
            elitism_individuals: 3,
        };

        let individual = Individual::from_genome(Genome::initialized(&parameters));

        let species = Species::new(individual);

        let expected_offspring = 5;

        let offspring: Vec<Individual> = species
            .reproduce(
                &mut thread_rng(),
                &parameters,
                &reproduction,
                expected_offspring,
            )
            .collect();

        assert_eq!(offspring.len(), expected_offspring);
    }
}
