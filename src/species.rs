use rand::prelude::SliceRandom;

use crate::individual::{genes::IdGenerator, Individual};
use crate::parameters::Parameters;
use crate::rng::NeatRng;

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
        // set fittest member as new representative
        self.representative = self.members[0].clone();
        // remove all members
        self.members.clear();
    }

    pub fn reproduce<'a>(
        &'a self,
        rng: &'a mut NeatRng,
        id_gen: &'a mut IdGenerator,
        parameters: &'a Parameters,
        offspring: usize,
    ) -> impl Iterator<Item = Individual> + 'a {
        self.members
            .iter()
            .cycle()
            .take(offspring)
            .map(move |member| {
                let mut offspring =
                    member.crossover(self.members.choose(&mut rng.small).unwrap(), &mut rng.small);
                offspring.mutate(rng, id_gen, parameters);
                offspring
            })
            // add top x members to offspring
            .chain(
                self.members
                    .iter()
                    .take(parameters.reproduction.elitism_individuals)
                    .cloned(),
            )
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        individual::{genes::IdGenerator, scores::Score},
        parameters::Parameters,
        rng::NeatRng,
        Individual,
    };

    use super::Species;

    #[test]
    fn order_and_truncate_members() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();
        let parameters: Parameters = Default::default();

        let individual_0 = Individual::initial(&mut id_gen, &parameters);

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
        // create id book-keeping
        let mut id_gen = IdGenerator::default();
        let parameters: Parameters = Default::default();

        let mut individual_0 = Individual::initial(&mut id_gen, &parameters);

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
        // create id book-keeping
        let mut id_gen = IdGenerator::default();
        let parameters: Parameters = Default::default();

        // create random source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let individual_0 = Individual::initial(&mut id_gen, &parameters);

        let species = Species::new(individual_0);

        let expected_offspring = 5;

        let offspring: Vec<Individual> = species
            .reproduce(&mut rng, &mut id_gen, &parameters, expected_offspring)
            .collect();

        assert!(offspring.len() == expected_offspring);
    }

    #[test]
    fn check_reproduction_with_elitism() {
        // create id book-keeping
        let mut id_gen = IdGenerator::default();
        let mut parameters: Parameters = Default::default();

        parameters.reproduction.elitism_individuals = 1;

        // create random source
        let mut rng = NeatRng::new(
            parameters.setup.seed,
            parameters.mutation.weight_perturbation_std_dev,
        );

        let individual_0 = Individual::initial(&mut id_gen, &parameters);

        let species = Species::new(individual_0);

        let expected_offspring = 5;

        let offspring: Vec<Individual> = species
            .reproduce(&mut rng, &mut id_gen, &parameters, expected_offspring)
            .collect();

        assert!(offspring.len() == expected_offspring + 1);
    }
}
