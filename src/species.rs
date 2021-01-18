use rand::prelude::SliceRandom;

use crate::parameters::Parameters;
use crate::rng::NeatRng;
use crate::{genes::IdGenerator, individual::Individual};

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
    use super::Species;
}
