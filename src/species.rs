use rand::prelude::SliceRandom;

use crate::context::Context;
use crate::genome::Genome;
use crate::parameters::Parameters;

#[derive(Debug, Clone)]
pub struct Species {
    pub representative: Genome,
    pub members: Vec<Genome>,
    pub score: f64,
    pub stale: usize,
}

// public API
impl Species {
    pub fn new(first_member: Genome) -> Self {
        Species {
            representative: first_member.clone(),
            members: vec![first_member],
            score: 0.0,
            stale: 0,
        }
    }

    fn order_surviving_members(&mut self, context: &Context, parameters: &Parameters) {
        // sort members by descending score, i.e. fittest first
        self.members.sort_by(|genome_0, genome_1| {
            genome_1
                .score(context)
                .partial_cmp(&genome_0.score(context))
                .unwrap()
        });
        // reduce to surviving members
        self.members.truncate(
            (self.members.len() as f64 * parameters.reproduction.surviving).ceil() as usize,
        );

        dbg!(self.members.len());
    }

    fn compute_score(&mut self, context: &Context) {
        let factor = self.members.len() as f64;
        let old_score = self.score;

        // we set the species fitness as the average of the members
        self.score = self
            .members
            .iter()
            .map(|member| member.score(context) / factor)
            .sum();

        // did score increase ?
        if self.score > old_score {
            self.stale = 0;
        } else {
            self.stale += 1;
        }
    }

    pub fn adjust(&mut self, context: &Context, parameters: &Parameters) {
        self.order_surviving_members(context, parameters);
        self.compute_score(context);
    }

    pub fn prepare_next_generation(&mut self) {
        // set fittest member as new representative
        self.representative = self.members[0].clone();
        // remove all members
        self.members.clear();
    }

    pub fn reproduce<'a>(
        &'a self,
        context: &'a mut Context,
        parameters: &'a Parameters,
        offspring: usize,
    ) -> impl Iterator<Item = Genome> + 'a {
        self.members
            .iter()
            .cycle()
            .take(offspring)
            .map(move |member| {
                let mut offspring = member.crossover(
                    self.members.choose(&mut context.small_rng).unwrap(),
                    context,
                );
                offspring.mutate(context, parameters);
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
    use crate::genome::Genome;
    use crate::{
        activations::Activation,
        genes::{
            connections::{Connection, FeedForward},
            nodes::{Input, Node, Output},
            Genes, Id, Weight,
        },
    };
}
