use rand::{prelude::SmallRng, Rng};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, hash::Hash, iter::FromIterator, ops::Deref, ops::DerefMut};

pub mod activations;
mod connection;
pub mod connections;
mod id_generator;
mod node;
pub mod nodes;
mod weights;

pub use activations::Activation;
pub use connection::ConnectionGene;
pub use id_generator::{Id, IdGenerator};
pub use node::NodeGene;
pub use weights::{Weight, WeightDistribution, WeightInitialization, WeightPerturbator};

pub trait Gene {}

impl<U: Gene, T: Deref<Target = U>> Gene for T {}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Genes<T: Gene + Hash + Eq>(pub HashSet<T>);

impl<T: Gene + Hash + Eq> Default for Genes<T> {
    fn default() -> Self {
        Genes(Default::default())
    }
}

impl<T: Gene + Hash + Eq> Deref for Genes<T> {
    type Target = HashSet<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Gene + Hash + Eq> DerefMut for Genes<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Gene + Hash + Eq> Genes<T> {
    pub fn iterate_with_random_offset(&self, rng: &mut SmallRng) -> impl Iterator<Item = &T> {
        self.iter()
            .cycle()
            .skip((rng.gen::<f64>() * self.len() as f64).floor() as usize)
            .take(self.len())
    }
}

impl<T: Gene + Eq + Hash> FromIterator<T> for Genes<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Genes(iter.into_iter().collect())
    }
}

impl<U: Ord, T: Gene + Hash + Eq + Deref<Target = U>> Genes<T> {
    pub fn as_sorted_vec(&self) -> Vec<&U> {
        let mut vec: Vec<&U> = self.iterate_unwrapped().collect();
        vec.sort_unstable();
        vec
    }
}

impl<T: Gene + Eq + Hash> Genes<T> {
    pub fn iterate_matches<'a>(
        &'a self,
        other: &'a Genes<T>,
    ) -> impl Iterator<Item = (&'a T, &'a T)> {
        self.intersection(other)
            // we know item exists in other as we are iterating the intersection
            .map(move |item_self| (item_self, other.get(item_self).unwrap()))
    }

    pub fn iterate_unmatches<'a>(&'a self, other: &'a Genes<T>) -> impl Iterator<Item = &'a T> {
        self.symmetric_difference(other)
    }
}

impl<'a, U: 'a, T: Gene + Hash + Eq + Deref<Target = U>> Genes<T> {
    pub fn iterate_unwrapped(&'a self) -> impl Iterator<Item = &'a U> + Sized + Clone {
        self.iter().map(|value| value.deref())
    }
}
