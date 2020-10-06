use rand::{prelude::SmallRng, Rng};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, hash::Hash, iter::FromIterator, ops::Deref, ops::DerefMut};

pub mod activations;
mod connection;
pub mod connections;
mod node;
pub mod nodes;
mod weights;

pub use activations::Activation;
pub use connection::ConnectionGene;
pub use node::NodeGene;
pub use weights::{Weight, WeightDistribution, WeightInitialization, WeightPerturbator};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Id(pub usize);

#[derive(Debug, Clone)]
pub struct Genes<T>(pub HashSet<T>);

impl<T> Deref for Genes<T> {
    type Target = HashSet<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Genes<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Genes<T> {
    pub fn iterate_with_random_offset(&self, rng: &mut SmallRng) -> impl Iterator<Item = &T> {
        self.iter()
            .cycle()
            .skip((rng.gen::<f64>() * self.len() as f64).floor() as usize)
            .take(self.len())
    }
}

impl<T: Eq + Hash> FromIterator<T> for Genes<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Genes(iter.into_iter().collect())
    }
}

impl<T: Clone + Ord> Genes<T> {
    pub fn as_sorted_vec(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.0.iter().collect();
        vec.sort_unstable();
        vec
    }
}

impl<T: Eq + Hash> Genes<T> {
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

impl<'a, U: Sized + 'a, T: Deref<Target = U>> Genes<T> {
    pub fn iterate_unwrapped(&'a self) -> impl Iterator<Item = &'a U> {
        self.iter().map(|value| value.deref())
    }
}
