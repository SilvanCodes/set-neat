use rand::{prelude::IteratorRandom, prelude::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, hash::Hash, iter::FromIterator, ops::Deref, ops::DerefMut};

pub mod activations;
pub mod connections;
mod id;
pub mod nodes;

pub use activations::Activation;
pub use id::{id_generator::IdGenerator, Id};

pub trait Gene: Eq + Hash {}

impl<U: Gene, T: Eq + Hash + Deref<Target = U>> Gene for T {}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Genes<T: Gene>(pub HashSet<T>);

impl<T: Gene> Default for Genes<T> {
    fn default() -> Self {
        Genes(Default::default())
    }
}

impl<T: Gene> Deref for Genes<T> {
    type Target = HashSet<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Gene> DerefMut for Genes<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Gene> Genes<T> {
    pub fn iterate_with_random_offset(&self, rng: &mut impl Rng) -> impl Iterator<Item = &T> {
        self.iter()
            .cycle()
            .skip((rng.gen::<f64>() * self.len() as f64).floor() as usize)
            .take(self.len())
    }

    pub fn random(&self, rng: &mut impl Rng) -> Option<&T> {
        self.iter().choose(rng)
    }

    pub fn drain_into_random(&mut self, rng: &mut impl Rng) -> impl Iterator<Item = T> {
        let mut random_vec = self.drain().collect::<Vec<T>>();
        random_vec.shuffle(rng);
        random_vec.into_iter()
    }

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

impl<T: Gene> FromIterator<T> for Genes<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Genes(iter.into_iter().collect())
    }
}

impl<'a, U: 'a, T: Gene + Deref<Target = U>> Genes<T> {
    pub fn iterate_unwrapped(&'a self) -> impl Iterator<Item = &'a U> + Sized + Clone {
        self.iter().map(|value| value.deref())
    }
}

/* impl<U: Ord, T: Gene + Deref<Target = U>> Genes<T> {
    pub fn as_sorted_vec(&self) -> Vec<&U> {
        let mut vec: Vec<&U> = self.iterate_unwrapped().collect();
        vec.sort_unstable();
        vec
    }
} */

impl<T: Gene + Ord> Genes<T> {
    pub fn as_sorted_vec(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.iter().collect();
        vec.sort_unstable();
        vec
    }
}

impl<T: Gene + Clone> Genes<T> {
    pub fn cross_in(&self, other: &Self, rng: &mut impl Rng) -> Self {
        self.iterate_matches(other)
            .map(|(gene_self, gene_other)| {
                if rng.gen::<f64>() < 0.5 {
                    gene_self.clone()
                } else {
                    gene_other.clone()
                }
            })
            .chain(self.difference(other).cloned())
            .collect()
    }
}
