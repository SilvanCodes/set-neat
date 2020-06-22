use rand::rngs::SmallRng;
use rand::Rng;
use std::hash::Hasher;
use std::hash::Hash;
use rand::random;
use super::Id;

use favannat::network::EdgeLike;

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub input: Id,
    pub output: Id,
    pub weight: Weight,
}

impl ConnectionGene {
    pub fn new(input: Id, output: Id, weight: Option<Weight>) -> Self {
        ConnectionGene {
            input,
            output,
            weight: weight.unwrap_or_else(Weight::new),
        }
    }
}

impl EdgeLike for ConnectionGene {
    fn start(&self) -> usize {
        self.input.0
    }
    fn end(&self) -> usize {
        self.output.0
    }
    fn weight(&self) -> f64 {
        self.weight.0
    }
}

impl PartialEq for ConnectionGene {
    fn eq(&self, other: &Self) -> bool {
        self.input.0 == other.input.0 && self.output.0 == other.output.0
    }
}

impl Eq for ConnectionGene {}

impl Hash for ConnectionGene {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.input.0, self.output.0).hash(state);
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Weight(pub f64);

impl Weight {
    pub fn new() -> Self {
        Weight(random::<f64>() * 2.0 - 1.0)
    }

    pub fn difference(&self, other: &Weight) -> f64 {
        (self.0 - other.0).abs()
    }

    #[inline]
    pub fn perturbate(&mut self, rng: &mut SmallRng, range: f64) {
        self.0 = self.0 + rng.gen::<f64>() * range * 2.0 - range;
    }

    #[inline]
    pub fn random(&mut self, rng: &mut SmallRng) {
        self.0 = rng.gen::<f64>() * 2.0 - 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::Weight;
    use super::ConnectionGene;
    use crate::genes::Id;
    use std::collections::HashSet;

    #[test]
    fn generate_random_weight() {
        for _ in 0..100 {
            let Weight(w) = Weight::new();
            assert!(w < 1.0)
        }
    }

    #[test]
    fn eq_is_hash() {
        let mut set = HashSet::new();

        let connection_gene_0 = ConnectionGene::new(Id(0), Id(1), None);
        let connection_gene_1 = ConnectionGene::new(Id(0), Id(1), None);
        let connection_gene_2 = ConnectionGene::new(Id(1), Id(1), None);

        assert_eq!(connection_gene_0, connection_gene_1);
        assert_ne!(connection_gene_0, connection_gene_2);
        assert_ne!(connection_gene_1, connection_gene_2);

        assert!(set.insert(connection_gene_0));
        assert!(!set.insert(connection_gene_1));
        assert!(set.insert(connection_gene_2));
    }
}
