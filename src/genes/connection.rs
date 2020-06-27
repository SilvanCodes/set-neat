use super::Id;
use crate::genes::weights::Weight;
use std::hash::Hash;
use std::hash::Hasher;

use favannat::network::EdgeLike;

use serde::{Deserialize, Serialize};

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
            weight: weight.unwrap_or_default(),
        }
    }

    pub fn id(&self) -> (Id, Id) {
        (self.input, self.output)
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

#[cfg(test)]
mod tests {
    use super::ConnectionGene;
    use crate::genes::Id;
    use std::collections::HashSet;

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
