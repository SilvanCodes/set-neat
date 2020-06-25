use favannat::network::NodeLike;
use std::borrow::Borrow;
use std::hash::Hasher;
use std::hash::Hash;
use super::Id;
use super::activations::{self, Activation};


use serde::{Serialize, Deserialize};

#[derive(PartialEq, Eq, Hash, Clone, Debug, Serialize, Deserialize)]
pub enum NodeKind {
    Input,
    Output,
    Hidden,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: Id,
    pub kind: NodeKind,
    pub activation: Activation
}

impl NodeGene {
    pub fn new(id: Id, kind: Option<NodeKind>, activation: Option<Activation>) -> Self {
        NodeGene {
            id,
            kind: kind.unwrap_or(NodeKind::Hidden),
            activation: activation.unwrap_or_default()
        }
    }
}

impl NodeLike for NodeGene {
    fn id(&self) -> usize {
        self.id.0
    }
    fn activation(&self) -> fn(f64) -> f64 {
        match self.activation {
            Activation::Linear => activations::LINEAR,
            Activation::Sigmoid => activations::SIGMOID,
            Activation::Gaussian => activations::GAUSSIAN,
            Activation::Tanh => activations::TANH
        }
    }
}

impl Borrow<Id> for &NodeGene {
    fn borrow(&self) -> &Id {
        &self.id
    }
}

impl Hash for NodeGene {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for NodeGene {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialEq<Id> for NodeGene {
    fn eq(&self, other: &Id) -> bool {
        self.id == *other
    }
}

impl Eq for NodeGene {}
