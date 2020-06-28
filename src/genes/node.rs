use super::activations::{self, Activation};
use super::Id;
use favannat::network::NodeLike;
use std::hash::Hash;
use std::hash::Hasher;

use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Hash, Clone, Debug, Serialize, Deserialize)]
pub enum NodeKind {
    Input,
    Output,
    Hidden,
}

impl Default for NodeKind {
    fn default() -> Self {
        NodeKind::Hidden
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: Id,
    pub kind: NodeKind,
    pub activation: Activation,
}

impl NodeGene {
    pub fn new(id: Id, kind: Option<NodeKind>, activation: Option<Activation>) -> Self {
        NodeGene {
            id,
            kind: kind.unwrap_or_default(),
            activation: activation.unwrap_or_default(),
        }
    }

    pub fn input(id: Id) -> Self {
        NodeGene {
            id,
            kind: NodeKind::Input,
            activation: Default::default(),
        }
    }

    pub fn output(id: Id, activation: Option<Activation>) -> Self {
        NodeGene {
            id,
            kind: NodeKind::Output,
            activation: activation.unwrap_or_default(),
        }
    }

    pub fn is_input(&self) -> bool {
        self.kind == NodeKind::Input
    }

    pub fn is_output(&self) -> bool {
        self.kind == NodeKind::Output
    }

    pub fn update_activation(&mut self, activation: Option<Activation>) {
        activation.map(|activation| self.activation = activation);
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
            Activation::Tanh => activations::TANH,
        }
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
