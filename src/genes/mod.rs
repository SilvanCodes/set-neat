mod node;
mod connection;
mod activations;
mod weights;

pub use node::NodeGene;
pub use connection::ConnectionGene;
pub use weights::{Weight, WeightDistribution, Perturbator};
pub use activations::{Activation, ActivationStrategy};

use serde::{Serialize, Deserialize};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct Id(pub usize);