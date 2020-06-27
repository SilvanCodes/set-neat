mod activations;
mod connection;
mod node;
mod weights;

pub use activations::{Activation, ActivationStrategy};
pub use connection::ConnectionGene;
pub use node::NodeGene;
pub use weights::{Perturbator, Weight, WeightDistribution};

use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct Id(pub usize);
