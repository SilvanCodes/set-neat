mod activations;
mod connection;
mod node;
mod weights;

pub use activations::Activation;
pub use connection::ConnectionGene;
pub use node::NodeGene;
pub use weights::{Weight, WeightDistribution, WeightInitialization, WeightPerturbator};

use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct Id(pub usize);
