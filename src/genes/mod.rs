pub mod node;
pub mod connection;
pub mod activations;
pub mod weights;

use serde::{Serialize, Deserialize};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct Id(pub usize);