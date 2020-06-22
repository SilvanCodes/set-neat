pub mod node;
pub mod connection;

use serde::{Serialize, Deserialize};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize)]
pub struct Id(pub usize);