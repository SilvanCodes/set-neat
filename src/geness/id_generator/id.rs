use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Id(pub usize);
