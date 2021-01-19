use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use super::{Activation, Gene, Id};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Node {
    pub id: Id,
    pub activation: Activation,
}

impl Node {
    pub fn new(id: Id, activation: Activation) -> Self {
        Node { id, activation }
    }
}

impl Gene for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialEq<Id> for Node {
    fn eq(&self, other: &Id) -> bool {
        &self.id == other
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}
