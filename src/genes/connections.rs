use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, hash::Hash, hash::Hasher};

use super::{Gene, Id};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub input: Id,
    pub output: Id,
    pub weight: f64,
}

impl Connection {
    pub fn new(input: Id, weight: f64, output: Id) -> Self {
        Self {
            input,
            output,
            weight,
        }
    }
    pub fn id(&self) -> (Id, Id) {
        (self.input, self.output)
    }
}

impl Gene for Connection {}

impl PartialEq for Connection {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for Connection {}

impl Hash for Connection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

impl PartialOrd for Connection {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Connection {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id().cmp(&other.id())
    }
}
