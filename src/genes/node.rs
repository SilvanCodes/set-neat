use std::borrow::Borrow;
use std::hash::Hasher;
use std::hash::Hash;
use super::Id;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum NodeKind {
    Input,
    Output,
    Hidden,
}

#[derive(Clone, Debug)]
pub struct NodeGene {
    pub id: Id,
    pub kind: NodeKind,
}

impl NodeGene {
    pub fn new(id: Id, kind: Option<NodeKind>) -> Self {
        NodeGene {
            id,
            kind: kind.unwrap_or(NodeKind::Hidden)
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
