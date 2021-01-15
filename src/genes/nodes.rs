use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};

use super::{Activation, Gene, Id};

pub trait NodeSpecifier {}

pub trait NodeMarker {}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Node(pub Id, pub Activation);

impl NodeMarker for Node {}

impl Node {
    pub fn id(&self) -> Id {
        self.0
    }
}

impl Gene for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialEq<Id> for Node {
    fn eq(&self, other: &Id) -> bool {
        &self.0 == other
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
        self.0.cmp(&other.0)
    }
}

macro_rules! makeNodeSpecifier {
    ( $( $name:ident ),* ) => {
        $(
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
            pub struct $name<T: NodeMarker>(pub T);

            impl<T: NodeMarker> NodeSpecifier for $name<T> {}

            impl<T: NodeMarker> Deref for $name<T> {
                type Target = T;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl<T: NodeMarker> DerefMut for $name<T> {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }
        )*
    };
}

makeNodeSpecifier!(Input, Hidden, Output);
