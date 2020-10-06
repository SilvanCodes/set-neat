use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    iter::FromIterator,
    ops::{Deref, DerefMut},
};

use super::{Activation, Id};

pub trait NodeType {}

pub trait NodeValue {}

#[derive(Debug, Copy, Clone)]
pub struct Node(pub Id, pub Activation);

impl NodeValue for Node {}

impl Node {
    pub fn id(&self) -> Id {
        self.0
    }
}

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
        Some(self.0.cmp(&other.0))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

macro_rules! makeNodeType {
    ( $( $name:ident ),* ) => {
        $(
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
            pub struct $name<T: NodeValue>(T);

            impl<T: NodeValue> NodeType for $name<T> {}

            impl<T: NodeValue> Deref for $name<T> {
                type Target = T;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl<T: NodeValue> DerefMut for $name<T> {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }

            impl<'a, T: NodeValue> FromIterator<&'a $name<T>> for Vec<&'a T> {
                fn from_iter<I: IntoIterator<Item=&'a $name<T>>>(iter: I) -> Self {
                    iter
                    .into_iter()
                    .map(|i| &i.0)
                    .collect()
                }
            }
        )*
    };
}

makeNodeType!(Input, Hidden, Output);
