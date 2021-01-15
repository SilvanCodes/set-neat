use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    hash::Hash,
    hash::Hasher,
    ops::{Deref, DerefMut},
};

use super::{Gene, Id, Weight};

pub trait ConnectionSpecifier {}

pub trait ConnectionMarker {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection(pub Id, pub Weight, pub Id);

impl ConnectionMarker for Connection {}

impl Connection {
    pub fn id(&self) -> (Id, Id) {
        (self.0, self.2)
    }
    pub fn input(&self) -> Id {
        self.0
    }
    pub fn output(&self) -> Id {
        self.2
    }
    pub fn weight(&mut self) -> &mut Weight {
        &mut self.1
    }
}
impl Gene for Connection {}

impl PartialEq for Connection {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.2 == other.2
    }
}

impl Eq for Connection {}

impl Hash for Connection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0, self.2).hash(state);
    }
}

impl PartialOrd for Connection {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Connection {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0).then(self.2.cmp(&other.2))
    }
}

macro_rules! makeConnectionSpecifier {
    ( $( $name:ident ),* ) => {
        $(
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
            pub struct $name<T: ConnectionMarker>(pub T);

            impl<T: ConnectionMarker> ConnectionSpecifier for $name<T> {}

            impl<T: ConnectionMarker> Deref for $name<T> {
                type Target = T;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl<T: ConnectionMarker> DerefMut for $name<T> {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }
        )*
    };
}

makeConnectionSpecifier!(FeedForward, Recurrent);
