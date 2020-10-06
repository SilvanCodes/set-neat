use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    hash::Hash,
    hash::Hasher,
    ops::{Deref, DerefMut},
};

use super::{Id, Weight};

pub trait ConnectionType {}

pub trait ConnectionValue {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection(pub Id, pub Weight, pub Id);

impl ConnectionValue for Connection {}

impl Connection {
    pub fn id(&self) -> (Id, Id) {
        (self.0, self.2)
    }
}

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

macro_rules! makeConnectionType {
    ( $( $name:ident ),* ) => {
        $(
            #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
            pub struct $name<T: ConnectionValue>(pub T);

            impl<T: ConnectionValue> ConnectionType for $name<T> {}

            impl<T: ConnectionValue> Deref for $name<T> {
                type Target = T;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl<T: ConnectionValue> DerefMut for $name<T> {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }
        )*
    };
}

makeConnectionType!(FeedForward, Recurrent);
