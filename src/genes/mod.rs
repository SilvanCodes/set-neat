pub mod node;
pub mod connection;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub struct Id(pub usize);