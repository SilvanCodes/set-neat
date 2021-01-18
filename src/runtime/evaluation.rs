use crate::{individual::Individual, statistics::Statistics};

pub enum Evaluation {
    Progress(Statistics),
    Solution(Individual),
}
