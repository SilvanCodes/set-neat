use crate::individual::{behavior::Behavior, Individual};

#[derive(Debug)]
pub enum Progress {
    Empty,
    Fitness(f64),
    Novelty(Behavior),
    Status(f64, Behavior),
    Solution(Option<f64>, Option<Behavior>, Box<Individual>),
}

impl Progress {
    pub fn new(fitness: f64, behavior: Vec<f64>) -> Self {
        Progress::Status(fitness, Behavior(behavior))
    }

    pub fn empty() -> Self {
        Progress::Empty
    }

    pub fn solved(self, genome: Individual) -> Self {
        match self {
            Progress::Fitness(fitness) => Progress::Solution(Some(fitness), None, Box::new(genome)),
            Progress::Novelty(behavior) => {
                Progress::Solution(None, Some(behavior), Box::new(genome))
            }
            Progress::Status(fitness, behavior) => {
                Progress::Solution(Some(fitness), Some(behavior), Box::new(genome))
            }
            Progress::Solution(fitness, behavior, _) => {
                Progress::Solution(fitness, behavior, Box::new(genome))
            }
            Progress::Empty => Progress::Solution(None, None, Box::new(genome)),
        }
    }

    pub fn fitness(fitness: f64) -> Self {
        Self::Fitness(fitness)
    }

    pub fn novelty(behavior: Vec<f64>) -> Self {
        Self::Novelty(Behavior(behavior))
    }

    pub fn behavior(&self) -> Option<&Behavior> {
        match self {
            Progress::Status(_, behavior) => Some(behavior),
            Progress::Solution(_, behavior, _) => behavior.as_ref(),
            Progress::Novelty(behavior) => Some(behavior),
            Progress::Fitness(_) => None,
            Progress::Empty => None,
        }
    }

    pub fn raw_fitness(&self) -> Option<f64> {
        match *self {
            Progress::Status(fitness, _) => Some(fitness),
            Progress::Solution(fitness, _, _) => fitness,
            Progress::Fitness(fitness) => Some(fitness),
            Progress::Novelty(_) => None,
            Progress::Empty => None,
        }
    }

    pub fn is_solution(&self) -> Option<&Individual> {
        match self {
            Progress::Solution(_, _, genome) => Some(genome),
            _ => None,
        }
    }
}
