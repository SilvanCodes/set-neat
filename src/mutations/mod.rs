use crate::Genome;

// needed mutations:
// - weight mutation
// - node mutation
// - connection mutation

trait Mutation {
    fn mutate(&mut self, genome: &mut Genome);
}

pub struct Mutations {
    mutations: Vec<Box<dyn Mutation>>,
}
