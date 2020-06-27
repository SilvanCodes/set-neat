// std imports
use std::ops::RangeFrom;
use std::collections::HashMap;
// external imports
use rand::{SeedableRng, Rng};
use rand::rngs::{ThreadRng, SmallRng};
// crate imports
use crate::parameters::Parameters;
use crate::genes::{Id, Perturbator};


pub struct IdIter<'a> {
    index: usize,
    ids: &'a mut Vec<Id>,
    gen: &'a mut RangeFrom<usize>
}

impl<'a> Iterator for IdIter<'a> {
    type Item = Id;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.ids.len() {
            let index = self.index;
            self.index += 1;
            Some(self.ids[index])
        } else {
            let id = self.gen.next().map(Id);
            self.ids.push(id.unwrap());
            id
        }
    }
}

impl<'a> IdIter<'a> {
    pub fn new(ids: &'a mut Vec<Id>, gen: &'a mut RangeFrom<usize>) -> Self {
        IdIter {
            index: 0,
            ids,
            gen
        }
    }
}

pub struct Context {
    // build unique ids
    id_gen: RangeFrom<usize>,
    // keep track of nodes that evolved as result of splitting connection
    cached_node_genes: HashMap<(Id, Id), Vec<Id>>,
    pub compatability_threshold: f64,
    pub last_num_species: usize,
    pub small_rng: SmallRng,
    pub weight_pertubator: Perturbator
}

impl Context {
    pub fn new(parameters: &Parameters) -> Self {
        Context {
            id_gen: 0..,
            cached_node_genes: HashMap::new(),
            compatability_threshold: parameters.compatability.threshold,
            last_num_species: 0,
            small_rng: SmallRng::from_rng(&mut ThreadRng::default()).unwrap(),
            weight_pertubator: Perturbator::new(
                &parameters.mutation.weight_distribution,
                parameters.mutation.weight_perturbation
            )
        }
    }

    pub fn set_id(&mut self, id: usize) {
        self.id_gen = id..;
    }

    pub fn get_id(&mut self) -> Id {
        self.id_gen.next().map(Id).unwrap()
    }

    pub fn gamble(&mut self, chance: f64) -> bool {
        self.small_rng.gen::<f64>() < chance
    }

    pub fn sample(&mut self) -> f64 {
        self.weight_pertubator.sample(&mut self.small_rng)
    }

    // checks if same structure evolved already and return corresponding id
    pub fn get_id_iter(&mut self, connection_id: (Id, Id)) -> IdIter {
        if self.cached_node_genes.contains_key(&connection_id) {
            IdIter::new(self.cached_node_genes.get_mut(&connection_id).unwrap(), &mut self.id_gen)
        } else {
            self.cached_node_genes.insert(connection_id, Vec::new());
            IdIter::new(self.cached_node_genes.get_mut(&connection_id).unwrap(), &mut self.id_gen)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Context;
    use super::IdIter;
    use crate::parameters::Parameters;
    use crate::genes::Id;

    #[test]
    fn get_same_id_for_same_node_mutation() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        let connection_id = (Id(0), Id(1));

        let id_0 = context.get_id_iter(connection_id).next().unwrap();
        let id_1 = context.get_id_iter(connection_id).next().unwrap();

        assert_eq!(id_0, id_1);
    }

    #[test]
    fn id_iter() {
        let mut parameters: Parameters = Default::default();
        parameters.mutation.weight_perturbation = 1.0;
        let mut context = Context::new(&parameters);

        let mut ids = vec![Id(8)];

        let mut id_iter_0 = IdIter::new(&mut ids, &mut context.id_gen);

        assert_eq!(id_iter_0.next(), Some(Id(8)));
        assert_eq!(id_iter_0.next(), Some(Id(0)));
        assert_eq!(ids.last(), Some(&Id(0)));

        let mut ids = vec![Id(9)];

        let mut id_iter_1 = IdIter::new(&mut ids, &mut context.id_gen);

        assert_eq!(id_iter_1.next(), Some(Id(9)));
        assert_eq!(id_iter_1.next(), Some(Id(1)));
        assert_eq!(ids.last(), Some(&Id(1)));
    }
}
