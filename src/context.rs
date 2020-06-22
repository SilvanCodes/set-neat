// std imports
use std::ops::RangeFrom;
use std::collections::HashMap;
// external imports
use rand::{SeedableRng, Rng};
use rand::rngs::{ThreadRng, SmallRng};
// crate imports
use crate::parameters::Parameters;
use crate::genes::Id;


pub struct Context {
    // build unique ids
    id_gen: RangeFrom<usize>,
    // keep track of nodes that evolved as result of splitting connection
    pub cached_node_genes: HashMap<(Id, Id), Id>,
    pub compatability_threshold: f64,
    pub last_num_species: usize,
    pub small_rng: SmallRng
}

impl Context {
    pub fn new(parameters: &Parameters) -> Self {
        Context {
            id_gen: 0..,
            cached_node_genes: HashMap::new(),
            compatability_threshold: parameters.compatability.threshold,
            last_num_species: 0,
            small_rng: SmallRng::from_rng(&mut ThreadRng::default()).unwrap()
        }
    }

    // checks if same structure evolved already and return corresponding id
    pub fn get_id(&mut self, connection_id: Option<(Id, Id)>) -> Id {
        if let Some(connection_id) = connection_id {
            if self.cached_node_genes.contains_key(&connection_id) {
                *self.cached_node_genes.get(&connection_id).unwrap()
            } else {
                let new_id = Id(self.id_gen.next().unwrap());
                self.cached_node_genes.insert(connection_id, new_id);
                new_id
            }
            /* *self.cached_node_genes.entry(connection_id)
                .or_insert(Id(self.id_gen.next().unwrap())) */
        } else {
            Id(self.id_gen.next().unwrap())
        }
    }

    pub fn gamble(&mut self, chance: f64) -> bool {
        self.small_rng.gen::<f64>() < chance
    }
}

#[cfg(test)]
mod tests {
    use crate::parameters::Parameters;
use super::Context;
    use crate::genes::Id;

    #[test]
    fn get_same_id_for_same_node_mutation() {
        let parameters: Parameters = Default::default();
        let mut context = Context::new(&parameters);

        let connection_id = (Id(0), Id(1));

        assert_eq!(context.get_id(Some(connection_id)), context.get_id(Some(connection_id)))
    }

    #[test]
    fn do_not_skip_ids() {
        let parameters: Parameters = Default::default();
        let mut context = Context::new(&parameters);

        let connection_id_0 = (Id(0), Id(1));
        let connection_id_1 = (Id(0), Id(2));

        assert_eq!(context.get_id(Some(connection_id_0)), Id(0));
        assert_eq!(context.get_id(Some(connection_id_0)), context.get_id(Some(connection_id_0)));
        assert_eq!(context.get_id(Some(connection_id_1)), Id(1));
    }
}
