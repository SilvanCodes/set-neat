use std::{collections::HashMap, ops::RangeFrom};

use super::{id_iter::IdIter, Id};

#[derive(Debug)]
pub struct IdGenerator {
    id_gen: RangeFrom<usize>,
    id_cache: HashMap<(Id, Id), Vec<Id>>,
}

impl Default for IdGenerator {
    fn default() -> Self {
        IdGenerator {
            id_gen: 0..,
            id_cache: HashMap::new(),
        }
    }
}

impl IdGenerator {
    pub fn next_id(&mut self) -> Id {
        self.id_gen.next().map(Id).unwrap()
    }
    pub fn cached_id_iter(&mut self, cache_key: (Id, Id)) -> IdIter {
        let cache_entry = self.id_cache.entry(cache_key).or_insert_with(Vec::new);
        IdIter::new(cache_entry, &mut self.id_gen)
    }
}

#[cfg(test)]
mod tests {

    use super::{Id, IdGenerator};

    #[test]
    fn get_new_id() {
        let mut test_id_manager = IdGenerator::default();

        assert_eq!(test_id_manager.next_id(), Id(0));
        assert_eq!(test_id_manager.next_id(), Id(1));
        assert_eq!(test_id_manager.next_id(), Id(2));
    }

    #[test]
    fn iter_cached_ids() {
        let mut test_id_manager = IdGenerator::default();

        let mut test_id_iter_0 = test_id_manager.cached_id_iter((Id(4), Id(2)));

        assert_eq!(test_id_iter_0.next(), Some(Id(0)));
        assert_eq!(test_id_iter_0.next(), Some(Id(1)));

        let mut test_id_iter_1 = test_id_manager.cached_id_iter((Id(4), Id(2)));

        assert_eq!(test_id_iter_1.next(), Some(Id(0))); // cached entry
        assert_eq!(test_id_iter_1.next(), Some(Id(1))); // cached entry
        assert_eq!(test_id_iter_1.next(), Some(Id(2))); // new entry
    }
}
