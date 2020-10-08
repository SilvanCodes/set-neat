use std::ops::RangeFrom;

use super::id::Id;

pub struct IdIter<'a> {
    index: usize,
    ids: &'a mut Vec<Id>,
    gen: &'a mut RangeFrom<usize>,
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
            self.index += 1;
            id
        }
    }
}

impl<'a> IdIter<'a> {
    pub fn new(ids: &'a mut Vec<Id>, gen: &'a mut RangeFrom<usize>) -> Self {
        IdIter { index: 0, ids, gen }
    }
}

#[cfg(test)]
mod tests {

    use super::{Id, IdIter};

    #[test]
    fn iterate_till_new() {
        let mut ids = vec![Id(4), Id(2)];
        let mut gen = 0..;

        let mut test_id_iter = IdIter::new(&mut ids, &mut gen);

        assert_eq!(test_id_iter.next(), Some(Id(4)));
        assert_eq!(test_id_iter.next(), Some(Id(2)));
        assert_eq!(test_id_iter.next(), Some(Id(0))); // generate and cache new id
        assert_eq!(test_id_iter.next(), Some(Id(1))); // generate and cache new id
    }
}
