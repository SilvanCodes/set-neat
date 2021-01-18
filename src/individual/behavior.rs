use std::ops::Deref;

use ndarray::{Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};

use crate::utility::gym::StandardScaler;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Behavior(pub Vec<f64>);

impl Deref for Behavior {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Behaviors<'a>(Vec<&'a Behavior>);

impl<'a> Deref for Behaviors<'a> {
    type Target = Vec<&'a Behavior>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> From<Vec<&'a Behavior>> for Behaviors<'a> {
    fn from(behaviors: Vec<&'a Behavior>) -> Self {
        Behaviors(behaviors)
    }
}

impl<'a> Behaviors<'a> {
    pub fn compute_novelty(&self, nearest_neighbors: usize) -> Vec<f64> {
        let width = self[0].len();
        let height = self.len();

        let mut behavior_iter = self.iter();

        let mut behavior_arr: Array2<f64> = Array2::zeros((width, height));
        for mut row in behavior_arr.axis_iter_mut(Axis(1)) {
            row += &ArrayView1::from(behavior_iter.next().unwrap().as_slice());
        }

        let standard_scaler = StandardScaler::new(behavior_arr.view().t());

        let mut z_scores_arr: Array2<f64> = Array2::zeros((width, height));

        for (index, row) in behavior_arr.axis_iter(Axis(1)).enumerate() {
            let mut z_row = z_scores_arr.index_axis_mut(Axis(1), index);
            z_row += &standard_scaler.scale(row);
        }

        let mut raw_novelties = Vec::new();

        for z_score in z_scores_arr.axis_iter(Axis(1)) {
            let mut distances = z_scores_arr
                .axis_iter(Axis(1))
                // build euclidian distance to neighbor
                .map(|neighbor| {
                    neighbor
                        .iter()
                        .zip(z_score.iter())
                        .map(|(n, z)| (n - z).powi(2))
                        .sum::<f64>()
                })
                .map(|sum| sum.sqrt())
                .collect::<Vec<f64>>();

            distances.sort_by(|dist_0, dist_1| {
                dist_0
                    .partial_cmp(&dist_1)
                    .unwrap_or_else(|| panic!("failed to compare {} and {}", dist_0, dist_1))
            });

            // take k nearest neighbors, calculate and assign spareseness
            let sparseness = distances
                .iter()
                // skip self with zero distance
                .skip(1)
                .take(nearest_neighbors)
                .sum::<f64>()
                / nearest_neighbors as f64;

            raw_novelties.push(sparseness);
        }

        raw_novelties
    }
}

#[cfg(test)]
mod tests {
    use super::{Behavior, Behaviors};

    #[test]
    fn compute_z_score() {
        let behavior_a = Behavior(vec![0.0, 1.0, 2.0]);
        let behavior_b = Behavior(vec![2.0, 1.0, 0.0]);

        let behaviors = Behaviors(vec![&behavior_a, &behavior_b]);

        let novelty = behaviors.compute_novelty(1);

        dbg!(novelty);

        // assert_eq!(novelty, vec![]);
    }
}
