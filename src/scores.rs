use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub trait ScoreType {}

pub trait ScoreValue {
    type Value;
    fn value(&self) -> Self::Value;
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct FitnessScore {
    pub raw: Raw<Fitness>,
    pub shifted: Shifted<Fitness>,
    pub normalized: Normalized<Fitness>,
    // pub adjusted: Adjusted<Fitness>,
}

impl FitnessScore {
    pub fn new(raw: f64, baseline: f64, with: f64) -> Self {
        let raw = Raw::new(raw);
        let shifted = raw.shift(baseline);
        let normalized = shifted.normalize(with);
        Self {
            raw,
            shifted,
            normalized,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, Deserialize, Serialize, PartialEq)]
pub struct Fitness(f64);

impl ScoreValue for Fitness {
    type Value = f64;

    fn value(&self) -> Self::Value {
        self.0
    }
}

/* impl<U: ScoreValue, T: ScoreType + Deref<Target = U>> ScoreValue for T {
    type Value = U::Value;

    fn value(&self) -> Self::Value {
        self.deref().value()
    }
} */

macro_rules! makeScoreType {
    ( $( $name:ident ),* ) => {
        $(
            #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
            pub struct $name<T: ScoreValue>(T);

            impl<T: ScoreValue> ScoreType for $name<T> {}

            impl<T: ScoreValue> Deref for $name<T> {
                type Target = T;

                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl<T: ScoreValue> DerefMut for $name<T> {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }
        )*
    };
}

makeScoreType!(Raw, Normalized, Adjusted, Shifted);

impl Raw<Fitness> {
    pub fn new(fitness: f64) -> Self {
        Self(Fitness(fitness))
    }
    pub fn shift(self, baseline: f64) -> Shifted<Fitness> {
        Shifted(Fitness((self.0).0 - baseline))
    }
}

impl Shifted<Fitness> {
    pub fn normalize(self, with: f64) -> Normalized<Fitness> {
        Normalized(Fitness((self.0).0 / with))
    }
}

impl Normalized<Fitness> {
    pub fn adjust(self, factor: f64) -> Adjusted<Fitness> {
        Adjusted(Fitness((self.0).0 / factor))
    }
}

#[cfg(test)]
mod tests {
    use super::{Adjusted, Fitness, Normalized, Raw, Shifted};

    #[test]
    fn normalize_raw() {
        let raw = Raw::new(1.0);

        let normalized = raw.shift(2.0);

        assert_eq!(normalized, Shifted(Fitness(3.0)))
    }

    #[test]
    fn adjust_normal() {
        let normal = Normalized(Fitness(1.0));

        let adjusted = normal.adjust(2.0);

        assert_eq!(adjusted, Adjusted(Fitness(0.5)))
    }

    /* #[test]
    fn nest() {
        let normal = Normalized(Fitness(1.0));

        let adjusted = normal.adjust(2.0);



        assert_eq!(adjusted, Adjusted(Fitness(0.5)))
    } */
}
