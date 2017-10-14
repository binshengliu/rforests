pub mod dcg;
pub use self::dcg::DCGScorer;

pub trait MetricScorer {
    fn new(truncation_level: usize) -> Self;
    fn score(&self, labels: &[f64]) -> f64;

    /// The change in score values by swaping any two of the labels.
    fn delta(&self, labels: &[f64]) -> Vec<Vec<f64>>;
}

