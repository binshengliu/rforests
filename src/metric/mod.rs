pub mod dcg;
pub mod ndcg;
pub use self::dcg::DCGScorer;
pub use self::ndcg::NDCGScorer;

pub trait Measure: Sync {
    fn get_k(&self) -> usize;

    fn measure(&self, labels: &[f64]) -> f64;

    /// The changes in score values by swaping any two of the labels.
    fn swap_changes(&self, labels: &[f64]) -> Vec<Vec<f64>>;

    /// Name of the scorer. For display.
    fn name(&self) -> String;
}

pub fn new(name: &str, k: usize) -> Option<Box<Measure>> {
    match name {
        "NDCG" => Some(Box::new(NDCGScorer::new(k))),
        "DCG" => Some(Box::new(DCGScorer::new(k))),
        _ => None,
    }
}
