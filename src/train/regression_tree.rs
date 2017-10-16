use train::dataset::DataSet;
use util::*;

pub struct Node {
    fid: Id,
    threashold: Value,
}

impl Node {
    pub fn new(fid: Id, threashold: Value) -> Node {
        Node {
            fid: fid,
            threashold: threashold,
        }
    }

    /// Evaluate an input
    pub fn evaluate() {}
}

pub struct RegressionTree {
    root: Node,
}

impl RegressionTree {
    pub fn new() -> RegressionTree {
        unimplemented!();
    }

    pub fn fit(dataset: &DataSet) {}

    pub fn best_split(dataset: &DataSet, threasholds: &[f64]) {
        // let histogram = dataset.histogram(10);
    }

    pub fn feature_best_split(
        dataset: &DataSet,
        fid: Id,
        threasholds: &[Value],
    ) {
        // let histogram = dataset.feature_histogram(fid, 10);
    }
}
