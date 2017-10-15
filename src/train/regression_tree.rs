use train::dataset::DataSet;

pub struct Node {
    fid: u64,
    threashold: f64,
}

impl Node {
    pub fn new(fid: u64, threashold: f64) -> Node {
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
        fid: u64,
        threasholds: &[f64],
    ) {
        // let histogram = dataset.feature_histogram(fid, 10);
    }
}
