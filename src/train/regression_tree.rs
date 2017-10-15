use train::dataset::DataSet;

pub struct Node<'a> {
    dataset: &'a DataSet,
    indices: Vec<usize>,

    /// Valid only after spliting
    fid: Option<u64>,
    threashold: Option<f64>,
}

impl<'a> Node<'a> {
    pub fn new(dataset: &'a DataSet, indices: Vec<usize>) -> Node<'a> {
        Node {
            dataset: dataset,
            indices: indices,
            fid: None,
            threashold: None,
        }
    }

    pub fn split(&mut self) {
        // (fid, threashold, s)
        let mut splits: Vec<(u64, f64, f64)> = Vec::new();
        for fid in self.dataset.fid_iter() {
            let histogram = self.dataset.feature_histogram(fid, 10);
            let split = histogram.best_split(1);
            if split.is_none() {
                continue;
            }

            let (threshold, s) = split.unwrap();
            splits.push((fid, threshold, s));
        }

        // Find the split with the best s value;
        let best_split =
            splits.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    }

    /// Evaluate an input
    pub fn evaluate() {}
}

pub struct RegressionTree<'a> {
    root: Node<'a>,
}

impl<'a> RegressionTree<'a> {
    pub fn new() -> RegressionTree<'a> {
        unimplemented!();
    }

    pub fn fit(dataset: &DataSet) {}

    pub fn best_split(dataset: &DataSet, threasholds: &[f64]) {
        let histogram = dataset.histogram(10);
    }

    pub fn feature_best_split(
        dataset: &DataSet,
        fid: u64,
        threasholds: &[f64],
    ) {
        let histogram = dataset.feature_histogram(fid, 10);
    }
}
