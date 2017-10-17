use std;
use train::dataset::*;
use util::*;

/// A node in the regression tree.
struct Node {
    fid: Option<Id>,
    threshold: Option<Value>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    output: Option<f64>,
}

impl Node {
    /// Create a new node.
    pub fn new() -> Node {
        Node {
            fid: None,
            threshold: None,
            left: None,
            right: None,
            output: None,
        }
    }

    /// Split on the data set sample and creates children, until self
    /// becomes the leaf node.
    pub fn split(&mut self, sample: TrainingSample, min_leaf_count: usize) {
        let split_result = sample.split(min_leaf_count);
        if split_result.is_none() {
            self.output = Some(sample.newton_output());
            return;
        }

        let (fid, threshold, _s_value, left_sample, right_sample) =
            split_result.unwrap();

        self.fid = Some(fid);
        self.threshold = Some(threshold);

        let mut left = Node::new();
        left.split(left_sample, min_leaf_count);

        let mut right = Node::new();
        right.split(right_sample, min_leaf_count);

        self.left = Some(Box::new(left));
        self.right = Some(Box::new(right));
    }

    /// Evaluate an input
    pub fn evaluate(&self, instance: &Instance) -> f64 {
        if let Some(value) = self.output {
            return value;
        }

        if instance.value(self.fid.unwrap()) <= self.threshold.unwrap() {
            self.left.as_ref().unwrap().evaluate(instance)
        } else {
            self.right.as_ref().unwrap().evaluate(instance)
        }
    }
}

/// A regression tree.
pub struct RegressionTree {
    learning_rate: f64,
    min_leaf_count: usize,
    root: Option<Box<Node>>,
}

impl RegressionTree {
    /// Create a new regression tree, with at least min_leaf_count
    /// training instances on the leaves.
    pub fn new(learning_rate: f64, min_leaf_count: usize) -> RegressionTree {
        RegressionTree {
            learning_rate: learning_rate,
            min_leaf_count: min_leaf_count,
            root: None,
        }
    }

    /// Fit to a training.
    pub fn fit(&mut self, training: &TrainingSet) {
        let mut root = Node::new();
        let sample = TrainingSample::from(training);
        root.split(sample, self.min_leaf_count);
    }

    /// Evaluate an input.
    pub fn evaluate(&self, instance: &Instance) -> f64 {
        self.root.as_ref().unwrap().evaluate(instance) * self.learning_rate
    }
}

pub struct Ensemble {
    trees: Vec<RegressionTree>,
}

impl Ensemble {
    pub fn new() -> Ensemble {
        Ensemble { trees: Vec::new() }
    }
}

impl std::ops::Deref for Ensemble {
    type Target = Vec<RegressionTree>;

    fn deref(&self) -> &Vec<RegressionTree> {
        &self.trees
    }
}

impl std::ops::DerefMut for Ensemble {
    fn deref_mut(&mut self) -> &mut Vec<RegressionTree> {
        &mut self.trees
    }
}
