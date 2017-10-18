use std;
use train::dataset::*;
use util::*;

/// A node in the regression tree.
struct Node {
    fid: Option<Id>,
    threshold: Option<Value>,
    output: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
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
            let value = sample.newton_output();
            self.output = Some(value);
            sample.update_output(value);
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

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Node {{ fid: {:?}, threshold: {:?}, output: {:?}, left: {:?}, right: {:?} }}",
            self.fid,
            self.threshold,
            self.output,
            self.left,
            self.right
        )
    }
}

/// A regression tree.
#[derive(Debug)]
pub struct RegressionTree {
    learning_rate: f64,
    min_leaf_count: usize,
    root: Option<Node>,
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
        self.root = Some(root);
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

    pub fn evaluate(&self, instance: &Instance) -> f64 {
        let mut result = 0.0;
        for tree in &self.trees {
            result += tree.evaluate(instance);
        }

        result
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tree_fitting() {
        // (label, qid, feature_values)
        let data = vec![
            (3.0, 1, vec![3.0, 0.0]), // 0
            (2.0, 1, vec![2.0, 0.0]), // 1
            (1.0, 1, vec![1.0, 0.0]), // 2
            (1.0, 1, vec![1.0, 0.0]), // 3
            (3.0, 1, vec![3.0, 0.0]), // 4
            (2.0, 1, vec![2.0, 0.0]), // 5
        ];

        let mut dataset = DataSet::new(3);
        dataset.from_iter(data.into_iter());

        let mut training = TrainingSet::from(&dataset);
        // training.init_model_scores(&[3.0, 2.0]);
        let learning_rate = 0.1;
        let min_leaf_count = 1;

        for _ in 0..10 {
            training.update_lambdas_weights();

            // println!("{:?}", training.lambdas);
            // println!("{:?}", training.weights);

            let mut tree = RegressionTree::new(learning_rate, min_leaf_count);
            tree.fit(&training);

            // println!("{:?}", tree);
            // println!("{:?}", training.model_scores);
            // println!("-----------------------------------");
        }
    }
}
