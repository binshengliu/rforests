use std;
use train::dataset::*;
use util::*;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use train::lambdamart::training_set::*;

/// A node in the regression tree.
struct Node {
    fid: Option<Id>,
    threshold: Option<Value>,
    output: Option<f64>,
    left: Option<Rc<RefCell<Node>>>,
    right: Option<Rc<RefCell<Node>>>,
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

    /// Evaluate an input
    pub fn evaluate(&self, instance: &Instance) -> f64 {
        if let Some(value) = self.output {
            return value;
        }

        if instance.value(self.fid.unwrap()) <= self.threshold.unwrap() {
            self.left.as_ref().unwrap().borrow().evaluate(instance)
        } else {
            self.right.as_ref().unwrap().borrow().evaluate(instance)
        }
    }

    pub fn print(&self, indent: usize) {
        print!("{:width$}", "", width = indent);
        if let Some(output) = self.output {
            println!(
                "{{ output: {:?} }}",
                output,
            );
        } else {
            println!(
                "{{ fid: {:?}, threshold: {:?} }}",
                option_to_string(&self.fid),
                option_to_string(&self.threshold)
            );
        }

        if let Some(ref left) = self.left {
            left.borrow().print(indent + 2);
        }
        if let Some(ref right) = self.right {
            right.borrow().print(indent + 2);
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

fn option_to_string<T: ToString>(option: &Option<T>) -> String {
    match option {
        &Some(ref value) => value.to_string(),
        &None => "None".to_string(),
    }
}

/// A regression tree.
#[derive(Debug)]
pub struct RegressionTree {
    learning_rate: f64,
    // Minimal count of samples per leaf.
    min_samples_per_leaf: usize,
    max_leaves: usize,
    root: Option<Rc<RefCell<Node>>>,
}

struct NodeData<'t, 'd: 't> {
    node: Rc<RefCell<Node>>,
    sample: TrainingSample<'t, 'd>,
}

impl<'t, 'd: 't> NodeData<'t, 'd> {
    pub fn new(
        node: Rc<RefCell<Node>>,
        sample: TrainingSample<'t, 'd>,
    ) -> NodeData<'t, 'd> {
        NodeData {
            node: node,
            sample: sample,
        }
    }
}

impl<'t, 'd: 't> PartialEq for NodeData<'t, 'd> {
    fn eq(&self, other: &NodeData) -> bool {
        self.sample.variance() == other.sample.variance()
    }
}

impl<'t, 'd: 't> PartialOrd for NodeData<'t, 'd> {
    fn partial_cmp(&self, other: &NodeData) -> Option<Ordering> {
        self.sample.variance().partial_cmp(&other.sample.variance())
    }
}

impl<'t, 'd: 't> Eq for NodeData<'t, 'd> {}

impl<'t, 'd: 't> Ord for NodeData<'t, 'd> {
    fn cmp(&self, other: &NodeData) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl RegressionTree {
    /// Create a new regression tree, with at least min_samples_per_leaf
    /// training instances on the leaves.
    pub fn new(
        learning_rate: f64,
        max_leaves: usize,
        min_samples_per_leaf: usize,
    ) -> RegressionTree {
        RegressionTree {
            learning_rate: learning_rate,
            min_samples_per_leaf: min_samples_per_leaf,
            max_leaves: max_leaves,
            root: None,
        }
    }

    /// Fit to a training.
    pub fn fit(&mut self, training: &TrainingSet) {
        let sample = TrainingSample::from(training);
        let mut leaves = 0;

        let root = Rc::new(RefCell::new(Node::new()));
        self.root = Some(root.clone());

        let mut queue: BinaryHeap<NodeData> =
            BinaryHeap::with_capacity(self.max_leaves);
        queue.push(NodeData::new(root.clone(), sample));

        while !queue.is_empty() {
            let NodeData { node, sample } = queue.pop().unwrap();

            // We have reached leaves count limitation.
            if 1 + leaves + queue.len() >= self.max_leaves {
                let value = sample.newton_output();
                node.borrow_mut().output = Some(value);
                sample.update_output(value * self.learning_rate);
                leaves += 1;
                continue;
            }

            let split_result = sample.split(self.min_samples_per_leaf);
            if split_result.is_none() {
                let value = sample.newton_output();
                node.borrow_mut().output = Some(value);
                sample.update_output(value * self.learning_rate);
                leaves += 1;
                continue;
            }

            let (fid, threshold, _s_value, left_sample, right_sample) =
                split_result.unwrap();

            let left = Rc::new(RefCell::new(Node::new()));
            let right = Rc::new(RefCell::new(Node::new()));

            let mut node = node.borrow_mut();
            node.fid = Some(fid);
            node.threshold = Some(threshold);
            node.left = Some(left.clone());
            node.right = Some(right.clone());

            debug!(
                "Split {} at fid {}, threshold {}, s {} : {} + {}",
                sample.len(),
                fid,
                threshold,
                _s_value,
                left_sample.len(),
                right_sample.len()
            );

            queue.push(NodeData::new(left.clone(), left_sample));
            queue.push(NodeData::new(right.clone(), right_sample));
        }
    }

    /// Evaluate an input.
    pub fn evaluate(&self, instance: &Instance) -> f64 {
        self.root.as_ref().unwrap().borrow().evaluate(instance) *
            self.learning_rate
    }

    pub fn print(&self) {
        if let Some(ref root) = self.root {
            root.borrow().print(0);
        } else {
            println!("Empty");
        }
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
    use metric;

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

        let dataset: DataSet = data.into_iter().collect();

        let mut training = TrainingSet::new(&dataset, 3);
        // training.init_model_scores(&[3.0, 2.0]);
        let learning_rate = 0.1;
        let min_samples_per_leaf = 1;
        let max_leaves = 10;

        for _ in 0..10 {
            training.update_lambdas_weights(&metric::new("NDCG", 10).unwrap());

            // println!("{:?}", training.lambdas);
            // println!("{:?}", training.weights);

            let mut tree = RegressionTree::new(
                learning_rate,
                max_leaves,
                min_samples_per_leaf,
            );
            tree.fit(&training);

            // println!("{:?}", tree);
            // println!("{:?}", training.model_scores);
            // println!("-----------------------------------");
        }
    }
}
