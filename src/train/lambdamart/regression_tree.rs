use std;
use train::dataset::*;
use util::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use train::lambdamart::training_set::*;

/// A node in the regression tree.
struct Node {
    fid: Option<Id>,
    threshold: Option<Value>,
    output: Option<f64>,
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
}

impl Node {
    /// Create a new node.
    pub fn new(parent: Option<usize>) -> Node {
        Node {
            fid: None,
            threshold: None,
            parent: parent,
            left: None,
            right: None,
            output: None,
        }
    }

    pub fn set_non_leaf(
        &mut self,
        fid: Id,
        threshold: Value,
        left: usize,
        right: usize,
    ) {
        self.fid = Some(fid);
        self.threshold = Some(threshold);
        self.left = Some(left);
        self.right = Some(right);
    }

    pub fn set_leaf(&mut self, output: f64) {
        self.output = Some(output);
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
    min_leaf_samples: usize,
    max_leaves: usize,
    nodes: Vec<Node>,
}

struct NodeData<'a> {
    index: usize,
    sample: TrainingSample<'a>,
}

impl<'a> NodeData<'a> {
    pub fn new(
        index: usize,
        sample: TrainingSample<'a>,
    ) -> NodeData<'a> {
        NodeData {
            index: index,
            sample: sample,
        }
    }
}

impl<'a> PartialEq for NodeData<'a> {
    fn eq(&self, other: &NodeData) -> bool {
        self.sample.variance() == other.sample.variance()
    }
}

impl<'a> PartialOrd for NodeData<'a> {
    fn partial_cmp(&self, other: &NodeData) -> Option<Ordering> {
        self.sample.variance().partial_cmp(&other.sample.variance())
    }
}

impl<'a> Eq for NodeData<'a> {}

impl<'a> Ord for NodeData<'a> {
    fn cmp(&self, other: &NodeData) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl RegressionTree {
    /// Create a new regression tree, with at least min_leaf_samples
    /// training instances on the leaves.
    pub fn new(
        learning_rate: f64,
        max_leaves: usize,
        min_leaf_samples: usize,
    ) -> RegressionTree {
        RegressionTree {
            learning_rate: learning_rate,
            min_leaf_samples: min_leaf_samples,
            max_leaves: max_leaves,
            nodes: Vec::new(),
        }
    }

    fn split_node(
        &mut self,
        index: usize,
        fid: usize,
        threshold: f64,
    ) -> (usize, usize) {
        let left_index = self.nodes.len();
        let mut left = Node::new(Some(index));
        left.parent = Some(index);
        self.nodes.push(left);
        let right_index = self.nodes.len();
        let mut right = Node::new(Some(index));
        right.parent = Some(index);
        self.nodes.push(right);

        let node = &mut self.nodes[index];
        node.set_non_leaf(fid, threshold, left_index, right_index);

        (left_index, right_index)
    }

    fn set_leaf_node(&mut self, index: usize, output: f64) {
        self.nodes[index].set_leaf(output);
    }

    /// Fit to a training.
    pub fn fit(&mut self, training: &TrainSet) -> Vec<Value> {
        let sample = TrainingSample::from(training);
        let mut leaves = 0;
        let mut leaf_output: Vec<Value> = vec![0.0; training.len()];

        let root = Node::new(None);
        self.nodes.push(root);

        let mut queue: BinaryHeap<NodeData> =
            BinaryHeap::with_capacity(self.max_leaves);
        queue.push(NodeData::new(0, sample));

        while !queue.is_empty() {
            let NodeData { index, sample } = queue.pop().unwrap();
            // We have reached leaves count limitation.
            if 1 + leaves + queue.len() >= self.max_leaves {
                let value = sample.newton_output();
                let output = value * self.learning_rate;
                self.set_leaf_node(index, value);
                sample.update_output(&mut leaf_output, output);
                leaves += 1;
                continue;
            }

            let split_result = sample.split(self.min_leaf_samples);
            if split_result.is_none() {
                let value = sample.newton_output();
                let output = value * self.learning_rate;
                self.set_leaf_node(index, value);
                sample.update_output(&mut leaf_output, output);
                leaves += 1;
                continue;
            }

            let split = split_result.unwrap();
            let left_len = split.left.len();
            let right_len = split.right.len();

            // Split node at `index`.
            let (left, right) =
                self.split_node(index, split.fid, split.threshold);

            queue.push(NodeData::new(left, split.left));
            queue.push(NodeData::new(right, split.right));

            debug!(
                "Split: fid:{} threshold:{} s:{}",
                split.fid,
                split.threshold,
                split.s
            );
            debug!("Split: {} => {} + {}", sample.len(), left_len, right_len);
        }

        leaf_output
    }

    pub fn print(&self) {
        if self.nodes.is_empty() {
            println!("Empty tree");
            return;
        }

        // (index, indent)
        let mut queue: Vec<(usize, usize)> = vec![(0, 0)];
        while !queue.is_empty() {
            let (index, indent) = queue.pop().unwrap();
            let node = &self.nodes[index];
            print!("{:width$}", "", width = indent);
            if let Some(output) = node.output {
                println!("{{ output: {:?} }}", output);
            } else {
                println!(
                    "{{ fid: {:?}, threshold: {:?} }}",
                    option_to_string(&node.fid),
                    option_to_string(&node.threshold)
                );
                queue.push((node.left.unwrap(), indent + 2));
                queue.push((node.right.unwrap(), indent + 2));
            }
        }
    }
}

impl ::train::Evaluate for RegressionTree {
    /// Evaluate an input.
    fn evaluate(&self, instance: &Instance) -> f64 {
        let mut node = &self.nodes[0];
        while node.output.is_none() {
            if instance.value(node.fid.unwrap()) <= node.threshold.unwrap() {
                node = &self.nodes[node.left.unwrap()];
            } else {
                node = &self.nodes[node.right.unwrap()];
            }
        }

        assert!(node.output.is_some());
        node.output.unwrap() * self.learning_rate
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

impl ::train::Evaluate for Ensemble {
    fn evaluate(&self, instance: &Instance) -> f64 {
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

        let mut training = TrainSet::new(&dataset, 3);
        // training.init_model_scores(&[3.0, 2.0]);
        let learning_rate = 0.1;
        let min_leaf_samples = 1;
        let max_leaves = 10;

        for _ in 0..10 {
            training.update_lambdas_weights(&metric::new("NDCG", 10).unwrap());

            // println!("{:?}", training.lambdas);
            // println!("{:?}", training.weights);

            let mut tree = RegressionTree::new(
                learning_rate,
                max_leaves,
                min_leaf_samples,
            );
            tree.fit(&training);

            // println!("{:?}", tree);
            // println!("{:?}", training.model_scores);
            // println!("-----------------------------------");
        }
    }
}
