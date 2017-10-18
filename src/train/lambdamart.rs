use train::regression_tree::*;
use train::dataset::*;
use util::*;
use metric::*;

pub struct LambdaMART<M> {
    dataset: DataSet,
    trees: usize,
    metric: M,
}

impl<M> LambdaMART<M>
where
    M: MetricScorer,
{
    pub fn new(dataset: DataSet, trees: usize, metric: M) -> LambdaMART<M> {
        LambdaMART {
            dataset: dataset,
            trees: trees,
            metric: metric,
        }
    }

    pub fn init(&self) -> Result<()> {
        Ok(())
    }

    pub fn learn(&self) -> Result<()> {
        let learning_rate = 0.1;
        let min_leaf_count = 1;
        let mut ensemble = Ensemble::new();
        let mut training = TrainingSet::from(&self.dataset);
        for i in 0..self.trees {
            training.update_lambdas_weights();

            let mut tree = RegressionTree::new(learning_rate, min_leaf_count);
            tree.fit(&training);
            ensemble.push(tree);

            let score = training.evaluate(&self.metric);

            println!("{}\t{}", i, score);
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs::File;

    #[test]
    fn test_lambda_mart() {
        let path = "/home/lbs/code/rforests/data/train-lite.txt";
        let max_bins = 256;
        let f = File::open(path).unwrap();
        let mut dataset = DataSet::new(max_bins);
        dataset.load(f).unwrap();

        let trees = 1;
        let ndcg = NDCGScorer::new(10);
        let lambdamart = LambdaMART::new(dataset, trees, ndcg);
        lambdamart.init().unwrap();
        lambdamart.learn().unwrap();
    }
}
