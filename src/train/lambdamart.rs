use train::regression_tree::*;
use train::dataset::*;
use util::*;
use metric::*;

pub struct LambdaMART<M> {
    dataset: DataSet,
    config: Config<M>,
}

pub struct Config<M>
{
    pub trees: usize,
    pub learning_rate: f64,
    pub max_leaves: usize,
    pub min_samples_per_leaf: usize,
    pub thresholds: usize,
    pub metric: M,
}

impl<M> LambdaMART<M>
where
    M: MetricScorer,
{
    pub fn new(dataset: DataSet, config: Config<M>) -> LambdaMART<M> {
        LambdaMART {
            dataset: dataset,
            config: config,
        }
    }

    pub fn init(&self) -> Result<()> {
        Ok(())
    }

    pub fn learn(&self) -> Result<()> {
        let learning_rate = 0.1;
        let min_samples_per_leaf = 10;
        let max_leaves = 10;
        let mut ensemble = Ensemble::new();
        let mut training = TrainingSet::from(&self.dataset);
        println!(
            "{:<7} | {:>9} | {:>9}",
            "#iter",
            self.config.metric.name() + "-T",
            self.config.metric.name() + "-V"
        );
        for i in 0..self.config.trees {
            training.update_lambdas_weights();

            let mut tree = RegressionTree::new(
                learning_rate,
                max_leaves,
                min_samples_per_leaf,
            );
            tree.fit(&training);
            ensemble.push(tree);

            let score = training.evaluate(&self.config.metric);

            println!("{:<7} | {:>9.4} | {:>9.4}", i, score, "");
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
