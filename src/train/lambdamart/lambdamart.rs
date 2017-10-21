use super::regression_tree::*;
use super::dataset::*;
use util::*;
use metric::*;

/// A instance of LambdaMART algorithm.
pub struct LambdaMART {
    config: Config,
}

/// Configurable options for LambdaMART.
pub struct Config {
    pub train: DataSet,
    pub validate: Option<DataSet>,
    pub test: Option<DataSet>,

    pub metric: Box<MetricScorer>,
    pub trees: usize,
    pub max_leaves: usize,
    pub learning_rate: f64,
    pub thresholds: usize,
    pub min_samples_per_leaf: usize,
    pub early_stop: usize,
    pub print_metric: bool,
    pub print_tree: bool,
}

impl LambdaMART {
    /// Create a new LambdaMART instance.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rforests::util::Result;
    /// # pub fn lambdamart(train_path: &str, valid_path: &str) -> Result<()> {
    ///     use std::fs::File;
    ///     use rforests::train::dataset::*;
    ///     use rforests::train::lambdamart::*;
    ///     use rforests::metric::*;
    ///
    ///     let max_bins = 256;
    ///     let f = File::open(train_path)?;
    ///     let mut dataset = DataSet::new(max_bins);
    ///     dataset.load(f).unwrap();
    ///
    ///     let v = File::open(valid_path)?;
    ///     let mut validate = DataSet::new(256);
    ///     validate.load(v).unwrap();
    ///
    ///     let config = Config {
    ///         trees: 1000,
    ///         learning_rate: 0.1,
    ///         max_leaves: 10,
    ///         min_samples_per_leaf: 1,
    ///         thresholds: 256,
    ///         print_metric: true,
    ///         print_tree: false,
    ///         metric: NDCGScorer::new(10),
    ///         validate: Some(validate),
    ///     };
    ///     let lambdamart = LambdaMART::new(dataset, config);
    ///     lambdamart.init()?;
    ///     lambdamart.learn()?;
    /// #    Ok(())
    /// # }
    /// ```
    pub fn new(config: Config) -> LambdaMART {
        LambdaMART { config: config }
    }

    /// Initializes LambdaMART algorithm.
    pub fn init(&self) -> Result<()> {
        Ok(())
    }

    /// Learns from the given training data, using the configuration
    /// specified when creating LambdaMART instance.
    pub fn learn(&self) -> Result<()> {
        let mut ensemble = Ensemble::new();
        let mut training = TrainingSet::from(&self.config.train);
        self.print_metric_header();
        for i in 0..self.config.trees {
            training.update_lambdas_weights();

            let mut tree = RegressionTree::new(
                self.config.learning_rate,
                self.config.max_leaves,
                self.config.min_samples_per_leaf,
            );

            // The scores of the model are updated when the tree node
            // does not split and becomes a leaf.
            tree.fit(&training);

            if self.config.print_tree {
                tree.print();
            }

            ensemble.push(tree);

            self.print_metric(i, &training, &ensemble);
        }
        Ok(())
    }

    /// Print metric header.
    fn print_metric_header(&self) {
        if self.config.print_metric {
            println!(
                "{:<7} | {:>9} | {:>9}",
                "#iter",
                self.config.metric.name() + "-T",
                self.config.metric.name() + "-V"
            );
        }
    }

    /// Print metric of each iteration.
    fn print_metric(
        &self,
        iteration: usize,
        training: &TrainingSet,
        ensemble: &Ensemble,
    ) {
        if self.config.print_metric {
            let train_score = training.evaluate(&self.config.metric);
            let mut validation_score = None;
            if let Some(ref validate) = self.config.validate {
                validation_score =
                    Some(validate.validate(&ensemble, &self.config.metric));
            }

            if let Some(validation_score) = validation_score {
                println!(
                    "{:<7} | {:>9.4} | {:>9.4}",
                    iteration,
                    train_score,
                    validation_score
                );
            } else {
                println!(
                    "{:<7} | {:>9.4} | {:>9.4}",
                    iteration,
                    train_score,
                    ""
                );
            }
        }
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

        let config = Config {
            train: dataset,
            test: None,
            trees: 1,
            early_stop: 100,
            learning_rate: 0.1,
            max_leaves: 10,
            min_samples_per_leaf: 1,
            thresholds: 256,
            print_metric: true,
            print_tree: false,
            metric: Box::new(NDCGScorer::new(10)),
            validate: None,
        };
        let lambdamart = LambdaMART::new(config);
        lambdamart.init().unwrap();
        lambdamart.learn().unwrap();
    }
}
