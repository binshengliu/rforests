use super::regression_tree::*;
use train::dataset::*;
use util::*;
use metric::*;
use super::training_set::*;
use train::lambdamart::validate_set::*;

/// A instance of LambdaMART algorithm.
pub struct LambdaMART {
    config: Config,
    ensemble: Ensemble,
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
    pub min_leaf_samples: usize,
    pub early_stop: usize,
    pub print_metric: bool,
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
    ///     use rforests::train::lambdamart::lambdamart::*;
    ///     use rforests::metric;
    ///
    ///     let f = File::open(train_path)?;
    ///     let dataset = DataSet::load(f).unwrap();
    ///
    ///     let v = File::open(valid_path)?;
    ///     let mut validate = DataSet::load(v).unwrap();
    ///
    ///     let config = Config {
    ///         train: dataset,
    ///         trees: 1000,
    ///         learning_rate: 0.1,
    ///         max_leaves: 10,
    ///         min_leaf_samples: 1,
    ///         thresholds: 256,
    ///         print_metric: true,
    ///         metric: metric::new("NDCG", 10).unwrap(),
    ///         validate: Some(validate),
    ///         test: None,
    ///         early_stop: 100,
    ///     };
    ///     let mut lambdamart = LambdaMART::new(config);
    ///     lambdamart.init()?;
    ///     lambdamart.learn()?;
    /// #    Ok(())
    /// # }
    /// ```
    pub fn new(config: Config) -> LambdaMART {
        LambdaMART {
            config: config,
            ensemble: Ensemble::new(),
        }
    }

    /// Initializes LambdaMART algorithm.
    pub fn init(&self) -> Result<()> {
        Ok(())
    }

    /// Learns from the given training data, using the configuration
    /// specified when creating LambdaMART instance.
    pub fn learn(&mut self) -> Result<()> {
        let mut training =
            TrainingSet::new(&self.config.train, self.config.thresholds);
        let mut validate =
            self.config.validate.as_ref().map(|v| ValidateSet::from(v));

        self.print_metric_header();
        for i in 0..self.config.trees {
            training.update_lambdas_weights(&self.config.metric);

            let mut tree = RegressionTree::new(
                self.config.learning_rate,
                self.config.max_leaves,
                self.config.min_leaf_samples,
            );

            // The scores of the model are updated when the tree node
            // does not split and becomes a leaf.
            let leaf_output = tree.fit(&training);

            // Update the scores fitted by the regression tree.
            training.update_result(&leaf_output);

            // Evaluate on the training data set.
            let train_score = training.evaluate(&self.config.metric);

            // Update scores on validate set.
            validate.as_mut().map(|v| v.update(&tree));

            // Evaluate on validate set.
            let validate_score =
                validate.as_ref().map(|v| v.evaluate(&self.config.metric));

            self.ensemble.push(tree);

            self.print_metric(i, train_score, validate_score);
        }
        Ok(())
    }

    pub fn evaluate(&self, dataset: &DataSet) -> f64 {
        dataset.evaluate(&self.ensemble, &self.config.metric)
    }

    fn print(&self, msg: &str) {
        if self.config.print_metric {
            println!("{}", msg);
        }
    }

    /// Print metric header.
    fn print_metric_header(&self) {
        self.print(&format!(
            "{:<7} | {:>9} | {:>9}",
            "#iter",
            self.config.metric.name() + "-T",
            self.config.metric.name() + "-V"
        ));
    }

    /// Print metric of each iteration.
    fn print_metric(
        &self,
        iteration: usize,
        train_score: f64,
        validate_score: Option<f64>,
    ) {
        if let Some(v_score) = validate_score {
            let s = format!(
                "{:<7} | {:>9.4} | {:>9.4}",
                iteration,
                train_score,
                v_score
            );
            self.print(&s);
        } else {
            let s = format!(
                "{:<7} | {:>9.4} | {:>9.4}",
                iteration,
                train_score,
                ""
            );
            self.print(&s);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs::File;

    #[test]
    fn test_lambda_mart() {
        // CWD of cargo test is the root of the project.
        let path = "./data/train-lite.txt";
        let f = File::open(path).unwrap();
        let dataset = DataSet::load(f).unwrap();
        let validate_set = dataset.clone();

        let config = Config {
            train: dataset,
            test: None,
            trees: 10,
            early_stop: 100,
            learning_rate: 0.1,
            max_leaves: 10,
            min_leaf_samples: 1,
            thresholds: 256,
            print_metric: false,
            metric: Box::new(NDCGScorer::new(10)),
            validate: None,
        };
        let mut lambdamart = LambdaMART::new(config);
        lambdamart.init().unwrap();
        lambdamart.learn().unwrap();
        // This is a verified result. Use as a guard for future
        // modifications.
        assert_eq!(lambdamart.evaluate(&validate_set), 0.5694960535660895);
    }
}
