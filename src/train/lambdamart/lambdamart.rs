use super::regression_tree::*;
use train::dataset::*;
use util::*;
use metric::*;
use super::training_set::*;

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
    ///         min_samples_per_leaf: 1,
    ///         thresholds: 256,
    ///         print_metric: true,
    ///         print_tree: false,
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
        LambdaMART { config: config, ensemble: Ensemble::new() }
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
        print_metric_header(&self.config.metric);
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

            self.ensemble.push(tree);

            print_metric(
                i,
                &self.ensemble,
                &training,
                &self.config.validate,
                &self.config.metric,
            );
        }
        Ok(())
    }
}

/// Print metric header.
fn print_metric_header(metric: &Box<MetricScorer>) {
    println!(
        "{:<7} | {:>9} | {:>9}",
        "#iter",
        metric.name() + "-T",
        metric.name() + "-V"
    );
}

/// Print metric of each iteration.
fn print_metric(
    iteration: usize,
    ensemble: &Ensemble,
    training: &TrainingSet,
    validate: &Option<DataSet>,
    metric: &Box<MetricScorer>,
) {
    let train_score = training.evaluate(&metric);
    if let &Some(ref validate) = validate {
        let validation_score = validate.validate(&ensemble, metric);
        println!(
            "{:<7} | {:>9.4} | {:>9.4}",
            iteration,
            train_score,
            validation_score
        );
    } else {
        println!("{:<7} | {:>9.4} | {:>9.4}", iteration, train_score, "");
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

        let config = Config {
            train: dataset,
            test: None,
            trees: 10,
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
        let mut lambdamart = LambdaMART::new(config);
        lambdamart.init().unwrap();
        lambdamart.learn().unwrap();
    }
}
