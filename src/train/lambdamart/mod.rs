use clap::{App, Arg, ArgMatches, SubCommand};
use std::fs::File;
use self::lambdamart::*;
use std;
use std::process::exit;
use metric;
use train::dataset::*;

pub mod training_set;
pub mod lambdamart;
pub mod regression_tree;
pub mod histogram;

struct LambdaMARTParameter<'a> {
    train_file_path: &'a str,
    validate_file_path: Option<&'a str>,
    test_file_path: Option<&'a str>,
    metric: &'a str,
    metric_k: usize,
    trees: usize,
    leaves: usize,
    shrinkage: f64,
    thresholds_count: usize,
    min_leaf_support: usize,
    early_stop: usize,
}

impl<'a> LambdaMARTParameter<'a> {
    pub fn parse(matches: &'a ArgMatches<'a>) -> LambdaMARTParameter<'a> {
        // Defaults to 256
        let train_file_path = matches.value_of("train-file").unwrap();
        let validate_file_path = matches.value_of("validate-file");
        let test_file_path = matches.value_of("test-file");
        let metric = matches.value_of("metric").unwrap();
        let metric_k = value_t!(matches.value_of("metric-k"), usize)
            .unwrap_or_else(|e| e.exit());
        let trees = value_t!(matches.value_of("trees"), usize).unwrap_or_else(
            |e| e.exit(),
        );
        let leaves = value_t!(matches.value_of("leaves"), usize)
            .unwrap_or_else(|e| e.exit());
        let shrinkage = value_t!(matches.value_of("shrinkage"), f64)
            .unwrap_or_else(|e| e.exit());
        let thresholds_count = value_t!(matches.value_of("thresholds"), usize)
            .unwrap_or_else(|e| e.exit());
        let min_leaf_support =
            value_t!(matches.value_of("min-leaf-support"), usize)
                .unwrap_or_else(|e| e.exit());
        let early_stop = value_t!(matches.value_of("early-stop"), usize)
            .unwrap_or_else(|e| e.exit());

        LambdaMARTParameter {
            train_file_path: train_file_path,
            validate_file_path: validate_file_path,
            test_file_path: test_file_path,
            metric: metric,
            metric_k: metric_k,
            trees: trees,
            leaves: leaves,
            shrinkage: shrinkage,
            thresholds_count: thresholds_count,
            min_leaf_support: min_leaf_support,
            early_stop: early_stop,
        }
    }

    pub fn config(&self) -> Config {
        let train_file =
            File::open(self.train_file_path).unwrap_or_else(|_e| exit(1));
        let train_dataset = DataSet::load(train_file).unwrap_or_else(|_e| exit(1));

        let validate = self.validate_file_path.map(|path| {
            let file = File::open(path).unwrap_or_else(|_e| exit(1));
            let dataset = DataSet::load(file).unwrap_or_else(|_e| exit(1));
            dataset
        });

        let test = self.test_file_path.map(|path| {
            let file = File::open(path).unwrap_or_else(|_e| exit(1));
            let dataset = DataSet::load(file).unwrap_or_else(|_e| exit(1));
            dataset
        });

        // The param is valid.
        let metric = metric::new(self.metric, self.metric_k).unwrap();

        Config {
            train: train_dataset,
            test: test,
            trees: 1000,
            learning_rate: 0.1,
            max_leaves: 10,
            min_samples_per_leaf: 1,
            thresholds: 256,
            print_metric: true,
            print_tree: false,
            metric: metric,
            validate: validate,
            early_stop: self.early_stop,
        }
    }

    pub fn print(&self) {
        fn print_param<T: std::fmt::Display>(name: &str, value: T) {
            println!("{:<20}: {}", name, value);
        }

        print_param("Training file", self.train_file_path);
        print_param(
            "Validating file",
            match self.validate_file_path {
                Some(path) => path,
                None => "None",
            },
        );
        print_param(
            "Testing file",
            match self.test_file_path {
                Some(path) => path,
                None => "None",
            },
        );
        print_param(
            "Metric",
            self.metric.to_owned() + "@" + &self.metric_k.to_string(),
        );
        print_param("Trees", self.trees);
        print_param("Leaves", self.leaves);
        print_param("Shrinkage", self.shrinkage);
        print_param("Thresholds count", self.thresholds_count);
        print_param("Min leaf support", self.min_leaf_support);
        print_param("Early stop", self.early_stop);
    }
}

pub fn main<'a>(matches: &ArgMatches<'a>) {
    let param = LambdaMARTParameter::parse(matches);
    param.print();

    let lambdamart = LambdaMART::new(param.config());
    lambdamart.init().unwrap();
    lambdamart.learn().unwrap();
}

pub fn clap_command<'a, 'b>() -> App<'a, 'b> {
    let train_common_args = super::common_args();
    // LambdaMART args
    let lambdamart_command = SubCommand::with_name("lambdamart")
        .about("Train LambdaMART")
        .args(&train_common_args)
        .arg(
            Arg::with_name("trees")
                .required_if("type", "lambdamart")
                .long("trees")
                .takes_value(true)
                .value_name("NUM")
                .default_value("1000")
                .display_order(101)
                .help("Number of trees"),
        )
        .arg(
            Arg::with_name("leaves")
                .required_if("type", "lambdamart")
                .long("leaves")
                .takes_value(true)
                .value_name("NUM")
                .default_value("10")
                .display_order(102)
                .help("Number of leaves for each tree"),
        )
        .arg(
            Arg::with_name("shrinkage")
                .required_if("type", "lambdamart")
                .long("shrinkage")
                .value_name("FACTOR")
                .takes_value(true)
                .default_value("0.1")
                .display_order(103)
                .help("Shrinkage, or learning rate"),
        )
        .arg(
            Arg::with_name("thresholds")
                .required_if("type", "lambdamart")
                .long("thresholds")
                .takes_value(true)
                .value_name("NUM")
                .default_value("256")
                .display_order(104)
                .help("Number of threshold candidates for tree spliting"),
        )
        .arg(
            Arg::with_name("min-leaf-support")
                .required_if("type", "lambdamart")
                .long("min-leaf-support")
                .takes_value(true)
                .value_name("NUM")
                .default_value("1")
                .display_order(105)
                .help("Min leaf support -- minimum #samples each leaf has to contain"),
        )
        .arg(
            Arg::with_name("early-stop")
                .required_if("type", "lambdamart")
                .long("early-stop")
                .takes_value(true)
                .value_name("NUM")
                .default_value("100")
                .display_order(106)
                .help("Stop early when no improvement is observed on validaton data in e consecutive rounds"),
        );
    lambdamart_command
}
