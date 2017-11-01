pub mod dataset;
pub mod lambdamart;

use clap::{App, Arg, ArgMatches, SubCommand};
use train::dataset::Instance;

pub fn main<'a>(matches: &ArgMatches<'a>) {
    match matches.subcommand_name() {
        Some("lambdamart") => lambdamart::main(
            matches.subcommand_matches("lambdamart").unwrap(),
        ),
        _ => (),
    }
}

/// Returns the train command.
pub fn clap_command<'a, 'b>() -> App<'a, 'b> {
    let train_command = SubCommand::with_name("train")
        .about("Train an learning algorithm")
        .subcommand(lambdamart::clap_command());

    train_command
}

/// Returns the common arguments for a learning algorithm. The display
/// order of this type of arguments ranges from 1 to 100.
fn common_args<'a, 'b>() -> Vec<Arg<'a, 'b>> {
    let common_args = vec![
        Arg::with_name("train-file")
            .short("t")
            .long("train")
            .value_name("FILE")
            .takes_value(true)
            .empty_values(false)
            .required(true)
            .display_order(1)
            .help("Training file"),
        Arg::with_name("validate-file")
            .short("v")
            .long("validate")
            .value_name("FILE")
            .takes_value(true)
            .empty_values(false)
            .display_order(2)
            .help("Validating file"),
        Arg::with_name("test-file")
            .short("T")
            .long("test")
            .value_name("FILE")
            .takes_value(true)
            .empty_values(false)
            .display_order(3)
            .help("Testing file"),
        Arg::with_name("metric")
            .short("m")
            .long("metric")
            .possible_values(&["NDCG", "DCG"])
            .default_value("NDCG")
            .display_order(4)
            .help("Metric to optimize on the training data"),
        Arg::with_name("metric-k")
            .short("k")
            .long("metric-k")
            .value_name("NUM")
            .requires("metric")
            .default_value("10")
            .display_order(5)
            .help("K value for metrics"),
    ];

    common_args
}

/// Evaluate on an instance.
pub trait Evaluate {
    fn evaluate(&self, instance: &Instance) -> f64;
}
