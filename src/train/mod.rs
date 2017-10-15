// pub mod ranknet;
pub mod histogram;
pub mod lambdamart;
pub mod regression_tree;

use util::Result;

#[derive(Debug, Deserialize)]
pub struct Args {
    flag_ranking: bool,
    flag_config: String,
    flag_train: String,
    flag_validation: String,
    flag_output: String,

    flag_help: bool,
}

pub const USAGE: &'static str = "
Train a leaner

Usage:
    rforests train [--ranking] --config <file> --train <file> --validation <file> --output <file>
    rforests train (-h | --help)

Options:
    -h, --help                             Display this message
    -r, --ranking                          Support ranking
    -c <file>, --config <file>             Specify config file
    -o <file>, --output <file>             Specify output file
    -t <file>, --train <file>              Specify training input file
    -v <file>, --validation <validation>   Specify validation input file
";

pub fn execute(args: Args) -> Result<()> {
    debug!("rforests train args: {:?}", args);
    Ok(())
}
