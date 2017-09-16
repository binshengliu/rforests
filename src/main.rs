#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate env_logger;
extern crate docopt;
extern crate rforests;

use std::env;
use std::path::{Path, PathBuf};

use docopt::Docopt;

use rforests::util::Result;

#[derive(Debug, Deserialize)]
pub struct Args {
    arg_command: String,
    arg_args: Vec<String>,
}

const USAGE: &'static str = "
Rust implementation of jforests

Usage:
    rforests <command> [<args>...]
    rforests [-h | --help | --version]

Options:
    -h, --help              Display this message
    --version               Print version info and exit

Subcommands:
    genbin      Convert data set to binary format
    train       Train a learner
    predict     Predict

See 'rforests <command> -h' for more information on a specific command.
";

fn main() {
    env_logger::init().unwrap();

    let mut argv = env::args().collect::<Vec<_>>();

    debug!("rforests command arguments: {:?}", argv);

    if argv.len() == 1 {
        argv.push("-h".to_string());
    }

    // Setting options_first to true makes options for sub commands be
    // parsed as arguments here. Otherwise Docopt will complain a
    // "Unknown flag"" error.
    if let Err(e) = rforests::call_entry_with_args(execute, USAGE, &argv, true) {
        error!("Error occured {}", e);
        std::process::exit(1);
    }
}

pub fn execute(args: Args) -> Result<()> {
    let argv = env::args().collect::<Vec<_>>();
    execute_sub_command(&args.arg_command, &argv)
}

macro_rules! each_subcommand{
    ($mac:ident) => {
        $mac!(genbin);
        $mac!(train);
        $mac!(predict);
    }
}

macro_rules! declare_mod {
    ($name:ident) => ( pub mod $name; )
}
each_subcommand!(declare_mod);

fn execute_sub_command(command: &str, argv: &[String]) -> Result<()> {
    macro_rules! cmd {
        ($name:ident) => (if command == stringify!($name).replace("_", "-") {
            return rforests::call_entry_with_args($name::execute, $name::USAGE, &argv, false);
        })
    }

    each_subcommand!(cmd);

    Err("Unknown command!")?
}
