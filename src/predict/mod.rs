use rforests::util::{Result};

#[derive(Debug, Deserialize)]
pub struct Args {
    flag_model: String,
    flag_tree: String,
    flag_test: String,
    flag_output: String,
}

pub const USAGE: &'static str = "
Rust implementation of jforests

Usage:
    rforests predict --model <file> --tree <type> --test <file> --output <file>
    rforests predict (-h | --help | --version)

Options:
    -m <mode>, --model <model>  Specify model file
    -t <tree>, --tree <type>    Specify tree type
    -s <file>, --test <file>    Specify test file
    -o <file>, --output <file>  Specify output file
    -h, --help                  Display this message
";

pub fn execute(args: Args) -> Result<()> {
    debug!("rforests predict args: {:?}", args);
    Ok(())
}
