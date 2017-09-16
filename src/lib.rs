use std::process;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs;

extern crate serde;
extern crate docopt;

use docopt::Docopt;

use util::{Result};
pub mod util;

use serde::Deserialize;
pub fn call_entry_with_args<'de, Args: Deserialize<'de>>(
    exec: fn(Args) -> Result<()>,
    usage: &str,
    argv: &[String],
    options_first: bool,
) -> Result<()> {
    let docopt = Docopt::new(usage)
        .unwrap()
        .options_first(options_first)
        .argv(argv.iter().map(|s| &s[..]))
        .help(true)
        .version(Some(String::from("0.1.0")));

    let args = docopt.deserialize().unwrap_or_else(|e| e.exit());

    exec(args)
}
