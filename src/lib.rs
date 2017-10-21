#![feature(ord_max_min)]
#![feature(conservative_impl_trait)]
#![feature(unboxed_closures)]

#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate docopt;
extern crate env_logger;

use std::env;
use clap::{App, Arg, SubCommand};

pub mod util;
pub mod format;
pub mod metric;
mod train;

use util::Result;

pub fn main() {
    env_logger::init().unwrap();

    let train_command = train::clap_command();

    let matches = App::new("rforests")
        .version(crate_version!())
        .author(crate_authors!())
        .about("A Rust library of tree-based learning algorithms")
        .subcommand(train_command)
        .get_matches();

    match matches.subcommand_name() {
        Some("train") => train::main(
            matches.subcommand_matches("train").unwrap(),
        ),
        _ => (),
    }
}
