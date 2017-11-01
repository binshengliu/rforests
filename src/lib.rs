#![feature(ord_max_min)]
#![feature(conservative_impl_trait)]
#![feature(unboxed_closures)]
#![feature(test)]

#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate test;
extern crate scoped_threadpool;
#[macro_use]
extern crate lazy_static;
extern crate num_cpus;

use clap::App;

pub mod util;
pub mod format;
pub mod metric;
pub mod train;

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
