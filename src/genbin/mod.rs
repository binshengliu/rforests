use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
// use std::error::Error;
use rforests::util::{Result};
use std::collections::HashMap;

pub mod parse;
pub mod feature;

use self::parse::*;

#[derive(Debug, Deserialize)]
pub struct Args {
    arg_file: Vec<String>,
    flag_ranking: bool,

    flag_help: bool,
}

pub const USAGE: &'static str = "
Generate binary files

Usage:
    rforests genbin [--ranking] <file>...
    rforests genbin (-h | --help)

Options:
    -r, --ranking               Support ranking
    -h, --help                  Display this message
";

const MAX_FEATURE_VALUE: u32 = ::std::i16::MAX as u32 - 1;
pub fn execute(args: Args) -> Result<()> {
    debug!("rforests genbin args: {:?}", args);
    let mut stats = SampleStats::parse(&args.arg_file)?;

    // stats.iter().map(|(feature_index, stat)| {
    //     0
    // });

    // for (feature_index, stat) in &mut stats {
    //     let range = stat.max - stat.min;
    //     if range < MAX_FEATURE_VALUE as f64 {
    //         stat.factor = MAX_FEATURE_VALUE as f64 / range;
    //     } else {
    //         stat.factor = MAX_FEATURE_VALUE as f64 / (range + 1.0).ln();
    //         stat.log = true;
    //     }
    // }
    Ok(())
}

fn write_stats(stats: HashMap<u32, FeatureStat>) -> Result<()> {
    let mut sorted: Vec<(u32, FeatureStat)> = stats.iter().map(|(index, stat)| (*index, *stat)).collect();
    sorted.sort_by_key(|&(index, _)| index);

    println!("{:?}", sorted);

    let mut f = File::create("data/stats.txt")?;
    f.write_all("FeatureIndex\tName\tMin\tMax\n".as_bytes())?;
    for (index, stat) in sorted {
        // let s = format!("{}\t{}\t{}\t{}\n", index, "null", stat.min, stat.max);
        // f.write_all(s.as_bytes())?;
    }
    Ok(())
}

// pub fn run<'de, Flags: Deserialize<'de>>(
//             exec: fn(Flags, &Config) -> Result<()>,
//             config: &Config,
//             usage: &str,
//             args: &[String],
//             options_first: bool) -> Result<()> {
//     let docopt = Docopt::new(usage).unwrap()
//         .options_first(options_first)
//         .argv(args.iter().map(|s| &s[..]))
//         .help(true);

//     let flags = docopt.deserialize().map_err(|e| {
//         let code = if e.fatal() {1} else {0};
//         CliError::new(e.to_string().into(), code)
//     })?;

//     exec(flags, config);

//     // let mut f = File::open(config.filename)?;

//     // let mut contents = String::new();
//     // f.read_to_string(&mut contents)?;

//     // println!("With text:\n{}", contents);

//     // let mut results: Vec<String> = Vec::new();

//     // for line in contents.lines() {}

//     unimplemented!()
// }

