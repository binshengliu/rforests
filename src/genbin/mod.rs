// use std::fs::File;
// use std::io::prelude::*;
// use std::io::BufReader;
// use std::error::Error;
// use std::collections::HashMap;

use util::Result;
use format::svmlight;

pub mod feature;

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

pub struct FeatureScale {
    logarithm: bool,
    scale: f64,
}

const MAX_FEATURE_VALUE: f64 = ::std::i16::MAX as f64 - 1.0;
pub fn execute(args: Args) -> Result<()> {
    debug!("rforests genbin args: {:?}", args);
    let mut stats = svmlight::SampleStats::parse(&args.arg_file)?;

    println!("Stats: {:?}", stats);

    let scales: Vec<_> = stats.feature_stats().map(|fstats| {
        let range = fstats.max - fstats.min;
        if range < MAX_FEATURE_VALUE {
            FeatureScale {logarithm: false, scale: MAX_FEATURE_VALUE / range}
        } else {
            FeatureScale {logarithm: true, scale: MAX_FEATURE_VALUE / (range + 1.0).ln()}
        }
    }).collect();

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

fn convert(input: &str, output: &str, stats: &svmlight::SampleStats) -> Result<()> {
    let file = svmlight::SvmLightFile::open(input)?;

    // 1. Scale the values according to svmlight
    for line in file.instances() {

    }

    // Load the values into a hash map
    // Convert the hash map into a sorted vec of values
    // Update each feature to contain index into the vec
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

