use std::fs::File;
use std::path::Path;
// use std::io::prelude::*;
// use std::io::BufReader;
// use std::error::Error;
use std::collections::HashMap;

use util::Result;
use format::svmlight;
use format::svmlight::SvmLightFile;

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

pub fn append_to_file_name(origin: &str, s: &str) -> String {
    let path = Path::new(origin);
    let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
    let mut new_name = stem + s;

    if let Some(ext) = path.extension() {
        new_name += ext.to_str().unwrap();
    }

    path.with_file_name(new_name).to_str().unwrap().to_string()
}

pub fn change_extension(origin: &str, new_ext: &str) -> String {
    Path::new(origin)
        .with_extension(new_ext)
        .to_str()
        .unwrap()
        .to_string()
}

pub fn execute(args: Args) -> Result<()> {
    debug!("rforests genbin args: {:?}", args);
    let input_files = args.arg_file.clone();

    // Generate statistics from the files
    let stats = svmlight::SampleStats::parse(&input_files)?;
    let feature_scales = stats.feature_scales();
    let output_files: Vec<_> = input_files
        .iter()
        .map(|input| append_to_file_name(input, "-compact"))
        .collect();

    // Scale the input file and trim zeros
    for (input_name, output_name) in
        input_files.iter().zip(output_files.iter())
    {
        info!("Converting {} to {}", input_name, output_name);

        let input = File::open(input_name.as_str())?;
        let output = File::create(output_name)?;
        SvmLightFile::write_compact_format(input, output, &feature_scales)?;
    }

    let mut feature_value_hash: Vec<HashMap<u32, u32>> = Vec::new();
    feature_value_hash.resize(stats.max_feature_id, HashMap::default());
    // Load value maps from output files
    for output_name in output_files {
        let output = File::open(output_name.as_str())?;
        for instance in SvmLightFile::instances(output) {
            let instance = instance?;

            for feature in instance.features() {
                let hash = &mut feature_value_hash[feature.id - 1];
                *hash.entry(feature.value.round() as u32).or_insert(0) += 1;
            }
        }
    }

    // Generate bin names
    let bin_files: Vec<_> = input_files
        .iter()
        .map(|input| change_extension(input, "bin"))
        .collect();
    for bin_name in bin_files {}

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

fn convert(
    input: &str,
    output: &str,
    stats: &svmlight::SampleStats,
) -> Result<()> {
    // let file = svmlight::SvmLightFile::open(input)?;

    // 1. Scale the values according to svmlight
    // for line in file.instances() {}

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
