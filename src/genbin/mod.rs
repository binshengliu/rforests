use std::fs::File;
use std::path::{Path, PathBuf};
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
    let mut new_name = path.file_stem().unwrap().to_os_string();
    new_name.push(s);
    let mut file_name = PathBuf::from(new_name);
    if let Some(ext) = path.extension() {
        file_name.set_extension(ext);
    };

    path.with_file_name(file_name).to_str().unwrap().to_string()
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
    let stats = svmlight::FilesStats::parse(&input_files)?;
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

    // Load value maps from output files
    let mut feature_value_hash: Vec<HashMap<u32, u32>> = Vec::new();
    feature_value_hash.resize(stats.max_feature_id, HashMap::default());
    for output_name in &output_files {
        let output = File::open(&output_name)?;
        for instance in SvmLightFile::instances(output) {
            let instance = instance?;

            for feature in instance.features() {
                let hash = &mut feature_value_hash[feature.id - 1];
                *hash.entry(feature.value.round() as u32).or_insert(0) += 1;
            }
        }
    }

    // Turn hash table into vector
    let value_table: Vec<_> = feature_value_hash
        .into_iter()
        .map(|hash| {
            // The hash does not contains 0 as its key. Add it.
            let mut values =
                (0..1).chain(hash.keys().cloned()).collect::<Vec<_>>();
            values.sort();
            // println!("Sorted values: {:?}", values);
            values
        })
        .collect();

    // Find indices for each value
    let mut feature_indices: Vec<Vec<u32>> = Vec::new();
    feature_indices.resize(stats.max_feature_id, Vec::new());
    for output_name in output_files {
        let output = File::open(&output_name)?;
        for (instance_index, instance) in
            SvmLightFile::instances(output).enumerate()
        {
            let instance = instance?;

            // does not comiple // TODO some features are skipped
            for feature in instance.features() {
                let values = &value_table[feature.id - 1];
                let index = values.binary_search(&&(feature.value as u32));
                feature_indices[feature.id - 1].push(index.unwrap() as u32);
                // assert_eq!(
                //     feature_indices[feature.id - 1].len(),
                //     instance_index
                // );
            }
        }
    }

    println!("Value table 0: {:?}", value_table[0]);
    println!("Index table 0: {:?}", feature_indices[0]);

    for dist in &value_table {
        let len = dist.len();
        if len <= 2 {
            // std::collections::BitVec;
        } else if len <= ::std::u8::MAX as usize {
        } else if len <= ::std::u16::MAX as usize {
        } else if len <= ::std::u32::MAX as usize {
        }

        println!("len {}", len);
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
    stats: &svmlight::FilesStats,
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
