use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::error::Error;
// use std::io::Result;

mod parse;

pub struct Config {
    pub query: String,
    pub filename: String,
}

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 3 {
            return Err("not enough arguments");
        }

        let query = args[1].clone();
        let filename = args[2].clone();

        Ok(Config { query, filename })
    }
}

pub fn run(config: Config) -> std::io::Result<()> {
    let mut f = File::open(config.filename)?;

    let mut contents = String::new();
    f.read_to_string(&mut contents)?;

    println!("With text:\n{}", contents);

    let mut results: Vec<String> = Vec::new();

    for line in contents.lines() {}

    Ok(())
}

pub fn generate_statistics(filename: &str) -> std::io::Result<()> {
    let f = File::open(filename)?;
    let f = BufReader::new(f);

    println!("Reading file");
    for line in f.lines().take(3) {
        println!("{}", line.unwrap());
        println!("");
    }

    Ok(())
}
