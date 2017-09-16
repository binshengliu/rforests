use std::error;
type Result<T> = ::std::result::Result<T, Box<error::Error>>;

#[derive(Default, Debug, PartialEq)]
struct FeaturePair {
    index: u32,
    value: f64,
}

impl FeaturePair {
    pub fn new(index: u32, value: f64) -> FeaturePair {
        FeaturePair {
            index: index,
            value: value,
        }
    }
}

use std::str::FromStr;
impl FromStr for FeaturePair {
    type Err = Box<error::Error>;

    fn from_str(s: &str) -> ::std::result::Result<Self, Self::Err> {
        let v: Vec<&str> = s.split(':').collect();
        if v.len() != 2 {
            Err(format!("Invalid string: {}", s))?;
        }

        let index = v[0].parse::<u32>()?;
        let value = v[1].parse::<f64>()?;

        Ok(FeaturePair {
            index: index,
            value: value,
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct Line {
    target: u32,
    qid: u64,
    pairs: Vec<FeaturePair>,
}

impl Line {
    fn parse_target(target: &str) -> Result<u32> {
        let target = target.parse::<u32>()?;
        Ok(target)
    }

    fn parse_qid(qid: &str) -> Result<u64> {
        let v: Vec<&str> = qid.split(':').collect();
        if v.len() != 2 {
            Err(format!("Invalid qid field: {}", qid))?;
        }

        if v[0] != "qid" {
            Err(format!("Invalid qid field: {}", v[0]))?;
        }

        let qid = v[1].parse::<u64>()?;

        Ok(qid)
    }

    fn parse_pairs(fields: &[&str]) -> Result<Vec<FeaturePair>> {
        fields
            .iter()
            .map(|s| FeaturePair::from_str(s))
            .collect::<Result<_>>()
    }
}

impl FromStr for Line {
    type Err = Box<error::Error>;

    fn from_str(s: &str) -> ::std::result::Result<Self, Self::Err> {
        let line = s.trim();
        let line: &str = line.split('#').next().unwrap().trim();
        if line.starts_with('@') {
            Err(format!("Meta line not supported yet"))?
        } else {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 2 {
                Err(format!("Invalid line"))?;
            }

            let target = Line::parse_target(fields[0])?;
            let qid = Line::parse_qid(fields[1])?;
            let pairs: Vec<FeaturePair> = Line::parse_pairs(&fields[2..])?;

            Ok(Line {
                target: target,
                qid: qid,
                pairs: pairs,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureStat {
    min: f64,
    max: f64,
    factor: f64,
    log: bool,
}

use std;
use std::fs::File;
use std::io::{self, BufReader};
use std::io::prelude::*;
use std::collections::HashMap;
const MAX_FEATURE_VALUE: u32 = std::i16::MAX as u32 - 1;

pub fn generate_statistics(filename: &str) -> Result<HashMap<u32, FeatureStat>> {
    let f = File::open(filename)?;
    let f = BufReader::new(f);
    let mut stats: HashMap<u32, FeatureStat> = HashMap::new();
    let mut max_feature_index = 0;

    println!("Processing file {}", filename);
    for (line_index, line) in f.lines().enumerate() {
        let line = line?;
        let line = Line::from_str(line.as_str())?;
        for pair in line.pairs.iter() {
            let mut stat = stats.entry(pair.index).or_insert(
                FeatureStat { min: 0.0, max: 0.0, factor: 0.0, log: false },
            );

            if pair.index > max_feature_index {
                max_feature_index = pair.index;
            }

            if pair.value > stat.max {
                stat.max = pair.value;
            }

            if pair.value < stat.min {
                stat.min = pair.value
            }
        }

        // Notify the user every 5000 lines.
        if (line_index + 1) % 5000 == 0 {
            println!("Processed {} lines", line_index + 1);
        }
    }

    for (id, stat) in &mut stats {
        println!("{}: {:?}", id, stat);
        let range = stat.max - stat.min;
        if range < MAX_FEATURE_VALUE as f64 {
            stat.factor = MAX_FEATURE_VALUE as f64 / range;
        } else {
            stat.factor = MAX_FEATURE_VALUE as f64 / (range + 1.0).ln();
            stat.log = true;
        }
    }

    let mut sorted: Vec<(u32, FeatureStat)> = stats.iter().map(|(index, stat)| (*index, *stat)).collect();
    sorted.sort_by_key(|&(index, _)| index);

    println!("{:?}", sorted);

    let mut f = File::create("data/stats.txt")?;
    f.write_all("FeatureIndex\tName\tMin\tMax\n".as_bytes())?;
    for (index, stat) in sorted {
        let s = format!("{}\t{}\t{}\t{}\n", index, "null", stat.min, stat.max);
        f.write_all(s.as_bytes())?;
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_new() {
        let p = FeaturePair::from_str("1:3").unwrap();
        assert_eq!(p.index, 1);
        assert_eq!(p.value, 3.0);
    }

    #[test]
    fn test_pair_only_index() {
        assert!(FeaturePair::from_str("1:").is_err());
    }

    #[test]
    fn test_pair_only_value() {
        assert!(FeaturePair::from_str(":3").is_err());
    }

    #[test]
    fn test_pair_too_many_colons() {
        assert!(FeaturePair::from_str("1:2:3").is_err());
    }

    #[test]
    fn test_pair_no_colons() {
        let p = FeaturePair::from_str("1");
        assert!(p.is_err());
    }

    #[test]
    fn test_line_parse() {
        let s = "0 qid:3864 1:3.000000 2:9.000000 # 3:10.0";
        let p = Line::from_str(s).unwrap();
        assert_eq!(p.target, 0);
        assert_eq!(p.qid, 3864);
        assert_eq!(
            p.pairs,
            vec![FeaturePair::new(1, 3.0), FeaturePair::new(2, 9.0)]
        );
    }

    #[test]
    fn test_line_meta() {
        let s = "@feature";
        let p = Line::from_str(s);
        assert!(p.is_err());
    }

    #[test]
    fn test_generate_statistics() {
        let result = generate_statistics("data/train.txt").unwrap();
        println!("{:?}", result);
        // assert!(false);
    }
}

// @Feature index:2 name:abc
// Record min and max value for each feature.
// Max feature Id.
