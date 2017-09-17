use std;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::collections::HashMap;
use std::str::FromStr;
use util::Result;

// Format of the example file. http://svmlight.joachims.org/
// <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
// <target> .=. +1 | -1 | 0 | <float>
// <feature> .=. <integer> | "qid"
// <value> .=. <float>
// <info> .=. <string>

#[derive(Default, Debug, PartialEq)]
struct Feature {
    id: u32,
    feature_value: f64,
}

impl Feature {
    fn new(id: u32, feature_value: f64) -> Feature {
        Feature {
            id: id,
            feature_value: feature_value,
        }
    }
}

impl FromStr for Feature {
    type Err = Box<std::error::Error>;

    fn from_str(s: &str) -> ::std::result::Result<Self, Self::Err> {
        let v: Vec<&str> = s.split(':').collect();
        if v.len() != 2 {
            Err(format!("Invalid string: {}", s))?;
        }

        let id = v[0].parse::<u32>()?;
        let feature_value = v[1].parse::<f64>()?;

        Ok(Feature {
            id: id,
            feature_value: feature_value,
        })
    }
}

#[derive(Debug, PartialEq)]
struct Instance {
    target: u32,
    qid: u64,
    features: Vec<Feature>,
}

impl Instance {
    fn features(&self) -> std::slice::Iter<Feature> {
        self.features.iter()
    }

    fn features_mut(&mut self) -> std::slice::IterMut<Feature> {
        self.features.iter_mut()
    }

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

    fn parse_features(fields: &[&str]) -> Result<Vec<Feature>> {
        fields
            .iter()
            .map(|s| Feature::from_str(s))
            .collect::<Result<_>>()
    }
}

impl FromStr for Instance {
    type Err = Box<std::error::Error>;

    fn from_str(s: &str) -> ::std::result::Result<Self, Self::Err> {
        let instance = s.trim();
        let instance: &str = instance.split('#').next().unwrap().trim();
        if instance.starts_with('@') {
            Err(format!("Meta instance not supported yet"))?
        } else {
            let fields: Vec<&str> = instance.split_whitespace().collect();
            if fields.len() < 2 {
                Err(format!("Invalid instance"))?;
            }

            let target = Instance::parse_target(fields[0])?;
            let qid = Instance::parse_qid(fields[1])?;
            let features: Vec<Feature> = Instance::parse_features(&fields[2..])?;

            Ok(Instance {
                target: target,
                qid: qid,
                features: features,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FeatureStat {
    id: u32,
    min: f64,
    max: f64,
    factor: f64,
    log: bool,
}

#[derive(Debug)]
pub struct SampleStats {
    min_feature_id: u32,
    max_feature_id: u32,
    feature_stats: HashMap<u32, FeatureStat>,
}

impl SampleStats {
    pub fn parse(files: &[String]) -> Result<SampleStats> {
        let mut stats = SampleStats {
            min_feature_id: std::u32::MAX,
            max_feature_id: 0,
            feature_stats: HashMap::new(),
        };

        for file in files {
            stats.update_stats_from_file(file)?;
        }

        Ok(stats)
    }

    fn update(&mut self, feature_id: u32, feature_value: f64) {
        let stat = self.feature_stats.entry(feature_id).or_insert(FeatureStat {
            id: feature_id,
            min: std::f64::MAX,
            max: std::f64::MIN,
            factor: 0.0,
            log: false,
        });

        self.max_feature_id = self.max_feature_id.max(feature_id);
        self.min_feature_id = self.min_feature_id.min(feature_id);
        stat.max = stat.max.max(feature_value);
        stat.min = stat.min.min(feature_value);
    }

    fn update_stats_from_file(&mut self, filename: &str) -> Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        debug!("Processing file {}", filename);
        for (line_index, line) in reader.lines().enumerate() {
            let line = line?;
            let instance = match Instance::from_str(line.as_str()) {
                Ok(instance) => instance,
                Err(e) => {
                    error!("Parse instance error: {}", e.description());
                    continue;
                }
            };

            for feature in instance.features() {
                self.update(feature.id, feature.feature_value);
            }

            // Notify the user every 5000 lines.
            if (line_index + 1) % 5000 == 0 {
                info!("Processed {} lines", line_index + 1);
            }
        }

        Ok(())
    }
}

pub struct SvmLightFile {
    filename: String,
}

impl SvmLightFile {
    pub fn new(filename: &str) -> Result<SvmLightFile> {
        Ok(SvmLightFile { filename: filename.to_string() })
    }

    pub fn lines(&self) -> Result<std::io::Lines<std::io::BufReader<File>>> {
        let file = File::open(&self.filename)?;
        let reader = BufReader::new(file);
        Ok(reader.lines())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_new() {
        let p = Feature::from_str("1:3").unwrap();
        assert_eq!(p.id, 1);
        assert_eq!(p.feature_value, 3.0);
    }

    #[test]
    fn test_pair_only_id() {
        assert!(Feature::from_str("1:").is_err());
    }

    #[test]
    fn test_pair_only_value() {
        assert!(Feature::from_str(":3").is_err());
    }

    #[test]
    fn test_pair_too_many_colons() {
        assert!(Feature::from_str("1:2:3").is_err());
    }

    #[test]
    fn test_pair_no_colons() {
        let p = Feature::from_str("1");
        assert!(p.is_err());
    }

    #[test]
    fn test_line_parse() {
        let s = "0 qid:3864 1:3.000000 2:9.000000 # 3:10.0";
        let p = Instance::from_str(s).unwrap();
        assert_eq!(p.target, 0);
        assert_eq!(p.qid, 3864);
        assert_eq!(p.features, vec![Feature::new(1, 3.0), Feature::new(2, 9.0)]);
    }

    #[test]
    fn test_line_meta() {
        let s = "@feature";
        let p = Instance::from_str(s);
        assert!(p.is_err());
    }
}

// fn write_stats(stats: HashMap<u32, FeatureStat>) -> Result<()> {
//     let mut sorted: Vec<(u32, FeatureStat)> = stats.iter().map(|(index, stat)| (*index, *stat)).collect();
//     sorted.sort_by_key(|&(index, _)| index);

//     println!("{:?}", sorted);

//     let mut f = File::create("data/stats.txt")?;
//     f.write_all("FeatureIndex\tName\tMin\tMax\n".as_bytes())?;
//     for (index, stat) in sorted {
//         // let s = format!("{}\t{}\t{}\t{}\n", index, "null", stat.min, stat.max);
//         // f.write_all(s.as_bytes())?;
//     }
//     Ok(())
// }

// @Feature id:2 name:abc
// Record min and max feature_value for each feature.
// Max feature Id.
