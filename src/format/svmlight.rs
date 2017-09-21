use std;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::str::FromStr;
use util::Result;

// Format of the example file. http://svmlight.joachims.org/
// <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
// <target> .=. +1 | -1 | 0 | <float>
// <feature> .=. <integer> | "qid"
// <value> .=. <float>
// <info> .=. <string>

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Feature {
    pub id: usize,
    pub value: f64,
}

impl Feature {
    pub fn new(id: usize, value: f64) -> Feature {
        Feature {
            id: id,
            value: value,
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

        let id = v[0].parse::<usize>()?;
        let value = v[1].parse::<f64>()?;

        Ok(Feature {
            id: id,
            value: value,
        })
    }
}

impl ToString for Feature {
    fn to_string(&self) -> String {
        format!("{}:{}", self.id, self.value as u32)
    }
}

const MAX_SCALE_VALUE: f64 = ::std::i16::MAX as f64 - 1.0;

pub struct FeatureScale {
    logarithm: bool,
    scale: f64,
    min: f64,
}

impl FeatureScale {
    pub fn scale(&self, value: f64) -> f64 {
        let output = if self.logarithm {
            (value - self.min + 1.0).ln() * self.scale
        } else {
            (value - self.min) * self.scale
        };
        output.round()
    }
}

impl<'a> From<&'a FeatureStat> for FeatureScale {
    fn from(fstat: &'a FeatureStat) -> Self {
        let range = fstat.max - fstat.min;
        if range < MAX_SCALE_VALUE {
            FeatureScale {
                logarithm: false,
                scale: MAX_SCALE_VALUE / range,
                min: fstat.min,
            }
        } else {
            FeatureScale {
                logarithm: true,
                scale: MAX_SCALE_VALUE / (range + 1.0).ln(),
                min: fstat.min,
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Instance {
    target: u32,
    qid: u64,
    features: Vec<Feature>,
}

impl Instance {
    pub fn features(&self) -> std::slice::Iter<Feature> {
        self.features.iter()
    }

    pub fn features_mut(&mut self) -> std::slice::IterMut<Feature> {
        self.features.iter_mut()
    }

    pub fn scale_and_trim_zeros(&mut self, scaling: &Vec<FeatureScale>) {
        let scale_f = |f: &Feature| {
            Feature {
                id: f.id,
                value: scaling[f.id - 1].scale(f.value),
            }
        };

        self.features = self.features
            .iter()
            .map(scale_f)
            .filter(|f| f.value.round() as u32 != 0)
            .collect();
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
        let instance: &str = s.trim().split('#').next().unwrap().trim();
        if instance.starts_with('@') {
            Err(format!("Meta instance not supported yet"))?
        } else {
            let fields: Vec<&str> = instance.split_whitespace().collect();
            if fields.len() < 2 {
                Err(format!("Invalid instance"))?;
            }

            let target = Instance::parse_target(fields[0])?;
            let qid = Instance::parse_qid(fields[1])?;
            let features: Vec<Feature> =
                Instance::parse_features(&fields[2..])?;

            Ok(Instance {
                target: target,
                qid: qid,
                features: features,
            })
        }
    }
}

impl ToString for Instance {
    fn to_string(&self) -> String {
        let features = self.features
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        if features.is_empty() {
            vec![
                self.target.to_string(),
                "qid:".to_string() + &self.qid.to_string(),
            ].join(" ")
        } else {
            vec![
                self.target.to_string(),
                "qid:".to_string() + &self.qid.to_string(),
                features,
            ].join(" ")
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct FeatureStat {
    pub id: usize,
    pub min: f64,
    pub max: f64,
}

#[derive(Default, Debug)]
pub struct FilesStats {
    pub max_feature_id: usize,
    feature_stats: Vec<FeatureStat>,
    instances_count: Vec<(String, usize)>,
}

impl FilesStats {
    pub fn parse(files: &[String]) -> Result<FilesStats> {
        let mut stats = FilesStats::default();

        for file in files {
            debug!("Performing statistics analysis of {}", file);
            stats.update_stats_from_file(file)?;
            debug!("Completed perform statistical analysis of {}", file);
        }

        Ok(stats)
    }

    pub fn instances_count(&self, file_name: &str) -> usize {
        let result = self.instances_count.iter().find(
            |tuple| tuple.0 == file_name,
        );
        match result {
            Some(&(_, count)) => count,
            None => 0,
        }
    }

    pub fn feature_count(&self) -> usize {
        self.max_feature_id
    }

    pub fn feature_stats(&self) -> std::slice::Iter<FeatureStat> {
        self.feature_stats.iter()
    }

    pub fn feature_scales(&self) -> Vec<FeatureScale> {
        self.feature_stats().map(FeatureScale::from).collect()
    }

    fn update(&mut self, feature_id: usize, value: f64) {
        // feature_id-1 is used as vec index
        if feature_id > self.feature_stats.len() {
            self.feature_stats.resize(
                feature_id,
                FeatureStat::default(),
            );
        }

        let stat = &mut self.feature_stats[feature_id - 1];

        stat.id = feature_id;
        stat.max = stat.max.max(value);
        stat.min = stat.min.min(value);

        self.max_feature_id = self.max_feature_id.max(feature_id);
    }

    fn update_stats_from_file(&mut self, filename: &str) -> Result<()> {
        let file = File::open(filename)?;

        let mut instance_count = 0;
        for (line_index, instance) in
            SvmLightFile::instances(file).enumerate()
        {
            let instance = instance?;
            instance_count += 1;

            for feature in instance.features() {
                self.update(feature.id, feature.value);
            }

            // Notify the user every 5000 lines.
            if (line_index + 1) % 5000 == 0 {
                info!("Processed {} lines", line_index + 1);
            }
        }

        self.instances_count.push((filename.to_string(), instance_count));

        Ok(())
    }
}

pub struct SvmLightFile {}

impl SvmLightFile {
    // Returning an abstract type is not well supported now. The Rust
    // team is working on it:
    // https://stackoverflow.com/questions/27535289/correct-way-to-return-an-iterator/27535594#27535594
    // https://github.com/rust-lang/rfcs/blob/master/text/1522-conservative-impl-trait.md
    pub fn instances(file: File) -> Box<Iterator<Item = Result<Instance>>> {
        // Bring Error::description() into scope
        use std::error::Error;

        let buf_reader = BufReader::new(file);

        let iter = buf_reader
            .lines()
            // Filter empty line and comment line
            .filter(|result| match result {
                &Ok(ref line) => {
                    let trimed = line.trim();
                    !trimed.is_empty() && !trimed.starts_with("#")
                }
                &Err(_) => true,
            })
            .map(|result| {
                result
                // Change the error type to match the function signature
                .map_err(|e| e.description().into())
                .and_then(|line| {
                    Instance::from_str(line.as_str())
                })
            });
        Box::new(iter)
    }

    pub fn write_compact_format(
        input: File,
        mut output: File,
        scales: &Vec<FeatureScale>,
    ) -> Result<()> {
        for (index, instance) in SvmLightFile::instances(input).enumerate() {
            let mut instance = instance?;
            instance.scale_and_trim_zeros(scales);
            let line = instance.to_string() + "\n";
            output.write_all(line.as_bytes())?;

            if (index + 1) % 5000 == 0 {
                info!("Written {} lines", index + 1);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_new() {
        let p = Feature::from_str("1:3").unwrap();
        assert_eq!(p.id, 1);
        assert_eq!(p.value, 3.0);
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
        assert_eq!(
            p.features,
            vec![Feature::new(1, 3.0), Feature::new(2, 9.0)]
        );
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
// Record min and max value for each feature.
// Max feature Id.
