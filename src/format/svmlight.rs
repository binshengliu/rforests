use std;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use util::*;
use train::dataset::*;

// Format of the example file. http://svmlight.joachims.org/
// <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
// <target> .=. +1 | -1 | 0 | <float>
// <feature> .=. <integer> | "qid"
// <value> .=. <float>
// <info> .=. <string>

// #[derive(Copy, Clone, Default, Debug, PartialEq)]
// pub struct Feature {
//     pub id: usize,
//     pub value: f64,
// }

// impl Feature {
//     pub fn new(id: usize, value: f64) -> Feature {
//         Feature {
//             id: id,
//             value: value,
//         }
//     }
// }

// impl FromStr for Feature {
//     type Err = Box<std::error::Error>;

//     fn from_str(s: &str) -> ::std::result::Result<Self, Self::Err> {
//         let v: Vec<&str> = s.split(':').collect();
//         if v.len() != 2 {
//             Err(format!("Invalid string: {}", s))?;
//         }

//         let id = v[0].parse::<usize>()?;
//         let value = v[1].parse::<f64>()?;

//         Ok(Feature {
//             id: id,
//             value: value,
//         })
//     }
// }

// impl ToString for Feature {
//     fn to_string(&self) -> String {
//         format!("{}:{}", self.id, self.value as u32)
//     }
// }

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

    fn update(&mut self, feature_id: Id, value: Value) {
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

            for (id, value) in instance.value_iter() {
                self.update(id, value);
            }

            // Notify the user every 5000 lines.
            if (line_index + 1) % 5000 == 0 {
                info!("Processed {} lines", line_index + 1);
            }
        }

        self.instances_count.push(
            (filename.to_string(), instance_count),
        );

        Ok(())
    }
}

pub struct SvmLightFile;

impl SvmLightFile {
    /// Read from reader and create (label, qid, values) tuple for
    /// each line.
    pub fn parse_reader<R>(
        reader: R,
    ) -> impl Iterator<Item = Result<(Value, Id, Vec<Value>)>>
    where
        R: std::io::Read,
    {
        // Bring Error::description() into scope
        use std::error::Error;

        let buf_reader = BufReader::new(reader);

        buf_reader
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
                    SvmLightFile::parse_str(line.as_str())
                })
            })
    }

    /// Read from reader and create Instance struct for each line.
    pub fn instances<R>(reader: R) -> impl Iterator<Item = Result<Instance>>
    where
        R: std::io::Read,
    {
        SvmLightFile::parse_reader(reader).map(|parse_result| {
            parse_result.map(|(label, qid, values)| {
                Instance::new(label, qid, values)
            })
        })
    }

    /// Parse "3".
    fn parse_label(label: &str) -> Result<Value> {
        let label = label.parse::<Value>()?;
        Ok(label)
    }

    /// Parse "qid:3333".
    fn parse_qid(qid: &str) -> Result<Id> {
        let v: Vec<&str> = qid.split(':').collect();
        if v.len() != 2 {
            Err(format!("Invalid qid field: {}", qid))?;
        }

        if v[0] != "qid" {
            Err(format!("Invalid qid field: {}", v[0]))?;
        }

        let qid = v[1].parse::<Id>()?;

        Ok(qid)
    }

    /// Parse &["1:3.0" "3:4.0"] into Vec of values. Absent indices
    /// are filled with 0.0. The example above would result vec![0.0,
    /// 3.0, 0.0, 4.0].
    fn parse_values(fields: &[&str]) -> Result<Vec<f64>> {
        fn parse(s: &str) -> Result<(Id, Value)> {
            let v: Vec<&str> = s.split(':').collect();
            if v.len() != 2 {
                Err(format!("Invalid string: {}", s))?;
            }

            let id = v[0].parse::<Id>()?;
            let value = v[1].parse::<Value>()?;

            Ok((id, value))
        }

        // (id, value) pairs
        let v: Vec<(Id, Value)> =
            fields.iter().map(|&s| parse(s)).collect::<Result<_>>()?;
        let max_id = v.iter().max_by_key(|e| e.0).unwrap().0;
        let mut ret: Vec<f64> = Vec::with_capacity(max_id + 1);
        ret.resize(max_id, 0.0);
        for &(id, value) in v.iter() {
            ret[(id - 1) as usize] = value;
        }

        Ok(ret)
    }

    /// Parse "3.0 qid:3864 1:3.000000 2:9.000000 4:3.0 # 3:10.0".
    pub fn parse_str(s: &str) -> Result<(Value, Id, Vec<Value>)> {
        let line: &str = s.trim().split('#').next().unwrap().trim();
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            Err(format!("Invalid line"))?;
        }

        let label = SvmLightFile::parse_label(fields[0])?;
        let qid = SvmLightFile::parse_qid(fields[1])?;
        let values: Vec<Value> = SvmLightFile::parse_values(&fields[2..])?;

        Ok((label, qid, values))
    }

    // pub fn write_compact_format(
    //     input: File,
    //     mut output: File,
    //     scales: &Vec<FeatureScale>,
    // ) -> Result<()> {
    //     for (index, instance) in SvmLightFile::instances(input).enumerate() {
    //         let mut instance = instance?;
    //         instance.scale_and_trim_zeros(scales);
    //         let line = instance.to_string() + "\n";
    //         output.write_all(line.as_bytes())?;

    //         if (index + 1) % 5000 == 0 {
    //             info!("Written {} lines", index + 1);
    //         }
    //     }
    //     Ok(())
    // }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_parse() {
        let s = "3.0 qid:3864 1:3.000000 2:9.000000 4:3.0 # 3:10.0";
        let (label, qid, values) = SvmLightFile::parse_str(s).unwrap();
        assert_eq!(label, 3.0);
        assert_eq!(qid, 3864);
        assert_eq!(values, vec![3.0, 9.0, 0.0, 3.0]);
    }
}
// @Feature id:2 name:abc
// Record min and max value for each feature.
// Max feature Id.
