use std;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use util::Result;
use std::collections::HashMap;
use num;

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

#[derive(PartialEq)]
pub struct Instance {
    qid: u64,
    label: f64, // or label
    values: Vec<f64>, // index from 1
}

impl Instance {
    pub fn values(&self) -> std::slice::Iter<f64> {
        self.values.iter()
    }

    // See https://github.com/rust-lang/rust/issues/38615 for the
    // reason that 'a is required.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (usize, f64)> + 'a {
        self.values.iter().enumerate().skip(1).map(
            |(index, &value)| {
                (index, value)
            },
        )
    }

    /// Return feature value of feature id.
    pub fn value(&self, id: u64) -> f64 {
        self.values.get(id as usize).map_or(0.0, |v| *v)
    }

    pub fn max_feature_id(&self) -> u64 {
        self.values.len() as u64
    }

    pub fn label(&self) -> f64 {
        self.label
    }

    pub fn qid(&self) -> u64 {
        self.qid
    }

    fn parse_label(label: &str) -> Result<f64> {
        let label = label.parse::<f64>()?;
        Ok(label)
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

    /// Parse &["1:3.0" "3:4.0"] into Vec of values. Absent indices
    /// are filled with 0.0. The example above would result vec![0.0,
    /// 3.0, 0.0, 4.0].
    fn parse_values(fields: &[&str]) -> Result<Vec<f64>> {
        fn parse(s: &str) -> Result<(u64, f64)> {
            let v: Vec<&str> = s.split(':').collect();
            if v.len() != 2 {
                Err(format!("Invalid string: {}", s))?;
            }

            let id = v[0].parse::<u64>()?;
            let value = v[1].parse::<f64>()?;

            Ok((id, value))
        }

        // (id, value) pairs
        let v: Vec<(u64, f64)> =
            fields.iter().map(|&s| parse(s)).collect::<Result<_>>()?;
        let max_id = v.iter().max_by_key(|e| e.0).unwrap().0;
        let mut ret: Vec<f64> = Vec::with_capacity(max_id as usize + 1);
        ret.resize(max_id as usize + 1, 0.0);
        for &(id, value) in v.iter() {
            ret[id as usize] = value;
        }

        Ok(ret)
    }

    pub fn from_str(s: &str) -> Result<Self> {
        let line: &str = s.trim().split('#').next().unwrap().trim();
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            Err(format!("Invalid line"))?;
        }

        let label = Instance::parse_label(fields[0])?;
        let qid = Instance::parse_qid(fields[1])?;
        let values: Vec<f64> = Instance::parse_values(&fields[2..])?;

        Ok(Instance {
            label: label,
            qid: qid,
            values: values,
        })
    }
}

impl std::fmt::Display for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut values = self.values
            .iter()
            .enumerate()
        // skip index 0
            .skip(1)
            .map(|(index, value)| format!("{}:{}", index, value))
            .collect::<Vec<_>>();

        let mut v = vec![self.label.to_string(), format!("qid:{}", self.qid)];
        v.append(&mut values);
        write!(f, "{}", v.join(" "))
    }
}

impl std::ops::Deref for Instance {
    type Target = Vec<f64>;

    fn deref(&self) -> &Vec<f64> {
        &self.values
    }
}

pub struct Query<'a> {
    dataset: &'a DataSet,

    // qid of this Query
    qid: u64,

    // beginning index in DataSet
    start: usize,

    // length in DataSet
    len: usize,
}

impl<'a> Query<'a> {
    pub fn new(
        qid: u64,
        dataset: &'a DataSet,
        start: usize,
        len: usize,
    ) -> Query<'a> {
        Query {
            qid: qid,
            dataset: dataset,
            start: start,
            len: len,
        }
    }

    pub fn qid(&self) -> u64 {
        self.qid
    }

    pub fn iter<'i>(&'i self) -> impl Iterator<Item = &Instance> + 'i {
        self.dataset[self.start..(self.start + self.len)].iter()
    }
}

impl<'a> std::fmt::Display for Query<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let v = self.iter()
            .map(|inst| inst.to_string())
            .collect::<Vec<String>>();

        write!(f, "{}", v.join("\n"))
    }
}

pub struct DataSet {
    nfeatures: usize,
    instances: Vec<Instance>,
}

impl DataSet {
    pub fn load<R>(reader: R) -> Result<DataSet>
    where
        R: ::std::io::Read,
    {
        let mut nfeatures = 0;
        let instances: Vec<Instance> = SvmLightFile::instances(reader)
            .map(|i| if let Ok(instance) = i {
                nfeatures = u64::max(nfeatures, instance.max_feature_id());
                return Ok(instance);
            } else {
                i
            })
            .collect::<Result<Vec<Instance>>>()?;

        Ok(DataSet {
            nfeatures: nfeatures as usize,
            instances: instances,
        })
    }

    /// Generate a vector of Query. Each Query keeps indices into the
    /// DataSet.
    pub fn group_by_queries<'a>(&'a self) -> Vec<Query<'a>> {
        let mut queries: HashMap<u64, Query> = HashMap::new();

        let mut prev_qid = None;
        let mut start = 0;
        let mut count = 0;
        for (index, instance) in self.iter().enumerate() {
            let qid = instance.qid();
            if prev_qid == Some(qid) {
                count += 1;
                continue;
            }

            if count != 0 {
                queries.entry(prev_qid.unwrap()).or_insert(Query::new(
                    prev_qid.unwrap(),
                    self,
                    start,
                    count,
                ));
            }

            prev_qid = Some(qid);
            start = index;
            count = 1;
        }

        if count != 0 {
            queries.entry(prev_qid.unwrap()).or_insert(Query::new(
                prev_qid.unwrap(),
                self,
                start,
                count,
            ));
        }

        let queries: Vec<_> =
            queries.into_iter().map(|(_key, value)| value).collect();

        queries
    }

    pub fn feature_sorted_indices(&self, fid: u64) -> Vec<usize> {
        use std::cmp::Ordering;

        let n_instances = self.len();
        let mut indices: Vec<usize> = (0..n_instances).collect();
        indices.sort_by(|&index1, &index2| {
            let value1 = self[index1].value(fid);
            let value2 = self[index2].value(fid);
            value1.partial_cmp(&value2).unwrap_or(Ordering::Equal)
        });
        indices
    }

    pub fn feature_sorted_values(&self, fid: u64) -> Vec<f64> {
        let indices = self.feature_sorted_indices(fid);
        indices
            .into_iter()
            .map(|index| self[index].value(fid))
            .collect()
    }

    /// Create at most max_threasholds intervals, so the return value
    /// is at most of length (max_threasholds+1).
    pub fn feature_threasholds(
        &self,
        fid: u64,
        max_threasholds: usize,
    ) -> Vec<f64> {
        let mut values = self.feature_sorted_values(fid);
        values.dedup();
        if values.len() <= max_threasholds {
            values.push(std::f64::MAX);
            values
        } else {
            let max = values.last().unwrap();
            let min = values.first().unwrap();
            let step = (max - min) / max_threasholds as f64;
            let mut v: Vec<f64> = (0..max_threasholds)
                .map(|n| min + n as f64 * step)
                .collect();
            v.push(std::f64::MAX);
            v
        }
    }
}

impl std::ops::Deref for DataSet {
    type Target = Vec<Instance>;

    fn deref(&self) -> &Vec<Instance> {
        &self.instances
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

            for (id, value) in instance.iter() {
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
    // Returning an abstract type is not well supported now. The Rust
    // team is working on it:
    // https://stackoverflow.com/questions/27535289/correct-way-to-return-an-iterator/27535594#27535594
    // https://github.com/rust-lang/rfcs/blob/master/text/1522-conservative-impl-trait.md
    pub fn instances<R>(reader: R) -> impl Iterator<Item = Result<Instance>>
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
                    Instance::from_str(line.as_str())
                })
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_parse() {
        let s = "3.0 qid:3864 1:3.000000 2:9.000000 4:3.0 # 3:10.0";
        let p = Instance::from_str(s).unwrap();
        assert_eq!(p.label, 3.0);
        assert_eq!(p.qid, 3864);
        assert_eq!(p.values, vec![0.0, 3.0, 9.0, 0.0, 3.0]);
    }

    #[test]
    fn test_sorted_feature() {
        let s = "0 qid:1 1:3.0 2:0.0 3:1.0\n2 qid:2 1:1.0 2:1.0 3:3.0\n0 qid:3 1:0.0 2:2.0 3:2.0";
        let dataset = DataSet::load(::std::io::Cursor::new(s)).unwrap();

        let sorted_indices = dataset.feature_sorted_indices(1);
        assert_eq!(sorted_indices, vec![2, 1, 0]);

        let sorted_indices = dataset.feature_sorted_indices(2);
        assert_eq!(sorted_indices, vec![0, 1, 2]);

        let sorted_indices = dataset.feature_sorted_indices(3);
        assert_eq!(sorted_indices, vec![0, 2, 1]);
    }

    #[test]
    fn test_feature_sorted_values() {
        let s = "0 qid:1 1:3.0 2:0.0 3:1.0\n2 qid:2 1:1.0 2:1.0 3:3.0\n0 qid:3 1:0.0 2:2.0 3:2.0";
        let dataset = DataSet::load(::std::io::Cursor::new(s)).unwrap();

        let sorted_indices = dataset.feature_sorted_values(1);
        assert_eq!(sorted_indices, vec![0.0, 1.0, 3.0]);
    }

    #[test]
    fn test_feature_threasholds() {
        let s = "0 qid:1 1:3.0 2:0.0 3:1.0\n2 qid:2 1:1.0 2:1.0 3:3.0\n0 qid:3 1:0.0 2:2.0 3:2.0";
        let dataset = DataSet::load(::std::io::Cursor::new(s)).unwrap();

        let threasholds = dataset.feature_threasholds(1, 10);
        assert_eq!(threasholds, vec![0.0, 1.0, 3.0, std::f64::MAX]);

        let threasholds = dataset.feature_threasholds(1, 3);
        assert_eq!(threasholds, vec![0.0, 1.0, 3.0, std::f64::MAX]);
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
