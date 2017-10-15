use std::collections::HashMap;
use metric::MetricScorer;
use train::histogram::*;
use util::Result;
use format::svmlight::*;
use std;

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

    pub fn iter(&'a self) -> impl Iterator<Item = &'a Instance> {
        self.dataset[self.start..(self.start + self.len)].iter()
    }

    /// Return Vec of &Instances sorted by the original labels.
    pub fn sorted_by_labels(&self) -> Vec<&Instance> {
        use std::cmp::Ordering;

        let mut indices: Vec<usize> = (self.start..(self.start + self.len))
            .collect();
        indices.sort_by(|&index1, &index2| {
            let label1 = self.dataset[index1].label();
            let label2 = self.dataset[index2].label();

            // Descending
            label2.partial_cmp(&label1).unwrap_or(Ordering::Equal)
        });

        indices
            .into_iter()
            .map(move |index| &self.dataset[index])
            .collect()
    }

    /// Return Vec of &Instances sorted by the model scores.
    pub fn sorted_by_model_scores(
        &self,
        model_scores: &Vec<f64>,
    ) -> Vec<&Instance> {
        use std::cmp::Ordering;

        let mut indices: Vec<usize> = (self.start..(self.start + self.len))
            .collect();
        indices.sort_by(|&index1, &index2| {
            let label1 = model_scores[index1];
            let label2 = model_scores[index2];

            // Descending
            label2.partial_cmp(&label1).unwrap_or(Ordering::Equal)
        });

        indices
            .into_iter()
            .map(move |index| &self.dataset[index])
            .collect()
    }

    pub fn get_lambda<S>(
        &self,
        model_scores: &Vec<f64>,
        metric: &S,
    ) -> Vec<(usize, f64, f64)>
    where
        S: MetricScorer,
    {
        use std::cmp::Ordering;

        // indices into DataSet
        let mut indices: Vec<usize> = (self.start..(self.start + self.len))
            .collect();

        indices.sort_by(|&index1, &index2| {
            let label1 = model_scores[index1];
            let label2 = model_scores[index2];

            // Descending
            label2.partial_cmp(&label1).unwrap_or(Ordering::Equal)
        });

        let labels_sorted_by_scores: Vec<f64> = indices
            .iter()
            .map(|&index| self.dataset[index].label())
            .collect();
        let metric_delta = metric.delta(&labels_sorted_by_scores);

        // hashmap: index -> (lambda, weight)
        let mut result: HashMap<usize, (f64, f64)> = HashMap::new();
        for &index1 in indices.iter() {
            let instance1 = &self.dataset[index1];
            for &index2 in indices.iter() {
                let instance2 = &self.dataset[index2];
                if instance1.label() <= instance2.label() {
                    continue;
                }

                let metric_delta_value = metric_delta[index1][index2].abs();
                let rho = 1.0 /
                    (1.0 + (model_scores[index1] - model_scores[index2]).exp());
                let lambda = metric_delta_value * rho;
                let weight = rho * (1.0 - rho) * metric_delta_value;

                result.entry(index1).or_insert((0.0, 0.0));
                result.get_mut(&index1).unwrap().0 += lambda;
                result.get_mut(&index1).unwrap().1 += weight;

                result.entry(index2).or_insert((0.0, 0.0));
                result.get_mut(&index2).unwrap().0 -= lambda;
                result.get_mut(&index2).unwrap().1 += weight;

            }
        }

        result
            .into_iter()
            .map(|(key, value)| (key, value.0, value.1))
            .collect()
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

    pub fn fid_iter(&self) -> impl Iterator<Item = u64> {
        (1..(self.nfeatures + 1)).map(|i| i as u64)
    }

    pub fn labels(&self) -> Vec<f64> {
        self.instances
            .iter()
            .map(|instance| instance.label)
            .collect()
    }

    pub fn labels_iter<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.instances.iter().map(|instance| instance.label)
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

    pub fn instance_sorted_by_feature<'a>(
        &'a self,
        fid: u64,
    ) -> impl Iterator<Item = &Instance> + 'a {
        let indices = self.feature_sorted_indices(fid);
        indices.into_iter().map(move |index| &self[index])
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

    /// Return sorted values of a specific feature.
    pub fn feature_sorted_values(&self, fid: u64) -> Vec<f64> {
        let indices = self.feature_sorted_indices(fid);
        indices
            .into_iter()
            .map(|index| self[index].value(fid))
            .collect()
    }

    /// Return sorted values of a specific feature, with the original
    /// indices in the dataset.
    pub fn feature_sorted_values_with_indices(
        &self,
        fid: u64,
    ) -> Vec<(usize, f64)> {
        let indices = self.feature_sorted_indices(fid);
        indices
            .into_iter()
            .map(|index| (index, self[index].value(fid)))
            .collect()
    }

    pub fn feature_histogram(
        &self,
        fid: u64,
        max_bins: usize,
    ) -> FeatureHistogram {
        let indices = self.feature_sorted_indices(fid);
        let values: Vec<(usize, f64, f64)> = indices
            .into_iter()
            .map(|index| (index, self[index].label(), self[index].value(fid)))
            .collect();
        FeatureHistogram::new(&values, max_bins)
    }

    pub fn histogram(&self, max_bins: usize) -> Histogram {
        Histogram::new(self, max_bins)
    }
}

impl std::ops::Deref for DataSet {
    type Target = Vec<Instance>;

    fn deref(&self) -> &Vec<Instance> {
        &self.instances
    }
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
}

