use std::collections::HashMap;
use metric::MetricScorer;
use train::histogram::*;
use util::Result;
use format::svmlight::*;
use std;

#[derive(Debug, PartialEq)]
pub struct Instance {
    qid: u64,
    label: f64, // or label
    values: Vec<f64>, // index from 0
}

impl Instance {
    pub fn new(label: f64, qid: u64, values: Vec<f64>) -> Instance {
        Instance {
            label: label,
            qid: qid,
            values: values,
        }
    }

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
        self.values.get((id - 1) as usize).map_or(0.0, |v| *v)
    }

    pub fn max_feature_id(&self) -> u64 {
        (self.values.len() + 1) as u64
    }

    pub fn label(&self) -> f64 {
        self.label
    }

    pub fn qid(&self) -> u64 {
        self.qid
    }
}

impl From<(f64, u64, Vec<f64>)> for Instance {
    fn from((label, qid, values): (f64, u64, Vec<f64>)) -> Instance {
        Instance::new(label, qid, values)
    }
}

impl std::fmt::Display for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut values = self.values
            .iter()
            .enumerate()
            .map(|(index, value)| format!("{}:{}", index + 1, value))
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

/// A collection type containing a data set.
pub struct DataSet {
    nfeatures: usize,
    instances: Vec<Instance>,
}

impl DataSet {
    /// Load data set from a reader.
    pub fn load<R>(reader: R) -> Result<DataSet>
    where
        R: ::std::io::Read,
    {
        let mut nfeatures = 0;
        let instances: Vec<Instance> = SvmLightFile::instances(reader)
            .map(|instance| if let Ok(instance) = instance {
                nfeatures = u64::max(nfeatures, instance.max_feature_id());
                return Ok(instance);
            } else {
                instance
            })
            .collect::<Result<Vec<Instance>>>()?;

        Ok(DataSet {
            nfeatures: nfeatures as usize,
            instances: instances,
        })
    }

    /// Returns the number of instances in the data set, also referred
    /// to as its 'length'.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Returns an iterator over the feature ids in the data set.
    pub fn fid_iter(&self) -> impl Iterator<Item = u64> {
        (1..(self.nfeatures + 1)).map(|i| i as u64)
    }

    /// Returns an iterator over the labels in the data set.
    pub fn label_iter<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
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

    // pub fn feature_histogram(
    //     &self,
    //     fid: u64,
    //     max_bins: usize,
    // ) -> FeatureHistogram {
    //     let indices = self.feature_sorted_indices(fid);
    //     let values: Vec<(usize, f64, f64)> = indices
    //         .into_iter()
    //         .map(|index| (index, self[index].label(), self[index].value(fid)))
    //         .collect();
    //     FeatureHistogram::new(&values, max_bins)
    // }
}

use std::iter::FromIterator;
impl FromIterator<(f64, u64, Vec<f64>)> for DataSet {
    fn from_iter<T>(iter: T) -> DataSet
    where
        T: IntoIterator<Item = (f64, u64, Vec<f64>)>,
    {
        let mut instances = Vec::new();
        let mut nfeatures = 0;
        for (label, qid, values) in iter {
            let instance = Instance::from((label, qid, values));
            nfeatures = u64::max(nfeatures, instance.max_feature_id());
            instances.push(instance);
        }

        DataSet {
            nfeatures: nfeatures as usize,
            instances: instances,
        }
    }
}

impl std::ops::Deref for DataSet {
    type Target = Vec<Instance>;

    fn deref(&self) -> &Vec<Instance> {
        &self.instances
    }
}

/// A collection type containing part of a data set.
pub struct DataSetSample<'a> {
    /// Original data
    dataset: &'a DataSet,

    /// Indices into Dataset
    indices: Vec<usize>,
}

impl<'a> DataSetSample<'a> {
    /// Returns the number of instances in the data set sample, also
    /// referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns an iterator over the data set sample.
    pub fn iter(&'a self) -> impl Iterator<Item = &Instance> + 'a {
        self.indices.iter().map(move |&index| &self.dataset[index])
    }

    /// Creates an iterator which gives the index of the Instance as
    /// well as the Instance.
    ///
    /// The iterator returned yields pairs (i, instance), where i is
    /// the index of Instance and instance is the reference to the
    /// Instance returned by the iterator.
    pub fn enumerate(
        &'a self,
    ) -> impl Iterator<Item = (usize, &Instance)> + 'a {
        self.indices.iter().map(move |&index| {
            (index, &self.dataset[index])
        })
    }

    /// Returns an iterator over the feature ids in the data set
    /// sample.
    pub fn fid_iter(&self) -> impl Iterator<Item = u64> {
        self.dataset.fid_iter()
    }

    /// Returns an iterator over the labels in the data set sample.
    pub fn label_iter(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.iter().map(|instance| instance.label)
    }

    /// Returns an iterator over the values of the given feature in
    /// the data set sample.
    pub fn value_iter(&'a self, fid: u64) -> impl Iterator<Item = f64> + 'a {
        self.iter().map(move |instance| instance.value(fid))
    }

    /// Returns a copy of the data set sample, sorted by the given
    /// feature.
    pub fn sorted_by_feature(&self, fid: u64) -> DataSetSample {
        let indices = self.sorted_indices_by_feature(fid);
        DataSetSample {
            dataset: self.dataset,
            indices: indices,
        }
    }

    /// Returns a copy of the data set sample, sorted by the given
    /// feature.
    fn sorted_indices_by_feature(&self, fid: u64) -> Vec<usize> {
        use std::cmp::Ordering::Equal;
        let mut indices = self.indices.clone();
        indices.sort_by(|&index1, &index2| {
            let value1 = self.dataset[index1].value(fid);
            let value2 = self.dataset[index2].value(fid);
            value1.partial_cmp(&value2).unwrap_or(Equal)
        });
        indices
    }

    /// Returns a histogram of the feature of the data set sample.
    pub fn feature_histogram(
        &self,
        fid: u64,
        max_bins: usize,
    ) -> FeatureHistogram {
        let sorted = self.sorted_by_feature(fid);

        FeatureHistogram::new(&sorted, fid, max_bins)
    }

    /// Returns histograms of all the features of the data set sample.
    fn histogram(&self, max_bins: usize) -> Histogram {
        Histogram::new(self, max_bins)
    }

    pub fn split(&self) -> Option<(DataSetSample, DataSetSample)> {
        // (fid, threashold, s, sorted data)
        let mut splits: Vec<(u64, f64, f64, DataSetSample)> = Vec::new();
        for fid in self.fid_iter() {
            let sorted: DataSetSample = self.sorted_by_feature(fid);
            let feature_histogram = FeatureHistogram::new(&sorted, fid, 256);
            let split = feature_histogram.best_split(1);
            if split.is_none() {
                continue;
            }

            let (threshold, s) = split.unwrap();
            splits.push((fid, threshold, s, sorted));
        }

        // Find the split with the best s value;
        let (fid, threashold, s, sorted) = splits
            .into_iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();

        //
        let (left_indices, right_indices) =
            sorted.enumerate().fold(
                (Vec::new(), Vec::new()),
                |(mut left, mut right),
                 (index, instance)| if instance.value(fid) <=
                    threashold
                {
                    left.push(index);
                    (left, right)
                } else {
                    right.push(index);
                    (left, right)
                },
            );

        unimplemented!()
    }
}

impl<'a> From<&'a DataSet> for DataSetSample<'a> {
    fn from(dataset: &'a DataSet) -> DataSetSample<'a> {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();
        DataSetSample {
            dataset: dataset,
            indices: indices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
