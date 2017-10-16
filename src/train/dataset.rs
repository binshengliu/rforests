use std::collections::HashMap;
use metric::MetricScorer;
use train::histogram::*;
use util::{Result, Id, Value};
use format::svmlight::*;
use std;
use std::cmp::Ordering::*;

/// An instance of a label, a qid, and a group of feature values.
#[derive(Debug, PartialEq)]
pub struct Instance {
    qid: Id,
    label: Value, // or label
    values: Vec<Value>, // index from 0
}

impl Instance {
    /// Creates a new instance.
    pub fn new(label: Value, qid: Id, values: Vec<Value>) -> Instance {
        Instance {
            label: label,
            qid: qid,
            values: values,
        }
    }

    /// Returns an iterator over the feature values of the instance.
    pub fn values<'a>(&'a self) -> impl Iterator<Item = Value> + 'a {
        self.values.iter().cloned()
    }

    // See https://github.com/rust-lang/rust/issues/38615 for the
    // reason that 'a is required.
    /// Returns an iterator over the (feature id, value) pairs.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Id, Value)> + 'a {
        self.values.iter().enumerate().map(|(index, &value)| {
            (index + 1, value)
        })
    }

    /// Returns the value of the given feature id.
    pub fn value(&self, id: Id) -> Value {
        self.values.get(id - 1).map_or(0.0, |v| *v)
    }

    /// Returns the max feature id.
    pub fn max_feature_id(&self) -> Id {
        self.values.len() as Id
    }

    /// Returns the label of the instance.
    pub fn label(&self) -> Value {
        self.label
    }

    /// Returns the qid of the instance.
    pub fn qid(&self) -> Id {
        self.qid
    }
}

impl From<(Value, Id, Vec<Value>)> for Instance {
    fn from((label, qid, values): (Value, Id, Vec<Value>)) -> Instance {
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
    type Target = Vec<Value>;

    fn deref(&self) -> &Vec<Value> {
        &self.values
    }
}

/// A collection of Instances with the same qid.
pub struct Query<'a> {
    dataset: &'a DataSet,

    // qid of this Query
    qid: Id,

    // beginning index in DataSet
    start: usize,

    // length in DataSet
    len: usize,
}

impl<'a> Query<'a> {
    /// Create a new Query.
    pub fn new(
        qid: Id,
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

    /// Create the qid of the Query.
    pub fn qid(&self) -> Id {
        self.qid
    }

    /// Returns an iterator over the `Instance`s of the Query.
    pub fn iter(&'a self) -> impl Iterator<Item = &'a Instance> {
        self.dataset[self.start..(self.start + self.len)].iter()
    }

    /// Returns Vec of &Instances sorted by the labels.
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

    /// Returns Vec of &Instances sorted by the model scores.
    pub fn sorted_by_model_scores(
        &self,
        model_scores: &Vec<Value>,
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

    /// Compute the lambda value of this Query.
    pub fn get_lambda<S>(
        &self,
        model_scores: &Vec<Value>,
        metric: &S,
    ) -> Vec<(usize, Value, Value)>
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

        let labels_sorted_by_scores: Vec<Value> = indices
            .iter()
            .map(|&index| self.dataset[index].label())
            .collect();
        let metric_delta = metric.delta(&labels_sorted_by_scores);

        // hashmap: index -> (lambda, weight)
        let mut result: HashMap<usize, (Value, Value)> = HashMap::new();
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

struct ThresholdMap {
    thresholds: Vec<Value>,
    map: Vec<usize>,
}

impl ThresholdMap {
    fn thresholds(sorted_values: Vec<Value>, max_bins: usize) -> Vec<Value> {
        let mut thresholds = sorted_values;

        // If too many values, generate at most max_bins thresholds.
        if thresholds.len() > max_bins {
            let max = *thresholds.last().unwrap();
            let min = *thresholds.first().unwrap();
            let step = (max - min) / max_bins as Value;
            thresholds = (0..max_bins).map(|n| min + n as Value * step).collect();
        }
        thresholds.push(std::f64::MAX);
        thresholds
    }

    pub fn new(values: Vec<Value>, max_bins: usize) -> ThresholdMap {
        let nvalues = values.len();

        let mut indexed_values: Vec<(usize, Value)> =
            values.iter().cloned().enumerate().collect();
        indexed_values.sort_by(|&(_, a), &(_, b)| {
            a.partial_cmp(&b).unwrap_or(Less)
        });

        let sorted_values = indexed_values
            .iter()
            .map(|&(_, value)| value)
            .collect::<Vec<Value>>();
        let thresholds = ThresholdMap::thresholds(sorted_values, max_bins);
        let mut map: Vec<usize> = Vec::new();
        map.resize(nvalues, 0);

        let mut value_pos = 0;
        for (threshold_index, &threshold) in thresholds.iter().enumerate() {
            for &(value_index, value) in indexed_values[value_pos..].iter() {
                if value > threshold {
                    break;
                }
                map[value_index] = threshold_index;
                value_pos += 1;
            }
        }
        ThresholdMap {
            thresholds: thresholds,
            map: map,
        }
    }
}

impl std::fmt::Debug for ThresholdMap {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "ThresholdMap {{ thresholds: {:?}, map: {:?} }}",
            self.thresholds
                .iter()
                .map(|&threshold| if threshold == std::f64::MAX {
                    "Value::MAX".to_string()
                } else {
                    threshold.to_string()
                })
                .collect::<Vec<String>>()
                .join(", "),
            self.map
        )
    }
}

/// A collection type containing a data set.
pub struct DataSet {
    nfeatures: usize,
    instances: Vec<Instance>,
    threshold_maps: Vec<ThresholdMap>,
}

impl DataSet {
    /// Create a DataSet.
    fn new(nfeatures: usize, instances: Vec<Instance>) -> DataSet {
        let mut threshold_maps = Vec::new();
        for fid in 1..(nfeatures + 1) {
            let values: Vec<Value> = instances
                .iter()
                .map(|instance| instance.value(fid))
                .collect();
            let map = ThresholdMap::new(values, 256);
            threshold_maps.push(map);
        }

        DataSet {
            nfeatures: nfeatures,
            instances: instances,
            threshold_maps: threshold_maps,
        }
    }

    /// Load data set from a reader.
    pub fn load<R>(reader: R) -> Result<DataSet>
    where
        R: ::std::io::Read,
    {
        let mut nfeatures = 0;
        let instances: Vec<Instance> = SvmLightFile::instances(reader)
            .map(|instance| if let Ok(instance) = instance {
                nfeatures =
                    usize::max(nfeatures, instance.max_feature_id() as usize);
                return Ok(instance);
            } else {
                instance
            })
            .collect::<Result<Vec<Instance>>>()?;

        Ok(DataSet::new(nfeatures, instances))
    }

    /// Returns the number of instances in the data set, also referred
    /// to as its 'length'.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Returns an iterator over the feature ids in the data set.
    pub fn fid_iter(&self) -> impl Iterator<Item = Id> {
        (1..(self.nfeatures + 1)).map(|i| i)
    }

    /// Returns an iterator over the labels in the data set.
    pub fn label_iter<'a>(&'a self) -> impl Iterator<Item = Value> + 'a {
        self.instances.iter().map(|instance| instance.label)
    }

    /// Generate a vector of Query. Each Query keeps indices into the
    /// DataSet.
    pub fn group_by_queries<'a>(&'a self) -> Vec<Query<'a>> {
        let mut queries: HashMap<Id, Query> = HashMap::new();

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
        fid: Id,
    ) -> impl Iterator<Item = &Instance> + 'a {
        let indices = self.feature_sorted_indices(fid);
        indices.into_iter().map(move |index| &self[index])
    }

    pub fn feature_sorted_indices(&self, fid: Id) -> Vec<usize> {
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
    pub fn feature_sorted_values(&self, fid: Id) -> Vec<Value> {
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
        fid: Id,
    ) -> Vec<(usize, Value)> {
        let indices = self.feature_sorted_indices(fid);
        indices
            .into_iter()
            .map(|index| (index, self[index].value(fid)))
            .collect()
    }

    // pub fn feature_histogram(
    //     &self,
    //     fid: Id,
    //     max_bins: usize,
    // ) -> FeatureHistogram {
    //     let indices = self.feature_sorted_indices(fid);
    //     let values: Vec<(usize, Value, Value)> = indices
    //         .into_iter()
    //         .map(|index| (index, self[index].label(), self[index].value(fid)))
    //         .collect();
    //     FeatureHistogram::new(&values, max_bins)
    // }
}

use std::iter::FromIterator;
impl FromIterator<(Value, Id, Vec<Value>)> for DataSet {
    fn from_iter<T>(iter: T) -> DataSet
    where
        T: IntoIterator<Item = (Value, Id, Vec<Value>)>,
    {
        let mut instances = Vec::new();
        let mut nfeatures = 0;
        for (label, qid, values) in iter {
            let instance = Instance::from((label, qid, values));
            nfeatures =
                usize::max(nfeatures, instance.max_feature_id() as usize);
            instances.push(instance);
        }

        DataSet::new(nfeatures, instances)
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
    pub fn fid_iter(&self) -> impl Iterator<Item = Id> {
        self.dataset.fid_iter()
    }

    /// Returns an iterator over the labels in the data set sample.
    pub fn label_iter(&'a self) -> impl Iterator<Item = Value> + 'a {
        self.iter().map(|instance| instance.label)
    }

    /// Returns an iterator over the values of the given feature in
    /// the data set sample.
    pub fn value_iter(&'a self, fid: Id) -> impl Iterator<Item = Value> + 'a {
        self.iter().map(move |instance| instance.value(fid))
    }

    /// Returns a copy of the data set sample, sorted by the given
    /// feature.
    pub fn sorted_by_feature(&self, fid: Id) -> DataSetSample {
        let indices = self.sorted_indices_by_feature(fid);
        DataSetSample {
            dataset: self.dataset,
            indices: indices,
        }
    }

    /// Returns a copy of the data set sample, sorted by the given
    /// feature.
    fn sorted_indices_by_feature(&self, fid: Id) -> Vec<usize> {
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
        fid: Id,
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
        let mut splits: Vec<(Id, Value, Value, DataSetSample)> = Vec::new();
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
    fn test_instance_interface() {
        let label = 3.0;
        let qid = 3333;
        let values = vec![1.0, 2.0, 3.0];
        let instance = Instance::new(label, qid, values);

        // values()
        let mut iter = instance.values();
        assert_eq!(iter.next(), Some(1.0));
        assert_eq!(iter.next(), Some(2.0));
        assert_eq!(iter.next(), Some(3.0));
        assert_eq!(iter.next(), None);

        // iter()
        let mut iter = instance.iter();
        assert_eq!(iter.next(), Some((1, 1.0)));
        assert_eq!(iter.next(), Some((2, 2.0)));
        assert_eq!(iter.next(), Some((3, 3.0)));
        assert_eq!(iter.next(), None);

        // value()
        assert_eq!(instance.value(1), 1.0);
        assert_eq!(instance.value(2), 2.0);
        assert_eq!(instance.value(3), 3.0);
        assert_eq!(instance.value(4), 0.0);

        // max_feature_id()
        assert_eq!(instance.max_feature_id(), 3);

        // label()
        assert_eq!(instance.label(), 3.0);

        // qid()
        assert_eq!(instance.qid(), 3333);
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
    fn test_threshold_map() {
        let values = vec![5.0, 7.0, 3.0, 2.0, 1.0, 8.0, 9.0, 4.0, 6.0];

        let map = ThresholdMap::new(values, 3);

        assert_eq!(
            map.thresholds,
            vec![
                1.0 + 0.0 * 8.0 / 3.0, // 1.0
                1.0 + 1.0 * 8.0 / 3.0, // 3.66
                1.0 + 2.0 * 8.0 / 3.0, // 6.33
                std::f64::MAX,
            ]
        );

        assert_eq!(map.map, vec![2, 3, 1, 1, 0, 3, 3, 2, 2]);
    }
}
