use std::collections::HashMap;
use metric::MetricScorer;
use train::histogram::*;
use util::{Id, Result, Value};
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

/// A Mapping from the index of a Instance in the DataSet into a
/// threshold interval.
struct ThresholdMap {
    /// Thresholds are ordered in ascending order. To test whether a
    /// value falls into the threshold, use `if value <= threshold`.
    thresholds: Vec<Value>,

    /// The index of the Vec is the index of the instances in the
    /// DataSet, which also means `map.len() == instances.len()`.
    ///
    /// The elements are the indices into the thresholds Vec.
    ///
    /// For example, if we have 100,000 instances, and 256 thresholds,
    /// then
    /// ```
    /// assert_eq!(map.len(), 100,000);
    /// assert!(map.iter().all(|&i| i <= 256));
    /// ```
    map: Vec<usize>,
}

impl ThresholdMap {
    /// Generate thresholds according to the given values and max
    /// bins. If the count of values exceeds max bins, thresholds are
    /// generated by averaging the difference of max and min of the
    /// values by max bins.
    fn thresholds(sorted_values: Vec<Value>, max_bins: usize) -> Vec<Value> {
        let mut thresholds = sorted_values;

        // If too many values, generate at most max_bins thresholds.
        if thresholds.len() > max_bins {
            let max = *thresholds.last().unwrap();
            let min = *thresholds.first().unwrap();
            let step = (max - min) / max_bins as Value;
            thresholds =
                (0..max_bins).map(|n| min + n as Value * step).collect();
        }
        thresholds.push(std::f64::MAX);
        thresholds
    }

    /// Create a map according to the given values and max bins.
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

    /// Generate a histogram for a series of values.
    ///
    /// The input is an iterator over (instance id, feature value,
    /// label value).
    ///
    /// There are two
    /// cases when we need to regenerate the histogram. First, after
    /// each iteration of learning, the label values are
    /// different. But this is a situation that we can update the
    /// histogram instead of constructing from scratch. Second, after
    /// a tree node is splited, each sub-node contains different part
    /// of data.
    pub fn histogram<I: Iterator<Item = (Id, Value, Value)>>(
        &self,
        iter: I,
    ) -> FeatureHistogram {
        // (threshold value, count, sum)
        let mut hist: Vec<(Value, usize, Value)> = self.thresholds
            .iter()
            .map(|&threshold| (threshold, 0, 0.0))
            .collect();

        for (id, feature_value, label) in iter {
            let threshold_index = self.map[id];

            let threshold = self.thresholds[threshold_index];
            assert!(feature_value <= threshold);

            hist[threshold_index].1 += 1;
            hist[threshold_index].2 += label;
        }

        for i in 1..hist.len() {
            hist[i].1 += hist[i - 1].1;
            hist[i].2 += hist[i - 1].2;
        }
        let feature_histogram = hist.into_iter().collect();
        feature_histogram
    }
}

impl std::fmt::Debug for ThresholdMap {
    // Avoid printing the very long f64::MAX value.
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

/// A collection type containing a data set. The DataSet is a static
/// data structure. See also TrainingDataSet which is a mutable data
/// structure that its label values get updated after each training.
pub struct DataSet {
    nfeatures: usize,
    instances: Vec<Instance>,
    threshold_maps: Vec<ThresholdMap>,
}

impl DataSet {
    const MAX_BINS_COUNT: usize = 256;

    /// Create an empty DataSet.
    pub fn new(instances: Vec<Instance>, nfeatures: usize) -> DataSet {
        DataSet {
            nfeatures: nfeatures,
            instances: instances,
            threshold_maps: Vec::new(),
        }
    }

    /// Generate thresholds. This interface is ugly. It introduces
    /// extra dependency that functions must be called in a specific
    /// order. But I haven't come up with a good workaround to support
    /// FromIterator. Basically, this is a issue how we customize the
    /// grouping of the data.
    pub fn generate_thresholds(&mut self, max_bin: usize) {
        for fid in self.fid_iter() {
            let values: Vec<Value> = self.instances
                .iter()
                .map(|instance| instance.value(fid))
                .collect();
            let map = ThresholdMap::new(values, max_bin);
            self.threshold_maps.push(map);
        }
    }

    /// Load data set from a reader.
    pub fn load<R>(reader: R) -> Result<DataSet>
    where
        R: ::std::io::Read,
    {
        let mut instances = Vec::new();
        let mut nfeatures = 0;
        for instance_result in SvmLightFile::instances(reader) {
            let instance = instance_result?;
            nfeatures =
                usize::max(nfeatures, instance.max_feature_id() as usize);
            instances.push(instance);
        }

        Ok(DataSet::new(instances, nfeatures))
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

    /// Returns an iterator over the instances sorted on the given
    /// feature.
    pub fn instance_sorted_by_feature<'a>(
        &'a self,
        fid: Id,
    ) -> impl Iterator<Item = &Instance> + 'a {
        let indices = self.feature_sorted_indices(fid);
        indices.into_iter().map(move |index| &self[index])
    }

    /// Returns a Vec of the indices, sorted on the given feature.
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

    /// Generate histogram for the specified instances.
    pub fn feature_histogram<I: Iterator<Item = (Id, Value)>>(
        &self,
        fid: Id,
        iter: I,
    ) -> FeatureHistogram {
        // Get the map by feature id.
        let threshold_map = &self.threshold_maps[fid - 1];
        let iter =
            iter.map(|(id, label)| (id, self.instances[id].value(fid), label));
        threshold_map.histogram(iter)
    }
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

        DataSet::new(instances, nfeatures)
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

    /// Returns the average value of the labels.
    pub fn label_avg(&self) -> f64 {
        self.label_iter().sum::<f64>() / (self.len() as f64)
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
    pub fn feature_histogram(&self, fid: Id) -> FeatureHistogram {
        let iter = self.indices.iter().map(|&index| {
            (index, self.dataset[index].label())
        });

        self.dataset.feature_histogram(fid, iter)
    }

    /// Returns histograms of all the features of the data set sample.
    fn histogram(&self, max_bins: usize) -> Histogram {
        Histogram::new(self, max_bins)
    }

    /// Split self. Returns (split feature, threshold, s value, left
    /// child, right child).
    pub fn split(
        &self,
        min_leaf_count: usize,
    ) -> Option<(Id, Value, f64, DataSetSample, DataSetSample)> {
        assert!(min_leaf_count > 0);
        // (fid, threshold, s)
        let mut splits: Vec<(Id, Value, f64)> = Vec::new();
        for fid in self.fid_iter() {
            let feature_histogram = self.feature_histogram(fid);
            let split = feature_histogram.best_split(min_leaf_count);
            match split {
                Some((threshold, s)) => splits.push((fid, threshold, s)),
                None => continue,
            }
        }

        // Find the split with the best s value;
        let (fid, threshold, s) = match splits.into_iter().max_by(|a, b| {
            a.2.partial_cmp(&b.2).unwrap()
        }) {
            Some((fid, threshold, s)) => (fid, threshold, s),
            None => return None,
        };

        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for (index, instance) in self.enumerate() {
            if instance.value(fid) <= threshold {
                left_indices.push(index);
            } else {
                right_indices.push(index);
            }
        }

        let left = DataSetSample {
            dataset: self.dataset,
            indices: left_indices,
        };
        let right = DataSetSample {
            dataset: self.dataset,
            indices: right_indices,
        };
        Some((fid, threshold, s, left, right))
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

impl<'a> std::fmt::Display for DataSetSample<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for &index in self.indices.iter() {
            write!(
                f,
                "{{index: {}, instance: {}}}\n",
                index,
                self.dataset[index]
            )?;
        }

        Ok(())
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

    #[test]
    fn test_data_set_sample_split() {
        // (label, qid, feature_values)
        let data = vec![
            (3.0, 1, vec![5.0]),
            (2.0, 1, vec![7.0]),
            (3.0, 1, vec![3.0]),
            (1.0, 1, vec![2.0]),
            (0.0, 1, vec![1.0]),
            (2.0, 1, vec![8.0]),
            (4.0, 1, vec![9.0]),
            (1.0, 1, vec![4.0]),
            (0.0, 1, vec![6.0]),
        ];

        let mut dataset: DataSet = data.into_iter().collect();
        dataset.generate_thresholds(3);

        let sample = DataSetSample::from(&dataset);
        let (fid, threshold, s, left, right) = sample.split(1).unwrap();
        assert_eq!(fid, 1);
        assert_eq!(threshold, 1.0 + 16.0 / 3.0);
        assert_eq!(s, 32.0);
        assert_eq!(left.indices, vec![0, 2, 3, 4, 7, 8]);
        assert_eq!(right.indices, vec![1, 5, 6]);
    }

    #[test]
    fn test_data_set_sample_non_split() {
        // (label, qid, feature_values)
        let data = vec![
            (3.0, 1, vec![5.0]), // 0
            (2.0, 1, vec![7.0]), // 1
            (3.0, 1, vec![3.0]), // 2
            (1.0, 1, vec![2.0]), // 3
            (0.0, 1, vec![1.0]), // 4
            (2.0, 1, vec![8.0]), // 5
            (4.0, 1, vec![9.0]), // 6
            (1.0, 1, vec![4.0]), // 7
            (0.0, 1, vec![6.0]), // 8
        ];

        let mut dataset: DataSet = data.into_iter().collect();
        dataset.generate_thresholds(3);

        // possible splits of feature values:
        // 1 | 2 3 4 5 6 7 8 9
        // 1 2 3 | 4 5 6 7 8 9
        // 1 2 3 4 5 6 | 7 8 9
        let sample = DataSetSample::from(&dataset);
        assert!(sample.split(9).is_none());
        assert!(sample.split(4).is_none());
        let (fid, threshold, s, left, right) = sample.split(3).unwrap();
        assert_eq!(fid, 1);
        assert_eq!(threshold, 1.0 + 16.0 / 3.0);
        assert_eq!(s, 32.0);
        assert_eq!(left.indices, vec![0, 2, 3, 4, 7, 8]);
        assert_eq!(right.indices, vec![1, 5, 6]);

        // (3.0, 1, vec![5.0]), // 0
        // (3.0, 1, vec![3.0]), // 2
        // (1.0, 1, vec![2.0]), // 3
        // (0.0, 1, vec![1.0]), // 4
        // (1.0, 1, vec![4.0]), // 7
        // (0.0, 1, vec![6.0]), // 8
        // possible splits of [0, 2, 3, 4, 7, 8]
        // 4 | 3 2 7 0 8
        // 4 3 2 | 7 0 8
        let (fid, threshold, s, left, right) = left.split(2).unwrap();
        assert_eq!(fid, 1);
        assert_eq!(threshold, 1.0 + 8.0 / 3.0);
        assert_eq!(s, 32.0/3.0);
        assert_eq!(left.indices, vec![2, 3, 4]);
        assert_eq!(right.indices, vec![0, 7, 8]);
    }
}
