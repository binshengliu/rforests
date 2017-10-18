use std::cell::Cell;
use metric::NDCGScorer;
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

    /// Returns the qid of the instance.
    pub fn qid(&self) -> Id {
        self.qid
    }

    /// Returns the label of the instance.
    pub fn label(&self) -> Value {
        self.label
    }

    /// Returns the value of the given feature id.
    pub fn value(&self, id: Id) -> Value {
        self.values.get(id - 1).map_or(0.0, |v| *v)
    }

    // See https://github.com/rust-lang/rust/issues/38615 for the
    // reason that 'a is required.
    /// Returns an iterator over the (feature id, value) pairs.
    pub fn value_iter<'a>(&'a self) -> impl Iterator<Item = (Id, Value)> + 'a {
        self.values.iter().enumerate().map(|(index, &value)| {
            (index + 1, value)
        })
    }

    /// Returns the max feature id.
    pub fn max_feature_id(&self) -> Id {
        self.values.len() as Id
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
    /// There are two cases when we need to regenerate the
    /// histogram. First, after each iteration of learning, the label
    /// values are different. But this is a situation that we can
    /// update the histogram instead of constructing from
    /// scratch. Second, after a tree node is splited, each sub-node
    /// contains different part of data.
    ///
    /// # Examples
    ///
    /// let data = vec![
    ///     // target value, feature values
    ///     (3.0, 5.0),
    ///     (2.0, 7.0),
    ///     (3.0, 3.0),
    ///     (1.0, 2.0),
    ///     (0.0, 1.0),
    ///     (2.0, 8.0),
    ///     (4.0, 9.0),
    ///     (1.0, 4.0),
    ///     (0.0, 6.0),
    /// ];
    ///
    /// let map = ThresholdMap::new(data.iter().map(|&(_, value)| value), 3);
    /// let histogram = map.histogram(data.iter().map(|&(target, _)| target));
    ///
    /// assert_eq!(histogram.variance(), 15.555555555555557);
    pub fn histogram<I: Iterator<Item = (Id, Value, Value)>>(
        &self,
        iter: I,
    ) -> Histogram {
        // (threshold value, count, sum, squared_sum)
        let mut hist: Vec<(Value, usize, Value, Value)> = self.thresholds
            .iter()
            .map(|&threshold| (threshold, 0, 0.0, 0.0))
            .collect();

        for (id, feature_value, label) in iter {
            let threshold_index = self.map[id];

            let threshold = self.thresholds[threshold_index];
            assert!(feature_value <= threshold);

            hist[threshold_index].1 += 1;
            hist[threshold_index].2 += label;
            hist[threshold_index].3 += label * label;
        }

        for i in 1..hist.len() {
            hist[i].1 += hist[i - 1].1;
            hist[i].2 += hist[i - 1].2;
            hist[i].3 += hist[i - 1].3;
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

pub struct QueryIter<'a> {
    dataset: &'a DataSet,
    index: usize,
}

impl<'a> Iterator for QueryIter<'a> {
    // (query id, Vec<instance index, label>)
    type Item = (Id, Vec<Id>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            return None;
        }

        let qid = self.dataset[self.index].qid();
        let queries = (self.index..self.dataset.len())
            .take_while(|&index| qid == self.dataset[index].qid())
            .collect::<Vec<Id>>();

        self.index += queries.len();
        Some((qid, queries))
    }
}

/// A collection type containing a data set. The DataSet is a static
/// data structure. See also TrainingDataSet which is a mutable data
/// structure that its label values get updated after each training.
pub struct DataSet {
    nfeatures: usize,
    // When making histograms, at most how many bins to split.
    max_bins: usize,
    instances: Vec<Instance>,
    threshold_maps: Vec<ThresholdMap>,
}

impl DataSet {
    /// Create an empty DataSet.
    /// # Examples
    ///
    /// ```
    /// use rforests::train::dataset::DataSet;
    ///
    /// let mut dataset = DataSet::new(3);
    /// ```
    pub fn new(max_bins: usize) -> DataSet {
        DataSet {
            nfeatures: 0,
            max_bins: max_bins,
            instances: Vec::new(),
            threshold_maps: Vec::new(),
        }
    }

    /// Generate thresholds. This interface is ugly. It introduces
    /// extra dependency that functions must be called in a specific
    /// order. But I haven't come up with a good workaround to support
    /// FromIterator. Basically, this is a issue how we customize the
    /// grouping of the data.
    fn generate_thresholds(&mut self) {
        for fid in self.fid_iter() {
            let values: Vec<Value> = self.instances
                .iter()
                .map(|instance| instance.value(fid))
                .collect();
            let map = ThresholdMap::new(values, self.max_bins);
            self.threshold_maps.push(map);
        }
    }

    /// Load data set from a reader.
    ///
    /// # Examples
    ///
    /// ```
    /// use rforests::train::dataset::DataSet;
    ///
    /// let s = "3.0 qid:1 1:5.0
    /// 2.0 qid:2 1:7.0
    /// 3.0 qid:3 1:3.0";
    ///
    /// let mut dataset = DataSet::new(3);
    /// dataset.load(::std::io::Cursor::new(s)).unwrap();
    ///
    /// assert_eq!(dataset[0].qid(), 1);
    /// assert_eq!(dataset[0].label(), 3.0);
    /// assert_eq!(dataset[0].value(1), 5.0);
    /// assert_eq!(dataset[1].qid(), 2);
    /// assert_eq!(dataset[2].qid(), 3);
    /// ```
    pub fn load<R>(&mut self, reader: R) -> Result<()>
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

        self.instances = instances;
        self.nfeatures = nfeatures;
        self.generate_thresholds();
        Ok(())
    }

    /// Load data from an Iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use rforests::train::dataset::DataSet;
    ///
    /// let data = vec![
    ///     // label, qid, values
    ///     (3.0, 1, vec![5.0]),
    ///     (2.0, 2, vec![7.0]),
    ///     (3.0, 3, vec![3.0]),
    /// ];
    ///
    /// let mut dataset = DataSet::new(3);
    /// dataset.from_iter(data.into_iter());
    ///
    /// assert_eq!(dataset[0].qid(), 1);
    /// assert_eq!(dataset[0].label(), 3.0);
    /// assert_eq!(dataset[0].value(1), 5.0);
    /// assert_eq!(dataset[1].qid(), 2);
    /// assert_eq!(dataset[2].qid(), 3);
    /// ```
    pub fn from_iter<T>(&mut self, iter: T)
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

        self.instances = instances;
        self.nfeatures = nfeatures;
        self.generate_thresholds();
    }

    /// Returns the number of instances in the data set, also referred
    /// to as its 'length'.
    ///
    /// # Examples
    ///
    /// ```
    /// use rforests::train::dataset::DataSet;
    ///
    /// let data = vec![
    ///     // label, qid, values
    ///     (3.0, 1, vec![5.0]),
    ///     (2.0, 2, vec![7.0]),
    ///     (3.0, 3, vec![3.0]),
    /// ];
    ///
    /// let mut dataset = DataSet::new(3);
    /// dataset.from_iter(data.into_iter());
    ///
    /// assert_eq!(dataset.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Returns an iterator over the feature ids in the data set.
    ///
    /// # Examples
    ///
    /// ```
    /// use rforests::train::dataset::DataSet;
    ///
    /// let data = vec![
    ///     // label, qid, values
    ///     (3.0, 1, vec![5.0, 6.0]),
    /// ];
    ///
    /// let mut dataset = DataSet::new(3);
    /// dataset.from_iter(data.into_iter());
    ///
    /// let mut iter = dataset.fid_iter();
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn fid_iter(&self) -> impl Iterator<Item = Id> {
        (1..(self.nfeatures + 1)).map(|i| i)
    }

    /// Returns an iterator over the labels in the data set.
    ///
    /// # Examples
    ///
    /// ```
    /// use rforests::train::dataset::DataSet;
    ///
    /// let data = vec![
    ///     // label, qid, values
    ///     (3.0, 1, vec![5.0, 6.0]),
    ///     (2.0, 2, vec![7.0, 8.0]),
    /// ];
    ///
    /// let mut dataset = DataSet::new(3);
    /// dataset.from_iter(data.into_iter());
    ///
    /// let mut iter = dataset.label_iter();
    /// assert_eq!(iter.next(), Some(3.0));
    /// assert_eq!(iter.next(), Some(2.0));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn label_iter<'a>(&'a self) -> impl Iterator<Item = Value> + 'a {
        self.instances.iter().map(|instance| instance.label)
    }

    /// Returns an iterator over the queries' indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use rforests::train::dataset::DataSet;
    ///
    /// let data = vec![
    ///     // label, qid, values
    ///     (3.0, 1, vec![5.0]), // 0
    ///     (2.0, 1, vec![7.0]), // 1
    ///     (3.0, 2, vec![3.0]), // 2
    ///     (1.0, 5, vec![2.0]), // 3
    /// ];
    ///
    /// let mut dataset = DataSet::new(3);
    /// dataset.from_iter(data.into_iter());
    ///
    /// let mut iter = dataset.query_iter();
    /// assert_eq!(iter.next(), Some((1, vec![0, 1])));
    /// assert_eq!(iter.next(), Some((2, vec![2])));
    /// assert_eq!(iter.next(), Some((5, vec![3])));
    /// ```
    pub fn query_iter(&self) -> QueryIter {
        QueryIter {
            dataset: self,
            index: 0,
        }
    }

    /// Generate histogram for the specified instances. `fid`
    /// specifies the feature used to split into histogram
    /// bins. `(index of the instance, value to do statistics)`.
    fn feature_histogram<I: Iterator<Item = (Id, Value)>>(
        &self,
        fid: Id,
        iter: I,
    ) -> Histogram {
        // Get the map by feature id.
        let threshold_map = &self.threshold_maps[fid - 1];
        let iter = iter.map(|(id, target)| {
            (id, self.instances[id].value(fid), target)
        });
        threshold_map.histogram(iter)
    }
}

impl std::ops::Deref for DataSet {
    type Target = Vec<Instance>;

    fn deref(&self) -> &Vec<Instance> {
        &self.instances
    }
}

/// A collection type containing a data set. The difference with
/// DataSet is that this data structure keeps the latest label values
/// after each training.
pub struct TrainingSet<'a> {
    dataset: &'a DataSet,
    // Fitting result of the model. We need to update the result at
    // each leaf node.
    model_scores: Vec<Cell<Value>>,
    // Gradients, or lambdas.
    lambdas: Vec<Value>,
    // Newton step weights
    weights: Vec<Value>,
}

impl<'a> TrainingSet<'a> {
    /// Returns the number of instances in the training set, also
    /// referred to as its 'length'.
    fn len(&self) -> usize {
        self.model_scores.len()
    }

    /// Get (label, instance) at given index.
    fn get(&self, index: usize) -> (Value, &'a Instance) {
        (self.model_scores[index].get(), &self.dataset[index])
    }

    /// Get (lambda, weight) at given index.
    fn get_lambda_weight(&self, index: usize) -> (Value, Value) {
        (self.lambdas[index], self.weights[index])
    }

    /// Returns an iterator over the feature ids in the training set.
    pub fn fid_iter(&self) -> impl Iterator<Item = Id> {
        self.dataset.fid_iter()
    }

    pub fn init_model_scores(&mut self, values: &[Value]) {
        assert_eq!(self.len(), values.len());
        for (score, &value) in self.model_scores.iter_mut().zip(values.iter()) {
            score.set(value);
        }
    }

    /// Returns an iterator over the labels in the data set.
    pub fn iter(&'a self) -> impl Iterator<Item = (Value, &Instance)> + 'a {
        self.model_scores.iter().map(|celled| celled.get()).zip(
            self.dataset.iter(),
        )
    }

    /// Returns an iterator over the labels in the data set.
    pub fn model_score_iter(&'a self) -> impl Iterator<Item = Value> + 'a {
        self.model_scores.iter().map(|celled| celled.get())
    }

    /// Returns the label value at given index.
    pub fn model_score(&self, index: usize) -> f64 {
        self.model_scores[index].get()
    }

    /// Adds delta to each label specified in `indices`.
    pub fn update_result(&self, indices: &[Id], delta: Value) {
        assert!(indices.len() <= self.model_scores.len());
        for &index in indices.iter() {
            let celled_score = &self.model_scores[index];
            celled_score.set(celled_score.get() + delta);
        }
    }

    /// Generate histogram for the specified instances. The input
    /// iterator specifies the indices of instance that we want to
    /// generate histogram on. For a training data set, the histogram
    /// is used to make statistics of the lambda values, which is
    /// actually the target value that we aims to fit to in the
    /// current iteration of learning.
    fn feature_histogram<I: Iterator<Item = Id>>(
        &self,
        fid: Id,
        iter: I,
    ) -> Histogram {
        // Get the map by feature id.
        let iter = iter.map(|id| (id, self.lambdas[id]));
        self.dataset.feature_histogram(fid, iter)
    }

    /// Updates the lambda and weight for each instance.
    pub fn update_lambdas_weights(&mut self) {
        let ndcg = NDCGScorer::new(10);

        for (_qid, query) in self.dataset.query_iter() {
            self.update_lambda_weight_by_query(&query, &ndcg);
        }
    }

    /// Updates the lambda and weight for each instance grouped by query.
    fn update_lambda_weight_by_query<S>(&mut self, query: &Vec<Id>, metric: &S)
    where
        S: MetricScorer,
    {
        use std::cmp::Ordering;

        let mut query = query.clone();

        // Rank the instances by the scores of our model.
        query.sort_by(|&index1, &index2| {
            // Descending
            self.model_scores[index2]
                .partial_cmp(&self.model_scores[index1])
                .unwrap_or(Ordering::Equal)
        });

        // Organize the original labels by the scores of our
        // model. For example, we have three instances.
        //
        // | Index | Label | Our Score |
        // |-------+-------+-----------|
        // |     0 |   5.0 |       2.0 |
        // |     1 |   4.0 |       5.0 |
        // |     2 |   3.0 |       4.0 |
        //
        // Ranked by our scores
        //
        // | Index | Label | Our Score |
        // |-------+-------+-----------|
        // |     1 |   4.0 |       5.0 |
        // |     2 |   3.0 |       4.0 |
        // |     0 |   5.0 |       2.0 |
        //
        // labes_sorted_by_scores: [4.0, 3.0, 5.0];
        let labels_sorted_by_scores: Vec<Value> = query
            .iter()
            .map(|&index| self.dataset[index].label())
            .collect();
        let metric_delta = metric.delta(&labels_sorted_by_scores);

        for (metric_index1, &index1) in query.iter().enumerate() {
            for (metric_index2, &index2) in query.iter().enumerate() {
                if self.dataset[index1].label() <=
                    self.dataset[index2].label()
                {
                    continue;
                }

                let metric_delta_value =
                    metric_delta[metric_index1][metric_index2].abs();
                let rho = 1.0 /
                    (1.0 +
                         (self.model_scores[index1].get() -
                              self.model_scores[index2].get())
                             .exp());
                let lambda = metric_delta_value * rho;
                let weight = rho * (1.0 - rho) * metric_delta_value;

                self.lambdas[index1] += lambda;
                self.weights[index1] += weight;
                self.lambdas[index2] -= lambda;
                self.weights[index2] += weight;
            }
        }
    }

    pub fn evaluate<S>(&self, metric: &S) -> f64
    where
        S: MetricScorer,
    {
        let mut score = 0.0;
        let mut count = 0;
        for (_qid, mut indices) in self.dataset.query_iter() {
            // Sort the indices by the score of the model, rank the
            // query based on the scores, then measure the output.

            indices.sort_by(|&index1, &index2| {
                self.model_score(index2)
                    .partial_cmp(&self.model_score(index1))
                    .unwrap()
            });

            let labels: Vec<Value> = indices
                .iter()
                .map(|&index| self.dataset[index].label())
                .collect();

            count += 1;
            score += metric.score(&labels);
        }

        score / count as f64
    }
}

impl<'a> From<&'a DataSet> for TrainingSet<'a> {
    fn from(dataset: &'a DataSet) -> TrainingSet<'a> {
        let len = dataset.len();
        let mut model_scores = Vec::with_capacity(len);
        model_scores.resize(len, Cell::new(0.0));
        let mut lambdas = Vec::with_capacity(len);
        lambdas.resize(len, 0.0);
        let mut weights = Vec::with_capacity(len);
        weights.resize(len, 0.0);
        TrainingSet {
            dataset: dataset,
            model_scores: model_scores,
            lambdas: lambdas,
            weights: weights,
        }
    }
}

/// A collection type containing part of a data set.
pub struct TrainingSample<'a> {
    /// Original data
    training: &'a TrainingSet<'a>,

    /// Indices into training
    indices: Vec<usize>,
}

impl<'a> TrainingSample<'a> {
    /// Returns the number of instances in the data set sample, also
    /// referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Creates an iterator which gives the index of the Instance as
    /// well as the Instance.
    ///
    /// The iterator returned yields pairs (index, value, instance),
    /// where `index` is the index of Instance, `value` is the label
    /// value, and `instance` is the reference to the Instance.
    pub fn iter(&'a self) -> impl Iterator<Item = (Id, Value, &Instance)> + 'a {
        self.indices.iter().map(move |&index| {
            let (label, instance) = self.training.get(index);
            (index, label, instance)
        })
    }

    /// Returns an iterator over the feature ids in the data set
    /// sample.
    pub fn fid_iter(&'a self) -> impl Iterator<Item = Id> + 'a {
        self.training.fid_iter()
    }

    /// Returns an iterator over the labels in the data set sample.
    pub fn label_iter(&'a self) -> impl Iterator<Item = Value> + 'a {
        self.iter().map(|(_index, label, _ins)| label)
    }

    /// Returns an iterator over the values of the given feature in
    /// the data set sample.
    pub fn value_iter(&'a self, fid: Id) -> impl Iterator<Item = Value> + 'a {
        self.iter().map(move |(_index, _label, ins)| ins.value(fid))
    }

    /// Returns the Newton step value.
    pub fn newton_output(&self) -> f64 {
        let (lambda_sum, weight_sum) = self.indices.iter().fold(
            (0.0, 0.0),
            |(lambda_sum,
              weight_sum),
             &index| {
                let (lambda, weight) = self.training.get_lambda_weight(index);
                (lambda_sum + lambda, weight_sum + weight)
            },
        );

        if weight_sum == 0.0 {
            0.0
        } else {
            lambda_sum / weight_sum
        }
    }

    pub fn update_output(&self, delta: Value) {
        self.training.update_result(&self.indices, delta);
    }

    /// Returns a histogram of the feature of the data set sample.
    fn feature_histogram(&self, fid: Id) -> Histogram {
        self.training.feature_histogram(
            fid,
            self.indices.iter().cloned(),
        )
    }

    /// Split self. Returns (split feature, threshold, s value, left
    /// child, right child). For each split, if its variance is zero,
    /// it's non-splitable.
    pub fn split(
        &self,
        min_leaf_count: usize,
    ) -> Option<(Id, Value, f64, TrainingSample, TrainingSample)> {
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
        for (index, _label, instance) in self.iter() {
            if instance.value(fid) <= threshold {
                left_indices.push(index);
            } else {
                right_indices.push(index);
            }
        }

        let left = TrainingSample {
            training: self.training,
            indices: left_indices,
        };
        let right = TrainingSample {
            training: self.training,
            indices: right_indices,
        };
        Some((fid, threshold, s, left, right))
    }
}

impl<'a> From<&'a TrainingSet<'a>> for TrainingSample<'a> {
    fn from(training: &'a TrainingSet) -> TrainingSample<'a> {
        let len = training.len();
        let indices: Vec<usize> = (0..len).collect();
        TrainingSample {
            training: training,
            indices: indices,
        }
    }
}

impl<'a> std::fmt::Display for TrainingSample<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for &index in self.indices.iter() {
            let (label, instance) = self.training.get(index);

            write!(
                f,
                "{{index: {}, label: {}, instance: {}}}\n",
                index,
                label,
                instance
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_queries() {
        let s = "0 qid:3864 1:1.0 2:0.0 3:0.0 4:0.0 5:0.0
2 qid:3864 1:1.0 2:0.007042 3:0.0 4:0.0 5:0.221591
0 qid:3865 1:0.289474 2:0.014085 3:0.4 4:0.0 5:0.085227";
        let mut dataset = DataSet::new(3);
        dataset.load(::std::io::Cursor::new(s)).unwrap();

        assert_eq!(dataset.nfeatures, 5);
        assert_eq!(
            dataset.instances[0],
            Instance::new(0.0, 3864, vec![1.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            dataset.instances[1],
            Instance::new(2.0, 3864, vec![1.0, 0.007042, 0.0, 0.0, 0.221591])
        );
        assert_eq!(
            dataset.instances[2],
            Instance::new(0.0, 3865, vec![0.289474, 0.014085, 0.4, 0.0, 0.085227])
        );
    }

    #[test]
    fn test_instance_interface() {
        let label = 3.0;
        let qid = 3333;
        let values = vec![1.0, 2.0, 3.0];
        let instance = Instance::new(label, qid, values);

        // value_iter()
        let mut iter = instance.value_iter();
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
    fn test_data_set_lambda_weight() {
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

        let mut dataset = DataSet::new(3);
        dataset.from_iter(data);

        let mut training = TrainingSet::from(&dataset);
        training.update_lambdas_weights();

        // The values are verified by hand. This test is kept as a
        // guard for future modifications.
        assert_eq!(
            training.lambdas,
            &[
                0.2959880583703105,
                -0.05406635038708441,
                0.06664831928002701,
                -0.10688704271796713,
                -0.1309783051272036,
                -0.056352467003334426,
                0.2573545140200802,
                -0.11687432957979353,
                -0.15483239685503464,
            ]
        );
        assert_eq!(
            training.weights,
            &[
                0.2503273430028968,
                0.07986338018045583,
                0.05890748809444887,
                0.056771982359676655,
                0.0654891525636018,
                0.037537655576830996,
                0.1286772570100401,
                0.06008388967286634,
                0.07741619842751732,
            ]
        );
    }

    #[test]
    fn test_data_set_sample_split() {
        // (label, qid, feature_values)
        let data = vec![         // lambda values to fit in the first iteration.
            (3.0, 1, vec![5.0]), // 0.2959880583703105,
            (2.0, 1, vec![7.0]), // -0.05406635038708441,
            (3.0, 1, vec![3.0]), // 0.06664831928002701,
            (1.0, 1, vec![2.0]), // -0.10688704271796713,
            (0.0, 1, vec![1.0]), // -0.1309783051272036,
            (2.0, 1, vec![8.0]), // -0.056352467003334426,
            (4.0, 1, vec![9.0]), // 0.2573545140200802,
            (1.0, 1, vec![4.0]), // -0.11687432957979353,
            (0.0, 1, vec![6.0]), // -0.15483239685503464,
        ];

        let mut dataset = DataSet::new(3);
        dataset.from_iter(data.into_iter());

        let mut training = TrainingSet::from(&dataset);
        training.update_lambdas_weights();

        let sample = TrainingSample::from(&training);
        let (fid, threshold, s, left, right) = sample.split(1).unwrap();
        assert_eq!(fid, 1);
        assert_eq!(threshold, 1.0);
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

        let mut dataset = DataSet::new(3);
        dataset.from_iter(data.into_iter());

        // possible splits of feature values:
        // 1 | 2 3 4 5 6 7 8 9
        // 1 2 3 | 4 5 6 7 8 9
        // 1 2 3 4 5 6 | 7 8 9
        let mut training = TrainingSet::from(&dataset);
        training.update_lambdas_weights();

        let sample = TrainingSample::from(&training);
        assert!(sample.split(9).is_none());
        assert!(sample.split(4).is_none());
        let (fid, threshold, _s, left, _right) = sample.split(3).unwrap();
        assert_eq!(fid, 1);
        assert_eq!(threshold, 3.0 + 2.0 / 3.0);

        assert!(left.split(2).is_none());
    }
}
