use format::svmlight::SvmLightFile;
use util::{Id, Result, Value};
use std;
use std::cmp::Ordering::*;
use train::Evaluate;
use metric::*;

/// An instance of a label, a qid, and a group of feature values.
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone)]
pub struct DataSet {
    nfeatures: usize,
    instances: Vec<Instance>,
    // Group by queries. (Start index, Query Length).
    queries: Vec<(usize, usize)>,
}

impl std::iter::FromIterator<(Value, Id, Vec<Value>)> for DataSet {
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
    /// let mut dataset: DataSet = data.into_iter().collect();
    ///
    /// assert_eq!(dataset[0].qid(), 1);
    /// assert_eq!(dataset[0].label(), 3.0);
    /// assert_eq!(dataset[0].value(1), 5.0);
    /// assert_eq!(dataset[1].qid(), 2);
    /// assert_eq!(dataset[2].qid(), 3);
    /// ```
    fn from_iter<T>(iter: T) -> DataSet
    where
        T: IntoIterator<Item = (Value, Id, Vec<Value>)>,
    {
        let mut instances = Vec::new();
        let mut nfeatures = 0;
        let mut queries = Vec::new();
        let mut query_start = 0;
        let mut query_len = 0;
        for (label, qid, values) in iter {
            let instance = Instance::from((label, qid, values));
            nfeatures =
                usize::max(nfeatures, instance.max_feature_id() as usize);
            instances.push(instance);

            if qid != instances[query_start].qid() {
                queries.push((query_start, query_len));
                query_start = instances.len() - 1;
                query_len = 1;
            } else {
                query_len += 1;
            }
        }
        queries.push((query_start, query_len));

        DataSet {
            instances: instances,
            nfeatures: nfeatures,
            queries: queries,
        }
    }
}

impl DataSet {
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
    /// let dataset = DataSet::load(::std::io::Cursor::new(s)).unwrap();
    ///
    /// assert_eq!(dataset[0].qid(), 1);
    /// assert_eq!(dataset[0].label(), 3.0);
    /// assert_eq!(dataset[0].value(1), 5.0);
    /// assert_eq!(dataset[1].qid(), 2);
    /// assert_eq!(dataset[2].qid(), 3);
    /// ```
    pub fn load<R>(reader: R) -> Result<DataSet>
    where
        R: ::std::io::Read,
    {
        let mut instances = Vec::new();
        let mut nfeatures = 0;
        let mut queries = Vec::new();
        let mut query_start = 0;
        let mut query_len = 0;
        debug!("Loading data...");
        for instance_result in SvmLightFile::instances(reader) {
            let instance = instance_result?;
            nfeatures =
                usize::max(nfeatures, instance.max_feature_id() as usize);
            let qid = instance.qid();
            instances.push(instance);

            if qid != instances[query_start].qid() {
                queries.push((query_start, query_len));
                query_start = instances.len() - 1;
                query_len = 1;
            } else {
                query_len += 1;
            }
        }
        queries.push((query_start, query_len));
        debug!(
            "Loaded {} instances, {} features.",
            instances.len(),
            nfeatures
        );

        Ok(DataSet {
            instances: instances,
            nfeatures: nfeatures,
            queries: queries,
        })
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
    /// let mut dataset: DataSet = data.into_iter().collect();
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
    /// let mut dataset: DataSet = data.into_iter().collect();
    ///
    /// let mut iter = dataset.label_iter();
    /// assert_eq!(iter.next(), Some(3.0));
    /// assert_eq!(iter.next(), Some(2.0));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn label_iter<'a>(&'a self) -> impl Iterator<Item = Value> + 'a {
        self.instances.iter().map(|instance| instance.label)
    }

    pub fn feature_value_iter<'a>(
        &'a self,
        fid: Id,
    ) -> impl Iterator<Item = Value> + 'a {
        self.instances.iter().map(
            move |instance| instance.value(fid),
        )
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
    /// let mut dataset: DataSet = data.into_iter().collect();
    ///
    /// let mut iter = dataset.query_iter();
    /// assert_eq!(iter.next(), Some((1, vec![0, 1])));
    /// assert_eq!(iter.next(), Some((2, vec![2])));
    /// assert_eq!(iter.next(), Some((5, vec![3])));
    /// ```
    pub fn query_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (Id, Vec<Id>)> + 'a {
        self.queries.iter().map(move |&(start, len)| {
            (
                self[start].qid(),
                (start..(start + len)).collect::<Vec<usize>>(),
            )
        })
    }

    pub fn evaluate<E: Evaluate>(
        &self,
        e: &E,
        metric: &Box<Measure>,
    ) -> f64 {
        let mut score = 0.0;
        let mut count: usize = 0;
        for (qid, query) in self.query_iter() {
            let mut model_scores: Vec<(Id, Value)> = query
                .iter()
                .map(|&id| (id, e.evaluate(&self.instances[id])))
                .collect();
            model_scores.sort_by(|&(_index1, score1), &(_index2, score2)| {
                score2.partial_cmp(&score1).unwrap_or(Equal)
            });

            let labels: Vec<f64> = model_scores
                .iter()
                .map(|&(id, _)| self.instances[id].label())
                .collect();
            let query_score = metric.measure(&labels);
            debug!("Model score for qid {}: {}", qid, score);

            count += 1;
            score += query_score;
        }

        let result = score / count as f64;
        debug!("Model score for validation data: {}", result);
        result
    }
}

impl std::ops::Deref for DataSet {
    type Target = Vec<Instance>;

    fn deref(&self) -> &Vec<Instance> {
        &self.instances
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_generate_queries() {
        let s = "0 qid:3864 1:1.0 2:0.0 3:0.0 4:0.0 5:0.0
2 qid:3864 1:1.0 2:0.007042 3:0.0 4:0.0 5:0.221591
0 qid:3865 1:0.289474 2:0.014085 3:0.4 4:0.0 5:0.085227";
        let dataset = DataSet::load(::std::io::Cursor::new(s)).unwrap();

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
        assert_eq!(dataset.queries[0], (0, 2));
        assert_eq!(dataset.queries[1], (2, 1));
    }
}
