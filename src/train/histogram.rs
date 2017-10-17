use std;
use util::*;

#[derive(PartialEq)]
struct HistogramBin {
    // Max value of this bin
    threashold: f64,

    // Accumulated count of all the values of this and preceding bins.
    acc_count: usize,

    // Accumulated sum of all the labels of this and preceding bins.
    acc_sum: f64,
}

impl HistogramBin {
    pub fn new(
        threashold: f64,
        acc_count: usize,
        acc_sum: f64,
    ) -> HistogramBin {
        HistogramBin {
            threashold: threashold,
            acc_count: acc_count,
            acc_sum: acc_sum,
        }
    }
}

impl std::fmt::Debug for HistogramBin {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "HistogramBin {{ threashold: {}, acc_count: {}, acc_sum: {} }}",
            if self.threashold == std::f64::MAX {
                "f64::MAX".to_string()
            } else {
                self.threashold.to_string()
            },
            self.acc_count,
            self.acc_sum.to_string()
        )
    }
}

#[derive(Debug)]
pub struct Histogram {
    // [from, to]
    bins: Vec<HistogramBin>,
}

impl Histogram {
    fn new(bins: Vec<HistogramBin>) -> Histogram {
        Histogram { bins: bins }
    }

    /// Return the best splitting point. The returned value is of the
    /// form (threashold, s value).
    pub fn best_split(&self, min_leaf: usize) -> Option<(Value, f64)> {
        let sum = self.bins.last().unwrap().acc_sum;
        let count = self.bins.last().unwrap().acc_count;
        let mut split: Option<(f64, f64)> = None;
        for bin in self.bins.iter() {
            let count_left = bin.acc_count;
            let count_right = count - count_left;
            if count_left < min_leaf || count_right < min_leaf {
                continue;
            }

            let sum_left = bin.acc_sum;
            let sum_right = sum - sum_left;

            let s_value = sum_left * sum_left / count_left as f64 +
                sum_right * sum_right / count_right as f64;

            match split {
                Some((old_s, _old_threashold)) => {
                    if s_value > old_s {
                        split = Some((bin.threashold, s_value));
                    }
                }
                None => split = Some((bin.threashold, s_value)),
            }
        }

        split
    }
}

use std::iter::FromIterator;
impl FromIterator<(Value, usize, Value)> for Histogram {
    fn from_iter<T>(iter: T) -> Histogram
    where
        T: IntoIterator<Item = (Value, usize, Value)>,
    {
        let bins: Vec<HistogramBin> = iter.into_iter()
            .map(|(threshold, acc_count, acc_sum)| {
                HistogramBin::new(threshold, acc_count, acc_sum)
            })
            .collect();

        Histogram::new(bins)
    }
}

#[cfg(test)]
mod test {
    use train::dataset::*;
    use super::*;

    #[test]
    fn test_feature_histogram() {
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

        // The label values are zero by default.
        let mut training = TrainingSet::from(&dataset);
        training.add(&[3.0, 2.0, 3.0, 1.0, 0.0, 2.0, 4.0, 1.0, 0.0]);

        let sample = TrainingSample::from(&training);

        let histogram = sample.feature_histogram(1);
        assert_eq!(
            histogram.bins,
            vec![
                // threashold: 1.0, values: [1.0], labels: [0.0]
                HistogramBin::new(1.0 + 0.0 * 8.0 / 3.0, 1, 0.0),
                // threashold: 3.66, values: [1.0, 2.0, 3.0], labels:
                // [0.0, 1.0, 3.0]
                HistogramBin::new(1.0 + 1.0 * 8.0 / 3.0, 3, 4.0),
                // threashold: 6.33, values: [1.0, 2.0, 3.0, 4.0, 5.0,
                // 6.0], labels: [0.0, 1.0, 3.0, 1.0, 3.0, 0.0]
                HistogramBin::new(1.0 + 2.0 * 8.0 / 3.0, 6, 8.0),
                // threashold: MAX, values: [1.0, 2.0, 3.0, 4.0, 5.0,
                // 6.0, 7.0, 8.0, 9.0], labels: [0.0, 1.0, 3.0, 1.0,
                // 3.0, 0.0, 2.0, 2.0, 4.0]
                HistogramBin::new(std::f64::MAX, 9, 16.0),
            ]
        );
    }
}
