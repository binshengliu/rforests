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
    ///
    /// The best split minimizes the sum of the variance of the left
    /// part and right part.
    ///
    /// Minimize sum_of_variance = sum((left_labels - left_label_avg)
    /// ^ 2) + sum((right_labels - right_label_avg) ^ 2)
    ///
    /// where left_label_avg = sum(left_labels) / left_count, and
    /// right_label_avg = sum(right_labels) / right_count.
    ///
    /// expand the right:
    ///
    /// sum_of_variance = sum(left_labels ^ 2) - sum(left_labels) ^ 2
    /// / left_count + sum(right_labels ^ 2) - sum(right_labels) ^ 2 /
    /// right_count
    ///
    /// sum_of_variance = sum(all_labels ^ 2) - sum(left_labels) ^ 2 -
    /// sum(right_labels) ^ 2
    ///
    /// To minimize the result, we just need to find a point that
    /// maximizes sum(left_label) ^ 2 + sum(right_labels) ^ 2
    pub fn best_split(&self, min_leaf: usize) -> Option<(Value, f64)> {
        let sum = self.bins.last().unwrap().acc_sum;
        let count = self.bins.last().unwrap().acc_count;
        let mut split: Option<(f64, f64)> = None;
        for (index, bin) in self.bins.iter().enumerate() {
            let count_left = bin.acc_count;
            let count_right = count - count_left;
            if count_left < min_leaf || count_right < min_leaf {
                continue;
            }

            let sum_left = bin.acc_sum;
            let sum_right = sum - sum_left;

            let s_value = sum_left * sum_left / count_left as f64 +
                sum_right * sum_right / count_right as f64;

            debug!(
                "Bin {}, threshold {}, left sum {}, left count {}, right sum {}, right count {}, s {}",
                index,
                bin.threashold,
                sum_left,
                count_left,
                sum_right,
                count_right,
                s_value
            );
            match split {
                Some((_old_threashold, old_s)) => {
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
                HistogramBin::new(
                    threshold,
                    acc_count,
                    acc_sum,
                )
            })
            .collect();

        Histogram::new(bins)
    }
}

#[cfg(test)]
mod test {
    // use train::dataset::*;
    // use super::*;

    // #[test]
    // fn test_feature_histogram() {
    //     // (label, qid, feature_values)
    //     let data = vec![
    //         (3.0, 1, vec![5.0]),
    //         (2.0, 1, vec![7.0]),
    //         (3.0, 1, vec![3.0]),
    //         (1.0, 1, vec![2.0]),
    //         (0.0, 1, vec![1.0]),
    //         (2.0, 1, vec![8.0]),
    //         (4.0, 1, vec![9.0]),
    //         (1.0, 1, vec![4.0]),
    //         (0.0, 1, vec![6.0]),
    //     ];

    //     let lambdas = vec![3.0, 2.0, 3.0, 1.0, 0.0, 2.0, 4.0, 1.0, 0.0];

    //     let mut dataset = DataSet::new(3);
    //     dataset.from_iter(data.into_iter());

    //     let histogram =
    //         dataset.feature_histogram(1, lambdas.iter().cloned().enumerate());

    //     assert_eq!(
    //         histogram.bins[0],
    //         // threashold: 1.0, values: [1.0], labels: [0.0]
    //         HistogramBin::new(1.0 + 0.0 * 8.0 / 3.0, 1, 0.0, 0.0)
    //     );

    //     assert_eq!(
    //         histogram.bins[1],
    //         // threashold: 3.66, values: [1.0, 2.0, 3.0], labels:
    //         // [0.0, 1.0, 3.0]
    //         HistogramBin::new(1.0 + 1.0 * 8.0 / 3.0, 3, 4.0, 10.0)
    //     );

    //     assert_eq!(
    //         histogram.bins[2],
    //         // threashold: 6.33, values: [1.0, 2.0, 3.0, 4.0, 5.0,
    //         // 6.0], labels: [0.0, 1.0, 3.0, 1.0, 3.0, 0.0]
    //         HistogramBin::new(1.0 + 2.0 * 8.0 / 3.0, 6, 8.0, 20.0)
    //     );

    //     assert_eq!(
    //         histogram.bins[3],
    //         // threashold: MAX, values: [1.0, 2.0, 3.0, 4.0, 5.0,
    //         // 6.0, 7.0, 8.0, 9.0], labels: [0.0, 1.0, 3.0, 1.0,
    //         // 3.0, 0.0, 2.0, 2.0, 4.0]
    //         HistogramBin::new(std::f64::MAX, 9, 16.0, 44.0)
    //     );

    //     // 15.555555555555557
    //     assert_eq!(histogram.variance(), 15.555555555555557);
    // }
}
