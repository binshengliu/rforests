use std;
use train::dataset::*;

#[derive(Debug, PartialEq)]
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

#[derive(Debug)]
pub struct FeatureHistogram {
    // [from, to]
    bins: Vec<HistogramBin>,

    // Index into a bin given the index in the dataset
    bin_index_map: Vec<usize>,
}

impl FeatureHistogram {
    /// Generate threasholds vec from values and max bins.
    fn threasholds(values: &[(usize, f64, f64)], max_bins: usize) -> Vec<f64> {
        let mut threasholds: Vec<f64> = values
            .iter()
            .map(|&(_index, _label, value)| value)
            .collect();
        threasholds.dedup();

        // If too many threasholds, generate at most max_bins
        // threasholds.
        if threasholds.len() > max_bins {
            let max = *threasholds.last().unwrap();
            let min = *threasholds.first().unwrap();
            let step = (max - min) / max_bins as f64;
            threasholds =
                (0..max_bins).map(|n| min + n as f64 * step).collect();
        }
        threasholds.push(std::f64::MAX);
        threasholds
    }

    /// Construct histograms for given values. Generate a map from the
    /// original indices into histogram bins.
    pub fn new(
        values: &[(usize, f64, f64)],
        max_bins: usize,
    ) -> FeatureHistogram {
        let nvalues = values.len();
        let mut bin_index_map: Vec<usize> = Vec::new();
        bin_index_map.resize(nvalues, 0);
        let mut bins: Vec<HistogramBin> = Vec::new();

        let mut pos = 0;
        let mut acc_count = 0;
        let mut acc_sum = 0.0;

        let threasholds: Vec<f64> =
            FeatureHistogram::threasholds(values, max_bins);

        for threashold in threasholds.iter() {
            let index_in_bins = bins.len();
            for &(index, label, value) in values[pos..].iter() {
                if value > *threashold {
                    break;
                }
                acc_count += 1;
                acc_sum += label;
                bin_index_map[index] = index_in_bins;
            }
            bins.push(HistogramBin::new(*threashold, acc_count, acc_sum));

            pos = acc_count;
        }

        FeatureHistogram {
            bins: bins,
            bin_index_map: bin_index_map,
        }
    }

    // Update the values' sum for each bin
    pub fn update(&mut self, labels: &[f64]) {
        for bin in self.bins.iter_mut() {
            bin.acc_sum = 0.0;
        }

        for (index, &label) in labels.iter().enumerate() {
            let bin_index = self.bin_index_map[index];
            let bin = &mut self.bins[bin_index];
            bin.acc_sum += label;
        }

        // Accumulate all the preceding values of each bin.
        let mut acc = 0.0;
        for bin in self.bins.iter_mut() {
            bin.acc_sum += acc;
            acc = bin.acc_sum;
        }
    }
}

pub struct Histogram {
    /// Histogram for each feature
    hists: Vec<FeatureHistogram>,

    /// Sum of labels
    label_sum: f64,

    /// Sum of squared labels
    label_sq_sum: f64,
}

impl Histogram {
    pub fn new(dataset: &DataSet, max_bins: usize) -> Histogram {
        // Sum of labels and sum of squared labels.
        let (sum, sq_sum) = dataset.labels_iter().fold(
            (0.0, 0.0),
            |(sum, sq_sum), label| {
                (sum + label, sq_sum + label * label)
            },
        );
        let mut hists: Vec<FeatureHistogram> = Vec::new();
        for fid in dataset.fid_iter() {
            let feature_hist = dataset.feature_histogram(fid, max_bins);
            hists.push(feature_hist);
        }

        Histogram {
            hists: hists,
            label_sum: sum,
            label_sq_sum: sq_sum,
        }
    }

    pub fn update(&mut self, labels: &[f64]) {
        for hist in self.hists.iter_mut() {
            hist.update(labels);
        }

        let (sum, sq_sum) =
            labels.iter().fold((0.0, 0.0), |(sum, sq_sum), label| {
                (sum + label, sq_sum + label * label)
            });
        self.label_sum = sum;
        self.label_sq_sum = sq_sum;
    }
}

impl std::ops::Deref for Histogram {
    type Target = Vec<FeatureHistogram>;

    fn deref(&self) -> &Vec<FeatureHistogram> {
        &self.hists
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_feature_histogram() {
        // (label, feature_value)
        let data = vec![
            (3.0, 5.0),
            (2.0, 7.0),
            (3.0, 3.0),
            (1.0, 2.0),
            (0.0, 1.0),
            (2.0, 8.0),
            (4.0, 9.0),
            (1.0, 4.0),
            (0.0, 6.0),
        ];
        // sorted by featuer_value, (index, label, feature_value):
        // (4, 0.0, 1.0),
        // (3, 1.0, 2.0),
        // (2, 3.0, 3.0),
        // (7, 1.0, 4.0),
        // (0, 3.0, 5.0),
        // (8, 0.0, 6.0),
        // (1, 2.0, 7.0),
        // (5, 2.0, 8.0),
        // (6, 4.0, 9.0),

        let mut data: Vec<(usize, f64, f64)> = data.iter()
            .enumerate()
            .map(|(index, &(label, value))| (index, label, value))
            .collect();

        // Sort by values in non-descending order.
        data.sort_by(|&(_index1, _label1, value1),
         &(_index2, _label2, value2)| {
            value1.partial_cmp(&value2).unwrap_or(
                std::cmp::Ordering::Equal,
            )
        });

        let mut histogram = FeatureHistogram::new(&data, 3);
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

        let new_labels = vec![3.0, 2.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0, 4.0];
        // (3.0, 5.0),
        // (2.0, 7.0),
        // (1.0, 3.0),
        // (3.0, 2.0),
        // (1.0, 1.0),
        // (0.0, 8.0),
        // (1.0, 9.0),
        // (2.0, 4.0),
        // (4.0, 6.0),
        // ->
        // sorted by featuer_value, (index, label, feature_value):
        // (1.0, 1.0),
        // (3.0, 2.0),
        // (1.0, 3.0),
        // (2.0, 4.0),
        // (3.0, 5.0),
        // (4.0, 6.0),
        // (2.0, 7.0),
        // (0.0, 8.0),
        // (1.0, 9.0),
        histogram.update(&new_labels);
        assert_eq!(
            histogram.bins,
            vec![
                // threashold: 1.0, values: [1.0], labels: [1.0]
                HistogramBin::new(1.0 + 0.0 * 8.0 / 3.0, 1, 1.0),
                // threashold: 3.66, values: [1.0, 2.0, 3.0], labels:
                // [1.0, 3.0, 1.0]
                HistogramBin::new(1.0 + 1.0 * 8.0 / 3.0, 3, 5.0),
                // threashold: 6.33, values: [1.0, 2.0, 3.0, 4.0, 5.0,
                // 6.0], labels: [1.0, 3.0, 1.0, 2.0, 3.0, 4.0]
                HistogramBin::new(1.0 + 2.0 * 8.0 / 3.0, 6, 14.0),
                // threashold: MAX, values: [1.0, 2.0, 3.0, 4.0, 5.0,
                // 6.0, 7.0, 8.0, 9.0], labels: [1.0, 3.0, 1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0]
                HistogramBin::new(std::f64::MAX, 9, 17.0),
            ]
        );
    }

}
