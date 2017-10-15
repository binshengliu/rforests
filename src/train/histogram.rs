use std;

#[derive(Debug, PartialEq)]
struct HistogramBin {
    // Max value of this bin
    threashold: f64,

    // Accumulated count of all the values less than or equal to
    // threashold.
    acc_count: usize,

    // Accumulated sum of all the values less than or equal to
    // threashold.
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
    map_from_dataset_to_bins: Vec<usize>,
}

impl FeatureHistogram {
    /// Construct histograms for given values. Generate a map from the
    /// original indices into histogram bins.
    pub fn new(
        sorted_indices_values: Vec<(usize, f64)>,
        max_bins_count: usize,
    ) -> FeatureHistogram {
        let mut threasholds: Vec<f64> = sorted_indices_values
            .iter()
            .map(|&(_index, value)| value)
            .collect();
        threasholds.dedup();

        // If too many threasholds, generate at most max_bins_count
        // threasholds. For example, to split "2, 3, 4, 5, 6" into 5
        // bins, we compute step = (6 - 2) / (5 - 1) = 1, and get
        // threasholds "2, 3, 4, 5, 6".
        if threasholds.len() > max_bins_count {
            let max = *threasholds.last().unwrap();
            let min = *threasholds.first().unwrap();
            let step = (max - min) / max_bins_count as f64;
            threasholds =
                (0..max_bins_count).map(|n| min + n as f64 * step).collect();
        }
        threasholds.push(std::f64::MAX);

        let nvalues = sorted_indices_values.len();
        let mut map_from_dataset_to_bins: Vec<usize> = Vec::new();
        map_from_dataset_to_bins.resize(nvalues, 0);
        let mut pos = 0;
        let mut acc_count = 0;
        let mut acc_sum = 0.0;
        let mut bins: Vec<HistogramBin> = Vec::new();
        for threashold in threasholds.iter() {
            let index_in_bins = bins.len();
            for &(original_index, value) in
                sorted_indices_values[pos..].iter()
            {
                if value > *threashold {
                    break;
                }
                acc_count += 1;
                acc_sum += value;
                map_from_dataset_to_bins[original_index] = index_in_bins;
            }
            bins.push(HistogramBin::new(*threashold, acc_count, acc_sum));

            pos = acc_count;
        }

        FeatureHistogram {
            bins: bins,
            map_from_dataset_to_bins: map_from_dataset_to_bins,
        }
    }

    // Update the values' sum for each bin
    pub fn update_sum(&mut self, labels: &[f64]) {
        for (index, &label) in labels.iter().enumerate() {
            let bin_index = self.map_from_dataset_to_bins[index];
            let bin = &mut self.bins[bin_index];
            bin.acc_sum += label;

        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_feature_histogram() {
        // original: vec![5, 7, 3, 2, 1, 8, 9, 4, 6]
        let sorted_indices_values = vec![
            (4, 1.0),
            (3, 2.0),
            (2, 3.0),
            (7, 4.0),
            (0, 5.0),
            (8, 6.0),
            (1, 7.0),
            (5, 8.0),
            (6, 9.0),
        ];

        let histogram = FeatureHistogram::new(sorted_indices_values, 3);
        assert_eq!(
            histogram.bins,
            vec![
                // threashold: 1.0, values: [1.0]
                HistogramBin::new(1.0 + 0.0 * 8.0 / 3.0, 1, 1.0),
                // threashold: 3.66, values: [1.0, 2.0, 3.0]
                HistogramBin::new(1.0 + 1.0 * 8.0 / 3.0, 3, 6.0),
                // threashold: 6.33, values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                HistogramBin::new(1.0 + 2.0 * 8.0 / 3.0, 6, 21.0),
                // threashold: MAX, values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                HistogramBin::new(std::f64::MAX, 9, 45.0),
            ]
        );
    }

}
