use super::MetricScorer;

pub struct DCGScorer {
    truncation_level: usize,
}

impl DCGScorer {
    // Maybe cache the values. But I haven't come up with a method to
    // share the cached values.
    fn discount(&self, i: usize) -> f64 {
        1.0 / (i as f64 + 2.0).log2()
    }

    fn gain(&self, score: f64) -> f64 {
        score.exp2() - 1.0
    }
}

impl MetricScorer for DCGScorer {
    fn new(truncation_level: usize) -> DCGScorer {
        DCGScorer { truncation_level: truncation_level }
    }

    fn score(&self, labels: &[f64]) -> f64 {
        let n = usize::min(labels.len(), self.truncation_level);
        (0..n)
            .map(|i| self.gain(labels[i]) * self.discount(i))
            .sum()
    }

    fn delta(&self, labels: &[f64]) -> Vec<Vec<f64>> {
        let nlabels = labels.len();
        let mut row: Vec<f64> = Vec::with_capacity(nlabels);
        row.resize(nlabels, 0.0);

        let mut delta = Vec::with_capacity(nlabels);
        delta.resize(nlabels, row);

        for i in 0..nlabels {
            for j in i + 1..nlabels {
                delta[i][j] = (self.gain(labels[i]) - self.gain(labels[j])) *
                    (self.discount(i) - self.discount(j));
                delta[j][i] = delta[i][j];
            }
        }

        delta
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dcg_score() {
        let dcg = DCGScorer::new(10);
        assert_eq!(
            dcg.score(&vec![3.0, 2.0, 4.0]),
            7.0 / 2.0_f64.log2() + 3.0 / 3.0_f64.log2() + 15.0 / 4.0_f64.log2()
        );
    }

    #[test]
    fn test_dcg_delta() {
        let dcg = DCGScorer::new(10);

        // 16.392789260714373
        let origin = 7.0 / 2.0_f64.log2() + 3.0 / 3.0_f64.log2() +
            15.0 / 4.0_f64.log2();

        // 14.916508275000202,
        let score_swap_0_1 = 3.0 / 2.0_f64.log2() + 7.0 / 3.0_f64.log2() +
            15.0 / 4.0_f64.log2();

        // 20.392789260714373
        let score_swap_0_2 = 15.0 / 2.0_f64.log2() + 3.0 / 3.0_f64.log2() +
            7.0 / 4.0_f64.log2();

        // 17.963946303571863
        let score_swap_1_2 = 7.0 / 2.0_f64.log2() + 15.0 / 3.0_f64.log2() +
            3.0 / 4.0_f64.log2();

        let result = dcg.delta(&vec![3.0, 2.0, 4.0]);
        let expected =
            vec![
                vec![0.0, origin - score_swap_0_1, origin - score_swap_0_2],
                vec![origin - score_swap_0_1, 0.0, origin - score_swap_1_2],
                vec![origin - score_swap_0_2, origin - score_swap_1_2, 0.0],
            ];
        let result: Vec<f64> = result.into_iter().flat_map(|row| row).collect();
        let expected: Vec<f64> =
            expected.into_iter().flat_map(|row| row).collect();

        let check =
            result.iter().zip(expected.iter()).all(|(value1, value2)| {
                (value1 - value2).abs() < 0.000001
            });
        assert!(check);
    }
}
