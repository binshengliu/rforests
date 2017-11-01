use super::MetricScorer;
use super::DCGScorer;

lazy_static! {
    static ref DISCOUNT: Vec<f64> = (0..128).map(|i| 1.0 / (i as f64 + 2.0).log2()).collect();
}

pub struct NDCGScorer {
    truncation_level: usize,
    dcg: DCGScorer,
}

impl NDCGScorer {
    pub fn new(truncation_level: usize) -> NDCGScorer {
        NDCGScorer {
            truncation_level: truncation_level,
            dcg: DCGScorer::new(truncation_level),
        }
    }

    // Maybe cache the values. But I haven't come up with a method to
    // share the cached values.
    fn discount(&self, i: usize) -> f64 {
        let len = DISCOUNT.len();
        if i >= len {
            1.0 / (i as f64 + 2.0).log2()
        } else {
            DISCOUNT[i]
        }
    }

    fn gain(&self, score: f64) -> f64 {
        score.exp2() - 1.0
    }

    fn max_dcg(&self, labels: &[f64]) -> f64 {
        use std::cmp::Ordering;

        let mut clone: Vec<f64> = labels.iter().cloned().collect();
        clone.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        self.dcg.score(&clone)
    }
}

impl MetricScorer for NDCGScorer {
    fn name(&self) -> String {
        format!("NDCG@{}", self.truncation_level)
    }

    fn get_k(&self) -> usize {
        self.truncation_level
    }

    fn score(&self, labels: &[f64]) -> f64 {
        let max = self.max_dcg(labels);
        if max.abs() == 0.0 {
            0.0
        } else {
            self.dcg.score(labels) / self.max_dcg(labels)
        }
    }

    fn delta(&self, labels: &[f64]) -> Vec<Vec<f64>> {
        let nlabels = labels.len();

        let mut delta = vec![vec![0.0; nlabels]; nlabels];

        let ideal_dcg = self.max_dcg(labels);

        let size = usize::min(self.truncation_level, nlabels);
        for i in 0..size {
            for j in i + 1..nlabels {
                delta[i][j] = (self.gain(labels[i]) - self.gain(labels[j])) *
                    (self.discount(i) - self.discount(j));
                delta[i][j] /= ideal_dcg;
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
    fn test_ndcg_score() {
        let ndcg = NDCGScorer::new(10);
        let dcg = 7.0 / 2.0_f64.log2() + 3.0 / 3.0_f64.log2() +
            15.0 / 4.0_f64.log2();
        let max_dcg = 15.0 / 2.0_f64.log2() + 7.0 / 3.0_f64.log2() +
            3.0 / 4.0_f64.log2();
        assert_eq!(ndcg.score(&vec![3.0, 2.0, 4.0]), dcg / max_dcg);
    }

    #[test]
    fn test_ndcg_score_zeros() {
        let ndcg = NDCGScorer::new(10);
        assert_eq!(ndcg.score(&vec![0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_ndcg_score_k_is_2() {
        let ndcg = NDCGScorer::new(2);
        let dcg = 7.0 / 2.0_f64.log2() + 3.0 / 3.0_f64.log2();
        let max_dcg = 15.0 / 2.0_f64.log2() + 7.0 / 3.0_f64.log2();
        assert_eq!(ndcg.score(&vec![3.0, 2.0, 4.0]), dcg / max_dcg);
    }

    #[test]
    fn test_ndcg_delta() {
        let ndcg = NDCGScorer::new(10);

        let max_dcg = 15.0 / 2.0_f64.log2() + 7.0 / 3.0_f64.log2() +
            3.0 / 4.0_f64.log2();

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

        let result = ndcg.delta(&vec![3.0, 2.0, 4.0]);
        let expected = vec![
            vec![
                0.0,
                (origin - score_swap_0_1) / max_dcg,
                (origin - score_swap_0_2) / max_dcg,
            ],
            vec![
                (origin - score_swap_0_1) / max_dcg,
                0.0,
                (origin - score_swap_1_2) / max_dcg,
            ],
            vec![
                (origin - score_swap_0_2) / max_dcg,
                (origin - score_swap_1_2) / max_dcg,
                0.0,
            ],
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
