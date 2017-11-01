use train::dataset::DataSet;
use train::Evaluate;
use metric::Measure;
use util::Value;
use std::cmp::Ordering;

pub struct ValidateSet<'d> {
    dataset: &'d DataSet,
    scores: Vec<f64>,
}

impl<'a> From<&'a DataSet> for ValidateSet<'a> {
    fn from(dataset: &'a DataSet) -> ValidateSet<'a> {
        let len = dataset.len();
        let scores = vec![0.0; len];
        ValidateSet {
            dataset: dataset,
            scores: scores,
        }
    }
}

impl<'a> ValidateSet<'a> {
    pub fn measure(&self, metric: &Box<Measure>) -> f64 {
        let mut score = 0.0;
        let mut count: usize = 0;
        for (_, query) in self.dataset.query_iter() {

            let mut model_scores: Vec<(Value, Value)> = query
                .iter()
                .map(|&id| (self.scores[id], self.dataset[id].label()))
                .collect();

            model_scores.sort_by(|&(score1, _), &(score2, _)| {
                score2.partial_cmp(&score1).unwrap_or(Ordering::Equal)
            });

            let labels: Vec<f64> =
                model_scores.iter().map(|&(_, label)| label).collect();
            let query_score = metric.measure(&labels);

            count += 1;
            score += query_score;
        }

        let result = score / count as f64;
        result
    }

    pub fn update<E: Evaluate>(&mut self, evaluator: &E) {
        for (instance, score) in
            self.dataset.iter().zip(self.scores.iter_mut())
        {
            *score += evaluator.evaluate(instance);
        }
    }
}
