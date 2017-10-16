use util::Result;
use std::fs::File;
use metric::NDCGScorer;
use super::regression_tree::RegressionTree;
use train::dataset::DataSet;
use util::*;

pub struct LambdaMART {
    dataset: DataSet,
}

impl LambdaMART {
    pub fn new() -> LambdaMART {
        let path = "/home/lbs/code/rforests/data/train-lite.txt";
        let f = File::open(path).unwrap();
        let dataset = DataSet::load(f).unwrap();
        LambdaMART { dataset: dataset }
    }

    pub fn init(&self) -> Result<()> {
        let queries = self.dataset.group_by_queries();
        let n_instance = self.dataset.len();

        let mut scores: Vec<Value> = Vec::with_capacity(n_instance);
        scores.resize(n_instance, 0.0);

        let mut resps: Vec<Value> = Vec::with_capacity(n_instance);
        resps.resize(n_instance, 0.0);

        let mut weights: Vec<Value> = Vec::with_capacity(n_instance);
        weights.resize(n_instance, 0.0);

        let mut sorted: Vec<Vec<usize>> = Vec::new();

        Ok(())
    }

    pub fn learn(&self) -> Result<()> {
        let ntrees = 2000;
        let ninstances = self.dataset.len();
        let mut model_scores: Vec<Value> = Vec::with_capacity(ninstances);
        model_scores.resize(ninstances, 0.0);

        let mut pseudo_response: Vec<Value> = Vec::with_capacity(ninstances);
        pseudo_response.resize(ninstances, 0.0);

        let mut weights: Vec<Value> = Vec::with_capacity(ninstances);
        weights.resize(ninstances, 0.0);

        for i in 0..ntrees {
            self.update_pseudo_response(
                &model_scores,
                &mut pseudo_response,
                &mut weights,
            );

            let tree = RegressionTree::fit(&self.dataset);
        }
        Ok(())
    }

    pub fn update_pseudo_response(
        &self,
        model_scores: &Vec<Value>,
        pseudo_response: &mut Vec<Value>,
        weights: &mut Vec<Value>,
    ) {
        let ndcg = NDCGScorer::new(10);
        for query in self.dataset.group_by_queries().iter() {
            let result = query.get_lambda(model_scores, &ndcg);
            for &(index, lambda, weight) in result.iter() {
                pseudo_response[index] += lambda;
                weights[index] += weight;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_generate_queries() {
        let s = "0 qid:3864 1:1.0 2:0.0 3:0.0 4:0.0 5:0.0\n2 qid:3864 1:1.0 2:0.007042 3:0.0 4:0.0 5:0.221591\n0 qid:3865 1:0.289474 2:0.014085 3:0.4 4:0.0 5:0.085227";
        let dataset = DataSet::load(::std::io::Cursor::new(s)).unwrap();
        let mut queries = dataset.group_by_queries();
        queries.sort_by_key(|q| q.qid());

        assert_eq!(
            queries[1].to_string(),
            "0 qid:3865 1:0.289474 2:0.014085 3:0.4 4:0 5:0.085227"
        );
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0].qid(), 3864);
        assert_eq!(queries[1].qid(), 3865);
    }
}
