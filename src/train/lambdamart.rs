use util::Result;
use std::fs::File;
use format::svmlight::{DataSet, Instance, Query, SvmLightFile};
use std::collections::HashMap;

struct LambdaMART {}

impl LambdaMART {
    pub fn new() -> LambdaMART {
        LambdaMART {}
    }

    pub fn init() -> Result<()> {
        Ok(())
    }

    pub fn learn() -> Result<()> {
        let path = "/home/lbs/code/rforests/data/train-lite.txt";
        let f = File::open(path)?;
        let dataset = DataSet::load(f)?;
        let queries = dataset.group_by_queries();
        let n_instance = dataset.len();

        let mut scores: Vec<f64> = Vec::with_capacity(n_instance);
        scores.resize(n_instance, 0.0);

        let mut resps: Vec<f64> = Vec::with_capacity(n_instance);
        resps.resize(n_instance, 0.0);

        let mut weights: Vec<f64> = Vec::with_capacity(n_instance);
        weights.resize(n_instance, 0.0);

        let mut sorted: Vec<Vec<usize>> = Vec::new();

        Ok(())
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
