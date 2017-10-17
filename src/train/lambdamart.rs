use std::fs::File;
use train::regression_tree::*;
use train::dataset::*;
use util::*;

pub struct LambdaMART {
    dataset: DataSet,
}

impl LambdaMART {
    pub fn new() -> LambdaMART {
        let path = "/home/lbs/code/rforests/data/train-lite.txt";
        let f = File::open(path).unwrap();
        let mut dataset = DataSet::load(f).unwrap();
        dataset.generate_thresholds(256);
        LambdaMART { dataset: dataset }
    }

    pub fn init(&self) -> Result<()> {
        Ok(())
    }

    pub fn learn(&self) -> Result<()> {
        let ntrees = 2000;

        let learning_rate = 0.1;
        let min_leaf_count = 1;
        let mut ensemble = Ensemble::new();
        let mut training = TrainingSet::from(&self.dataset);
        for _i in 0..ntrees {
            training.update_pseudo_response();

            let mut tree = RegressionTree::new(learning_rate, min_leaf_count);
            tree.fit(&training);
            ensemble.push(tree);
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {}
