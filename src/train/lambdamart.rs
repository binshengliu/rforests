use std::fs::File;
use train::regression_tree::RegressionTree;
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

        let mut training = TrainingSet::from(&self.dataset);
        for _i in 0..ntrees {
            training.update_pseudo_response();

            let min_leaf_count = 1;
            let mut tree = RegressionTree::new(min_leaf_count);
            tree.fit(&training);
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {}
