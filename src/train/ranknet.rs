use std::iter::FromIterator;
use std::fs::File;
use std::cmp::Ordering;
use format::svmlight::*;
use util::Result;

pub struct RankList {
    list: Vec<Instance>,
}

impl RankList {
    pub fn new() -> RankList {
        RankList { list: Vec::new() }
    }

    pub fn sort_by_target(&mut self) {
        self.list.sort_by(|instance1, instance2| {
            let (target1, target2) = (instance1.target(), instance2.target());
            target1.partial_cmp(&target2).unwrap_or(Ordering::Less)
        });
    }
}

impl FromIterator<Instance> for RankList {
    fn from_iter<I: IntoIterator<Item = Instance>>(iter: I) -> RankList {
        RankList { list: Vec::from_iter(iter) }
    }
}

/// A layer in neural network
pub struct Neuron {
    output: f64,

    /// Outputs for each propagation
    outputs: Vec<f64>,
}

pub struct Synapse {}

pub struct Layer {
    
}

pub struct RankNet {
    
}

impl RankNet {
    pub fn new() -> RankNet {
        RankNet {}
    }

    pub fn read_file(&self, filename: &str) -> Result<Vec<RankList>> {
        let filename = "";
        let file = File::open(&filename)?;
        let mut prev_qid = None;

        let lists = Vec::new();

        let mut data_points = Vec::new();
        for instance in SvmLightFile::instances(file) {
            let instance = instance?;

            if Some(instance.qid()) == prev_qid {
                data_points.push(instance);
            } else {
                lists.push(data_points.into_iter().collect::<RankList>());
            }
        }
        Ok(lists)
    }

    pub fn train(&self) -> Result<()> {
        let rank_lists = self.read_file("");
        Ok(())
    }

    pub fn init(&self) {
        debug!("Init ranknet");
        // self.layers.push();
    }

    pub fn learn(&self) {
        
    }
}
