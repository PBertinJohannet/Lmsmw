use backpropagation::BackPropTrainer;
use lvbm::LevembergMarquardtTrainer;
use train::Trainer;
use std::sync::Arc;
use network::LayerConfig;
use network::EvalFunc;
use network::Network;
use example::Test;
use example::GreaterThan;
use example::Sine;
use example::Square;
use example::Hole;
use example::Triangle;
use example::Round;
use example::TrainingData;

use rand::XorShiftRng;
use rand::Rng;


pub struct LearningParameters {
    structure: Vec<LayerConfig>,
    backprop_on_fail: bool,
    backprop_conf: BackPropParameters,
    lvbm_params: LevembergMarquardtParameters,
    examples: ExamplesConfig,
    lower_bound: f64,
    iters: usize,
}

struct AutoGeneratingExampleParameters {
    empty: (),
}

struct BackPropParameters {
    iters: usize,
    step: f64,
    nb_batches: usize,
}

impl BackPropParameters {
    pub fn new() -> Self {
        BackPropParameters {
            iters: 10,
            step: 0.1,
            nb_batches: 5,
        }
    }
}

pub enum ExamplesConfig {
    //    auto_gen(AutoGeneratingExampleParameters),
    Ready(Vec<Test>),
}
impl ExamplesConfig {
    fn get_tests(&self) -> Vec<Test> {
        match self {
            &ExamplesConfig::Ready(ref tests) => tests.clone(),
        }
    }
    fn len_tests(&self) -> usize {
        match self {
            &ExamplesConfig::Ready(ref tests) => tests.len(),
        }
    }
}



struct LevembergMarquardtParameters {
    pub lambda_start: f64,
    pub lambda_plus: f64,
    pub lambda_minus: f64,
    pub lambda_max: f64,
    pub batch_size: usize,
}

impl LevembergMarquardtParameters {
    pub fn new() -> Self {
        LevembergMarquardtParameters {
            lambda_max: 1_000_000_000.0,
            lambda_minus: 4.0,
            lambda_plus: 5.0,
            lambda_start: 1000.0,
            batch_size: 50,
        }
    }
}

impl LearningParameters {
    pub fn new(examples: ExamplesConfig, layers: Vec<LayerConfig>) -> Self {
        LearningParameters {
            iters: 100,
            lower_bound: 0.05,
            structure: layers,
            backprop_on_fail: true,
            backprop_conf: BackPropParameters::new(),
            examples: examples,
            lvbm_params: LevembergMarquardtParameters::new(),
        }
    }
    pub fn max_iter(&mut self, iter: usize) -> &mut Self {
        self.iters = iter;
        self
    }
    pub fn aim_score(&mut self, lower_bound: f64) -> &mut Self {
        self.lower_bound = lower_bound;
        self
    }
    pub fn lambda_start(&mut self, lam_start: f64) -> &mut Self {
        self.lvbm_params.lambda_start = lam_start;
        self
    }
    pub fn lambda_max(&mut self, lam_max: f64) -> &mut Self {
        self.lvbm_params.lambda_max = lam_max;
        self
    }
    pub fn lambda_plus(&mut self, lam_plus: f64) -> &mut Self {
        self.lvbm_params.lambda_plus = lam_plus;
        self
    }
    pub fn lambda_minus(&mut self, lam_minus: f64) -> &mut Self {
        self.lvbm_params.lambda_minus = lam_minus;
        self
    }
    pub fn batch_size(&mut self, size: usize) -> &mut Self {
        self.lvbm_params.batch_size = size;
        self
    }
    pub fn backprop_on_fail(&mut self, choice: bool) -> &mut Self {
        self.backprop_on_fail = choice;
        self
    }
    pub fn backprop_step(&mut self, step: f64) -> &mut Self {
        self.backprop_conf.step = step;
        self
    }
    pub fn backprop_iters(&mut self, iters: usize) -> &mut Self {
        self.backprop_conf.iters = iters;
        self
    }
    pub fn backprop_nb_batches(&mut self, nb_batches: usize) -> &mut Self {
        self.backprop_conf.nb_batches = nb_batches;
        self
    }
    pub fn start(&self) -> Network {
        let mut my_rand = XorShiftRng::new_unseeded();
        let tests = self.examples.get_tests();
        let mut net = Network::new(self.structure.clone(), &mut my_rand);
        let mut i = 0;
        while net.evaluate(&tests) > self.lower_bound && self.backprop_on_fail && self.iters > i {
            i += 1;
            let mut new_net = BackPropTrainer::new(
                Arc::new(self.examples.get_tests()),
                self.structure.clone(),
                &mut my_rand,
            ).number_of_batches(self.backprop_conf.nb_batches)
                .step(self.backprop_conf.step)
                .max_iterations(self.backprop_conf.iters)
                .set_net(net.clone())
                .lower_bound(self.lower_bound)
                .start()
                .get_net();
            net = LevembergMarquardtTrainer::new(
                Arc::new(self.examples.get_tests()),
                self.structure.clone(),
                &mut my_rand,
            ).number_of_batches(self.examples.len_tests() / self.lvbm_params.batch_size)
                .lambda(self.lvbm_params.lambda_start)
                .set_net(new_net.clone())
                .lower_bound(self.lower_bound)
                .start()
                .get_net();
            println!("lower bound : {}", self.lower_bound);
        }
        net.clone()
    }
}
