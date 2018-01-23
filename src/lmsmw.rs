//! # LMSMW module
//! This module contains the trainer.
//! The learning is done via levemberg marquardt algorithm
//! TODO : document examples
//!

use sgd::BackPropTrainer;
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

/// Learner
/// The struct used to train the network
///
#[derive(Debug)]
pub struct Learner {
    /// The structure of the network
    structure: Vec<LayerConfig>,
    /// Option to try some SGD when lvbm fails.
    gradient_descent_on_fail: bool,
    /// Configuration for sgd algorithm.
    gradient_descent_conf: BackPropParameters,
    /// Configuration for levemberg marquardt parameters.
    lvbm_params: LevembergMarquardtParameters,
    /// Configuration to generate examples.
    examples: ExamplesConfig,
    /// the aim score to reach
    lower_bound: f64,
    /// maximum number of iterations (fail lvbm sgd).
    iters: usize,
    /// verbose if the scores need to be printed during progression.
    verbose: bool,
    /// The network to train
    net : Option<Network>,

}
struct AutoGeneratingExampleParameters {
    empty: (),
}
/// Configuration for sgd.
#[derive(Debug)]
struct BackPropParameters {
    /// maximum number of iterations if the aim score is not reached.
    iters: usize,
    /// the learning rate of the algorithm.
    step: f64,
    /// number of mini batches.
    nb_batches: usize,
}
impl BackPropParameters {
    /// creates the default configuration for sgd.
    pub fn new() -> Self {
        BackPropParameters {
            iters: 10,
            step: 0.1,
            nb_batches: 5,
        }
    }
}
/// Enum to generate examples
/// currently only already generated examples
/// could be a trait in the future.
#[derive(Debug)]
pub enum ExamplesConfig {
    //    auto_gen(AutoGeneratingExampleParameters),
    ///already generated examples
    Ready(Vec<Test>),
}
impl ExamplesConfig {
    /// get training data
    fn get_tests(&self) -> Vec<Test> {
        match self {
            &ExamplesConfig::Ready(ref tests) => tests.clone(),
        }
    }
    /// get the number of different examples.
    fn len_tests(&self) -> usize {
        match self {
            &ExamplesConfig::Ready(ref tests) => tests.len(),
        }
    }
}

/// parameters for levemberg marquardt algorithm
#[derive(Debug)]
struct LevembergMarquardtParameters {
    /// lambda parameter at the start of the algorithm
    pub lambda_start: f64,
    /// augmentation of lambda if the score didnt improved.
    pub lambda_plus: f64,
    /// reduction of lambda if the score improved.
    pub lambda_minus: f64,
    /// maximum lambda value before quitting the alg.
    pub lambda_max: f64,
    /// Size of mini batches.
    pub batch_size: usize,
    /// the maximum number of iterations before exiting the algorithm
    max_iters : usize,
}

impl LevembergMarquardtParameters {
    /// Creates default configuration for levemberg marquardt algorithm.
    pub fn new() -> Self {
        LevembergMarquardtParameters {
            lambda_max: 1_000_000_000.0,
            lambda_minus: 4.0,
            lambda_plus: 5.0,
            lambda_start: 1000.0,
            batch_size: 50,
            max_iters : 10_000_000,
        }
    }
}

impl Learner {


    /// Creates a default learner from examples and network structure.
    pub fn new(examples: ExamplesConfig, layers: Vec<LayerConfig>) -> Self {
        Learner {
            iters: 100,
            lower_bound: 0.05,
            structure: layers,
            gradient_descent_on_fail: true,
            gradient_descent_conf: BackPropParameters::new(),
            examples: examples,
            lvbm_params: LevembergMarquardtParameters::new(),
            verbose: false,
            net : None,
        }
    }
    /// Sets the base network to train
    /// if no network is set, will train a randomly generated network.
    pub fn set_net(&mut self, net : Network) -> &mut Self {
        self.net = Some(net);
        self
    }
    /// Sets the maximum number of iterations of the algorithm.
    /// An iteration is composed of :
    ///     - try with levemberg marquardt until the algorithm diverges
    ///     - try with SGD.
    pub fn max_iter(&mut self, iter: usize) -> &mut Self {
        self.iters = iter;
        self
    }
    /// Sets the aim score. at this score the algorithm will stop.
    pub fn aim_score(&mut self, lower_bound: f64) -> &mut Self {
        self.lower_bound = lower_bound;
        self
    }
    /// set the starting value of lambda for lvbm
    pub fn lambda_start(&mut self, lam_start: f64) -> &mut Self {
        self.lvbm_params.lambda_start = lam_start;
        self
    }
    /// set the maximum value of lambda for lvbm
    /// after this valueis exeded, the lvbm alg will stop.
    pub fn lambda_max(&mut self, lam_max: f64) -> &mut Self {
        self.lvbm_params.lambda_max = lam_max;
        self
    }
    /// set the increasing factor of lambda for lvbm
    pub fn lambda_plus(&mut self, lam_plus: f64) -> &mut Self {
        self.lvbm_params.lambda_plus = lam_plus;
        self
    }
    /// set the decreasing factor of lambda for lvbm
    pub fn lambda_minus(&mut self, lam_minus: f64) -> &mut Self {
        self.lvbm_params.lambda_minus = lam_minus;
        self
    }
    /// sets the batch size for lvbm.
    pub fn lvbm_nb_batches(&mut self, size: usize) -> &mut Self {
        self.lvbm_params.batch_size = size;
        self
    }
    /// sets the maximum number of iterations for lvbm
    pub fn lvbm_max_iters(&mut self, iters: usize) -> &mut Self {
        self.lvbm_params.max_iters = iters;
        self
    }
    /// sets the use of sgd when lvbm fail to true/false
    pub fn gradient_descent_on_fail(&mut self, choice: bool) -> &mut Self {
        self.gradient_descent_on_fail = choice;
        self
    }
    /// sets the learning rate for sgd algorithm
    pub fn gradient_descent_step(&mut self, step: f64) -> &mut Self {
        self.gradient_descent_conf.step = step;
        self
    }
    /// set the number of iterations for sgd try
    pub fn gradient_descent_iters(&mut self, iters: usize) -> &mut Self {
        self.gradient_descent_conf.iters = iters;
        self
    }
    /// sets the batch size for sgd.
    pub fn gradient_descent_nb_batches(&mut self, nb_batches: usize) -> &mut Self {
        self.gradient_descent_conf.nb_batches = nb_batches;
        self
    }
    /// sets the batch size for sgd.
    pub fn verbose(&mut self) -> &mut Self {
        self.verbose = true;
        self
    }
    /// start the algorithm
    /// will return the Network if the aim score is reached or the maximum number of iterations is exeeded
    pub fn start(&mut self) -> Network {
        let mut my_rand = XorShiftRng::new_unseeded();
        let mut net = (self.net.as_mut().unwrap_or(&mut Network::new(self.structure.clone(), &mut my_rand)).clone());
        //let mut net = my_net.clone();
        let batch_size = usize::max(self.examples.len_tests() / self.lvbm_params.batch_size, 1);
        let tests = &self.examples.get_tests().clone();
        let mut i = 0;
        while net.evaluate(&tests) > self.lower_bound && self.gradient_descent_on_fail && self.iters > i {
            i += 1;
            let new_net = BackPropTrainer::new(
                Arc::new(self.examples.get_tests()),
                self.structure.clone(),
                &mut my_rand,
            ).number_of_batches(self.gradient_descent_conf.nb_batches)
                .verbose(self.verbose)
                .step(self.gradient_descent_conf.step)
                .max_iterations(self.gradient_descent_conf.iters)
                .set_net(net.clone())
                .lower_bound(self.lower_bound)
                .start()
                .get_net();
            net = LevembergMarquardtTrainer::new(
                Arc::new(self.examples.get_tests()),
                self.structure.clone(),
                &mut my_rand,
            ).number_of_batches(batch_size)
                .verbose(self.verbose)
                .lambda(self.lvbm_params.lambda_start)
                .set_net(new_net.clone())
                .max_iterations(self.lvbm_params.max_iters)
                .lower_bound(self.lower_bound)
                .start()
                .get_net();
        }
        net.clone()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn usage_test() {
        // first create some tests
        let tests_array = vec![
            Test::new(vector![2.0, 3.0, 0.1], vector![0.1, 0.5]),
            Test::new(vector![1.0, 1.0, 0.7], vector![0.5, 1.0]),
        ];


        // Create the network structure


        let layers = layers![3, 3, 2]; //  3 input layers, 3 hidden and 2 output layers.



        // Create and launch the trainer

        let net = Learner::new(ExamplesConfig::Ready(tests_array.clone()), layers)
            .gradient_descent_iters(20) // every time levemberg marquardt fails, run 20 iterations of GD
            .gradient_descent_step(0.1) // with learning rate 0.1
            .lvbm_nb_batches(2)    // run lvbm on 2 unit batches
            .aim_score(0.005)  // try to go until 0.005 score
            .max_iter(50)       // but if it still diverges everytime stop after 50 algorithm swaps
            .start(); // start learning
        assert!(net.evaluate(&tests_array) < 0.005)
    }
}