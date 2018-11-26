use std::sync::Arc;
use crate::netstruct::NetStruct;
use crate::netstruct::NetStructTrait;
use crate::example::Test;
use crate::train::Trainer;
use crate::train::CoefCalculator;
use crate::network::Network;
use rand::prelude::ThreadRng;
use crate::network::LayerConfig;
pub struct BackPropTrainer {
    tests: Arc<Vec<Test>>,
    back_prop_calc: GradientDescentCalculator,
    lower_bound: f64,
    step: f64,
    mini_batches: usize,
    max_iters: usize,
    verbose: bool,
}

impl Trainer<NetStruct, GradientDescentCalculator> for BackPropTrainer {
    fn new(tests: Arc<Vec<Test>>, structure: Vec<LayerConfig>, my_rand: &mut ThreadRng) -> Self {
        BackPropTrainer {
            tests: tests,
            back_prop_calc: Network::new(structure, my_rand),
            lower_bound: 0.2,
            step: 0.1,
            mini_batches: 1,
            max_iters: 1000_000,
            verbose: false,
        }
    }
    fn get_tests<'a>(&'a self) -> &'a Arc<Vec<Test>> {
        &self.tests
    }
    fn number_of_batches(&mut self, mini_batch_size: usize) -> &mut Self {
        self.mini_batches = mini_batch_size;
        self
    }
    fn get_cloned_calculator(&self) -> GradientDescentCalculator {
        self.back_prop_calc.clone()
    }
    fn get_calculator(&self) -> &GradientDescentCalculator {
        &self.back_prop_calc
    }
    fn start(&mut self) -> &mut Self {
        let mut i = 0;
        #[allow(dead_code)]
        let mut score = self.get_net().evaluate(&self.tests);
        while score > self.lower_bound && i < self.max_iters {
            i += 1;
            for batch in self.get_mini_batches() {
                let l = batch.len();
                if l > 0 {
                    let glob_grad = self.train(&Arc::new(batch));
                    let step = self.step;
                    self.get_mut_net().add_gradient(
                        &glob_grad,
                        step / (l as f64),
                    );
                }
            }
            if self.get_net().evaluate(&self.tests) > score {
                self.step *= 0.95;
            }
            score = self.get_net().evaluate(&self.tests);
            if i % 2 == 0 && self.verbose {
                /*println!("glob grad max : {:?}", self.get_net().get_weights().to_vector().iter().fold(0.0 as f64, |acc, &x| match acc > x {
                    true => acc,
                    false => x
                }));*/
                println!("epoch : {}, eval after : {}", i, score);
            }
        }
        self
    }
    fn get_net(&self) -> Network {
        self.back_prop_calc.clone()
    }
    fn get_mut_net(&mut self) -> &mut Network {
        &mut self.back_prop_calc
    }
    fn get_number_of_batches(&self) -> usize {
        self.mini_batches
    }
    fn get_empty_val(&self) -> NetStruct {
        self.back_prop_calc.get_empty_grad()
    }
    fn lower_bound(&mut self, bound: f64) -> &mut Self {
        self.lower_bound = bound;
        self
    }
    fn set_net(&mut self, net: Network) -> &mut Self {
        self.back_prop_calc = net;
        self
    }
    fn verbose(&mut self, verb: bool) -> &mut Self {
        self.verbose = verb;
        self
    }

    fn is_verbose(&mut self) -> bool {
        self.verbose
    }
}

impl BackPropTrainer {
    pub fn step(&mut self, step: f64) -> &mut Self {
        self.step = step;
        self
    }
    pub fn max_iterations(&mut self, max: usize) -> &mut Self {
        self.max_iters = max;
        self
    }
}



type GradientDescentCalculator = Network;

impl CoefCalculator<NetStruct> for GradientDescentCalculator {
    fn get_net(&self) -> Network {
        self.clone()
    }
    fn calc_result(&self, tests: &[Test]) -> NetStruct {
        self.global_gradient(tests, true)
    }
    fn add_result(&self, first: &NetStruct, second: &NetStruct) -> NetStruct {
        first.add(second)
    }
    fn get_empty_val(&self) -> NetStruct {
        self.get_empty_grad()
    }
}
