use std::sync::Arc;
use netstruct::NetStruct;
use netstruct::NetStructTrait;
use example::Test;
use train::Trainer;
use train::CoefCalculator;
use network::Network;
use rand::XorShiftRng;
use network::LayerConfig;
pub struct BackPropTrainer {
    tests: Arc<Vec<Test>>,
    back_prop_calc: BackPropagationCalculator,
    lower_bound: f64,
    step: f64,
    mini_batches: usize,
}

impl Trainer<NetStruct, BackPropagationCalculator> for BackPropTrainer {
    fn new(tests: Arc<Vec<Test>>, structure: Vec<LayerConfig>, my_rand: &mut XorShiftRng) -> Self {
        BackPropTrainer {
            tests: tests,
            back_prop_calc: Network::new(structure, my_rand),
            lower_bound: 0.2,
            step: 0.1,
            mini_batches: 1,
        }
    }
    fn get_tests<'a>(&'a self) -> &'a Arc<Vec<Test>> {
        &self.tests
    }
    fn number_of_batches(&mut self, mini_batch_size: usize) -> &mut Self {
        self.mini_batches = mini_batch_size;
        self
    }
    fn get_cloned_calculator(&self) -> BackPropagationCalculator {
        self.back_prop_calc.clone()
    }
    fn get_calculator(&self) -> &BackPropagationCalculator {
        &self.back_prop_calc
    }
    fn start(&mut self) -> &mut Self {
        println!("start net : {:?}", self.get_net());
        let mut i = 0;
        #[allow(dead_code)]
        let mut score = self.get_net().evaluate(&self.tests);
        while score > self.lower_bound  {
            i += 1;
            for batch in self.get_mini_batches() {
                let l = batch.len();
                let glob_grad = self.train(&Arc::new(batch));
                let step = self.step;
                self.get_mut_net().add_gradient(
                    &glob_grad,
                    step / (l as f64),
                );
            }
            score = self.get_net().evaluate(&self.tests);
            if i % 50 == 0 {
                println!("glob grad max : {:?}", self.get_net().get_weights().to_vector().iter().fold(0.0 as f64, |acc, &x| match acc > x {
                    true => acc,
                    false => x
                }));
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
}

impl BackPropTrainer {
    pub fn step(&mut self, step: f64) -> &mut Self {
        self.step = step;
        self
    }
}



type BackPropagationCalculator = Network;

impl CoefCalculator<NetStruct> for BackPropagationCalculator {
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



