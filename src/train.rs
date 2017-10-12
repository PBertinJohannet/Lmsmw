use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use network::NetStruct;
use network::Network;
use network::NetStructTrait;
use rulinalg::vector::Vector;
use rand::XorShiftRng;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Test {
    pub inputs: Vector<f64>,
    pub outputs: Vector<f64>,
}
impl Test {
    pub fn new(inputs: Vector<f64>, outputs: Vector<f64>) -> Self {
        Test {
            inputs: inputs,
            outputs: outputs,
        }
    }
}


pub trait CoefCalculator<T : Send + 'static> {
    fn calc_result(&self, tests: &[Test]) -> T;
    fn add_result(&self, &T, &T) -> T;
    fn get_net(&self)->Network;
    fn get_empty_val(&self)->T;
}

type BackPropagationCalculator = Network;

impl CoefCalculator<NetStruct> for BackPropagationCalculator {
    fn get_net(&self)->Network{
        self.clone()
    }
    fn calc_result(&self, tests: &[Test]) -> NetStruct{
        self.global_gradient(tests)
    }
    fn add_result(&self, first : &NetStruct, second : &NetStruct) -> NetStruct{
        first.add(second)
    }
    fn get_empty_val(&self)->NetStruct{
        self.get_empty_grad()
    }
}

pub trait Trainer<T : Send+ 'static, U: CoefCalculator<T> + Sync + Send+ 'static>{
    fn start(&mut self) -> &mut Self;
    fn calc_result(&self, tests: &[Test]) -> T;
    fn add_result(&self, &T, &T) -> T;
    fn get_net(&self)->Network;
    fn get_empty_val(&self)->T;
    fn new(tests: Arc<Vec<Test>>, structure: Vec<usize>, my_rand: &mut XorShiftRng) -> Self;
    fn get_number_of_batches(&self) -> usize;
    fn get_tests<'a>(&'a self) -> &'a Arc<Vec<Test>>;
    fn get_cloned_calculator(&self) -> U;
    fn get_mut_net(&mut self)-> &mut Network;
    fn get_mini_batches(&self) -> Vec<Vec<Test>>{
        let num_tests_per_batches = self.get_tests().len() / self.get_number_of_batches();
        let num_tougher_batches = self.get_tests().len() % self.get_number_of_batches();
        let mut offset = 0;
        (0..self.get_number_of_batches()).map(|id| {
            let chunksize = match id < num_tougher_batches {
                true => num_tests_per_batches + 1,
                false => num_tests_per_batches,
            };
            let to_ret = self.get_tests()[offset..offset + chunksize].to_vec();
            offset += chunksize;
            to_ret
        }).collect::<Vec<Vec<Test>>>()
    }
    fn train(&self, tests: &Arc<Vec<Test>>) -> T {
        const NTHREADS: usize = 8;
        let (tx, rx) = mpsc::channel();
        {
            let num_tasks_per_thread = tests.len() / NTHREADS;
            let num_tougher_threads = tests.len() % NTHREADS;
            let mut offset = 0;
            for id in 0..NTHREADS {
                let chunksize = if id < num_tougher_threads {
                    num_tasks_per_thread + 1
                } else {
                    num_tasks_per_thread
                };
                let my_tests = tests.clone();
                let my_tx = tx.clone();
                let my_calc = self.get_cloned_calculator();
                thread::spawn(move || {
                    let end = offset + chunksize;
                    let grad = my_calc.calc_result(&my_tests[offset..end]);
                    my_tx.send(grad).unwrap();
                });
                offset += chunksize;
            }
        }
        drop(tx);
        rx.iter().fold(
            self.get_empty_val(),
            |base, grad| self.add_result(&base, &grad),
        )
    }
}


pub struct BackPropTrainer {
    tests: Arc<Vec<Test>>,
    back_prop_calc: BackPropagationCalculator,
    lower_bound: f64,
    step: f64,
    mini_batches: usize,
}

impl Trainer<NetStruct, BackPropagationCalculator> for BackPropTrainer {
    fn new(tests: Arc<Vec<Test>>, structure: Vec<usize>, my_rand: &mut XorShiftRng) -> Self {
        BackPropTrainer {
            tests: tests,
            back_prop_calc: Network::new(structure, my_rand),
            lower_bound: 0.2,
            step: 0.1,
            mini_batches: 1,
        }
    }
    fn get_tests<'a>(&'a self) -> &'a Arc<Vec<Test>>{
        &self.tests
    }
    fn get_cloned_calculator(&self)->BackPropagationCalculator{
        self.back_prop_calc.clone()
    }
    fn start(&mut self) -> &mut Self {
        println!("start net : {:?}", self.get_net());
        let mut i = 0;
        #[allow(dead_code)]
        let mut score = self.get_net().evaluate(&self.tests);
        while self.get_net().evaluate(&self.tests) > self.lower_bound {
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
            println!(
                "epoch : {}, eval after : {}",
                i,
                score
            );
        }
        self
    }
    fn get_net(&self)->Network{
        self.back_prop_calc.clone()
    }
    fn get_mut_net(&mut self)->&mut Network{
        &mut self.back_prop_calc
    }
    fn calc_result(&self, tests: &[Test]) -> NetStruct{
        self.back_prop_calc.global_gradient(tests)
    }
    fn add_result(&self, first : &NetStruct, second : &NetStruct) -> NetStruct{
        first.add(second)
    }
    fn get_number_of_batches(&self) -> usize{
        self.mini_batches
    }
    fn get_empty_val(&self)-> NetStruct {
        self.back_prop_calc.get_empty_grad()
    }
}

impl BackPropTrainer {
    pub fn lower_bound(&mut self, bound: f64) -> &mut Self {
        self.lower_bound = bound;
        self
    }
    pub fn step(&mut self, step: f64) -> &mut Self {
        self.step = step;
        self
    }
    pub fn number_of_batches(&mut self, mini_batch_size: usize) -> &mut Self {
        self.mini_batches = mini_batch_size;
        self
    }
}

