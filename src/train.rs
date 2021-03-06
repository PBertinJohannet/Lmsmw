use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use crate::network::Network;
use rand::prelude::ThreadRng;
use crate::network::LayerConfig;
use std::cmp::min;
use crate::example::Test;

pub trait Trainer<T: Send + 'static, U: CoefCalculator<T> + Sync + Send + 'static>
     {
    fn verbose(&mut self, verbose : bool) -> &mut Self;
    fn is_verbose(&mut self) -> bool;
    fn start(&mut self) -> &mut Self;
    fn get_net(&self) -> Network;
    fn get_empty_val(&self) -> T;
    fn new(tests: Arc<Vec<Test>>, structure: Vec<LayerConfig>, my_rand: &mut ThreadRng) -> Self;
    fn get_number_of_batches(&self) -> usize;
    fn get_tests<'a>(&'a self) -> &'a Arc<Vec<Test>>;
    fn get_cloned_calculator(&self) -> U;
    fn get_calculator<'a>(&'a self) -> &'a U;
    fn get_mut_net(&mut self) -> &mut Network;
    fn number_of_batches(&mut self, mini_batch_size: usize) -> &mut Self;
    fn set_net(&mut self, net: Network) -> &mut Self;
    fn lower_bound(&mut self, bound: f64) -> &mut Self;
    fn calc_result(&self, tests: &[Test]) -> T {
        self.get_calculator().calc_result(tests)
    }
    fn show_me(&self) {
        let net = self.get_net();
        for i in self.get_tests().iter() {
            println!(
                "error : {}",
                (&i.outputs - net.feed_forward(&i.inputs)).sum()
            );
            println!(
                "squared : {}",
                (&i.outputs - net.feed_forward(&i.inputs))
                    .apply(&|x| x * x * 0.5)
                    .sum()
            )
        }
    }
    fn add_result(&self, first: &T, second: &T) -> T {
        self.get_calculator().add_result(first, second)
    }
    fn get_mini_batches(&self) -> Vec<Vec<Test>> {
        let num_tests_per_batches = self.get_tests().len() / self.get_number_of_batches();
        let num_tougher_batches = self.get_tests().len() % self.get_number_of_batches();
        let mut offset = 0;
        (0..self.get_number_of_batches())
            .map(|id| {
                let chunksize = match id < num_tougher_batches {
                    true => num_tests_per_batches + 1,
                    false => num_tests_per_batches,
                };
                let to_ret = self.get_tests()[offset..offset + chunksize].to_vec();
                offset += chunksize;
                to_ret
            })
            .collect::<Vec<Vec<Test>>>()
    }
    fn train(&self, tests: &Arc<Vec<Test>>) -> T {
        let nb_threads: usize = min(8, tests.len());
        let (tx, rx) = mpsc::channel();
        if nb_threads > 0 {
            let num_tasks_per_thread = tests.len() / nb_threads;
            let num_tougher_threads = tests.len() % nb_threads;
            let mut offset = 0;
            for id in 0..nb_threads {
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
        rx.iter().fold(self.get_empty_val(), |base, grad| {
            self.add_result(&base, &grad)
        })
    }
}


pub trait CoefCalculator<T: Send + 'static> {
    fn calc_result(&self, tests: &[Test]) -> T;
    fn add_result(&self, first : &T, second : &T) -> T;
    fn get_net(&self) -> Network;
    fn get_empty_val(&self) -> T;
}
