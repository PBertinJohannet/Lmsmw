#[macro_use]
extern crate rulinalg;
extern crate rand;
extern crate num_traits;

use rand::XorShiftRng;
use rand::Rng;

#[macro_use]
mod network;
mod train;
mod netstruct;
mod backpropagation;
mod lvbm;
mod example;

use backpropagation::BackPropTrainer;
use lvbm::LevembergMarquardtTrainer;
use train::Trainer;
use std::sync::Arc;
use network::LayerConfig;
use network::EvalFunc;
use example::GreaterThan;
use example::Sine;
use example::Square;
use example::Hole;
use example::Triangle;
use example::Round;
use example::TrainingData;
// I need a real test now ! -> 1 / n**2 * n * machin

fn main() {
    let mut my_rand = XorShiftRng::new_unseeded();
    let mut tests_array = Round::create_tests(1000);
    my_rand.shuffle(&mut tests_array);
    let tests = Arc::new(tests_array);

    println!("tests : {:?}", tests);
    let mut layers = layers![64,8];
    for l in 0..layers.len() {
        layers[l].eval_function(EvalFunc::Sigmoid);
    }
    let net = match false {
        true => BackPropTrainer::new(tests, layers, &mut my_rand)
            .number_of_batches(10)
            .step(5.6)
            .lower_bound(0.1)
            .start()
            .get_net(),
        false => {
            LevembergMarquardtTrainer::new(tests, layers, &mut my_rand)
                .lambda(1000.0)
                .lower_bound(0.0015)
                .start()
                .get_net()
        }
    };
    Square::show_me(&net);
}
