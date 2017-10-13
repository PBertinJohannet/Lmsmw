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
use train::Test;
use backpropagation::BackPropTrainer;
use train::Trainer;
use std::sync::Arc;
use network::LayerConfig;


trait TrainingData {
    fn create_tests(nb: i32) -> Vec<Test>;
    fn lower_bound() -> f64;
}
struct GreaterThan {}
impl TrainingData for GreaterThan {
    fn create_tests(nb: i32) -> Vec<Test> {
        let mut my_rand = XorShiftRng::new_unseeded();
        (0..nb)
            .map(|_| {
                (my_rand.gen_range(-5.0, 5.0), my_rand.gen_range(-5.0, 5.0))
            })
            .map(|(a, b)| {
                Test::new(
                    vector![a, b],
                    vector![(a > b) as i32 as f64, (a < b) as i32 as f64],
                )
            })
            .collect::<Vec<Test>>()
    }
    fn lower_bound() -> f64 {
        0.2
    }
}
struct Sine {}
impl TrainingData for Sine {
    fn create_tests(nb: i32) -> Vec<Test> {
        (0..nb)
            .map(|n| 0.5 * 6.28 * n as f64 / nb as f64 - 3.14)
            .map(|a| Test::new(vector![a, 1.0], vector![a.sin()]))
            .collect::<Vec<Test>>()
    }
    fn lower_bound() -> f64 {
        0.2
    }
}
fn main() {
    let mut my_rand = XorShiftRng::new_unseeded();
    let tests = Arc::new(GreaterThan::create_tests(500));
    println!("tests : {:?}", tests);
    BackPropTrainer::new(tests, layers![2, 2], &mut my_rand)
     //   .levemberg_marquardt()
        .number_of_batches(1)
        .step(5.6)
        .lower_bound(0.7)
        .start()
        .step(0.2)
        .lower_bound(0.65)
        .start()
        .step(15.0)
        .lower_bound(0.64)
        .start()
        .step(30.0)
        .lower_bound(0.001)
        .start()
        .get_net();
}
