#[macro_use]
extern crate rulinalg;
extern crate rand;
extern crate num_traits;
use rand::XorShiftRng;
use rand::Rng;
mod network;
use network::Test;
use network::Trainer;
use std::sync::Arc;


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
    let tests = Arc::new(Sine::create_tests(50_000));
    println!("tests : {:?}", tests);
    Trainer::new(tests, vec![2, 3, 3, 1], &mut my_rand)
     //   .levemberg_marquardt()
        .number_of_batches(100)
        .step(5.6)
        .lower_bound(0.5)
        .start()
        .step(0.2)
        .lower_bound(0.5)
        .start()
        .step(2.8)
        .lower_bound(0.05)
        .start()
        .step(0.6)
        .lower_bound(0.001)
        .start()
        .get_net();

}
