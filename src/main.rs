#[macro_use]
extern crate rulinalg;
extern crate rand;
extern crate num_traits;
use rand::XorShiftRng;
use rand::Rng;
use rulinalg::matrix::BaseMatrix;
#[macro_use]
mod network;
mod train;
mod netstruct;
mod backpropagation;
mod lvbm;
use train::Test;
use backpropagation::BackPropTrainer;
use lvbm::LevembergMarquardtTrainer;
use train::Trainer;
use std::sync::Arc;
use netstruct::NetStruct;
use network::LayerConfig;
use network::EvalFunc;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use netstruct::NetStructTrait;


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
                (my_rand.gen_range(-1.0, 1.0), my_rand.gen_range(-1.0, 1.0))
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
            .map(|n| 6.28 * n as f64 / nb as f64 )
            .map(|a| Test::new(vector![a/6.28, 1.0], vector![a.sin()*0.5+0.5]))
            .collect::<Vec<Test>>()
    }
    fn lower_bound() -> f64 {
        0.2
    }
}
fn main() {
    let mut my_rand = XorShiftRng::new_unseeded();
    let mut tests_array = Sine::create_tests(60);
    my_rand.shuffle(&mut tests_array);
    let tests = Arc::new(tests_array);
    /*let tests = Arc::new(vec![Test::new(vector![1.0,3.0], vector![0.3]),
        Test::new(vector![2.0,1.0], vector![0.4]),
        Test::new(vector![3.0,2.0], vector![0.5])]);*/

    println!("tests : {:?}", tests);
    let mut layers = layers![2, 4, 1];
    for l in 0..layers.len() {
        layers[l].eval_function(EvalFunc::Sigmoid);
    }
    let net = match false {
        true => BackPropTrainer::new(tests, layers, &mut my_rand)
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
            .lower_bound(0.002)
            .start()
            .get_net(),
        false => {
            LevembergMarquardtTrainer::new(tests, layers, &mut my_rand)
                .lambda(10.0)
                .lower_bound(0.00005)
                .start()
                .get_net()
        }
    };
    for n in 0..50 {
        let inp =  6.28 * n as f64 / 50.0 as f64;
        let out = (inp.sin()*0.5+0.5)*50.0;
        let out_real = net.feed_forward(&vector![inp/6.28, 1.0])[0]*50.0;
        println!("out : {}", out as i32);
        for j in 0..100 {
            if j as i32 == out as i32 {
                print!("O");
            } else if j as i32 == out_real as i32{
                print!(".");
            } else {
                print!(" ");
            }
        }

    }
    /*let tests = Arc::new(vec![Test::new(vector![1.0,3.0], vector![-0.3]),
                              Test::new(vector![2.0,1.0], vector![0.4]),
                              Test::new(vector![3.0,2.0], vector![0.5])]);*/
    /*let mut layers = layers![2, 3, 3, 1];
    for l in 0..layers.len() {
        layers[l].eval_function(EvalFunc::Identity);
    }
    let mut lvb = LevembergMarquardtTrainer::new(tests, layers, &mut my_rand);

   // lvb.get_mut_net().set_weights(
     //   NetStruct::from_vector(&vector![1.0, 1.0], &vec![2, 3, 1]));
    lvb.prev_score = lvb.lvbm_calc.evaluate(&lvb.tests);
    lvb
        .lambda(10.0)
        .lower_bound(0.00005)
        .start()
        .show_me();*/

    /*let to_inv = Matrix::new(4,4,vec![0.2,0.2,0.1,0.5,
                                      0.3,0.5,1.1,1.0,
                                      0.15,0.9,0.7,0.01,
                                       0.2,0.25,0.31,0.4]);
    println!("inversed : \n{}", &to_inv.clone().inverse().expect("inversed"));
    // a = i - T
    // T = i - a
    let t : Matrix<f64>= Matrix::identity(4) - &to_inv;
    println!("t : \n{}", t);
    let mut acc : Matrix<f64>= Matrix::identity(4);
    let mut pow = t.clone();
    for i in 0..100 {
        acc+=&pow;
        pow=&pow*&t;
    }
    println!("n : \n{}", acc);
    println!("diff : \n{}", &to_inv.clone().inverse().expect("inversed") -& acc);
    //println!("norm : {:?}", &to_inv.norm());*/
    // grad divisé par nb de tests ?
    // I+lambda mais si ca n'avance plus ? (eg : si lambda tres grand)
}
