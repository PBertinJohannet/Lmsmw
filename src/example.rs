use network::Network;
use rand::XorShiftRng;
use rand::Rng;
use rulinalg::vector::Vector;


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

pub trait TrainingData {
    fn create_tests(nb: i32) -> Vec<Test>;
    fn lower_bound() -> f64;
    fn show_me(&self, net : &Network);
}
pub struct GreaterThan {}
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
    fn show_me(&self, net : &Network) {
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
    }
}
pub struct Sine {}
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
    fn show_me(&self, net : &Network) {
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
    }
}