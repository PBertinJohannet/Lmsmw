use rulinalg::vector::Vector;
use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;
use rand::XorShiftRng;
use rand::Rng;
use num_traits::float::Float;
use std::iter::FromIterator;
use std::sync::Arc;


pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + 2.0.powf(-x))
}
pub fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[derive(Debug, Clone)]
pub struct Test {
    inputs: Vector<f64>,
    outputs: Vector<f64>,
}
impl Test {
    pub fn new(inputs: Vector<f64>, outputs: Vector<f64>) -> Self {
        Test {
            inputs: inputs,
            outputs: outputs,
        }
    }
}


pub type NetStruct = Vec<Vec<Vector<f64>>>;
pub trait NetStructTrait {
    fn random(structure: &Vec<usize>, my_rand: &mut XorShiftRng) -> Self;
    fn apply(&self, fun : &Fn(f64) -> f64) -> Self;
    fn add(&self, other  : &Self) -> Self;
    fn to_vector(&self)->Vector<f64>;
    fn from_vector(source : &Vector<f64>, structure : &Vec<usize>)->Self;
}
impl NetStructTrait for NetStruct {
    fn random(structure: &Vec<usize>, my_rand: &mut XorShiftRng) -> Self {
        (structure.iter().zip(structure.iter().skip(1)).map(|(&prev,
          &next)| {
            (0..next)
                .map(|_| {
                    Vector::from(my_rand.gen_iter().take(prev).collect::<Vec<f64>>())*0.1
                })
                .collect::<Vec<Vector<f64>>>()
        })).collect()
    }
    fn add(&self, other: &NetStruct) -> NetStruct {
        let mut net = self.clone();
        for layer in 0..net.len() {
            for neur in 0..net[layer].len() {
                let s =
                    Vector::from_iter(net[layer][neur].iter().zip(other[layer][neur].iter()).map(
                        |(a, b)| a + b,
                    ));
                net[layer][neur] = s;
            }
        }
        net
    }
    fn apply(&self, f: &Fn(f64) -> f64) -> NetStruct {
        let mut net = self.clone();
        for layer in 0..net.len() {
            for neur in 0..net[layer].len() {
                let s = net[layer][neur].clone().apply(f);
                net[layer][neur] = s;
            }
        }
        net
    }
    fn to_vector(&self)->Vector<f64>{
        let mut to_ret = vec![];
        for layer in 0..self.len() {
            for neur in 0..self[layer].len() {
                for coef in 0..self[layer][neur].size(){
                    to_ret.push(self[layer][neur][coef]);
                }
            }
        }
        Vector::from(to_ret)
    }
    fn from_vector(vec : &Vector<f64>, structure : &Vec<usize>)->Self{
        let mut i = 0;
        (structure.iter().zip(structure.iter().skip(1)).map(|(&prev,
                                                                 &next)| {
            (0..next)
                .map(|_| {
                    Vector::from((0..prev).map(|_|{i+=1;vec[i-1]}).collect::<Vec<f64>>())
                })
                .collect::<Vec<Vector<f64>>>()
        })).collect()
    }
}




#[derive(Debug, Clone)]
pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    weights: NetStruct,
    nb_coef : usize,
}


impl Network {
    pub fn new(structure: Vec<usize>, my_rand: &mut XorShiftRng) -> Self {
        Network {
            num_layers: structure.len(),
            sizes: structure.clone(),
            weights: NetStructTrait::random(&structure, my_rand),
            nb_coef:(structure.iter().zip(structure.iter().skip(1)).map(|(&prev, &next)| next * prev)).sum(),
        }
    }
    pub fn levemberg_marquardt(&mut self)->&mut Self {
        self
    }
    pub fn feed_forward(&self, input: &Vector<f64>) -> Vector<f64> {
        self.weights.iter().fold(input.clone(), |prev, layer| {
            Vector::from(
                layer
                    .iter()
                    .map(|neur| sigmoid(*&neur.elemul(&prev).sum()))
                    .collect::<Vec<f64>>(),
            )
        })
    }
    pub fn evaluate(&self, tests: &Vec<Test>) -> f64 {
        (tests
             .iter()
             .map(|test| {
            (&test.outputs - self.feed_forward(&test.inputs))
                .apply(&|x| x.abs())
                .sum() as f64
        })
             .sum::<f64>()) / (tests.len()) as f64
    }
    pub fn cost_derivative(&self, output: &Vector<f64>, desired: &Vector<f64>) -> Vector<f64> {
        output - desired
    }
    pub fn get_empty_grad(&self) -> NetStruct {
        (self.sizes.iter().zip(self.sizes.iter().skip(1)).map(
            |(&prev,
              &next)| {
                (0..next)
                    .map(|_| Vector::zeros(prev))
                    .collect::<Vec<Vector<f64>>>()
            },
        )).collect::<Vec<_>>()
    }
    pub fn global_gradient(&self, tests: &[Test]) -> NetStruct {
        tests.iter().fold(self.get_empty_grad(), |grad, test| {
                grad.add(&self.back_propagation(&test.inputs, &test.outputs))
            })
    }
    pub fn add_gradient(&mut self, grad: &NetStruct, coef: f64) {
        self.weights = self.weights.add(&grad.apply(&|a| -a * coef));
    }
    pub fn global_hessian(&self, tests: &[Test])->Matrix<f64>{
        let mut hessian = self.get_hessian(&tests[0].inputs, &tests[0].outputs);
        for i in 1..tests.len(){
            hessian += self.get_hessian(&tests[i].inputs, &tests[0].outputs);
        }
        hessian
    }
    pub fn get_hessian(&self, input : &Vector<f64>, output: &Vector<f64>)->Matrix<f64>{
        let grad = self.back_propagation(input, output).to_vector();
        Matrix::from_fn(grad.size(), grad.size(), |col, row| grad[col]*grad[row])
    }
    pub fn back_propagation(&self, input: &Vector<f64>, output: &Vector<f64>) -> NetStruct {
        let mut grad = self.get_empty_grad();
        let mut activations = vec![input.clone()];
        let mut z_vecs = vec![];
        for ref layer in self.weights.iter() {
            z_vecs.push(Vector::from(
                layer
                    .iter()
                    .map(|neur| *&neur.elemul(&activations.last().unwrap()).sum())
                    .collect::<Vec<f64>>(),
            ));
            activations.push(z_vecs.last().unwrap().clone().apply(&sigmoid));
        }
        let mut delta = z_vecs
            .last()
            .unwrap()
            .clone()
            .apply(&sigmoid_prime)
            .elemul(&self.cost_derivative(activations.last().unwrap(), &output));
        //println!("delta : {}", delta);
        grad[self.weights.len() - 1] = delta
            .iter()
            .map(|x| activations[activations.len() - 2].clone().apply(&|a| a * x))
            .collect::<Vec<Vector<f64>>>();
        //println!("last grad : {:?}", grad[self.weights.len() - 1]);
        //println!("backprog\n");
        for l_id in 2..z_vecs.len() + 1 {
            //println!("layer : {}", l_id);
            //println!("this z vec {}!", z_vecs[z_vecs.len() - l_id]);
            let sig_prim = z_vecs[z_vecs.len() - l_id].clone().apply(&sigmoid_prime);
            //println!("sig prim : {}", sig_prim);
            delta = (&Matrix::new(
                self.sizes[self.sizes.len() - l_id + 1],
                self.sizes[self.sizes.len() - l_id],
                self.weights[self.weights.len() - l_id + 1]
                    .clone()
                    .iter()
                    .map(|a| a.clone().into_vec())
                    .fold(vec![], |mut a, mut b| {
                        a.append(&mut b);
                        a
                    }),
            ).transpose() * &delta)
                .elemul(&sig_prim);
            //println!("new {:?}", delta);
            grad[self.weights.len() - l_id] = delta
                .iter()
                .map(|d| {
                    Vector::from(
                        activations[activations.len() - l_id - 1]
                            .iter()
                            .map(&|a| d * a)
                            .collect::<Vec<f64>>(),
                    )
                })
                .collect::<Vec<Vector<f64>>>();
            //println!("new grad : {:?}", grad[grad.len() - l_id]);
            //println!("next : \n");
        }
        grad
    }
}

pub struct Trainer {
    tests: Arc<Vec<Test>>,
    net: Network,
    lower_bound: f64,
    step: f64,
    mini_batches: usize,
}

impl Trainer {
    pub fn new(tests: Arc<Vec<Test>>, structure: Vec<usize>, my_rand: &mut XorShiftRng) -> Self {
        Trainer {
            tests: tests,
            net: Network::new(structure, my_rand),
            lower_bound: 0.2,
            step: 0.1,
            mini_batches: 1,
        }
    }
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
    pub fn get_mini_batches(&self) -> Vec<Vec<Test>>{
        let num_tests_per_batches = self.tests.len() / self.mini_batches;
        let num_tougher_batches = self.tests.len() % self.mini_batches;
        let mut offset = 0;
        (0..self.mini_batches).map(|id| {
            let chunksize = match id < num_tougher_batches {
                true => num_tests_per_batches + 1,
                false => num_tests_per_batches,
            };
            let to_ret = self.tests[offset..offset + chunksize].to_vec();
            offset += chunksize;
            to_ret
        }).collect::<Vec<Vec<Test>>>()
    }
    pub fn start(&mut self) -> &mut Self {
        println!("start net : {:?}", self.net);
        let mut i = 0;
        #[allow(dead_code)]
        let mut score = self.net.evaluate(&self.tests);
        while self.net.evaluate(&self.tests) > self.lower_bound {
            i += 1;
            for batch in self.get_mini_batches() {
                let l = batch.len();
                let glob_grad = train(&Arc::new(batch), &self.net);
                self.net.add_gradient(
                    &glob_grad,
                    self.step / (l as f64),
                );
            }
            score = self.net.evaluate(&self.tests);
            println!(
                "epoch : {}, eval after : {}",
                i,
                score
            );
        }
        self
    }
    #[allow(dead_code)]
    pub fn levemberg_marquardt(&mut self) -> &mut Self {
        self.net.levemberg_marquardt();
        self
    }
    pub fn get_net(&self)->Network{
        self.net.clone()
    }
}


pub fn train(tests: &Arc<Vec<Test>>, nn: &Network) -> NetStruct {
    use std::sync::mpsc;
    use std::thread;
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
            let new_net = nn.clone();
            thread::spawn(move || {
                let end = offset + chunksize;
                let grad = new_net.global_gradient(&my_tests[offset..end]);
                my_tx.send(grad).unwrap();
            });
            offset += chunksize;
        }
    }
    drop(tx);
    rx.iter().fold(
        nn.get_empty_grad(),
        |base, grad| base.add(&grad),
    )
}
