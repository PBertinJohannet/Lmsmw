use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;
use rulinalg::vector::Vector;
use std::sync::Arc;
use netstruct::NetStruct;
use netstruct::NetStructTrait;
use example::Test;
use train::Trainer;
use train::CoefCalculator;
use network::Network;
use rand::XorShiftRng;
use network::LayerConfig;

#[derive(Debug)]
pub struct Gradients {
    grads: Vec<Vector<f64>>,
    from_cost: Vector<f64>,
}

pub struct LevembergMarquardtTrainer {
    pub tests: Arc<Vec<Test>>,
    pub lvbm_calc: LevembergMarquardtCalculator,
    lower_bound: f64,
    mini_batches: usize,
    lambda: f64,
    pub prev_score: f64,
}

impl Trainer<Gradients, LevembergMarquardtCalculator> for LevembergMarquardtTrainer {
    fn new(tests: Arc<Vec<Test>>, structure: Vec<LayerConfig>, my_rand: &mut XorShiftRng) -> Self {
        let net = Network::new(structure, my_rand);
        let score = net.evaluate(&tests);
        LevembergMarquardtTrainer {
            tests: tests,
            lvbm_calc: net,
            lower_bound: 0.2,
            mini_batches: 1,
            lambda: 10000.0,
            prev_score: score,
        }
    }
    fn get_tests<'a>(&'a self) -> &'a Arc<Vec<Test>> {
        &self.tests
    }
    fn get_cloned_calculator(&self) -> LevembergMarquardtCalculator {
        self.lvbm_calc.clone()
    }
    fn get_calculator(&self) -> &LevembergMarquardtCalculator {
        &self.lvbm_calc
    }
    fn number_of_batches(&mut self, mini_batch_size: usize) -> &mut Self {
        self.mini_batches = mini_batch_size;
        self
    }
    fn start(&mut self) -> &mut Self {

        //println!("start net : {:?}", self.get_net());
        let mut i = 0;
        #[allow(dead_code)]
        while self.prev_score > self.lower_bound {
            i += 1;
            for batch in self.get_mini_batches() {
                let l = batch.len();
                let glob_grad = self.train(&Arc::new(batch));
                self.next_iter(&glob_grad);
            }
            //println!("glob grad max : {:?}", self.get_net().get_weights().to_vector().iter().fold(0.0 as f64,|acc,&x|match acc>x {true => acc, false =>x}));
            println!(
                "epoch : {}, eval after : {}                glob grad max : {}",
                i,
                self.prev_score,
                self.get_net().get_weights().to_vector().iter().fold(
                    0.0,
                    |acc, &x| {
                        match acc > x {
                            true => acc,
                            false => x,
                        }
                    },
                )
            );
            if self.lambda > 10_000_000_000.0 {
                break;
            }
        }
        self
    }
    fn get_net(&self) -> Network {
        self.lvbm_calc.clone()
    }
    fn get_mut_net(&mut self) -> &mut Network {
        &mut self.lvbm_calc
    }
    fn get_number_of_batches(&self) -> usize {
        self.mini_batches
    }
    fn get_empty_val(&self) -> Gradients {
        self.lvbm_calc.get_empty_val()
    }
    fn lower_bound(&mut self, bound: f64) -> &mut Self {
        self.lower_bound = bound;
        self
    }
    fn set_net(&mut self, net: Network) -> &mut Self {
        self.lvbm_calc = net;
        self
    }
}

impl LevembergMarquardtTrainer {
    fn fold_elimination(&self, vecs: &Vec<Vector<f64>>, sol: &Vector<f64>) -> Vector<f64> {
        let mut xzero = sol.clone().apply(&|x| x / self.lambda);
        let mut y = vec![
            vecs.iter()
                .map(|vec| Matrix::from(vec.clone().apply(&|x| x / self.lambda)))
                .collect::<Vec<Matrix<f64>>>(),
        ];
        let tests = vecs.len();
        for l in 1..tests + 1 {
            let u = Matrix::from(vecs[l - 1].clone());
            let lower = 1.0 / (1.0 + (&u.transpose() * &y[l - 1][l - 1]).col(0)[0]);
            let xone = &xzero - &y[l - 1][l - 1] * lower * ((&u.transpose() * &xzero));
            let nouv = (1..tests + 1)
                .map(|k| match k >= l - 1 {
                    true => {
                        &y[l - 1][k - 1] -
                            &y[l - 1][l - 1] * lower * ((&u.transpose() * &y[l - 1][k - 1]))
                    }
                    _ => Matrix::new(0, 0, vec![]),
                })
                .collect::<Vec<Matrix<f64>>>();
            y.push(nouv);
            xzero = xone;
        }
        return xzero;
    }

    fn normal_elimination(&self, grads: &Gradients) -> Vector<f64> {
        let len = self.get_net().get_empty_grad().to_vector().size();
        let mut h = Matrix::identity(len) * self.lambda;
        for g in grads.grads.iter() {
            h += Matrix::from_fn(len, len, |col, row| g[col] * g[row]);
        }
        h.solve(grads.from_cost.clone())
            .expect("should work")
            .clone()
    }
    fn calc_final(&self, grads: &Gradients) -> NetStruct {
        let len = self.get_net().get_empty_grad().to_vector().size();
        NetStruct::from_vector(
            &match len > grads.grads.len() {
                true => self.fold_elimination(&grads.grads, &grads.from_cost),
                _ => self.normal_elimination(grads),
            },
            &self.get_net().get_layers_structure(),
        )
    }
    fn next_iter(&mut self, grads: &Gradients) {
        loop {
            let final_grad = self.calc_final(&grads);
            self.lvbm_calc.add_gradient(&final_grad, 1.0);
            let score = self.lvbm_calc.evaluate(&self.tests);
            if score >= self.prev_score {
                self.lambda *= 9.5;
                self.lvbm_calc.add_gradient(&final_grad, -1.0);
            //println!("score : {}\n lambda : {}\n", self.prev_score, self.lambda);
            } else {
                if self.lambda > 0.000_000_000_1 {
                    self.lambda /= 1.5;
                }
                self.prev_score = score;
                break;
            }
            //assert_eq![self.lvbm_calc.evaluate(&self.tests), self.prev_score];
            if self.lambda > 1000_000_000.0 {
                //println!("tests : \n{:?}", &self.tests);
                //println!("grad : \n{:?}", grads.from_cost);
                //println!("hess : \n{:?}", grads.hessian);
                //println!("net : \n{:?}", &self.get_net());
                //println!("final grad : \n{:?}", &final_grad);
                self.lambda = 100_000_000_000.0;
                break;
                //  println!("net : \n{:?}", &self.get_net());
            }
        }
        //assert_eq![self.get_net().evaluate(&self.tests), score];
    }
    pub fn lambda(&mut self, lambda: f64) -> &mut Self {
        self.lambda = lambda;
        self
    }
}

type LevembergMarquardtCalculator = Network;

impl CoefCalculator<Gradients> for LevembergMarquardtCalculator {
    fn get_net(&self) -> Network {
        self.clone()
    }
    fn calc_result(&self, tests: &[Test]) -> Gradients {
        Gradients {
            from_cost: self.global_gradient(tests, true).to_vector(),
            grads: tests
                .iter()
                .map(|test| {
                    self.back_propagation(&test.inputs, &test.outputs, false)
                        .to_vector()
                })
                .collect::<Vec<Vector<f64>>>(),
        }
    }
    fn add_result(&self, first: &Gradients, second: &Gradients) -> Gradients {
        Gradients {
            grads: [&first.grads[..], &second.grads[..]].concat(),
            from_cost: &first.from_cost + &second.from_cost,
        }
    }
    fn get_empty_val(&self) -> Gradients {
        let grad = self.get_empty_grad().to_vector();
        let len = grad.size();
        Gradients {
            grads: vec![],
            from_cost: grad,
        }
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn new() {
        let mut my_rand = XorShiftRng::from_seed([1, 2, 3, 4]);
        let tests = Arc::new(vec![Test::new(vector![1.0, 1.1], vector![0.1])]);
        let my_trainer = LevembergMarquardtTrainer::new(tests, layers![2, 1], &mut my_rand);
        let my_net: Network = my_trainer.get_net();
        assert_eq![
            my_net.feed_forward(&vector![1.0, 0.5]),
            Network::new(layers![2, 1], &mut XorShiftRng::from_seed([1, 2, 3, 4]))
                .feed_forward(&vector![1.0, 0.5]),
        ];
    }
    #[test]
    fn one_iter() {
        let mut layers = layers![2, 1];
        for l in 0..layers.len() {
            layers[l].eval_function(EvalFunc::Identity);
        }
        let mut net = Network::new(layers.clone(), &mut XorShiftRng::from_seed([1, 2, 3, 4]));
        // test hessian calculation
        net.set_weights(NetStruct::from_vector(&vector![1.0, 1.0], &vec![2, 1]));
        assert_eq![
            net.get_hessian(&vector![1.0, 3.0], &vector![-0.3]),
            Matrix::new(2, 2, vec![1.0, 3.0, 3.0, 9.0]),
        ];
        assert_eq![
            net.get_hessian(&vector![2.0, 1.0], &vector![0.4]),
            Matrix::new(2, 2, vec![4.0, 2.0, 2.0, 1.0]),
        ];
        assert_eq![
            net.get_hessian(&vector![3.0, 2.0], &vector![0.5]),
            Matrix::new(2, 2, vec![9.0, 6.0, 6.0, 4.0]),
        ];
        assert_eq![
            net.global_hessian(&vec![
                Test::new(vector![1.0, 3.0], vector![-0.3]),
                Test::new(vector![2.0, 1.0], vector![0.4]),
                Test::new(vector![3.0, 2.0], vector![0.5]),
            ]),
            Matrix::new(2, 2, vec![14.0, 11.0, 11.0, 14.0]),
        ];
        // test dJ/dc
        assert_eq![
            net.back_propagation(&vector![1.0, 3.0], &vector![-0.3], true)
                .to_vector(),
            vector![-4.3, -12.899999999999999],
        ];
        assert_eq![
            net.back_propagation(&vector![2.0, 1.0], &vector![0.4], true)
                .to_vector(),
            vector![-5.2, -2.6],
        ];
        assert_eq![
            net.back_propagation(&vector![3.0, 2.0], &vector![0.5], true)
                .to_vector(),
            vector![-13.5, -9.0],
        ];
        assert_eq![
            net.global_gradient(
                &vec![
                    Test::new(vector![1.0, 3.0], vector![-0.3]),
                    Test::new(vector![2.0, 1.0], vector![0.4]),
                    Test::new(vector![3.0, 2.0], vector![0.5]),
                ],
                true
            ).to_vector(),
            vector![-23.0, -24.5],
        ];

        // test final :

        let lambda = 10.0;
        let dj = vector![-23.0, -24.5];
        let hessian = Matrix::new(2, 2, vec![14.0, 11.0, 11.0, 14.0]);
        let diag = lambda + hessian[[0, 0]];
        let mut hess = Matrix::from_fn(dj.size(), dj.size(), |i, j| match i == j {
            true => diag,
            false => hessian[[i, j]],
        });
        assert_eq![hess, Matrix::new(2, 2, vec![24.0, 11.0, 11.0, 24.0])];
        assert_eq![
            hess.clone().inverse().expect("matrix inversion failed"),
            Matrix::new(
                2,
                2,
                vec![
                    0.05274725274725275,
                    -0.024175824175824173,
                    -0.024175824175824173,
                    0.05274725274725274,
                ]
            ),
        ];
        NetStruct::from_vector(
            &(&hess.inverse().expect("matrix inversion failed") * &dj),
            &vec![2, 1],
        );



        // test method
        let mut lvb = LevembergMarquardtTrainer::new(
            Arc::new(vec![
                Test::new(vector![1.0, 3.0], vector![-0.3]),
                Test::new(vector![2.0, 1.0], vector![0.4]),
                Test::new(vector![3.0, 2.0], vector![0.5]),
            ]),
            layers.clone(),
            &mut XorShiftRng::from_seed([1, 2, 3, 4]),
        );
        lvb.lambda = 10.0;
        lvb.get_mut_net().set_weights(NetStruct::from_vector(
            &vector![1.0, 1.0],
            &vec![2, 1],
        ));
        let glob_grad = lvb.train(&lvb.tests);
        println!("{:?}", glob_grad);
        assert_eq![glob_grad.from_cost, vec![vec![vector![-23.0, -24.5]]]];
        assert_eq![
            glob_grad.grads,
            Matrix::new(2, 2, vec![14.0, 11.0, 11.0, 14.0]),
        ];
        let final_grad = lvb.calc_final(&glob_grad);
        assert_eq![
            final_grad.to_vector(),
            vector![-0.6208791208791209, -0.7362637362637362],
        ];
        lvb.get_mut_net().add_gradient(&final_grad, 1.0);
        assert_eq![
            lvb.get_net().get_weights().to_vector(),
            vector![0.3791208791208791, 0.2637362637362638],
        ];
        assert_eq![lvb.get_net().evaluate(&lvb.tests), 1.0857142857142856];
        lvb = LevembergMarquardtTrainer::new(
            Arc::new(vec![
                Test::new(vector![1.0, 3.0], vector![-0.3]),
                Test::new(vector![2.0, 1.0], vector![0.4]),
                Test::new(vector![3.0, 2.0], vector![0.5]),
            ]),
            layers,
            &mut XorShiftRng::from_seed([1, 2, 3, 4]),
        );
        lvb.lambda = 10.0;
        lvb.get_mut_net().set_weights(NetStruct::from_vector(
            &vector![1.0, 1.0],
            &vec![2, 1],
        ));
        lvb.prev_score = lvb.lvbm_calc.evaluate(&lvb.tests);
        let glob_grad = lvb.train(&lvb.tests);
        lvb.next_iter(&glob_grad);
        assert_eq![
            lvb.get_net().get_weights().to_vector(),
            vector![0.3791208791208791, 0.2637362637362638],
        ];
        assert_eq![lvb.get_net().evaluate(&lvb.tests), 1.0857142857142856];
        let glob_grad = lvb.train(&lvb.tests);
        lvb.next_iter(&glob_grad);
        println!(" nouv score : {}", lvb.get_net().evaluate(&lvb.tests));
        assert_eq![
            lvb.get_net().evaluate(&lvb.tests) < 1.0857142857142856,
            true,
        ];
    }
}
