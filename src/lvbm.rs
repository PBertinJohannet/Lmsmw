use rulinalg::matrix::Matrix;
use std::sync::Arc;
use netstruct::NetStruct;
use netstruct::NetStructTrait;
use train::Test;
use train::Trainer;
use train::CoefCalculator;
use network::Network;
use rand::XorShiftRng;
use network::LayerConfig;
use network::EvalFunc;

#[derive(Debug)]
pub struct Gradients{
    hessian : Matrix<f64>,
    from_cost : NetStruct
}

pub struct LevembergMarquardtTrainer {
    tests: Arc<Vec<Test>>,
    lvbm_calc: LevembergMarquardtCalculator,
    lower_bound: f64,
    mini_batches : usize,
    lambda : f64,
    prev_score : f64,
}

impl Trainer<Gradients, LevembergMarquardtCalculator> for LevembergMarquardtTrainer {
    fn new(tests: Arc<Vec<Test>>, structure: Vec<LayerConfig>, my_rand: &mut XorShiftRng) -> Self {
        let net = Network::new(structure, my_rand);
        let score = net.evaluate(&tests);
        LevembergMarquardtTrainer {
            tests: tests,
            lvbm_calc: net,
            lower_bound: 0.2,
            mini_batches : 1,
            lambda : 10000.0,
            prev_score :score
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
        unimplemented![];/*
        println!("start net : {:?}", self.get_net());
        let mut i = 0;
        #[allow(dead_code)]
        let mut score = self.get_net().evaluate(&self.tests);
        while score > self.lower_bound {
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
            println!("epoch : {}, eval after : {}", i, score);
        }
        self*/
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
}

impl LevembergMarquardtTrainer {
    fn calc_final(&self, grads : Gradients) -> NetStruct{
        let dj = grads.from_cost.to_vector();
        let diag = self.lambda+grads.hessian[[0,0]];
        let mut hess = Matrix::from_fn(dj.size(), dj.size(), |i,j| match i==j {true => diag, false => grads.hessian[[i,j]]});
        NetStruct::from_vector(&(&hess.inverse().expect("matrix inversion failed") * &dj),
                    &self.get_net().get_layers_structure())
    }
    fn lambda(&mut self, lambda : f64) -> &mut Self {
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
        Gradients{from_cost : self.global_gradient(tests, true),
            hessian : self.global_hessian(tests)}
    }
    fn add_result(&self, first: &Gradients, second: &Gradients) -> Gradients {
        Gradients{hessian : &first.hessian + &second.hessian,
                from_cost : first.from_cost.add(&second.from_cost)}
    }
    fn get_empty_val(&self) -> Gradients {
        let grad = self.get_empty_grad();
        let len = grad.to_vector().size();
        Gradients{hessian : Matrix::zeros(len,len), from_cost : grad}
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn new() {
        let mut my_rand = XorShiftRng::from_seed([1, 2, 3, 4]);
        let tests = Arc::new(vec![Test::new(vector![1.0,1.1], vector![0.1])]);
        let my_trainer = LevembergMarquardtTrainer::new(tests, layers![2, 1], &mut my_rand);
        let my_net : Network= my_trainer.get_net();
        assert_eq![
            my_net.feed_forward(&vector![1.0,0.5]),
            Network::new(layers![2, 1], &mut XorShiftRng::from_seed([1, 2, 3, 4])).feed_forward(&vector![1.0, 0.5])
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
        net.set_weights(
            NetStruct::from_vector(&vector![1.0, 1.0], &vec![2, 1]));
        assert_eq![net.get_hessian(&vector![1.0,3.0], &vector![-0.3]),
                Matrix::new(2,2,vec![1.0,3.0,3.0,9.0])];
        assert_eq![net.get_hessian(&vector![2.0,1.0], &vector![0.4]),
                Matrix::new(2,2,vec![4.0,2.0,2.0,1.0])];
        assert_eq![net.get_hessian(&vector![3.0,2.0], &vector![0.5]),
                Matrix::new(2,2,vec![9.0,6.0,6.0,4.0])];
        assert_eq![net.global_hessian(&vec![Test::new(vector![1.0,3.0], vector![-0.3]), Test::new(vector![2.0,1.0], vector![0.4]), Test::new(vector![3.0,2.0], vector![0.5])]),
            Matrix::new(2,2,vec![14.0,11.0,11.0,14.0])];
        // test dJ/dc
        assert_eq![
        net.back_propagation(&vector![1.0,3.0], &vector![-0.3], true)
            .to_vector(),
            vector![-4.3, -12.899999999999999]
        ];
        assert_eq![
        net.back_propagation(&vector![2.0,1.0], &vector![0.4], true)
            .to_vector(),
            vector![-5.2, -2.6]
        ];
        assert_eq![
        net.back_propagation(&vector![3.0,2.0], &vector![0.5], true)
            .to_vector(),
            vector![-13.5, -9.0]
        ];
        assert_eq![net.global_gradient(&vec![Test::new(vector![1.0,3.0], vector![-0.3]),
                                             Test::new(vector![2.0,1.0], vector![0.4]),
                                             Test::new(vector![3.0,2.0], vector![0.5])], true).to_vector(),
            vector![-23.0, -24.5]];

        // test final :

        let lambda = 10.0;
        let dj = vector![-23.0, -24.5];
        let hessian = Matrix::new(2,2,vec![14.0,11.0,11.0,14.0]);
        let diag = lambda+hessian[[0,0]];
        let mut hess = Matrix::from_fn(dj.size(), dj.size(), |i,j| match i==j {true => diag, false => hessian[[i,j]]});
        assert_eq![hess, Matrix::new(2,2,vec![24.0,11.0,11.0,24.0])];
        assert_eq![hess.clone().inverse().expect("matrix inversion failed"), Matrix::new(2,
                                                                                         2,
                                                                                         vec![0.05274725274725275,-0.024175824175824173
        ,-0.024175824175824173,0.05274725274725274])];
        NetStruct::from_vector(&(&hess.inverse().expect("matrix inversion failed") * &dj),
                               &vec![2,1]);



        // test method
        let mut lvb = LevembergMarquardtTrainer::new(Arc::new(vec![Test::new(vector![1.0,3.0], vector![-0.3]),
                                                           Test::new(vector![2.0,1.0], vector![0.4]),
                                                           Test::new(vector![3.0,2.0], vector![0.5])]),
                                                            layers,
                                                            &mut XorShiftRng::from_seed([1, 2, 3, 4]));
        lvb.lambda = 10.0;
        lvb.get_mut_net().set_weights(
            NetStruct::from_vector(&vector![1.0, 1.0], &vec![2, 1]));
        let glob_grad = lvb.train(&lvb.tests);
        println!("{:?}", glob_grad);
        assert_eq![glob_grad.from_cost, vec![vec![vector![-23.0, -24.5]]]];
        assert_eq![glob_grad.hessian, Matrix::new(2,2,vec![14.0,11.0,11.0,14.0])];
        let final_grad = lvb.calc_final(glob_grad);
        assert_eq![final_grad.to_vector(), vector![-0.6208791208791209,-0.7362637362637362]];
        lvb.get_mut_net().add_gradient(&final_grad, 1.0);
        assert_eq![lvb.get_net().get_weights().to_vector(), vector![0.3791208791208791, 0.2637362637362638]];
        assert_eq![lvb.get_net().evaluate(&vec![Test::new(vector![1.0,3.0], vector![-0.3]),
                                               Test::new(vector![2.0,1.0], vector![0.4]),
                                               Test::new(vector![3.0,2.0], vector![0.5])]), 1.3018556535040056]
    }
}

