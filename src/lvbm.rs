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
}

impl Trainer<Gradients, LevembergMarquardtCalculator> for LevembergMarquardtTrainer {
    fn new(tests: Arc<Vec<Test>>, structure: Vec<LayerConfig>, my_rand: &mut XorShiftRng) -> Self {
        LevembergMarquardtTrainer {
            tests: tests,
            lvbm_calc: Network::new(structure, my_rand),
            lower_bound: 0.2,
            mini_batches : 1
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
        let mut my_rand = XorShiftRng::from_seed([1, 2, 3, 4]);
        let tests = Arc::new(vec![Test::new(vector![1.0,1.1], vector![0.1])]);
        let my_trainer = LevembergMarquardtTrainer::new(tests, layers![2, 1], &mut my_rand);

    }
}

