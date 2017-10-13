use rulinalg::vector::Vector;
use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;
use rand::XorShiftRng;
use num_traits::float::Float;
use train::Test;
use netstruct::NetStruct;
use netstruct::NetStructTrait;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
pub fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[derive(Debug, Clone)]
pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    weights: NetStruct,
    nb_coef: usize,
}


impl Network {
    pub fn new(structure: Vec<usize>, my_rand: &mut XorShiftRng) -> Self {
        Network {
            num_layers: structure.len(),
            sizes: structure.clone(),
            weights: NetStructTrait::random(&structure, my_rand),
            nb_coef: (structure.iter().zip(structure.iter().skip(1)).map(|(&prev,
              &next)| {
                next * prev
            })).sum(),
        }
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
            grad.add(&self.back_propagation(&test.inputs, &test.outputs, true))
        })
    }
    pub fn add_gradient(&mut self, grad: &NetStruct, coef: f64) {
        self.weights = self.weights.add(&grad.apply(&|a| -a * coef));
    }
    pub fn global_hessian(&self, tests: &[Test]) -> Matrix<f64> {
        let mut hessian = self.get_hessian(&tests[0].inputs, &tests[0].outputs);
        for i in 1..tests.len() {
            hessian += self.get_hessian(&tests[i].inputs, &tests[0].outputs);
        }
        hessian
    }
    pub fn get_hessian(&self, input: &Vector<f64>, output: &Vector<f64>) -> Matrix<f64> {
        let grad = self.back_propagation(input, output, false).to_vector();
        Matrix::from_fn(grad.size(), grad.size(), |col, row| grad[col] * grad[row])
    }
    pub fn get_fed_layers(
        &self,
        input: &Vector<f64>,
        output: &Vector<f64>,
    ) -> (Vec<Vector<f64>>, Vec<Vector<f64>>) {
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
        (activations, z_vecs)
    }
    pub fn back_propagation(
        &self,
        input: &Vector<f64>,
        output: &Vector<f64>,
        from_cost: bool,
    ) -> NetStruct {
        let mut grad = self.get_empty_grad();
        let (activations, z_vecs) = self.get_fed_layers(input, output);
        let mut delta = z_vecs.last().unwrap().clone().apply(&sigmoid_prime);
        if from_cost {
            delta = delta.elemul(&self.cost_derivative(activations.last().unwrap(), &output));
        }
        grad[self.weights.len() - 1] = delta
            .iter()
            .map(|x| {
                activations[activations.len() - 2].clone().apply(&|a| a * x)
            })
            .collect::<Vec<Vector<f64>>>();



        for l_id in 2..z_vecs.len() + 1 {
            let sig_prim = z_vecs[z_vecs.len() - l_id].clone().apply(&sigmoid_prime);
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
        }
        grad
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn sigmoids() {
        assert_eq![sigmoid(-2.0), 0.11920292202211755];
        assert_eq![sigmoid(2.0), 0.8807970779778823];
        assert_eq![sigmoid(0.2), 0.549833997312478];
        assert_eq![sigmoid(0.8), 0.6899744811276125];
        assert_eq![sigmoid_prime(-2.0), 0.1049935854035065];
        assert_eq![sigmoid_prime(2.0), 0.10499358540350662];
        assert_eq![sigmoid_prime(0.2), 0.24751657271185995];
        assert_eq![sigmoid_prime(0.8), 0.2139096965202944];
    }
    #[test]
    fn feed_forward() {
        let mut net = Network::new(vec![2, 2, 1], &mut XorShiftRng::from_seed([1, 2, 3, 4]));
        net.weights =
            NetStruct::from_vector(&vector![1.0, -1.0, 0.5, -0.5, 1.0, 0.1], &vec![2, 2, 1]);
        assert_eq![
            net.feed_forward(&vector![0.5, -0.5]),
            vector![0.6885404325558554]
        ];
    }
    #[test]
    fn back_propagation() {
        let mut net = Network::new(vec![2, 2, 1], &mut XorShiftRng::from_seed([1, 2, 3, 4]));
        net.weights =
            NetStruct::from_vector(&vector![1.0, -1.0, 0.5, -0.5, 1.0, 0.1], &vec![2, 2, 1]);
        assert_eq![
            net.back_propagation(&vector![0.5, -0.5], &vector![0.6], true)[0][0],
            vector![0.0018666059307424513 as f64, -0.0018666059307424513 as f64]
        ];
        assert_eq![
            net.back_propagation(&vector![0.5, -0.5], &vector![-0.6], true)
                .to_vector(),
            vector![
                0.02716495892306482,
                -0.02716495892306482,
                0.0032469372959591913,
                -0.0032469372959591913,
                0.20201394626893662,
                0.17200463760873333
            ]
        ];
    }
    #[test]
    fn hessian() {
        let mut net = Network::new(vec![2, 1], &mut XorShiftRng::from_seed([1, 2, 3, 4]));
        net.weights = NetStruct::from_vector(&vector![1.0, -1.0], &vec![2, 1]);
        assert_eq![
            net.back_propagation(&vector![0.5, -0.5], &vector![0.6], true)
                .to_vector(),
            vector![0.012883840256163013, -0.012883840256163013]
        ];
        assert_eq![
            net.feed_forward(&vector![0.5, -0.5]),
            vector![0.7310585786300049]
        ];
        assert_eq![
            net.back_propagation(&vector![0.5, -0.5], &vector![0.6], false)
                .to_vector(),
            vector![0.09830596662074093, -0.09830596662074093]
        ];
        assert_eq![
            net.get_hessian(&vector![1.0, -0.5], &vector![0.6]),
            Matrix::new(
                2,
                2,
                vec![
                    0.0222446641651681,
                    -0.01112233208258405,
                    -0.01112233208258405,
                    0.005561166041292025,
                ]
            )
        ];
    }
}
