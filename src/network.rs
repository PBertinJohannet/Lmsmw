//! # Network
//! The module containing the
//! to use a network call the feed_forward function on the input.

use rulinalg::vector::Vector;
use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;
use rand::prelude::ThreadRng;
use crate::example::Test;
use crate::netstruct::NetStruct;
use crate::netstruct::NetStructTrait;


#[derive(Debug, Clone, Copy)]
/// Configuration for a layer
pub struct LayerConfig {
    /// the number of neurons in the layer
    nb_neurs: usize,
    /// the evaluation function of the layer.
    eval_fun: EvalFunc,
}

impl LayerConfig {
    /// creates a layer config with the desired number of neurons
    /// Default activation function is sigmoid.
    pub fn new(nb_neurs: usize) -> Self {
        LayerConfig {
            nb_neurs: nb_neurs,
            eval_fun: EvalFunc::Sigmoid,
        }
    }
    /// Sets the evaluation function for the given layer
    pub fn eval_function(&mut self, func: EvalFunc) -> &mut Self {
        self.eval_fun = func;
        self
    }
    fn get_num(&self) -> usize {
        self.nb_neurs
    }
}


#[derive(Debug, Clone, Copy)]
/// The possible activation functions for the layers.
pub enum EvalFunc {
    /// The sigmoid function :
    /// $$
    /// \frac{1,{1 + e^{-x}
    /// $$
    Sigmoid,
    /// The Identity function :
    /// $$
    /// x
    /// $$
    Identity,
}

fn get_eval(func: &EvalFunc) -> (fn(f64) -> f64) {
    match func {
        &EvalFunc::Sigmoid => |x| 1.0 / (1.0 + (-x).exp()),
        &EvalFunc::Identity => |x| x,
    }
}


fn get_prime(func: &EvalFunc) -> (fn(f64) -> f64) {
    match func {
        &EvalFunc::Sigmoid => |x| {
            let a = 1.0 / (1.0 + (-x).exp());
            a * (1.0 - a)
        },
        &EvalFunc::Identity => |_| 1.0,
    }
}

#[derive(Debug, Clone)]
/// The neural network structure
pub struct Network {
    /// number of layers
    num_layers: usize,
    /// comfigurations of layers
    layers: Vec<LayerConfig>,
    /// the weights of the network
    weights: NetStruct,
    /// the number of coeficient in the network.
    nb_coef: usize,
}


impl Network {
    /// Creates a new random network with the given randomgenerator and structure.
    pub fn new(structure: Vec<LayerConfig>, my_rand: &mut ThreadRng) -> Self {
        Network {
            num_layers: structure.len(),
            layers: structure.clone(),
            weights: NetStructTrait::random(
                &structure.iter().map(|x| x.get_num()).collect(),
                my_rand,
            ),
            nb_coef: (structure.iter().zip(structure.iter().skip(1)).map(
                |(ref prev, ref next)| next.get_num() * prev.get_num(),
            )).sum(),
        }
    }
    /// Returns the structure of the network's layers
    pub fn layers(&self) -> Vec<LayerConfig>{
        self.layers.clone()
    }
    /// sets the weights of the network.
    pub fn set_weights(&mut self, weights: NetStruct) {
        self.weights = weights;
    }
    /// Returns the weights of the network.
    pub fn get_weights(&self) -> &NetStruct {
        &self.weights
    }
    /// Feed forward the input trough the network and returns the output.
    pub fn feed_forward(&self, input: &Vector<f64>) -> Vector<f64> {
        self.weights.iter().enumerate().fold(input.clone(), |prev,
         (nb, layer)| {
            Vector::from(
                layer
                    .iter()
                    .map(|neur| {
                        get_eval(&self.layers[nb].eval_fun)(*&neur.elemul(&prev).sum())
                    })
                    .collect::<Vec<f64>>(),
            )
        })
    }
    /// evaluate the network on all tests.
    pub fn evaluate(&self, tests: &Vec<Test>) -> f64 {
        (tests
             .iter()
             .map(|test| {
            (&test.outputs - self.feed_forward(&test.inputs))
                .apply(&|x| x.abs())
                .sum()
        })
             .sum::<f64>()) / (tests.len()) as f64
    }
    /// returns the derivative of the score given the desired and actual outputs.
    pub fn cost_derivative(&self, output: &Vector<f64>, desired: &Vector<f64>) -> Vector<f64> {
        desired - output
    }
    /// returns the structure of the layers.
    pub fn get_layers_structure(&self) -> Vec<usize> {
        self.layers.iter().map(|x| x.get_num()).collect()
    }
    /// Returns a gradient of zeroes.
    pub fn get_empty_grad(&self) -> NetStruct {
        (self.layers.iter().zip(self.layers.iter().skip(1)).map(
            |(ref prev, ref next)| {
                (0..next.get_num())
                    .map(|_| Vector::zeros(prev.get_num()))
                    .collect::<Vec<Vector<f64>>>()
            },
        )).collect::<Vec<_>>()
    }
    /// Compute the global gradient of the score for all the tests.
    pub fn global_gradient(&self, tests: &[Test], from_cost: bool) -> NetStruct {
        tests
            .iter()
            .fold(self.get_empty_grad(), |grad, test| {
                grad.add(&self.back_propagation(
                    &test.inputs,
                    &test.outputs,
                    from_cost,
                ))
            })
            .apply(&|x| x)
    }
    /// add the given gradient to the network.
    pub fn add_gradient(&mut self, grad: &NetStruct, coef: f64) {
        self.weights = self.weights.add(&grad.apply(&|a| a * coef));
    }
    /// Compute the global hessian of the output for all the tests.
    pub fn global_hessian(&self, tests: &[Test]) -> Matrix<f64> {
        let mut hessian = self.get_hessian(&tests[0].inputs, &tests[0].outputs);
        for i in 1..tests.len() {
            hessian += self.get_hessian(&tests[i].inputs, &tests[0].outputs);
        }
        hessian
    }
    /// Return the hessian of the output for the given input.
    pub fn get_hessian(&self, input: &Vector<f64>, output: &Vector<f64>) -> Matrix<f64> {
        let grad = self.back_propagation(input, output, false).to_vector();
        Matrix::from_fn(grad.size(), grad.size(), |col, row| grad[col] * grad[row])
    }
    /// Returns all the activations and z values of the vectors for the given input.
    pub fn get_fed_layers(&self, input: &Vector<f64>) -> (Vec<Vector<f64>>, Vec<Vector<f64>>) {
        let mut activations = vec![input.clone()];
        let mut z_vecs = vec![];
        for (layer_id, ref layer) in self.weights.iter().enumerate() {
            z_vecs.push(Vector::from(
                layer
                    .iter()
                    .map(|neur| *&neur.elemul(&activations.last().unwrap()).sum())
                    .collect::<Vec<f64>>(),
            ));
            activations.push(z_vecs.last().unwrap().clone().apply(&get_eval(
                &self.layers[layer_id].eval_fun,
            )));
        }
        (activations, z_vecs)
    }
    /// The backpropagation algorithm.
    /// If from_cost is at true, computes the gradient of the score
    /// Else compute the gradient of the output.
    pub fn back_propagation(
        &self,
        input: &Vector<f64>,
        output: &Vector<f64>,
        from_cost: bool,
    ) -> NetStruct {
        // feed forward

        let mut grad = self.get_empty_grad();
        let (activations, z_vecs) = self.get_fed_layers(input);


        // output derivatives

        let mut delta = z_vecs.last().unwrap().clone().apply(&get_prime(
            &self.layers
                .last()
                .unwrap()
                .eval_fun,
        ));
        if from_cost {
            delta = delta.elemul(&self.cost_derivative(activations.last().unwrap(), &output));
        }
        grad[self.weights.len() - 1] = delta
            .iter()
            .map(|x| {
                activations[activations.len() - 2].clone().apply(&|a| a * x)
            })
            .collect::<Vec<Vector<f64>>>();

        // back pass

        for l_id in 2..z_vecs.len() + 1 {
            let sig_prim = z_vecs[z_vecs.len() - l_id].clone().apply(&get_prime(
                &self.layers[self.layers.len() - l_id +
                                 1]
                    .eval_fun,
            ));
            delta = (&Matrix::new(
                self.layers[self.layers.len() - l_id + 1].get_num(),
                self.layers[self.layers.len() - l_id].get_num(),
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
                            .map(&|a| (d * a))
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
    use rand::thread_rng;

    #[test]
    fn sigmoids() {
        let sigmoid = get_eval(&EvalFunc::Sigmoid);
        let sigmoid_prime = get_prime(&EvalFunc::Sigmoid);
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
        let mut net = Network::new(layers![2, 2, 1], &mut thread_rng());
        net.weights =
            NetStruct::from_vector(&vector![1.0, -1.0, 0.5, -0.5, 1.0, 0.1], &vec![2, 2, 1]);
        assert_eq![
            net.feed_forward(&vector![0.5, -0.5]),
            vector![0.6885404325558554],
        ];
    }

    #[test]
    fn back_propagation() {
        let mut layers = layers![2, 1];
        for l in 0..layers.len() {
            layers[l].eval_function(EvalFunc::Identity);
        }
        let mut net = Network::new(layers, &mut thread_rng());
        net.weights = NetStruct::from_vector(&vector![1.0, 1.0], &vec![2, 1]);
        assert_eq![net.feed_forward(&vector![1.0, 3.0]), vector![4.0]];
    }

    #[test]
    fn hessian() {
        let mut net = Network::new(layers![2, 1], &mut thread_rng());
        net.weights = NetStruct::from_vector(&vector![1.0, -1.0], &vec![2, 1]);
        assert_eq![
            net.back_propagation(&vector![0.5, -0.5], &vector![0.6], true)
                .to_vector(),
            vector![-0.012883840256163013, 0.012883840256163013],
        ];
        assert_eq![
            net.feed_forward(&vector![0.5, -0.5]),
            vector![0.7310585786300049],
        ];
        assert_eq![
            net.back_propagation(&vector![0.5, -0.5], &vector![0.6], false)
                .to_vector(),
            vector![0.09830596662074093, -0.09830596662074093],
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
            ),
        ];
        let mut layers = layers![2, 1];
        for l in 0..layers.len() {
            layers[l].eval_function(EvalFunc::Identity);
        }
        let mut net = Network::new(layers, &mut thread_rng());
        net.weights = NetStruct::from_vector(&vector![1.0, 1.0], &vec![2, 1]);
        assert_eq![
            net.get_hessian(&vector![1.0, 3.0], &vector![-0.3]),
            Matrix::new(2, 2, vec![1.0, 3.0, 3.0, 9.0]),
        ];
    }
}
