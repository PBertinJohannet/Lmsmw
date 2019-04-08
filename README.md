# lmsmw

Neural network training using levemberg marquardt algorithm with an optimisation using sherman morisson woodburry formula

Implemented entirely in Rust.

---

Can also use basic gradient descent when the previous algorithm diverges/is to slow.


Usage example :



 ```rust
    // first create some tests
    let mut tests_array = vec![Test::new(vector![2.0,3.0,0.1], vector![0.1, 0.5]),
                               Test::new(vector![1.0,1.0,0.7], vector![0.5, 1.0])];


    // Create the network structure


    let mut layers = layers![3, 3, 2]; //  3 input layers, 3 hidden and 2 output layers.



    // Create and launch the trainer

    let net = Learner::new(ExamplesConfig::Ready(tests_array.clone()), layers)
        .gradient_descent_iters(20) // every time levemberg marquardt fails, run 20 iterations of GD
        .gradient_descent_step(0.1) // with learning rate 0.1
        .lvbm_nb_batches(2)    // run lvbm on 2 unit batches
        .aim_score(0.005)  // try to go until 0.005 score
        .max_iter(50)       // but if it diverges stop after 50 algorithm changes
        .start();           // start learning
    assert!(net.evaluate(&tests_array) < 0.005)
 ```

This algorithm gives a way of using Levemberg marquardt algorithm with a linear time complexity on the number of parameters in the network.
On the other hand, the time complexity on the length of the batch is polynomial.


