use rulinalg::vector::Vector;
use rand::XorShiftRng;
use rand::Rng;
use std::iter::FromIterator;


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
                    Vector::from(my_rand.gen_iter().take(prev).collect::<Vec<f64>>())
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



#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn random() {
        let mut my_rand = XorShiftRng::from_seed([1,2,3,4]);
        assert_eq![NetStruct::random(&vec![3,2], &mut my_rand),
                    vec![vec![vector![0.0019655227674844067,
                                      0.0000038146990917198313,
                                      0.00007343478686872018],
                              vector![0.00013542734277605107,
                                      0.1278181086308321,
                                      0.15870571251203636]]]];
        let s = NetStruct::random(&vec![3,4,5,4,18,2], &mut my_rand);
        assert_eq![s[2][0].size(), 5];
        assert_eq![s[4][0].size(), 18];
    }
    #[test]
    fn to_vector(){
        let s = NetStruct::from_vector(&vector![1.0,2.0,3.0], &vec![2,1,1]);
        assert_eq![s, vec![vec![vector![1.0,2.0]],vec![vector![3.0]]]];
        assert_eq![s.to_vector(), vector![1.0,2.0,3.0]];
    }
    #[test]
    fn add(){
        let s = NetStruct::from_vector(&vector![1.0,2.0,3.0], &vec![2,1,1]);
        assert_eq![s.add(&NetStruct::from_vector(&vector![3.0,2.0,1.0], &vec![2,1,1])).to_vector(),
                    vector![4.0,4.0,4.0]];
    }
    #[test]
    fn apply(){
        let s = NetStruct::from_vector(&vector![1.0,2.0,3.0], &vec![2,1,1]);
        assert_eq![s.apply(&|x|(x*5.0).sin()).to_vector(),vector![-0.9589242746631385,
                                                                  -0.5440211108893698,
                                                                   0.6502878401571168]];
        assert_eq![s.apply(&|x|x*2.0).to_vector(),vector![2.0,4.0,6.0]];
    }
}

