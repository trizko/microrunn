#![allow(dead_code)]

use crate::engine::v1::Value;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::rc::Rc;

trait Module {
    fn zero_grad(&self) -> Value;
    fn parameters(&self) -> Vec<Value>;
}

#[derive(Debug)]
struct Neuron {
    weights: Vec<Rc<RefCell<Value>>>,
    bias: Rc<RefCell<Value>>,
    non_lin: bool,
}

impl Neuron {
    fn new(nin: usize, non_lin: bool) -> Neuron {
        let seed = 42; // Choose a seed value
        let mut rng = StdRng::seed_from_u64(seed);
        let generator = Uniform::from(0.01..=1.00);

        Neuron {
            weights: {
                let mut v = Vec::with_capacity(nin);
                (0..nin).for_each(|_| {
                    v.push(Rc::new(RefCell::new(Value::new(
                        generator.sample(&mut rng),
                    ))))
                });
                v
            },
            bias: Rc::new(RefCell::new(Value::new(generator.sample(&mut rng)))),
            non_lin,
        }
    }

    fn call(&self, x: Vec<Rc<RefCell<Value>>>) -> Value {
        let act = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(x, y)| (Rc::clone(&x), Rc::clone(&y)))
            .map(move |(x, y)| (*x.borrow()).clone() * (*y.borrow()).clone())
            .fold((*self.bias.borrow()).clone(), |a, b| a + b);

        if self.non_lin {
            return act.tanh();
        }

        act
    }

    fn _parameters(self) -> Vec<Rc<RefCell<Value>>> {
        let mut result: Vec<Rc<RefCell<Value>>> =
            self.weights.into_iter().map(|v| Rc::clone(&v)).collect();
        result.push(Rc::clone(&self.bias));

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_neuron() {
        let n = Neuron::new(6, true);

        assert_eq!(6, n.weights.len());
    }
    #[test]
    fn create_output_from_neuron() {
        let seed = 42; // Choose a seed value
        let mut rng = StdRng::seed_from_u64(seed);
        let generator = Uniform::from(0.01..=1.00);
        let x: Vec<Rc<RefCell<Value>>> = {
            let mut v = Vec::with_capacity(3);
            (0..3).for_each(|_| {
                v.push(Rc::new(RefCell::new(Value::new(
                    generator.sample(&mut rng),
                ))))
            });
            v
        };

        let n = Neuron::new(3, true);
        let out = n.call(x);

        assert_eq!(3, n.weights.len());
        assert_eq!(0.0, *out.borrow().grad.borrow());
    }
}
