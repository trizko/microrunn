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
            weights: vec![Rc::new(RefCell::new(Value::new(generator.sample(&mut rng)))); nin],
            bias: Rc::new(RefCell::new(Value::new(generator.sample(&mut rng)))),
            non_lin,
        }
    }

    fn call(&self, x: &[Value]) -> Value {
        unimplemented!()
    }

    fn _parameters(&self) -> Vec<Value> {
        unimplemented!()
    }
}
