use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Debug;

pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub _prev: Vec<Rc<RefCell<Value>>>,
    _op: Op,
}

#[derive(Copy, Clone, Debug)]
enum Op {
    Add,
    Mul,
    Powf(f64),
    Tanh,
    None,
}

impl Value {
    pub fn new(data: f64) -> Value {
        Value {
            data,
            grad: 0.0,
            _prev: vec![],
            _op: Op::None,
        }
    }

    pub fn tanh(self) -> Value {
        let data: f64 = self.data.tanh();
        let grad: f64 = 0.0;
        let left = Rc::new(RefCell::new(self));
        let _prev: Vec<Rc<RefCell<Value>>> = vec![left];
        let _op: Op = Op::Tanh;

        Value {
            data,
            grad,
            _prev,
            _op,
        }
    }
}