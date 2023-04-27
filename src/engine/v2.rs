use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::rc::Rc;

pub struct Value {
    pub data: RefCell<f64>,
    pub grad: RefCell<f64>,
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
            data: RefCell::new(data),
            grad: RefCell::new(0.0),
            _prev: vec![],
            _op: Op::None,
        }
    }

    pub fn tanh(self) -> Value {
        let data: RefCell<f64> = RefCell::new(self.data.borrow().tanh());
        let grad: RefCell<f64> = RefCell::new(0.0);
        let left: Rc<RefCell<Value>> = Rc::new(RefCell::new(self));
        let _prev: Vec<Rc<RefCell<Value>>> = vec![left];
        let _op: Op = Op::Tanh;

        Value {
            data,
            grad,
            _prev,
            _op,
        }
    }

    pub fn powf(self, n: f64) -> Value {
        let data: RefCell<f64> = RefCell::new(self.data.borrow().powf(n));
        let grad: RefCell<f64> = RefCell::new(0.0);
        let left = Rc::new(RefCell::new(self));
        let _prev: Vec<Rc<RefCell<Value>>> = vec![left];
        let _op: Op = Op::Powf(n);

        Value {
            data,
            grad,
            _prev,
            _op,
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        let data: RefCell<f64> = RefCell::new(*self.data.borrow() + *other.data.borrow());
        let grad: RefCell<f64> = RefCell::new(0.0);
        let left = Rc::new(RefCell::new(self));
        let right = Rc::new(RefCell::new(other));
        let _prev: Vec<Rc<RefCell<Value>>> = vec![left, right];
        let _op: Op = Op::Add;

        Value {
            data,
            grad,
            _prev,
            _op,
        }
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        let data: RefCell<f64> = RefCell::new(*self.data.borrow() * *other.data.borrow());
        let grad: RefCell<f64> = RefCell::new(0.0);
        let left = Rc::new(RefCell::new(self));
        let right = Rc::new(RefCell::new(other));
        let _prev: Vec<Rc<RefCell<Value>>> = vec![left, right];
        let _op: Op = Op::Mul;

        Value {
            data,
            grad,
            _prev,
            _op,
        }
    }
}