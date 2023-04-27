use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};
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

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Self) -> Self::Output {
        self + -other
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

impl Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Value::new(-1.0)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .finish()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        (*self.data.borrow(), *self.grad.borrow()) == (*other.data.borrow(), *other.grad.borrow())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_two_values() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let result = a + b;
        assert_eq!(result, Value::new(-1.0));
    }
    #[test]
    fn multiply_two_values() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let result = a * b;
        assert_eq!(result, Value::new(-6.0));
    }
    #[test]
    fn multiply_two_reference_values() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let result = a * b;
        assert_eq!(result, Value::new(-6.0));
    }
    #[test]
    fn tanh_one_value() {
        let a = Value::new(2.0);
        let result = a.tanh();
        let offset = 0.000009;
        assert!((0.96402 + offset) > *result.data.borrow() && *result.data.borrow() > (0.96402 - offset))
    }
    // #[test]
    // fn feed_forward() {
    //     let a = Value::new(2.0);
    //     let b = Value::new(-3.0);
    //     let c = Value::new(10.0);
    //     let d = a * b;
    //     let e = d + c;
    //     let mut f = e.tanh();

    //     f.grad = 1.0;
    //     let f_back = f.backward();

    //     assert_ne!(0.0, f_back.grad);
    // }
}
