use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Clone)]
pub struct Value {
    pub data: Rc<RefCell<f64>>,
    pub grad: Rc<RefCell<f64>>,
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
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(0.0)),
            _prev: vec![],
            _op: Op::None,
        }
    }

    pub fn tanh(self) -> Value {
        let data: Rc<RefCell<f64>> = Rc::new(RefCell::new(self.data.borrow().tanh()));
        let grad: Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));
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
        let data: Rc<RefCell<f64>> = Rc::new(RefCell::new(self.data.borrow().powf(n)));
        let grad: Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));
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

    pub fn backward(&self) {
        *self.grad.borrow_mut() = 1.0;

        fn recurse(root: &Value) {
            root._backward();
            for v in root._prev.iter() {
                recurse(&v.borrow());
            }
        }

        recurse(self);
    }

    fn _backward(&self) {
        match self._op {
            Op::Add => {
                *self._prev[0].borrow().grad.borrow_mut() = *self.grad.borrow();
                *self._prev[1].borrow().grad.borrow_mut() = *self.grad.borrow();
            }
            Op::Mul => {
                *self._prev[0].borrow().grad.borrow_mut() =
                    *self._prev[1].borrow().data.borrow() * *self.grad.borrow();
                *self._prev[1].borrow().grad.borrow_mut() =
                    *self._prev[0].borrow().data.borrow() * *self.grad.borrow();
            }
            Op::Powf(n) => {
                *self._prev[0].borrow().grad.borrow_mut() = (n
                    * (self._prev[0].borrow().data.borrow_mut().powf(n - 1.0)))
                    * *self.grad.borrow();
            }
            Op::Tanh => {
                *self._prev[0].borrow().grad.borrow_mut() = (1.0
                    * (self._prev[0].borrow().data.borrow_mut().powf(2.0)))
                    * *self.grad.borrow();
            }
            Op::None => {}
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        let data: Rc<RefCell<f64>> =
            Rc::new(RefCell::new(*self.data.borrow() + *other.data.borrow()));
        let grad: Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));
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
        let data: Rc<RefCell<f64>> =
            Rc::new(RefCell::new(*self.data.borrow() * *other.data.borrow()));
        let grad: Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));
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
            .field("data", &self.data.borrow())
            .field("grad", &self.grad.borrow())
            .field("_prev", &self._prev)
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
        assert!(
            (0.96402 + offset) > *result.data.borrow()
                && *result.data.borrow() > (0.96402 - offset)
        )
    }
    #[test]
    fn feed_forward() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = Value::new(10.0);
        let d = a * b;
        let e = d + c;
        let f = e.tanh();

        f.backward();

        println!("{:#?}", f);

        assert_ne!(0.0, *f.grad.borrow());
    }
}
