use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
use std::rc::Rc;

pub struct Value {
    pub data: f64,
    pub grad: f64,
    _prev: Vec<Rc<RefCell<Value>>>,
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
        let left = Rc::new(RefCell::new(self.clone()));
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
        let data: f64 = self.data.powf(n);
        let grad: f64 = 0.0;
        let left = Rc::new(RefCell::new(self.clone()));
        let _prev: Vec<Rc<RefCell<Value>>> = vec![left];
        let _op: Op = Op::Powf(n);

        Value {
            data,
            grad,
            _prev,
            _op,
        }
    }

    pub fn backward(self) -> Value {
        let mut out = self.clone();
        out.grad = 1.0;

        fn build_grads(root: &Value) -> Value {
            let mut result = root.clone()._backward();
            let mut temp_prev: Vec<Rc<RefCell<Value>>> = vec![];

            for v in result._prev.iter() {
                temp_prev.push(Rc::new(RefCell::new(build_grads(&v.borrow().clone()))));
            }

            result._prev = temp_prev;

            result
        }

        out = build_grads(&out);

        out
    }

    fn _backward(self) -> Value {
        let _prev: Vec<Rc<RefCell<Value>>> = match self._op {
            Op::Add => {
                let left = &*self._prev[0].borrow();
                let right = &*self._prev[1].borrow();

                let left_grad = self.grad;
                let right_grad = self.grad;

                vec![
                    Rc::new(RefCell::new(Value {
                        data: left.data,
                        grad: left_grad,
                        _prev: left._prev.clone(),
                        _op: left._op,
                    })),
                    Rc::new(RefCell::new(Value {
                        data: right.data,
                        grad: right_grad,
                        _prev: right._prev.clone(),
                        _op: right._op,
                    })),
                ]
            }
            Op::Mul => {
                let left = &*self._prev[0].borrow();
                let right = &*self._prev[1].borrow();

                let left_grad = right.data * self.grad;
                let right_grad = left.data * self.grad;

                vec![
                    Rc::new(RefCell::new(Value {
                        data: left.data,
                        grad: left_grad,
                        _prev: left._prev.clone(),
                        _op: left._op,
                    })),
                    Rc::new(RefCell::new(Value {
                        data: right.data,
                        grad: right_grad,
                        _prev: right._prev.clone(),
                        _op: right._op,
                    })),
                ]
            }
            Op::Powf(n) => {
                let left = &*self._prev[0].borrow();

                let left_grad = (n * left.data.powf(n - 1.0)) * self.grad;

                vec![Rc::new(RefCell::new(Value {
                    data: left.data,
                    grad: left_grad,
                    _prev: left._prev.clone(),
                    _op: left._op,
                }))]
            }
            Op::Tanh => {
                let left = &*self._prev[0].borrow();

                let left_grad = (1.0 - left.data.powf(2.0)) * self.grad;

                vec![Rc::new(RefCell::new(Value {
                    data: left.data,
                    grad: left_grad,
                    _prev: left._prev.clone(),
                    _op: left._op,
                }))]
            }
            Op::None => {
                vec![]
            }
        };

        Value {
            data: self.data,
            grad: self.grad,
            _prev,
            _op: self._op,
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        let data: f64 = self.data + other.data;
        let grad: f64 = 0.0;
        let left = Rc::new(RefCell::new(self.clone()));
        let right = Rc::new(RefCell::new(other.clone()));
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

impl Add for &Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        let data: f64 = self.data + other.data;
        let grad: f64 = 0.0;
        let left = Rc::new(RefCell::new(self.clone()));
        let right = Rc::new(RefCell::new(other.clone()));
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
        let data: f64 = self.data * other.data;
        let grad: f64 = 0.0;
        let left = Rc::new(RefCell::new(self.clone()));
        let right = Rc::new(RefCell::new(other.clone()));
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

impl Mul for &Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        let data: f64 = self.data * other.data;
        let grad: f64 = 0.0;
        let left = Rc::new(RefCell::new(self.clone()));
        let right = Rc::new(RefCell::new(other.clone()));
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

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Self) -> Self::Output {
        self + -other
    }
}

impl Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Value::new(-1.0)
    }
}

impl Clone for Value {
    fn clone(&self) -> Value {
        Value {
            data: self.data,
            grad: self.grad,
            _prev: self._prev.clone(),
            _op: self._op.clone(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_two_values() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let result = a + b;
        assert_eq!(result.data, -1.0);
    }
    #[test]
    fn multiply_two_values() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let result = a * b;
        assert_eq!(result.data, -6.0);
    }
    #[test]
    fn multiply_two_reference_values() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let result = &a * &b;
        assert_eq!(result.data, -6.0);
    }
    #[test]
    fn tanh_one_value() {
        let a = Value::new(2.0);
        let result = a.tanh();
        let offset = 0.000009;
        assert!((0.96402 + offset) > result.data && result.data > (0.96402 - offset))
    }
    #[test]
    fn feed_forward() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = Value::new(10.0);
        let d = a * b;
        let e = d + c;
        let mut f = e.tanh();

        f.grad = 1.0;
        let f_back = f.backward();

        assert_ne!(0.0, f_back.grad);
    }
}
