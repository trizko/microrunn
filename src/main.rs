use microrunn::engine::Value;
use microrunn::nn::MLP;

fn main() {
    let x = vec![Value::new(0.0), Value::new(1.0)];
    let y = vec![Value::new(1.0)];
    let model: MLP = MLP::new(2, vec![3, 3, 1]);

    let out = model.call(&x);

    print!("actual: {:#?} | expected: {:#?}", out[0].data, y[0].data);
}
