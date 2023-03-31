use microrunn::engine::Value;
use microrunn::nn::MLP;

fn main() {
    let inputs = vec![
        vec![Value::new(0.0), Value::new(0.0)],
        vec![Value::new(0.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(0.0)],
        vec![Value::new(1.0), Value::new(1.0)],
    ];
    let targets = vec![
        Value::new(0.0),
        Value::new(1.0),
        Value::new(1.0),
        Value::new(0.0),
    ];
    let model: MLP = MLP::new(2, vec![3, 3, 1]);
    let mut loss: Value = model.loss(inputs, targets);
    loss = loss.backward();
    println!("{:?}", loss);
}
