use rand::Rng;

fn generate_data(n: usize) -> Vec<(f64, f64)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();

    for _ in 0..n {
        let x: f64 = rng.gen_range(-10.0..10.0);
        let y = 2.0 * x + 1.0 + rng.gen_range(-1.0..1.0); // y = 2x + 1 + noise
        data.push((x, y));
    }

    data
}

use burn::{tensor::Tensor, nn::{Module, Linear}};
use burn::tensor::Tensor;

#[derive(Debug)]
struct LinearRegression {
    weight: Tensor,
    bias: Tensor,
}

impl LinearRegression {
    fn forward(&self, x: Tensor) -> Tensor {
        &x * &self.weight + &self.bias
    }
}

impl Module for LinearRegression {
    fn forward(&self, x: Tensor) -> Tensor {
        self.forward(x)
    }
}

fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let diff = predictions - targets;
    let squared_diff = diff * diff;
    squared_diff.mean()
}

fn train_model(model: &mut LinearRegression, data: Vec<(f64, f64)>, epochs: usize, learning_rate: f64) {
    for _ in 0..epochs {
        let mut total_loss = 0.0;
        
        for (x_val, y_val) in data.iter() {
            let x = Tensor::from(*x_val);
            let y = Tensor::from(*y_val);
            let prediction = model.forward(x);
            let loss = mean_squared_error(&prediction, &y);
            total_loss += loss.data()[0];
            
            model.weight -= learning_rate * loss.grad().unwrap();
            model.bias -= learning_rate * loss.grad().unwrap();
        }

        println!("Epoch loss: {}", total_loss);
    }
}

use textplots::plot;

fn plot_results(predictions: Vec<f64>, actual: Vec<f64>) {
    let data = predictions.iter().zip(actual.iter()).map(|(p, a)| (*p, *a)).collect::<Vec<_>>();
    plot::plot(&data);
}
