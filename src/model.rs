// https://burn.dev/book/basic-workflow/model.html
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Config, Module, Tensor};
use burn::tensor::activation::relu;

#[derive(Module, Debug)]
pub struct ClimSimModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct ClimSimModelConfig {
    hidden_size: usize,
}

impl ClimSimModelConfig {
    // Returns the initialized model
    pub fn init<B: Backend>(&self, device: &B::Device) -> ClimSimModel<B> {
        ClimSimModel {
            linear1: LinearConfig::new(556, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size / 2).init(device),
            linear3: LinearConfig::new(self.hidden_size / 2, 368).init(device),
            norm1: LayerNormConfig::new(self.hidden_size).init(device),
            norm2: LayerNormConfig::new(self.hidden_size / 2).init(device),
        }
    }
}

impl<B: Backend> ClimSimModel<B> {
    /// # Shapes
    ///   - Images [batch_size, climate_variables]
    ///   - Output [batch_size, climate_variables]
    pub fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(inputs);
        let x = self.norm1.forward(x);
        let x = relu(x);

        let x = self.linear2.forward(x);
        let x = self.norm2.forward(x);
        let x = relu(x);

        self.linear3.forward(x) // [batch_size, num_classes]
    }
}
