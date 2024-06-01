// https://burn.dev/book/basic-workflow/model.html
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::{Backend, Config, Module, Tensor};

#[derive(Module, Debug)]
pub struct ClimSimModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
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
            linear2: LinearConfig::new(self.hidden_size, 368).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> ClimSimModel<B> {
    /// # Shapes
    ///   - Images [batch_size, climate_variables]
    ///   - Output [batch_size, climate_variables]
    pub fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(inputs);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
