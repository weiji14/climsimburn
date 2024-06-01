// https://burn.dev/book/basic-workflow/backend.html
mod data;
mod model;
mod training;

use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;

use crate::model::ClimSimModelConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::BestAvailable;
    crate::training::train::<MyAutodiffBackend>(
        "/tmp",
        crate::training::TrainingConfig::new(ClimSimModelConfig::new(256), AdamConfig::new()),
        device,
    );
}
