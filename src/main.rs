// https://burn.dev/book/basic-workflow/backend.html
mod data;
mod inference;
mod model;
mod training;

use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;

use crate::model::ClimSimModelConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let artifact_dir = "artifacts/v02";
    let device = burn::backend::wgpu::WgpuDevice::BestAvailable;

    // Train model
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        crate::training::TrainingConfig::new(ClimSimModelConfig::new(1024), AdamConfig::new()),
        device.clone(),
    );

    // Produce submission.csv file
    println!("Starting inference...");
    crate::inference::infer::<MyAutodiffBackend>(artifact_dir, device);
}
