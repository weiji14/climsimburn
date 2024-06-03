use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::AdamConfig;
use burn::prelude::{Backend, Config, Module, Tensor};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{Adaptor, CpuUse, LossInput, LossMetric};
use burn::train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use derive_new::new;

use crate::data::{ClimSimBatch, ClimSimBatcher, ClimSimDataSplit, ClimSimDataset};
use crate::model::{ClimSimModel, ClimSimModelConfig};

/// Regression output adapted for multiple climate variables in the ClimSim dataset..
#[derive(new)]
pub struct ClimSimRegressionOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,
    /// The output [batch_size, num_classes]
    pub output: Tensor<B, 2>,
    /// The targets [batch_size, num_classes]
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Adaptor<LossInput<B>> for ClimSimRegressionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

/// Forward pass to get loss value
impl<B: Backend> ClimSimModel<B> {
    pub fn forward_regression(
        &self,
        inputs: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> ClimSimRegressionOutput<B> {
        let output = self.forward(inputs);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Reduction::Mean);

        ClimSimRegressionOutput::new(loss, output, targets)
    }
}

/// Training step
impl<B: AutodiffBackend> TrainStep<ClimSimBatch<B>, ClimSimRegressionOutput<B>>
    for ClimSimModel<B>
{
    fn step(&self, batch: ClimSimBatch<B>) -> TrainOutput<ClimSimRegressionOutput<B>> {
        let item = self.forward_regression(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

/// Validation step
impl<B: Backend> ValidStep<ClimSimBatch<B>, ClimSimRegressionOutput<B>> for ClimSimModel<B> {
    fn step(&self, batch: ClimSimBatch<B>) -> ClimSimRegressionOutput<B> {
        self.forward_regression(batch.inputs, batch.targets)
    }
}

/// Hyperparameters for the ClimSimModel
#[derive(Config)]
pub struct TrainingConfig {
    pub model: ClimSimModelConfig,
    pub optimizer: AdamConfig,

    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    // Setup dataloaders
    let batcher_train = ClimSimBatcher::<B>::new(device.clone());
    let batcher_valid = ClimSimBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ClimSimDataset::new(ClimSimDataSplit::Train).unwrap());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ClimSimDataset::new(ClimSimDataSplit::Valid).unwrap());

    // Setup learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    // Start training
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
