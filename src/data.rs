// https://burn.dev/book/basic-workflow/data.html
// https://github.com/tracel-ai/burn/blob/v0.13.2/crates/burn-dataset/src/vision/mnist.rs
use std::fs::File;
use std::io::Seek;
use std::sync::{Arc, RwLock};

use arrow::array::Float64Array;
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow_csv::reader::{Format, Reader, ReaderBuilder};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::{Backend, Tensor};

#[derive(Debug, Clone)]
pub(crate) struct ClimSimItem {
    /// Input as 1D array of floats.
    pub input: f64, //[[f64; WIDTH]; HEIGHT],
    /// Target value of climate variable.
    pub target: f64,
}

/// Reading ClimSim data from a CSV file into Arrow float arrays.
pub struct ClimSimDataset {
    dataset: RwLock<Reader<File>>,
}

impl ClimSimDataset {
    pub fn new() -> ArrowResult<Self> {
        let mut file =
            File::open("data/train.csv").map_err(|err| ArrowError::CsvError(err.to_string()))?;

        // Infer schema automatically
        let format = Format::default().with_header(true);
        let (schema, _) = format.infer_schema(&mut file, Some(10))?;
        file.rewind().unwrap();
        // println!("Schema of csv file {}", schema);

        let csv_reader = ReaderBuilder::new(Arc::new(schema))
            .with_header(true)
            .with_batch_size(1)
            .build(file)?;

        let dataset = Self {
            dataset: RwLock::new(csv_reader),
        };

        Ok(dataset)
    }
}

impl Dataset<ClimSimItem> for ClimSimDataset {
    fn get(&self, index: usize) -> Option<ClimSimItem> {
        // let item = self.dataset.nth(index);
        let mut rw_access = self.dataset.write().unwrap();
        let item = rw_access.nth(index);
        drop(rw_access);
        match item {
            Some(r) => {
                let rec_batch = r.unwrap();

                let input: f64 = rec_batch
                    .column_by_name("state_u_0") // TODO get more climate variables
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Failed to downcast")
                    .value(0);

                let target: f64 = rec_batch
                    .column_by_name("ptend_u_0") // TODO get more climate variables
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Failed to downcast")
                    .value(0);

                Some(ClimSimItem { input, target })
            }
            None => None,
        }
    }
    fn len(&self) -> usize {
        // let mut rw_access = self.dataset.write().unwrap();
        // let row_count = rw_access.count();
        // drop(rw_access);
        // row_count
        256
    }
}

/// Map ClimSim dataset items into batched tensors.
#[derive(Clone)]
pub struct ClimSimBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ClimSimBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ClimSimBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Batcher<ClimSimItem, ClimSimBatch<B>> for ClimSimBatcher<B> {
    fn batch(&self, items: Vec<ClimSimItem>) -> ClimSimBatch<B> {
        let inputs: Vec<_> = items
            .iter()
            .map(|item| Tensor::<B, 2>::from_floats([[item.input as f32]], &self.device))
            .collect();

        let targets: Vec<_> = items
            .iter()
            .map(|item| Tensor::<B, 2>::from_floats([[item.target as f32]], &self.device))
            .collect();

        let inputs = Tensor::cat(inputs, 0);
        let targets = Tensor::cat(targets, 0);

        ClimSimBatch { inputs, targets }
    }
}
