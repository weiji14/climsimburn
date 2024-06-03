// https://burn.dev/book/basic-workflow/data.html
// https://github.com/tracel-ai/burn/blob/v0.13.2/crates/burn-dataset/src/vision/mnist.rs
use std::fs::File;
use std::io::Seek;
use std::sync::{Arc, RwLock};

use arrow::array::cast::as_primitive_array;
use arrow::array::types::Float64Type;
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow_csv::reader::{Format, Reader, ReaderBuilder};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::{Backend, Data, Shape, Tensor};

#[derive(Debug, Clone)]
pub(crate) struct ClimSimItem {
    /// Input climate variables (555 columns).
    pub input: Vec<f64>, // [f64; 555]
    /// Target climate variables (368 columns).
    pub target: Vec<f64>, // [f64; 368]
}

pub enum ClimSimDataSplit {
    Train,
    Valid,
    Test,
}

/// Reading ClimSim data from a CSV file into Arrow float arrays.
pub struct ClimSimDataset {
    dataset: RwLock<Reader<File>>,
    split: ClimSimDataSplit,
}

impl ClimSimDataset {
    pub fn new(split: ClimSimDataSplit) -> ArrowResult<Self> {
        let filename = match split {
            ClimSimDataSplit::Train | ClimSimDataSplit::Valid => "data/train.csv",
            ClimSimDataSplit::Test => "data/test.csv",
        };

        let mut file = File::open(filename).map_err(|err| ArrowError::CsvError(err.to_string()))?;

        // Infer schema automatically
        let format = Format::default().with_header(true);
        let (schema, _) = format.infer_schema(&mut file, Some(10))?;
        file.rewind().unwrap();
        // println!("Schema of csv file {}", schema);

        let csv_reader = ReaderBuilder::new(Arc::new(schema))
            .with_header(true)
            .with_batch_size(1); // read one row at a time

        let csv_reader_subset = match split {
            ClimSimDataSplit::Train => csv_reader.with_bounds(0, 256000).build(file)?,
            ClimSimDataSplit::Valid => csv_reader.with_bounds(256000, 512000).build(file)?,
            ClimSimDataSplit::Test => csv_reader.build(file)?, // read all 625000 rows in test.csv
        };

        let dataset = Self {
            dataset: RwLock::new(csv_reader_subset),
            split,
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
                let mut rec_batch = r.unwrap();
                if let ClimSimDataSplit::Train = self.split {
                    assert_eq!(rec_batch.num_columns(), 925); // 1 index + 556 input + 368 target columns
                };

                let _ = rec_batch.remove_column(0); // drop first string index column
                if let ClimSimDataSplit::Train = self.split {
                    assert_eq!(rec_batch.num_columns(), 924); // 556 input + 368 target columns
                };

                // https://docs.rs/arrow-array/51.0.0/arrow_array/index.html#downcasting-an-array
                let row_buffer: Vec<f64> = rec_batch
                    .columns()
                    .iter()
                    .map(|arr| as_primitive_array::<Float64Type>(arr).values())
                    .map(|bar| bar.first().expect("should have one row").to_owned())
                    .collect();

                let input: Vec<f64> = row_buffer[0..556].to_vec(); // input has 556 columns
                let target: Vec<f64> = match self.split {
                    ClimSimDataSplit::Train | ClimSimDataSplit::Valid => {
                        row_buffer[556..924].to_vec() // target has 368 columns
                    }
                    ClimSimDataSplit::Test => vec![f64::NAN; 368], // NAN target values for test set
                };

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
        match self.split {
            ClimSimDataSplit::Train | ClimSimDataSplit::Valid => 256000,
            ClimSimDataSplit::Test => 625000,
        }
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
            .map(|item| Data::new(item.input.clone(), Shape::new([1, 556])))
            .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
            .collect();

        let targets: Vec<_> = items
            .iter()
            .map(|item| Data::new(item.target.clone(), Shape::new([1, 368])))
            .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
            .collect();

        let inputs = Tensor::cat(inputs, 0);
        let targets = Tensor::cat(targets, 0);

        ClimSimBatch { inputs, targets }
    }
}
