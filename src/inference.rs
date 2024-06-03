// https://burn.dev/book/basic-workflow/inference.html
use std::fs::File;
use std::sync::Arc;

use arrow::array::{make_array, Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, SchemaBuilder};
use arrow_csv::reader::Format;
use arrow_csv::writer::Writer;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::prelude::Tensor;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use num_traits::ToPrimitive;

use crate::data::{ClimSimBatcher, ClimSimDataSplit, ClimSimDataset};
use crate::training::TrainingConfig;

pub(crate) fn infer<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config
        .model
        .init::<B::InnerBackend>(&device)
        .load_record(record);

    // Get data from test.csv
    let batcher_test = ClimSimBatcher::<B::InnerBackend>::new(device);
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(ClimSimDataset::new(ClimSimDataSplit::Test).unwrap());

    // Obtain template CSV schema from sample_submission.csv file
    let mut sample_file =
        File::open("data/sample_submission.csv").expect("Error reading sample_submission.csv file");
    let (template_schema, _) = Format::default()
        .with_header(true)
        .infer_schema(&mut sample_file, Some(100))
        .expect("Failed to infer schema");
    // Change Int64 fields to Float64
    let mut builder = SchemaBuilder::new();
    for field in template_schema.fields.iter() {
        if field.data_type().is_integer() {
            let new_field = <Field as Clone>::clone(field).with_data_type(DataType::Float64);
            builder.push(new_field);
        } else {
            builder.push(<Field as Clone>::clone(field));
        }
    }
    let schema = builder.finish();

    // Create new submission.csv file for writing
    let mut file = File::create(format!("{artifact_dir}/submission.csv"))
        .expect("Failed to create new submission.csv file");
    let mut writer = Writer::new(&mut file);

    // Get predictions on each mini-batch
    for batch in dataloader_test.iter() {
        let predictions: Tensor<_, 2> = model.forward(batch.inputs);

        // println!("Predicted: {predictions}");
        // println!("Target {}", batch.targets);

        // Convert tensor to numerical columns
        let mut output_columns: Vec<ArrayRef> = predictions
            .iter_dim(1) // iterate over each column
            .map(|col| col.to_data().value)
            .map(|val| {
                val.into_iter()
                    .map(|e| e.to_f64().expect("conversion to f64 failed"))
                    .collect::<Vec<_>>()
            })
            .map(|vec| Float64Array::from(vec))
            .map(|arr| make_array(arr.to_data()))
            .collect();

        // Prepend index string column
        let index_col = make_array(StringArray::from(batch.indexes).to_data());
        output_columns.insert(0, index_col);

        // Write record batch to csv file
        // https://docs.rs/arrow-csv/51.0.0/arrow_csv/writer/index.html
        let rec_batch = RecordBatch::try_new(Arc::new(schema.clone()), output_columns)
            .expect("Failed to make record batch");
        // dbg!(&rec_batch);
        writer
            .write(&rec_batch.clone())
            .expect("Failed writing record batch");
    }
    println!(
        "Finished writing predictions to {}/submission.csv file!",
        artifact_dir
    )
}
