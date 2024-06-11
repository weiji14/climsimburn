// https://burn.dev/book/basic-workflow/inference.html
use std::fs::File;
use std::io::Seek;
use std::sync::Arc;

use arrow::array::{
    make_array, Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, SchemaBuilder};
use arrow_csv::reader::{Format, ReaderBuilder};
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
        .batch_size(625)
        .num_workers(1)
        .build(ClimSimDataset::new(ClimSimDataSplit::Test).unwrap());

    // Obtain template CSV schema from sample_submission.csv file
    let mut sample_file =
        File::open("data/sample_submission.csv").expect("Error opening sample_submission.csv file");
    let (sample_schema, _) = Format::default()
        .with_header(true)
        .infer_schema(&mut sample_file, Some(100))
        .expect("Failed to infer schema");
    sample_file.rewind().unwrap();
    // Read sample_submission.csv rows (to multiply with predictions later)
    let mut sample_reader = ReaderBuilder::new(Arc::new(sample_schema.clone()))
        .with_header(true)
        .with_batch_size(625)
        .build(sample_file)
        .expect("Error reading sample_submission.csv file");
    // Change Int64 fields to Float64 for the writer schema
    let mut builder = SchemaBuilder::new();
    for field in sample_schema.fields.iter() {
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
    for (i, batch) in dataloader_test.iter().enumerate() {
        let predictions: Tensor<_, 2> = model.forward(batch.inputs);
        // println!("Predicted: {predictions}");
        // println!("Target {}", batch.targets);

        // Convert predicted tensor to numerical values
        let mut pred_values: Vec<Vec<f64>> = predictions
            .iter_dim(1) // iterate over each column
            .map(|col| col.to_data().value)
            .map(|val| {
                val.into_iter()
                    .map(|e| e.to_f64().expect("conversion to f64 failed"))
                    .collect::<Vec<_>>()
            })
            .collect();

        // Prepare weight values from sample_submission.csv (to multiply against pred values later)
        let row_group = sample_reader.nth(0).expect("should have some rows");
        let mut sample_batch: RecordBatch = row_group.unwrap();
        let _ = sample_batch.remove_column(0); //  drop first string index column
        let weight_values: Vec<Vec<f64>> = sample_batch
            .columns()
            .iter()
            // Cast Int64 columns to Float64
            .map(|arr| {
                let dtype = arr.data_type();
                match dtype {
                    DataType::Float64 => arr
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .unwrap()
                        .values()
                        .to_vec(),
                    DataType::Int64 => arr
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values()
                        .into_iter()
                        .map(|v| v.to_f64().unwrap())
                        .collect(),
                    _ => panic!("Conversion for {dtype} not implemented"),
                }
            })
            .collect();

        // Multiply predicted values with weight values to get output columns
        for (pred, weight) in pred_values.iter_mut().zip(weight_values.iter()) {
            for (p, w) in pred.iter_mut().zip(weight.iter()) {
                // println!("Old (raw value): {} * weight {}", &p, &w);
                *p = *p * w;
                // println!("New (weighted): {}", p);
            }
        }

        let mut output_columns: Vec<ArrayRef> = pred_values
            .iter()
            .map(|vec| Float64Array::from(vec.clone()))
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
