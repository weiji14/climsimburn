# ClimSimBurn

A machine learning model implemented in Rust 🦀 using the [Burn](https://burn.dev) 🔥
deep learning framework for the LEAP - Atmospheric Physics using AI (ClimSim) challenge
on [Kaggle](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim).

## Getting started

### Dataset

Download the `train.csv`, `test.csv` and `sample_submission.csv` files from
https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data.

> [!IMPORTANT]
> The `train.csv` file is very big (181.72GB). I'd personally recommend clicking on the
> 'Download all' button to get the compressed `leap-atmospheric-physics-ai-climsim.zip`
> version which is only 77.8GB :slightly_smiling_face:, and you can partially decompress
> the `train.csv` to get a smaller subset with less rows for quick experimentation.

### Installation

Compile the project (including all dependencies) in dev mode.

    cargo build


## Usage

### Running the model

The neural network model can be ran by calling `src/main.rs` like so:

    cargo run --release

By default, the model will be trained using the `Wgpu`
[backend](https://burn.dev/book/basic-workflow/backend.html). The training should show
up as a Terminal User Interface (TUI) dashboard:

![ClimSim Burn model training progression visualized on a Terminal User Interface](https://github.com/weiji14/climsimburn/assets/23487320/99d027f6-5f76-4d56-bf07-5d2912c25baf)

Logs will be saved to `/artifacts/vXX/experiment.log` by default. Hyperparameters can be
adjusted by modifying the default values in the `TrainingConfig` struct in
`src/training.rs`. After training, the inference script will be ran automatically, and
the results saved to `artifacts/vXX/submission.csv`.

## References

- Yu, S., Hannah, W., Peng, L., Lin, J., Bhouri, M. A., Gupta, R., Lütjens, B.,
  Will, J. C., Behrens, G., Busecke, J., Loose, N., Stern, C. I., Beucler, T.,
  Harrop, B., Hillman, B. R., Jenney, A., Ferretti, S., Liu, N., Anandkumar, A., …
  Pritchard, M. (2023). ClimSim: A large multi-scale dataset for hybrid physics-ML
  climate emulation (Version 5). arXiv. https://doi.org/10.48550/ARXIV.2306.08754
