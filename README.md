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

TODO


## References

- Yu, S., Hannah, W., Peng, L., Lin, J., Bhouri, M. A., Gupta, R., Lütjens, B.,
  Will, J. C., Behrens, G., Busecke, J., Loose, N., Stern, C. I., Beucler, T.,
  Harrop, B., Hillman, B. R., Jenney, A., Ferretti, S., Liu, N., Anandkumar, A., …
  Pritchard, M. (2023). ClimSim: A large multi-scale dataset for hybrid physics-ML
  climate emulation (Version 5). arXiv. https://doi.org/10.48550/ARXIV.2306.08754
