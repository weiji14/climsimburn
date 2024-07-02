/// ClimSim model with a Mamba (Selective State Space Model) architecture.
///
/// Implemented using [Burn](https://github.com/tracel-ai/burn), adapted from:
/// - HuggingFace Candle implementation at https://github.com/huggingface/candle/blob/0.5.1/candle-examples/examples/mamba-minimal/model.rs
/// - Pytorch implementation at https://github.com/johnma2006/mamba-minimal/blob/03de542a36d873f6e6c4057ad687278cc6ae944d/model.py
///
/// References:
/// - Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces
///   (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2312.00752
/// - https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state
/// - https://thegradient.pub/mamba-explained
use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::loss::{HuberLoss, HuberLossConfig, Reduction};
use burn::nn::{Initializer, Linear, LinearConfig, PaddingConfig1d, RmsNorm, RmsNormConfig};
use burn::prelude::{Backend, Config, Module, Tensor};
use burn::tensor::activation::{silu, softplus};

/// A single Mamba block, as described in Figure 3 in Section 3.4 of the Mamba paper
/// https://github.com/huggingface/candle/blob/0.5.1/candle-examples/examples/mamba-minimal/model.rs#L40-L141
#[derive(Module, Debug)]
struct MambaBlock<B: Backend> {
    /* Layers in MambaBlock */
    in_proj: Linear<B>,
    conv1d: Conv1d<B>,
    out_proj: Linear<B>,
    /* Layers in State Space Model */
    x_proj: Linear<B>,
    dt_proj: Linear<B>,
    a_log: Param<Tensor<B, 2>>, // Param of shape (batch, d_state)
    d: Param<Tensor<B, 1>>,
    // rank of Δ, See Gu & Dao 2023 - Section 3.6 "Parameterization of ∆"
    // taken as ceil(d_model / 16)
    dt_rank: usize,
}

#[derive(Config, Debug)]
struct MambaBlockConfig {
    // Input size
    // #[config(default = 24)]
    d_model_in: usize,
    // Hidden linear layer size (D)
    // #[config(default = 48)] // d_model_in * expansion_factor (2)
    d_inner: usize,
    // Output size
    // #[config(default = 24)]
    d_model_out: usize,
    // Conv1D Kernel size
    #[config(default = 4)]
    d_conv: usize,
    // Hidden state size (N)
    #[config(default = 16)]
    d_state: usize,
}

impl MambaBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MambaBlock<B> {
        let dt_rank: usize = self.d_model_in.div_ceil(16);

        MambaBlock {
            /* Layers in MambaBlock */
            // projects block input from D to 2*ED (two branches)
            in_proj: LinearConfig::new(self.d_model_in, self.d_inner * 2).init(device),
            // conv1d layer
            conv1d: Conv1dConfig::new(self.d_inner, self.d_inner, self.d_conv)
                .with_padding(PaddingConfig1d::Explicit(self.d_conv - 1))
                .with_groups(self.d_inner)
                .init(device),
            // final output projection layer
            out_proj: LinearConfig::new(self.d_inner, self.d_model_out).init(device),

            /* Layers in State Space Model */
            // x_proj takes in `x` and outputs the input-specific Δ, B, C
            x_proj: LinearConfig::new(self.d_inner, dt_rank + self.d_state * 2)
                .with_bias(false)
                .init(device),
            // dt_proj projects Δ from dt_rank to d_in
            dt_proj: LinearConfig::new(dt_rank, self.d_inner).init(device),
            // A-state nn.Parameter of shape (d_inner, d_state)
            // note that original mamba implementation uses log() initialization
            a_log: Initializer::Uniform {
                min: 1.0,
                max: (self.d_inner as f64 + 1.0).ln(),
            }
            .init([self.d_inner, self.d_state], device),
            // D-state nn.Parameter of shape (d_inner,)
            d: Initializer::Ones.init([self.d_inner], device),

            // rank of Δ
            dt_rank,
        }
    }
}

impl<B: Backend> MambaBlock<B> {
    /// Mamba block forward. This looks the same as Figure 3 in Section 3.4 in Gu & Dao 2023.
    ///
    ///    inputs (B, L, D)
    ///      |
    ///      |-----------┐
    ///    in_proj     in_proj
    ///      |           |
    ///    Conv1D        |
    ///      |           |
    ///     SiLU        SiLU
    ///      |           |
    ///    [SSM]         |
    ///      |           |
    ///      x-----------┘
    ///      |
    ///   out_proj
    ///      |
    ///    outputs (B, L, D)
    ///
    /// # Shapes
    ///   - Inputs [batch_size, 1, climate_variables]
    ///   - Outputs [batch_size, 1, climate_variables]
    fn forward(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        // https://github.com/huggingface/candle/blob/0.5.1/candle-examples/examples/mamba-minimal/model.rs#L128-L141
        // (B, L, D) -> (B, L, 2*ED)
        let x_and_residual = self
            .in_proj
            .forward(inputs) // input and residual passes through same projection layer
            .chunk(2, 2); // split into 2 chunks at last dim
        let x: Tensor<B, 3> = x_and_residual[0].clone();
        let residual: Tensor<B, 3> = x_and_residual[1].clone();

        // First branch (Conv1D + SiLU + SSM block)
        let [_b_size, seq_length, _d_size] = x.dims();
        let x_: Tensor<B, 3> = x.transpose(); //(B, L, D) -> (B, D, L)
        let xs: Tensor<B, 3> = self.conv1d.forward(x_).narrow(2, 0, seq_length); // Conv1D, shape (B, D, L) -> (B, D, :L)
        let xs: Tensor<B, 3> = xs.transpose(); // (B, D, L) -> (B, L, D)

        let xs: Tensor<B, 3> = silu(xs); // SiLU activation, shape (B, L, D) -> (B, L, D)

        // Pass into State Space Model
        let ss = self.ssm(xs);

        // Second branch (SiLU)
        let rs = silu(residual);

        // Multiply output from SSM block with residual
        let ys = ss * rs;
        // Pass output through projection layer
        let output = self.out_proj.forward(ys);
        output
    }

    /// State Space Model
    ///
    /// State equation : h'(t) = Ah(t) + Bx(t)
    /// Output equation:  y(t) = Ch(t) + Dx(t)
    ///
    ///     ┌----------------> D >-----┐
    ///  x ---> B ------> h ------> C ---> y
    ///              └--< A <--┘
    ///
    /// Input shape              : (B: batch_size, L: sequence_length, D: size_of_input_vector)
    /// Step size (delta Δ) shape: (B: batch_size, L: sequence_length, D: size_of_input_vector)
    /// Matrix B shape           : (B: batch_size, L: sequence_length, N: hidden_state_size)
    /// Matrix C shape           : (B: batch_size, L: sequence_length, N: hidden_state_size)
    /// Output shape             : (B: batch_size, L: sequence_length, D: size_of_input_vector)
    fn ssm(&self, inputs: Tensor<B, 3>) -> Tensor<B, 3> {
        // Get A and D matrices (input independent)
        let a: Tensor<B, 2> = self.a_log.val().no_grad().exp().neg(); // shape (D, N)
        let d: Tensor<B, 1> = self.d.val().no_grad();
        let [_d_in, n] = self.a_log.dims();

        // Pass input x through linear projection layer
        let delta_b_c: Tensor<B, 3> = self.x_proj.forward(inputs.clone()); // (B, L, D) -> (B, L, dt_rank+2*N)

        // Get ∆, B, C matrices (input-dependent)
        let delta = delta_b_c.clone().narrow(2, 0, self.dt_rank); // (B, L, dt_rank)
        let b = delta_b_c.clone().narrow(2, self.dt_rank, n); // (B, L, N)
        let c = delta_b_c.narrow(2, self.dt_rank + n, n); // (B, L, N)

        // softplus without threshold
        let delta = self.dt_proj.forward(delta); // (B, L, dt_rank) -> (B, L, D)
        let delta = softplus(delta, 1.0);

        // Apply selective scan algorithm
        let ss = selective_scan(inputs, delta, a, b, c, d);
        ss
    }
}

/// Does selective scan algorithm
/// See Section 2 State Space Models and Algorithm 2 in Gu & Dao 2023.
/// https://github.com/huggingface/candle/blob/0.5.1/candle-examples/examples/mamba-minimal/model.rs#L99-L124
///
/// This is the classic discrete state space formula:
///             x(t + 1) = Ax(t) + Bu(t)
///             y(t)     = Cx(t) + Du(t)
/// except for the fact that the B and C matrices (and the step size delta, which is used for
/// discretization) are dependent on the input x(t).
///
/// Key: (B: batch_size, L: sequence_length, D: size_of_input_vector, N: hidden_state_size)
///
fn selective_scan<B: Backend>(
    u: Tensor<B, 3>,     // shape (B, L, D)
    delta: Tensor<B, 3>, // shape (B, L, D)
    a: Tensor<B, 2>,     // shape (D, N)
    b: Tensor<B, 3>,     // shape (B, L, N)
    c: Tensor<B, 3>,     // shape (B, L, N)
    d: Tensor<B, 1>,     // shape (D,)
) -> Tensor<B, 3> {
    // Discretize continuous parameters (Δ, A, B) to discrete parameters (A_bar, B_bar)
    // Using zero-order hold discretisation, see equation 4 in Gu & Dao 2023.
    let delta_: Tensor<B, 4> = delta.clone().unsqueeze_dim(3); //       (B, L, D)    -> (B, L, D, 1)
    let a_: Tensor<B, 4> = a.unsqueeze_dims(&[0, 1]); //                      (D, N) -> (1, 1, D, N)
    let delta_a: Tensor<B, 4> = delta_.mul(a_).exp(); // (B, L, D, 1) * (1, 1, D, N) -> (B, L, D, N)

    let delta_: Tensor<B, 4> = delta.clone().unsqueeze_dim(3); //       (B, L, D)    -> (B, L, D, 1)
    let b_: Tensor<B, 4> = b.unsqueeze_dim(2); //                       (B, L, N)    -> (B, L, 1, N)
    let delta_b: Tensor<B, 4> = delta_.mul(b_); //       (B, L, D, 1) * (B, L, 1, N) -> (B, L, D, N)

    let u_: Tensor<B, 4> = u.clone().unsqueeze_dim(3); //                  (B, L, D) -> (B, L, D, 1)
    let delta_b_u: Tensor<B, 4> = delta_b.mul(u_); //    (B, L, D, N) * (B, L, D, 1) -> (B, L, D, N)

    // Perform selective scan
    // Note that the implementation below is sequential, while the official Mamba code uses a much
    // faster parallel scan implementation that is also hardware aware (like FlashAttention)
    let [b_size, l_size, d_size, n_size] = delta_b_u.dims();
    let mut xs: Tensor<B, 3> = Tensor::zeros([b_size, d_size, n_size], &delta.device()); // (B, D, N)
    let mut y_vec: Vec<Tensor<B, 2>> = Vec::with_capacity(l_size); // (L,)
    for i in 0..l_size {
        // loop over sequence_length (L) dimension
        let dt_a_slice: Tensor<B, 3> = delta_a
            .clone()
            .slice([0..b_size, i..i + 1, 0..d_size, 0..n_size]) // (B, /L/, D, N) -> (B, D, N)
            .squeeze(1);
        let dt_b_u_slice: Tensor<B, 3> = delta_b_u
            .clone()
            .slice([0..b_size, i..i + 1, 0..d_size, 0..n_size]) // (B, /L/, D, N) -> (B, D, N)
            .squeeze(1);
        xs = dt_a_slice * xs + dt_b_u_slice; // (B, D, N)

        let c_slice: Tensor<B, 3> = c
            .clone()
            .slice([0..b_size, i..i + 1, 0..n_size]) // (B, /L/, N) -> (B, N)
            .squeeze::<2>(1)
            .unsqueeze_dim(2); // (B, N) -> (B, N, 1)
        let y: Tensor<B, 2> = xs.clone().matmul(c_slice).squeeze(2); // (B, D, N) * (B, N, 1) -> (B, D, 1) -> (B, D)

        y_vec.push(y)
    }

    let ys: Tensor<B, 3> = Tensor::stack(y_vec, 1); // L(B, D) -> (B, L, D)

    // Compute y(t) = Cx(t) + Du(t)
    let d_ = d.unsqueeze_dims(&[0, 1]); // (D,) -> (1, 1, D)
    let skip_connection: Tensor<B, 3> = u.mul(d_); // (B, L, D) * (1, 1, D) -> (B, L, D)
    ys + skip_connection // Output shape (B, L, D)|
}

/// ClimSim Mamba model
#[derive(Module, Debug)]
pub struct ClimSimModel<B: Backend> {
    // Input layers
    // embedding: Embedding<B>,
    linear_seq: Linear<B>,
    linear_in: Linear<B>,
    // Mamba layers
    norm1: RmsNorm<B>,
    mamba1: MambaBlock<B>,
    norm2: RmsNorm<B>,
    mamba2: MambaBlock<B>,
    norm3: RmsNorm<B>,
    // Output layers
    lm_head: Linear<B>,
    linear_out: Linear<B>,
    // Loss function
    loss_huber: HuberLoss<B>,
}

#[derive(Config, Debug)]
pub struct ClimSimModelConfig {
    hidden_size: usize,
}

impl ClimSimModelConfig {
    // Returns the initialized model
    pub fn init<B: Backend>(&self, device: &B::Device) -> ClimSimModel<B> {
        ClimSimModel {
            // Input size hardcoded to 556
            // embedding: EmbeddingConfig::new(556, self.hidden_size).init(device),
            linear_seq: LinearConfig::new(556, 256).init(device),
            linear_in: LinearConfig::new(1, self.hidden_size).init(device),
            // MambaBlocks with params (d_model_in, d_inner, d_model_out)
            mamba1: MambaBlockConfig::new(self.hidden_size, self.hidden_size * 2, self.hidden_size)
                .init(device),
            mamba2: MambaBlockConfig::new(self.hidden_size, self.hidden_size * 2, self.hidden_size)
                .init(device),
            norm1: RmsNormConfig::new(self.hidden_size).init(device),
            norm2: RmsNormConfig::new(self.hidden_size).init(device),
            norm3: RmsNormConfig::new(self.hidden_size).init(device),
            // Output size hardcoded to 368
            linear_out: LinearConfig::new(self.hidden_size, 1).init(device),
            lm_head: LinearConfig::new(256, 368).init(device),
            // Loss function
            loss_huber: HuberLossConfig::new(1.0).init(device),
        }
    }
}

impl<B: Backend> ClimSimModel<B> {
    /// # Shapes
    ///   - Inputs [batch_size, climate_variables]
    ///   - Output [batch_size, climate_variables]
    pub fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        // Not adding embedding layer because inputs are climate variables, not words
        // let x = self.embedding.forward(inputs);
        let x: Tensor<B, 2> = self.linear_seq.forward(inputs); // (B, L=556) -> (B, L=256)
        let x0: Tensor<B, 3> = x.unsqueeze_dim(2); // (B, L) -> (B, L, D=1)
        let x0: Tensor<B, 3> = self.linear_in.forward(x0); // (B, L, D=1) -> (B, L, D=24)

        // MambaBlock 1
        let x1 = self.norm1.forward(x0.clone());
        let x1 = self.mamba1.forward(x1);
        let x1 = x1 + x0;
        // MambaBlock 2
        let x2 = self.norm2.forward(x1.clone());
        let x2 = self.mamba2.forward(x2);
        let x2 = x2 + x1;

        // Pass through RmsNorm function
        let ys: Tensor<B, 3> = self.norm3.forward(x2); // (B, L, D) -> (B, L, D)

        // Pass through linear layer
        let ys: Tensor<B, 3> = self.linear_out.forward(ys); //(B, L, D=24) -> (B, L, 1)
        let ys: Tensor<B, 2> = ys.squeeze(2); // (B, L, 1) -> (B, L)

        // Pass through linear head layer to get 368 sequence_length
        let logits: Tensor<B, 2> = self.lm_head.forward(ys); // (B, L=256) -> (B, L=368)
        logits
    }

    pub fn loss(&self, predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        self.loss_huber
            .forward(predictions, targets, Reduction::Sum)
    }
}
