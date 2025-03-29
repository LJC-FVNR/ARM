# ARM: Refining Multivariate Forecasting with Adaptive Temporal-Contextual Learning

This is the implementation of the ARM methods, introduced in the paper [ARM](https://openreview.net/pdf?id=JWpwDdVbaM). The core model is located in `models/ARM.py`. This is a temporary implementation, modified from the ARMA Attention repository. An integrated version will be released in a future update.

## 1. Install Required Packages

First, install PyTorch with GPU support by following the instructions on [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally). Then, install the additional dependencies with:

```bash
pip install -r requirements.txt
```

## 2. Download Datasets

You can download the datasets used in the paper from the link provided by [itransformer](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view) [1]. Place the downloaded files in `dataset/`.

## 3. Run the Training Scripts

To train the ARM model, run:

```bash
bash ARM.sh
```

## 4. Track the Training Process

You can visualize the training process using TensorBoard by running the following command:

```bash
nohup tensorboard --logdir runs --port 6006 --bind_all > tensorb.log 2>&1 &
```

## 5. Training on Custom Data

To train the models on your own dataset, format the CSV file with the first column as `date` (timestamps), and the remaining columns as the time series values. Place your dataset in the `dataset/` folder. 

Next, update the following arrays in the scripts to include your dataset:

```bash
data_names=("weather/weather.csv" "ETT-small/ETTm1.csv" "ETT-small/ETTm2.csv" "ETT-small/ETTh1.csv" "ETT-small/ETTh2.csv" "Solar/solar_AL.txt" "electricity/electricity.csv" "PEMS/PEMS03.npz" "PEMS/PEMS04.npz" "PEMS/PEMS07.npz" "PEMS/PEMS08.npz" "traffic/traffic.csv")
data_alias=("Weather" "ETTm1" "ETTm2" "ETTh1" "ETTh2" "Solar" "ECL" "PEMS03" "PEMS04" "PEMS07" "PEMS08" "Traffic")
data_types=("custom" "ETTm1" "ETTm2" "ETTh1" "ETTh2" "Solar" "custom" "PEMS" "PEMS" "PEMS" "PEMS" "custom")
enc_ins=(21 7 7 7 7 137 321 358 307 883 170 862)
batch_sizes=(32 32 32 32 32 32 32 32 32 32 32 32) 
grad_accums=(1 1 1 1 1 1 1 1 1 1 1 1)
```

Modify these lists to match the configuration of your custom dataset.

## 6. Implementation of Key Modules

- AUEL:

```python
# models/ARM.py
# AUEL preprocessing
    def preprocessing(self, x, x_output=None, cutoff=None, preprocessing_method=None, preprocessing_detach=None, preprocessing_affine=None):
        eps = self.eps
        x_orig = x.clone()
        if preprocessing_method is None: preprocessing_method = self.preprocessing_method
        if preprocessing_detach is None: preprocessing_detach = self.preprocessing_detach
        if preprocessing_affine is None: preprocessing_affine = self.preprocessing_affine
        
        # ...
        elif preprocessing_method == 'AUEL':
            # Clipping operations are applied using .clip() and gate_activation to prevent the weights from becoming negative
            anchor_context = ema_3d(x[:, 0:(self.seq_len), :], alpha=self.ema_alpha[0:x.shape[-1]].clip(0.000001, 0.999999))
            std = weighted_std(x[:, 0:(self.seq_len), :], self.conv_size+[self.seq_len], gate_activation(self.std_weights*2).unsqueeze(0).repeat(x.shape[0], 1, 1)[:, :, 0:x.shape[-1]]).clip(1e-4, 5000)+eps
            mean = anchor_context
        if x_output is None:
            x_output = x_orig.clone()
        if self.preprocessing_detach:
            x_output = (x_output - mean.detach())/std.detach()
        else:
            x_output = (x_output - mean)/std
        if self.preprocessing_affine:
            x_output = x_output * self.channel_w + self.channel_b
        return x_output, mean, std

    # ...
    # In forward

    ### AUEL Preprocessing of Distribution
    if self.auel_global_flag and self.auel_distribution_flag:
        x, mean, std = self.preprocessing(x, x_output=None, cutoff=None)
    ### AUEL Preprocessing of Temporal Patterns with MoE
    if self.auel_global_flag and self.MoE_init_flag:
        x_moe_init = self.MoE_lambda(x.permute(0,2,1)).permute(0,2,1) # B Lp C
        x_enc = torch.cat([x[:, 0:self.seq_len], x_moe_init], dim=1) # B L C
    else:
        x_enc = x

    # ...
    ### AUEL Inverse Processing of Temporal Patterns with MoE
    if self.auel_global_flag and self.MoE_output_flag:
        moe_input = torch.cat([x[:, 0:self.seq_len], output], dim=1)
        output = self.MoE_lambda(moe_input.permute(0,2,1)).permute(0,2,1) + output
```

- Random Dropping:

```python
# exp/exp_main.py
# Random dropping implementation
drop_mask = None
if self.args.random_drop_training:
    if torch.rand(1).item() > 0:
        random_drop_rate = torch.rand(1).item()
        drop_mask = torch.rand(1, 1, batch_x.shape[2], device=batch_x.device) < 1-random_drop_rate
        batch_x = batch_x.masked_fill(drop_mask, 0)
        batch_y = batch_y.masked_fill(drop_mask, 0)
```

- MKLS:
```python
# blocks.py
class MultiConvLayer(nn.Module):
    """
    MKLS
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, padding_lens, dropout=0.25, 
                 pos_len=1024, alpha=1, groups=1, mtype='encoder',
                 pos_emb_flag=True, attention_flag=True, d_latent=None):
        super(MultiConvLayer, self).__init__()
        self.pos_emb_flag, self.attention_flag = pos_emb_flag, attention_flag
        self.kernel_sizes = kernel_sizes
        self.padding_lens = padding_lens
        self.conv_type = nn.Conv1d if mtype == 'encoder' else nn.ConvTranspose1d
        self.d_latent = d_latent
        if d_latent is not None:
            self.input_proj = nn.Linear(in_channels, d_latent)
            self.output_proj = nn.Linear(d_latent, in_channels)
            in_channels = out_channels = d_latent
        self.convs = nn.ModuleList([self.conv_type(in_channels=in_channels, out_channels=out_channels, kernel_size=k, padding=p)
            for k, p in zip(kernel_sizes, padding_lens)])
        size = len(self.convs)
        self.kernel_weights = nn.Parameter(torch.ones(size)/size)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = gate_activation
        
        # Attention Weights: Identity Mapping as Initialization
        self.WQ_w = nn.Parameter(torch.diag(torch.ones(out_channels)))
        self.WK_w = nn.Parameter(torch.diag(torch.ones(out_channels)))
        self.WQ_b = nn.Parameter(torch.zeros(out_channels))
        self.WK_b = nn.Parameter(torch.zeros(out_channels))
        self.WQ = lambda x: torch.matmul(x, self.WQ_w.to(x.device)) + self.WQ_b.to(x.device)
        self.WK = lambda x: torch.matmul(x, self.WK_w.to(x.device)) + self.WK_b.to(x.device)
        self.V = nn.Linear(out_channels, size)
        self.kernel_weight_bias = nn.Parameter(torch.ones(size)/size)
        torch.nn.init.xavier_uniform_(self.V.weight, 1)
        
        # Dropout
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Positional Embedding
        self.pos_emb = PositionalEmbeddingTSP(d_model=out_channels, maxlen=pos_len, init_gain=1e-5)
        self.mask = torch.ones(out_channels, size)
        
    def forward(self, x, *args, partial=None, exchange=False, repeat=False):
        if self.d_latent is not None:
            x = self.input_proj(x)
        x_in = x.permute(0, 2, 1)   # B D L
        res = []                   # x: B L D
        for conv in self.convs:
            res.append(conv(x_in).permute(0, 2, 1))        # N | B L D
        res = torch.stack(res).permute(1, 2, 3, 0)
        if self.attention_flag:
            if self.pos_emb_flag:
                pos_emb = self.pos_emb(device=x.device)
            else:
                pos_emb = 0
            current_mask = self.dropout(self.mask.to(x.device)) * 1.0
            Q, K = self.activation(self.WQ(x) + pos_emb), self.activation(self.WK((res*current_mask).mean(dim=-1)) + pos_emb)
            QK = torch.matmul(Q.permute(0, 2, 1), K)
            QK = QK/sqrt(x.shape[1])
            QK = torch.softmax(QK, dim=-1)
            QKV = self.V(QK)
            kernel_weights = self.activation(QKV).unsqueeze(1)
            res = torch.mean(res * kernel_weights * current_mask, dim=-1)
        else:
            res = torch.mean(res, dim=-1)
        if self.d_latent is not None:
            res = self.output_proj(x)
        return res
```
