import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Some External Basic Predictors
from models.PatchTST import Model as PatchTST
from models.DLinear import Model as DLinear
from models.Autoformer import Model as Autoformer
from models.Informer import Model as Informer

from functions import *
from blocks import *

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # Basic Dataset Settings
        self.seq_len = configs.seq_len                                # L_I
        self.pred_len = configs.pred_len                              # L_P
        self.input_len = self.seq_len + self.pred_len                 # L = L_I+L_P
        self.in_channels = configs.enc_in                             # Number of series (C)
        self.time_emb_dim = configs.time_emb_dim                      # Dimension of external timestep/date embedding (optional. set as 0 to disable this)
        self.in_channels_full = self.in_channels + self.time_emb_dim  # Full input number of channels (C)
        self.d_model = configs.d_model                                # Model dimension in the predictor/encoder-decoder
        self.eps = 1e-6
        self.preprocessing_method = 'AUEL'
        self.preprocessing_detach = False
        self.preprocessing_affine = True
        
        # AUEL (see A.4 and A.5 for explanation)
        self.auel_global_flag = configs.auel_global_flag
        # AUEL of Distribution
        self.auel_distribution_flag = configs.auel_distribution_flag
        self.conv_size = [49, 145, 385]
        self.conv_padding = [24, 72, 192]
        self.ema_alpha_init = configs.ema_alpha
        self.ema_alpha = nn.Parameter(self.ema_alpha_init*torch.ones(self.in_channels))
        self.std_weights = nn.Parameter(torch.cat([1*torch.ones(len(self.conv_size), self.in_channels), 
                                                      torch.ones(1, self.in_channels)]))
        self.channel_w = nn.Parameter(torch.ones(self.in_channels), requires_grad=True)
        self.channel_b = nn.Parameter(torch.zeros(self.in_channels), requires_grad=True)
        # MoE
        self.MoE_init_flag = configs.moe_flag
        self.MoE_output_flag = configs.moe_flag
        self.MoE_init = nn.Sequential(SwitchFeedForward(capacity_factor=1.0, 
                                           drop_tokens=False, 
                                           is_scale_prob=False, 
                                           n_experts=configs.n_experts, 
                                           expert=FeedForward(self.input_len, 4*self.input_len, 0.75, is_gated=False, d_output=self.pred_len), 
                                           d_model=self.input_len,
                                           d_output=self.pred_len)
                                     )
        self.MoE_lambda = lambda x: self.MoE_init(x)
        
        # MKLS
        self.mkls_global_flag = configs.mkls_global_flag
        self.mkls_pos_emb_flag = configs.mkls_pos_emb_flag
        self.pre_mkls_flag = configs.pre_mkls_flag
        self.post_mkls_flag = configs.post_mkls_flag
        self.mkls_attention_flag = configs.mkls_attention_flag
        
        # Predictor / Encoder Decoder
        # Projections
        self.predictor_mkls_flag = configs.predictor_mkls_flag
        predictor_configs = copy.deepcopy(configs)
        self.predictor = configs.predictor
        print(f'Current Predictor: {self.predictor}')

        # MKLS and channel projection for other predictors. See A.5.2 for explanation. 
        if configs.predictor not in ['Vanilla', 'Default']:
            if self.predictor_mkls_flag:
                # MKLS Blocks
                self.encoder_pre_mkls = MultiConvLayer(self.d_model-self.in_channels, self.d_model-self.in_channels, self.conv_size, self.conv_padding, pos_len=self.input_len, mtype='decoder', pos_emb_flag=self.mkls_pos_emb_flag, attention_flag=self.mkls_attention_flag, d_latent=self.d_model-self.in_channels if self.d_model-self.in_channels < 64 else 64, dropout=0.25)
                self.encoder_post_mkls = MultiConvLayer(self.d_model-self.in_channels, self.d_model-self.in_channels, self.conv_size, self.conv_padding, pos_len=self.input_len, mtype='decoder', pos_emb_flag=self.mkls_pos_emb_flag, attention_flag=self.mkls_attention_flag, d_latent=self.d_model-self.in_channels if self.d_model-self.in_channels < 64 else 64, dropout=0.25)
                # Input / Output Projection
                self.enc_in_proj = nn.Linear(self.in_channels, self.d_model-self.in_channels) if self.in_channels != self.d_model else nn.Identity()
                self.output_proj = nn.Linear(self.d_model, self.in_channels) if self.in_channels != self.d_model else nn.Identity()
                torch.nn.init.zeros_(self.output_proj.weight)
                predictor_configs.enc_in = self.d_model
                predictor_configs.dec_in = self.d_model
                predictor_configs.c_out = self.d_model
            else:
                predictor_configs.enc_in = self.in_channels
                predictor_configs.dec_in = self.in_channels
                predictor_configs.c_out = self.in_channels
        
        # For Vanilla Transformer
        if configs.predictor in ['Vanilla', 'Default']:
            # Input / Output Projection
            self.enc_in_proj = nn.Linear(self.in_channels_full, self.d_model)
            self.output_proj = nn.Linear(self.d_model, self.in_channels)
            # MKLS Blocks
            # Pre-MKLS, shared between Encoder and Decoder
            self.encoder_pre_mkls = MultiConvLayer(self.d_model, self.d_model, self.conv_size, self.conv_padding, pos_len=self.input_len, mtype='decoder', pos_emb_flag=self.mkls_pos_emb_flag, attention_flag=self.mkls_attention_flag, d_latent=self.d_model if self.d_model < 64 else 64, dropout=0.25)
            # Post-MKLS, for both Encoder and Decoder
            self.encoder_post_mkls = MultiConvLayer(self.d_model, self.d_model, self.conv_size, self.conv_padding, pos_len=self.input_len, mtype='decoder', pos_emb_flag=self.mkls_pos_emb_flag, attention_flag=self.mkls_attention_flag, d_latent=self.d_model if self.d_model < 64 else 64, dropout=0.25)
            self.decoder_post_mkls = MultiConvLayer(self.d_model, self.d_model, self.conv_size, self.conv_padding, pos_len=self.input_len, mtype='decoder', pos_emb_flag=self.mkls_pos_emb_flag, attention_flag=self.mkls_attention_flag, d_latent=self.d_model if self.d_model < 64 else 64, dropout=0.25)
            # Task Embedding & Positonal Embedding
            # Task Embedding to Distinguish Input and Output Part
            self.task_emb_input_part = nn.Parameter(torch.zeros(1, 1, self.d_model))
            self.task_emb_pred_part = nn.Parameter(torch.zeros(1, 1, self.d_model))
            # Positional Embedding
            self.pos_emb = nn.Parameter(torch.zeros(1, self.input_len, self.d_model))
            transformer_dropout = 0.0
            deep_initialization = 1 # Gain of the parameter initialization
            POST_NORM = True
            PRE_NORM = False
            self.n_heads = configs.n_heads
            self.encoder1 = EncoderLayerTSP(
                            attention=AttentionLayerTSP(FullAttentionTSP(mask_flag=0, attention_dropout=transformer_dropout, output_attention=False), d_model_q=self.d_model, d_model_k=self.d_model, d_model_v=self.d_model, n_heads=self.n_heads, d_keys=None, d_values=None, deep_beta=deep_initialization, d_output=self.d_model), 
                            d_ff = self.d_model*4, d_model=self.d_model, dropout=transformer_dropout, activation='gelu', POST_NORM=POST_NORM, PRE_NORM=PRE_NORM)
            self.encoder2 = EncoderLayerTSP(
                            attention=AttentionLayerTSP(FullAttentionTSP(mask_flag=0, attention_dropout=transformer_dropout, output_attention=False), d_model_q=self.d_model, d_model_k=self.d_model, d_model_v=self.d_model, n_heads=self.n_heads, d_keys=None, d_values=None, deep_beta=deep_initialization, d_output=self.d_model), 
                            d_ff = self.d_model*4, d_model=self.d_model, dropout=transformer_dropout, activation='gelu', POST_NORM=POST_NORM, PRE_NORM=PRE_NORM)
            self.decoder1 = EncoderLayerTSP(
                            attention=AttentionLayerTSP(FullAttentionTSP(mask_flag=0, attention_dropout=transformer_dropout, output_attention=False), d_model_q=self.d_model, d_model_k=self.d_model, d_model_v=self.d_model, n_heads=self.n_heads, d_keys=None, d_values=None, deep_beta=deep_initialization, d_output=self.d_model), 
                            d_ff = self.d_model*4, d_model=self.d_model, dropout=transformer_dropout, activation='gelu', POST_NORM=POST_NORM, PRE_NORM=PRE_NORM)
        # For Other Predictors
        elif self.predictor == 'PatchTST':
            predictor_configs.e_layers = 3
            predictor_configs.n_heads = 4
            predictor_configs.d_model = self.d_model
            predictor_configs.d_ff = 4*self.d_model
            predictor_configs.dropout = 0.3
            predictor_configs.fc_dropout = 0.3
            predictor_configs.head_dropout = 0
            predictor_configs.patch_len = 16
            predictor_configs.stride = 8
            predictor_configs.revin = 1
            self.Predictor = PatchTST(predictor_configs)
        elif self.predictor == 'DLinear':
            self.Predictor = DLinear(predictor_configs)
        elif self.predictor == 'Autoformer':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 1
            predictor_configs.factor = 3
            self.Predictor = Autoformer(predictor_configs)
        elif self.predictor == 'Informer':
            predictor_configs.e_layers = 2
            predictor_configs.d_layers = 1
            predictor_configs.factor = 3
            self.Predictor = Informer(predictor_configs)
        
        self.predictor_configs = predictor_configs
        
    # AUEL preprocessing
    def preprocessing(self, x, x_output=None, cutoff=None, preprocessing_method=None, preprocessing_detach=None, preprocessing_affine=None):
        eps = self.eps
        x_orig = x.clone()
        if preprocessing_method is None: preprocessing_method = self.preprocessing_method
        if preprocessing_detach is None: preprocessing_detach = self.preprocessing_detach
        if preprocessing_affine is None: preprocessing_affine = self.preprocessing_affine
        
        if preprocessing_method == 'lastvalue':
            mean = x[:,-1:,:]
            std = torch.Tensor([1]).to(x.device)
        elif preprocessing_method == 'standardization':
            if cutoff is None:
                mean = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True) + eps
            else:
                n_elements = cutoff.sum(dim=1, keepdim=True) # B L C -> B 1 C
                # x = x * cutoff # already done
                mean = x.sum(dim=1, keepdim=True) / (n_elements+eps).detach()
                sst = (cutoff*((torch.square(x - mean))**2)).sum(dim=1, keepdim=True)
                std = torch.sqrt(sst / (n_elements-1+eps)) + eps
                std = std.detach()
                #std = torch.Tensor([1]).to(x.device) 
        elif preprocessing_method == 'AUEL':
            # Clipping operations are applied using .clip() and gate_activation to prevent the weights from becoming negative
            anchor_context = ema_3d(x[:, 0:(self.seq_len), :], alpha=self.ema_alpha[0:x.shape[-1]].clip(0.000001, 0.999999))
            std = weighted_std(x[:, 0:(self.seq_len), :], self.conv_size+[self.seq_len], gate_activation(self.std_weights*2).unsqueeze(0).repeat(x.shape[0], 1, 1)[:, :, 0:x.shape[-1]]).clip(1e-4, 5000)+eps
            mean = anchor_context
        elif preprocessing_method == 'none':
            mean = torch.Tensor([0]).to(x.device)
            std = torch.Tensor([1]).to(x.device)
        if x_output is None:
            x_output = x_orig.clone()
        if self.preprocessing_detach:
            x_output = (x_output - mean.detach())/std.detach()
        else:
            x_output = (x_output - mean)/std
        if self.preprocessing_affine:
            x_output = x_output * self.channel_w + self.channel_b
        return x_output, mean, std
    
    #  AUEL inverse processing
    def inverse_processing(self, x, mean, std):
        x = x[:, -self.pred_len:, -self.in_channels:]
        if self.preprocessing_affine:
            x = (x - self.channel_b)/self.channel_w
        x = x*std + mean
        return x
    
    def forward(self, x, x_mark, drop_mask=None, train=True):
        # x: [Batch, Input Length (L_I+L_P), Channel], input data
        # x_mark: [Batch, Input Length (L_I+L_P), EmbeddingChannel], optional timestep/date embedding
        # drop_mask: mask of the random dropping, generated in the training script
        
        ### AUEL Preprocessing of Distribution
        if self.auel_global_flag and self.auel_distribution_flag:
            x, mean, std = self.preprocessing(x, x_output=None, cutoff=None)
        ### AUEL Preprocessing of Temporal Patterns with MoE, see details in the main text and A.5.1
        if self.auel_global_flag and self.MoE_init_flag:
            x_moe_init = self.MoE_lambda(x.permute(0,2,1)).permute(0,2,1) # B Lp C
            x_enc = torch.cat([x[:, 0:self.seq_len], x_moe_init], dim=1) # B L C
        else:
            x_enc = x
        
        ### Vanilla Encoder-Decoder
        if self.predictor in ['Vanilla', 'Default']:
            ### Optional Timestep Embedding
            x_enc = torch.cat([x_enc, x_mark], dim=-1) # B L C
            ##### Input Projection
            x_enc = self.enc_in_proj(x_enc) # B L d
            ### Task Embedding & Positonal Embedding
            task_emb = torch.cat([self.task_emb_input_part.repeat(1, self.seq_len, 1), self.task_emb_pred_part.repeat(1, self.pred_len, 1)], dim=1)
            x_enc = x_enc + task_emb + self.pos_emb
            ### Pre-MKLS
            if self.mkls_global_flag and self.pre_mkls_flag:
                mkls = self.encoder_pre_mkls(x_enc)
                x_enc = x_enc + mkls
            ### Encoder 1
            rep = self.encoder1(x_enc, x_enc, x_enc, add=0)
            ### Post-MKLS + Encoder 2
            if self.mkls_global_flag and self.post_mkls_flag:
                mkls = self.encoder_post_mkls(x_enc)
            else:
                mkls = 0
            rep = self.encoder2(rep, rep, rep, add=mkls)
            ### Post-MKLS + Decoder 1
            if self.mkls_global_flag and self.post_mkls_flag:
                mkls = self.decoder_post_mkls(x_enc)
            else:
                mkls = 0
            x_enc = self.decoder1(x_enc, rep, rep, add=mkls)
            ### Output Projection
            output = self.output_proj(x_enc[:, -self.pred_len:]) # B Lp C
        ### Other Predictors, see A.5.2 for explanation
        else:
            # Align Input Format for Different Predictors
            # Pre-MKLS
            if self.predictor_mkls_flag:
                x_enc_concat = self.enc_in_proj(x_enc) # B L C -> B L d-C
                x_enc_concat = x_enc_concat + self.encoder_pre_mkls(x_enc_concat) # B L d-C
                x_enc = torch.cat([x_enc_concat, x_enc], dim=-1) # B L d
            
            # Main Predictor
            output = self.Predictor(x_enc[:, 0:self.seq_len],    # Encoder Input X
                                    x_mark[:, 0:self.seq_len],   # Encoder Input Timestep Embedding
                                    x_enc[:, -(self.pred_len+self.predictor_configs.label_len):],  # Decoder Input X
                                    x_mark[:, -(self.predictor_configs.label_len+self.pred_len):]  # Decoder Input Timestep Embedding
                                    ) # B L d
            
            # Post-MKLS
            if self.predictor_mkls_flag:
                mkls_shortcut = torch.cat([x_enc_concat[:, 0:self.seq_len, :], output[:, :, 0:(self.d_model-self.in_channels)]], dim=1) # B L d-C
                output[:, :, 0:(self.d_model-self.in_channels)] = output[:, :, 0:(self.d_model-self.in_channels)] + self.encoder_post_mkls(mkls_shortcut)[:, -self.pred_len:, :]
                output_multi = self.output_proj(output) # B L d -> B L C
                output = output_multi + output[:, :, -self.in_channels:] # B L C
        
        ### AUEL Inverse Processing of Temporal Patterns with MoE, see details in the main text and A.5.1
        if self.auel_global_flag and self.MoE_output_flag:
            moe_input = torch.cat([x[:, 0:self.seq_len], output], dim=1)
            output = self.MoE_lambda(moe_input.permute(0,2,1)).permute(0,2,1) + output

        ### AUEL Inverse Processing of Distribution
        if self.auel_global_flag and self.auel_distribution_flag:
            output = self.inverse_processing(output, mean, std)
        return output # [Batch, Output length, Channel]