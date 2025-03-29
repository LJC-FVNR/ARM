import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import random
import math
import copy
from functions import *

# Epsilon
EPS = 1e-5

# Attention Layer
class FullAttentionTSP(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, mask_length=0, output_attention=False):
        super(FullAttentionTSP, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        if mask_flag:
            with torch.no_grad():
                self.mask_shape = [1, 1, mask_length, mask_length]
                self.mask = torch.triu(torch.ones(self.mask_shape, dtype=torch.bool), diagonal=1).cuda()

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(D)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Whether to apply causal mask
        if self.mask_flag:
            with torch.no_grad():
                if attn_mask is None:
                    attn_mask = self.mask.repeat(B, 1, 1, 1)
            scores.masked_fill_(attn_mask, -np.inf)
            
        # Dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # QKV
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayerTSP(nn.Module):
    def __init__(self, attention, d_model_q, d_model_k, d_model_v, n_heads, d_keys=None,
                 d_values=None, deep_initialization=1, d_output=None, 
                 ATTN_NORM=False, retain='min', channels=None, low_rank=None, **kwargs):
        super(AttentionLayerTSP, self).__init__()
        
        # Dimension Aligment
        d_queries = d_model_q // n_heads
        d_keys = d_model_k // n_heads
        d_values = d_model_v // n_heads
        retain = min if retain == 'min' else max
        if np.mean([d_queries, d_keys, d_values]) != d_queries:
            d_queries = retain([d_queries, d_keys, d_values])
            d_keys = d_queries
            d_values = d_queries
        
        if low_rank is not None and low_rank != 0:
            d_queries = low_rank if low_rank < d_queries else d_queries
            d_keys = low_rank if low_rank < d_keys else d_keys
            d_values = low_rank if low_rank < d_values else d_values
        
        self.D = d_queries
        
        # Attention LayerNorm
        self.pre_normq = nn.LayerNorm(d_queries * n_heads) if ATTN_NORM else nn.Identity()
        self.pre_normk = nn.LayerNorm(d_keys * n_heads) if ATTN_NORM else nn.Identity()
        self.pre_normv = nn.LayerNorm(d_values * n_heads) if ATTN_NORM else nn.Identity()
        
        self.inner_attention = attention
        # W_Q, W_K, W_V
        self.query_projection = nn.Linear(d_model_q, d_queries * n_heads)
        self.key_projection = nn.Linear(d_model_k, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model_v, d_values * n_heads)
        if d_output is None: d_output = d_model_q
        self.out_projection = nn.Linear(d_values * n_heads, d_output)
        self.n_heads = n_heads
        
        # Weight Initialization
        torch.nn.init.xavier_normal_(self.query_projection.weight, gain=deep_initialization)
        torch.nn.init.xavier_normal_(self.key_projection.weight, gain=deep_initialization)
        torch.nn.init.xavier_normal_(self.value_projection.weight, gain=deep_initialization)
        torch.nn.init.xavier_normal_(self.out_projection.weight, gain=deep_initialization)

    def forward(self, queries, keys, values, attn_mask, add_emb=None, concat_emb=None):
        B, LQK, _ = queries.shape
        _, LV, _ = values.shape
        D = self.D
        H = self.n_heads
        queries = self.pre_normq(self.query_projection(queries)).view(B, LQK, H, D)
        keys = self.pre_normk(self.key_projection(keys)).view(B, LQK, H, D)
        values = self.pre_normv(self.value_projection(values)).view(B, LV, H, D)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        out = out.view(B, LV, -1)
        return self.out_projection(out), attn

class EncoderLayerTSP(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", deep_initialization=1, 
                       PRE_NORM=True, POST_NORM=True, channels=None, ffn_bypass=False, ex_conv1=None, ex_conv2=None):
        super(EncoderLayerTSP, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1) if ex_conv1 is None else ex_conv1
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1) if ex_conv2 is None else ex_conv2
        self.ffn_bypass = ffn_bypass
        self.norm1 = nn.LayerNorm(d_model) if PRE_NORM else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if POST_NORM else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if PRE_NORM else nn.Identity()
        self.norm4 = nn.LayerNorm(d_model) if POST_NORM else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.deep_initialization = deep_initialization
        self.fuse_beta = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        torch.nn.init.xavier_normal_(self.conv1.weight, gain=self.deep_initialization)
        torch.nn.init.xavier_normal_(self.conv2.weight, gain=self.deep_initialization)

    def forward(self, q, k, v, attn_mask=None, add=0, cross_add_emb=None, cross_concat_emb=None):
        x = self.dropout(self.attention(
            q, k, v,
            attn_mask=attn_mask,
            add_emb=cross_add_emb,
            concat_emb=cross_concat_emb
        )[0])
        x = q + self.norm1(x)
        x = self.norm2(x)
        x_shortcut = x.clone()
        if not self.ffn_bypass:
            x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
            x = self.conv2(x).transpose(-1, 1)
            x = x_shortcut + self.norm3(x)
        return self.norm4(x*self.fuse_beta[0]+add)

class PositionalEmbeddingTSP(nn.Module):
    def __init__(self, d_model=128, maxlen=512, init_gain=0.01):
        super(PositionalEmbeddingTSP, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model
        self.pos = torch.arange(maxlen)
        self.emb = nn.Embedding(maxlen, d_model)
        torch.nn.init.xavier_uniform_(self.emb.weight, gain=init_gain)
        
    def forward(self, n=None, device=None):
        if device is None:
            res = self.emb(self.pos)
        else:
            res = self.emb(torch.arange(self.maxlen, device=device))
        return res
    
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
        self.activation = gate_activation # nn.GELU()
        
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

# Channel-independent Linear Layers
class IndependentLinears(nn.Module):	
    def __init__(self, in_d, out_d, n=16, zero_init=True, init=0, transpose=False):	
        super(IndependentLinears, self).__init__()	
        # self.model = nn.ModuleList([nn.Linear(in_d, out_d) for i in range(n)])	
        self.weight = nn.Parameter(torch.zeros(n, in_d, out_d)) # C L Lp
        self.bias = nn.Parameter(torch.zeros(n, in_d, out_d)) # C L Lp
        torch.nn.init.zeros_(self.weight) if zero_init else None
        self.init = init
        self.transpose = transpose
    	
    def forward(self, x):	
        # Input: B L C
        if self.transpose:
            x = x.permute(0,2,1)
        x = x.permute(0, 2, 1).unsqueeze(2) # B C 1 L
        w = self.weight.unsqueeze(0) # 1 C L Lp
        b = self.bias.unsqueeze(0) # 1 C L Lp
        x = torch.matmul(x, w) + b # B C 1 Lp
        x = x[:, :, 0, :] # B C Lp
        x = x.permute(0, 2, 1) # B Lp C
        x = x + self.init
        if self.transpose:
            x = x.permute(0,2,1)
        return x
    
# MoE Blocks
# Edited from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/__init__.py

class FeedForward(nn.Module):
    """
    ## FFN module
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.GELU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True,
                 d_output = None):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff, d_model if d_output is None else d_output, bias=bias2)
        torch.nn.init.zeros_(self.layer2.weight)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$
        # depending on whether it is gated
        return self.layer2(x)
    

class SwitchFeedForward(nn.Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(self, *,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_experts: int,
                 expert: FeedForward,
                 d_model: int,
                 d_output=None):
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # make copies of the FFNs
        self.experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(n_experts)])
        self.d_output = d_output
        # Routing layer and softmax
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """
        # current x: batch * seq_len * d_model
        x = x.permute(1, 0, 2)
        # Capture the shape to change shapes later
        seq_len, batch_size, d_model = x.shape
        # Flatten the sequence and batch dimensions
        x = x.reshape(-1, d_model)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.switch(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(seq_len*batch_size, self.d_output)

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        final_output = final_output.type_as(expert_output[0])
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        final_output = final_output.view(seq_len, batch_size, self.d_output)
        
        # change the sequence to b * s * d
        final_output = final_output.permute(1, 0, 2)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        return final_output #, counts, route_prob.sum(0), len(dropped), route_prob_max
    
    def l2_penalty(self):
        norm = 0
        for m in self.experts:
            norm = norm + m.layer1.weight.norm() + m.layer1.weight.norm()
        return norm


