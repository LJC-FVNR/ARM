import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import random
import math

# Calculating Multi-window Std
def weighted_std(tensor, conv_size, weights):
    weights = weights / weights.sum(dim=1, keepdim=True)
    res = []
    for s in conv_size:
        res.append(tensor[:, -s:, :].std(dim=1, keepdim=True))
    return (torch.cat(res, dim=1) * weights).mean(dim=1, keepdim=True)

# Calculating EMA Mean
def ema_3d(tensor, alpha):
    a, b, c = tensor.shape
    indices = torch.arange(0, b, device=tensor.device)
    indices = indices.view(1, b, 1).repeat(a, 1, c)
    alpha = alpha.view(1, 1, c)
    weights_raw = (1 - alpha) * torch.pow(alpha, indices)
    weights_normalized = weights_raw.flip(1) / weights_raw.sum(dim=1, keepdim=True)
    return (weights_normalized * tensor).sum(dim=1, keepdim=True)

# Weight Clipping
def gate_activation(x, alpha=5):
    #return ((torch.nn.functional.gelu(torch.tanh(x)*alpha)/alpha)*(1/0.9640)).abs()
    return x.clip(0, 1000)