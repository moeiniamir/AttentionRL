import numpy as np
import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from transformers.models.vit.modeling_vit import *


class Actor(nn.Module):
    def __init__(self, input_size, basenet):
        super().__init__()
        self.basenet = basenet
        self.curr_linear = nn.Linear(input_size, 1024)
        self.adj_linear = nn.Linear(input_size, 1024)
        self.layer_norm = nn.LayerNorm(1024)
        # self.t = nn.Parameter(torch.tensor(100, dtype=torch.float32))
        self.t = np.sqrt(1024)

    def forward(self, obs, **kwargs):
        curr, adj = self.basenet(obs)
        curr = self.curr_linear(curr)
        curr = self.layer_norm(curr)
        adj = self.adj_linear(adj)
        adj = self.layer_norm(adj)
        pi = einops.einsum(curr, adj, 'i k, i j k -> i j')
        pi = pi / self.t
        return pi, None


class Critic(nn.Module):
    def __init__(self, input_size, basenet):
        super().__init__()
        self.basenet = basenet
        self.linear = nn.Linear(input_size, 1)
        self.cross_attention = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            input_size, 4, input_size, 0.1, batch_first=True), 2)

    def forward(self, obs, **kwargs):
        curr, adj = self.basenet(obs)
        ca_out = self.cross_attention(curr.unsqueeze(1), adj).squeeze()
        value = self.linear(ca_out)
        return value
