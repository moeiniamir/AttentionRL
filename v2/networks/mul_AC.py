import numpy as np
import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from transformers.models.vit.modeling_vit import *

if os.environ['USER'] == 'server':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, basenet):
        super().__init__()
        self.basenet = basenet
        # self.curr_linear = nn.Linear(basenet.vit.config.hidden_size, basenet.vit.config.hidden_size)
        self.curr_linear = nn.Linear(basenet.vit.config.hidden_size, 1024)
        # self.adj_linear = nn.Linear(basenet.vit.config.hidden_size, basenet.vit.config.hidden_size)
        self.adj_linear = nn.Linear(basenet.vit.config.hidden_size, 1024)
        # self.t = nn.Parameter(torch.tensor(100, dtype=torch.float32))
        self.t = np.sqrt(1024)

    def forward(self, obs, **kwargs):
        curr, adj = self.basenet(obs, **kwargs)
        curr = self.curr_linear(curr)
        adj = self.adj_linear(adj)
        pi = einops.einsum(curr, adj, 'i k, i j k -> i j')
        pi = pi / self.t
        return pi, None


class Critic(nn.Module):
    def __init__(self, basenet):
        super().__init__()
        self.basenet = basenet
        self.linear = nn.Linear(basenet.vit.config.hidden_size, 1)
        self.cross_attention = nn.TransformerDecoder(nn.TransformerDecoderLayer(384, 4, 384, 0.1, batch_first=True), 2)

    def forward(self, obs, **kwargs):
        curr, adj = self.basenet(obs, **kwargs)
        ca_out = self.cross_attention(curr.unsqueeze(1), adj).squeeze()
        value = self.linear(ca_out)
        return value
