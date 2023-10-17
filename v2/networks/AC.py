import numpy as np
import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from transformers.models.vit.modeling_vit import *
import traceback


class MulActor(nn.Module):
    def __init__(self, input_size, basenet):
        super().__init__()
        self.basenet = basenet
        self.curr_linear = nn.Linear(input_size, 256)
        self.adj_linear = nn.Linear(input_size, 256)
        self.layer_norm = nn.LayerNorm(256)
        # self.t = nn.Parameter(torch.tensor(100, dtype=torch.float32))
        self.t = np.sqrt(256)

    def forward(self, obs, **kwargs):
        curr, adj = self.basenet(obs)
        curr = self.curr_linear(curr)
        curr = self.layer_norm(curr)
        adj = self.adj_linear(adj)
        adj = self.layer_norm(adj)
        pi = einops.einsum(curr, adj, 'i k, i j k -> i j')
        pi = pi / self.t
        return pi, None


class CAAdjCritic(nn.Module):
    def __init__(self, input_size, basenet, d_model=1024):
        super().__init__()
        self.basenet = basenet
        self.embedding = nn.Linear(input_size, d_model)
        self.linear = nn.Linear(d_model, 1)
        self.cross_attention = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dropout=0.1, batch_first=True), 2)

    def forward(self, obs, **kwargs):
        curr, adj = self.basenet(obs)
        curr = self.embedding(curr)
        adj = self.embedding(adj)
        ca_out = self.cross_attention(curr.unsqueeze(1), adj).squeeze()
        value = self.linear(ca_out)
        return value


class CrossAttentionActor(nn.Module):
    def __init__(self, preprocess, d_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(d_model, 4)
        with torch.no_grad():
            self.linear.weight /= 1000
            
        self.preprocess = preprocess
        self.cross_attention = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dropout=0.1, batch_first=True), 3)
    
    def forward(self, obs, **kwargs):
        lhs, kmask, last_position, *_ = self.preprocess(obs)
        tgt = lhs.gather(1, last_position.unsqueeze(-1).expand(-1, -1, lhs.shape[-1]))
        kmask[0][0] = False
        emb = self.cross_attention(tgt, lhs, memory_key_padding_mask=kmask).squeeze(1)
        logits = self.linear(emb)
        return logits, None


class CrossAttentionCritic(nn.Module):
    def __init__(self, preprocess, d_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(d_model, 1)
        
        self.preprocess = preprocess
        self.cross_attention = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dropout=0.1, batch_first=True), 3)
    
    def forward(self, obs, **kwargs):
        lhs, kmask, last_position, *_ = self.preprocess(obs)
        tgt = lhs.gather(1, last_position.unsqueeze(-1).expand(-1, -1, lhs.shape[-1]))
        emb = self.cross_attention(tgt, lhs, memory_key_padding_mask=kmask).squeeze(1)
        value = self.linear(emb)
        return value


class AdjExCAC(CrossAttentionCritic):
    def forward(self, obs, **kwargs):
        lhs, kmask, last_position, urdl, running_kmask = self.preprocess(obs)
        tgt = lhs.gather(1, last_position.unsqueeze(-1).expand(-1, -1, lhs.shape[-1]))
        emb = self.cross_attention(tgt, lhs, memory_key_padding_mask=running_kmask).squeeze(1)
        value = self.linear(emb)
        return value


class AdjCrossAttentionActor(nn.Module):
    def __init__(self, preprocess, d_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(d_model, 1)
        with torch.no_grad():
            self.linear.weight /= 1000
            
        self.preprocess = preprocess
        self.cross_attention = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dropout=0.1, batch_first=True), 3)        
        
    def forward(self, obs, **kwargs):
        lhs, kmask, last_position, urdl, running_kmask = self.preprocess(obs)
        tgt = lhs.gather(1, urdl.unsqueeze(-1).expand(-1, -1, lhs.shape[-1]))
        emb = self.cross_attention(tgt, lhs, memory_key_padding_mask=running_kmask).squeeze(1)
        logits = self.linear(emb).squeeze(-1)
        print(logits.shape)
        return logits, None