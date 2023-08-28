import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from transformers import ViTImageProcessor, ViTModel
from .vit import ViTTrailEncoder


class Q_network(nn.Module):
    def __init__(self, action_count, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vit_trail_encoder = ViTTrailEncoder()

        state_shape = 384
        action_shape = action_count
        self.dueling_head = ts.utils.net.common.Net(state_shape, action_shape, hidden_sizes=[512, 512], dueling_param=(
            {
                "hidden_sizes": [512],
            },
            {
                "hidden_sizes": [512],
            }
        ))


    def forward(self, obs, **kwargs):
        curr_enc = self.vit_trail_encoder(obs, **kwargs)
        duel_out = self.dueling_head(curr_enc)
        return duel_out