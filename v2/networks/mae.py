import numpy as np
import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from transformers.models.vit.modeling_vit import *
import numpy as np
import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from .modeling_vit_mae_custom import ViTMAEModel, ViTMAEConfig


class BaseNetwork(nn.Module):
    def __init__(self, patch_size, n_last_positions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = ViTMAEConfig.from_pretrained('facebook/vit-mae-base',
                                                    proxies={'http': '127.0.0.1:10809', 'https': '127.0.0.1:10809'},)
        config.n_last_positions = n_last_positions
        self.vit = ViTMAEModel.from_pretrained('facebook/vit-mae-base', config=config)        
        self.vit_patch_size = self.vit.config.patch_size
        self.patch_size = patch_size
        
        self.patch_h, self.patch_w = self.patch_size[0] // self.vit_patch_size, self.patch_size[
            1] // self.vit_patch_size
        self.output_dim = self.vit.config.hidden_size

        for param in self.vit.parameters():
            param.requires_grad = False

    def build_indices(self, row, col, canvas):
        w = canvas.shape[3] // self.vit_patch_size
        i, j = row * self.patch_h, col * self.patch_w
        curr_index = i * w + j + (self.patch_h // 2) * w + (self.patch_w // 2)
        return curr_index

    def forward(self, obs, state=-1, **kwargs):
        history = obs['history']
        kmask = history['kmask'][:, ::self.vit_patch_size,
                                 ::self.vit_patch_size].flatten(1)
        canvas = history['history']
        last_positions = history['last_positions']
        padded_mask = history['padded_mask']
        last_positions = self.build_indices(last_positions[..., 0], last_positions[..., 1], canvas).to(self.vit.device)
        out = self.vit(
            pixel_values=canvas.to(torch.float32).to(self.vit.device),
            last_positions=last_positions.to(self.vit.device),
            padded_mask=padded_mask.to(self.vit.device),
            kmask=kmask.to(self.vit.device),
        )
        lhs = out['last_hidden_state']
        return lhs[:, 0], None
