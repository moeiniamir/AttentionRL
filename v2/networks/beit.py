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
from transformers import BeitForMaskedImageModeling


class BaseNetwork(nn.Module):
    def __init__(self, patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit = BeitForMaskedImageModeling.from_pretrained('microsoft/beit-base-patch16-224-pt22k', 
                                                   proxies={'http': '127.0.0.1:10809', 'https': '127.0.0.1:10809'},)

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
        indices = history['curdl_indices']
        kmask = history['kmask'][:, ::self.vit_patch_size,
                                 ::self.vit_patch_size].flatten(1)
        pmask = history['pmask'][:, ::self.vit_patch_size,
                                 ::self.vit_patch_size].flatten(1)
        canvas = history['history']
        indices = self.build_indices(indices[..., 0], indices[..., 1], canvas).to(self.vit.device)
        out = self.vit(
            pixel_values=canvas.to(torch.float32).to(self.vit.device),
            bool_masked_pos=(~kmask).to(self.vit.device))
        lhs = out.logits
        gathered = lhs.gather(1, einops.repeat(indices, "b s -> b s h", h=lhs.shape[2]))
        if state is None:  # means it's being used as base for ts Actor/Critic
            return gathered[:, 0], None
        else:
            return gathered[:, 0], gathered[:, 1:5]
