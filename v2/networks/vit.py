import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from transformers import ViTImageProcessor, ViTModel



class ViTTrailEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.vit = ViTModel.from_pretrained('facebook/dino-vits8', use_mask_token=True,
        #                                     proxies={'http': '127.0.0.1:10809', 'https': '127.0.0.1:10809'}).to(device)
        self.vit = ViTModel.from_pretrained('facebook/dino-vits8', use_mask_token=True)

        self.vit_patch_size = self.vit.config.patch_size
        self.output_dim = self.vit.config.hidden_size

        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, obs, state=-1, **kwargs):
        history = obs['history']

        bool_masked_pos = history.pos_mask[:, ::self.vit_patch_size, ::self.vit_patch_size].flatten(start_dim=1)
        out = self.vit(history.history.to(device),
                       bool_masked_pos=bool_masked_pos,
                       interpolate_pos_encoding=True)

        patch_h, patch_w = history.patch_size[:, 0] // self.vit_patch_size, history.patch_size[:,
                                                                            1] // self.vit_patch_size  # todo optimize
        w = history.history.shape[3] // self.vit_patch_size
        i, j = history.curr_rel_row * patch_h, history.curr_rel_col * patch_w
        curr_index = i * w + j + (patch_h // 2) * w + (patch_w // 2)
        lhs = out.last_hidden_state
        indices = einops.repeat(curr_index, 'b -> b 1 h', h=lhs.shape[2])
        curr_enc = torch.gather(lhs, 1, indices.to(device))
        # curr_enc = lhs[torch.arange(lhs.shape[0]), curr_index, :]

        if state is None:
            return curr_enc, None
        else:
            return curr_enc

