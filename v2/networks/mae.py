import numpy as np
import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
import numpy as np
import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from .modeling_vit_mae_custom import ViTMAEModel, ViTMAEConfig, get_2d_sincos_pos_embed


class BaseNetwork(nn.Module):
    def __init__(self, patch_size, n_last_positions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = ViTMAEConfig.from_pretrained('facebook/vit-mae-base',
                                              proxies={'http': '127.0.0.1:10809', 'https': '127.0.0.1:10809'},)
        config.n_last_positions = n_last_positions
        self.vit = ViTMAEModel.from_pretrained(
            'facebook/vit-mae-base', config=config)
        self.vit_patch_size = self.vit.config.patch_size
        self.patch_size = patch_size

        self.patch_h, self.patch_w = self.patch_size[0] // self.vit_patch_size, self.patch_size[
            1] // self.vit_patch_size
        self.output_dim = self.vit.config.hidden_size

        for param in self.vit.parameters():
            param.requires_grad = False

        self.store_output = False
        self.stored_output = None

    def build_indices(self, row, col, canvas):
        w = canvas.shape[3] // self.vit_patch_size
        i, j = row * self.patch_h, col * self.patch_w
        curr_index = i * w + j + (self.patch_h // 2) * w + (self.patch_w // 2)
        return curr_index

    def forward(self, obs, state=-1, **kwargs):
        if self.stored_output is not None:
            stored_output = self.stored_output
            self.stored_output = None
            return stored_output, None

        history = obs['history']
        kmask = history['kmask'][:, ::self.vit_patch_size,
                                 ::self.vit_patch_size].flatten(1)
        canvas = history['history']
        last_positions = history['last_positions']
        padded_mask = history['padded_mask']
        last_positions = self.build_indices(
            last_positions[..., 0], last_positions[..., 1], canvas).to(self.vit.device)
        out = self.vit(
            pixel_values=canvas.to(torch.float32).to(self.vit.device),
            last_positions=last_positions.to(self.vit.device),
            padded_mask=padded_mask.to(self.vit.device),
            kmask=kmask.to(self.vit.device),
        )

        # print('after mae forward')
        # print(torch.cuda.memory_summary())

        lhs = out['last_hidden_state']

        # idx = last_positions[:, [-1]].unsqueeze(-1).expand(-1, -1, lhs.shape[2])
        # gathered = lhs.gather(1, idx)
        # out = gathered.squeeze(1)

        out = lhs[:, 0]

        if self.store_output:
            self.stored_output = out
            self.store_output = False

        return out, None


class OrderEmbeddingBaseNetwork(BaseNetwork):
    def __init__(self, patch_size, n_last_positions, *args, **kwargs):
        assert n_last_positions > 0
        super().__init__(patch_size, 0, *args, **kwargs)
        self.n_last_positions = n_last_positions
        if self.n_last_positions > 0:
            self.order_embeddings = nn.Parameter(
                torch.zeros(1, self.n_last_positions, self.vit.config.hidden_size), requires_grad=True
            )
        self.pos_emb = get_2d_sincos_pos_embed(self.vit.config.hidden_size, int(
            self.vit.embeddings.patch_embeddings.num_patches**0.5), add_cls_token=False) 
        self.pos_emb = torch.from_numpy(self.pos_emb).unsqueeze(0).to(torch.float32)

    def forward(self, obs, state=-1, **kwargs):
        if self.stored_output is not None:
            stored_output = self.stored_output
            self.stored_output = None
            return stored_output

        history = obs['history']
        kmask = history['kmask'][:, ::self.vit_patch_size,
                                 ::self.vit_patch_size].flatten(1).to(self.vit.device)
        canvas = history['history'].to(torch.float32).to(self.vit.device)
        out = self.vit(
            pixel_values=canvas,
            last_positions=None,
            padded_mask=None,
            kmask=kmask,
        )

        # add ord embedding
        padded_mask = history['padded_mask'].to(self.vit.device)
        last_positions = history['last_positions']
        last_positions = self.build_indices(
            last_positions[..., 0], last_positions[..., 1], canvas).to(self.vit.device)
        lhs = out['last_hidden_state'][:, 1:]
        idx = last_positions.unsqueeze(-1).repeat(1, 1, lhs.shape[-1])
        src = self.order_embeddings.repeat(
            lhs.shape[0], 1, 1) * (~padded_mask.unsqueeze(-1))
        lhs.scatter_add_(1, idx, src)
        
        # add pos embedding
        lhs = lhs + self.pos_emb
        
        output = (lhs, kmask, last_positions)
        if self.store_output:
            self.stored_output = output
            self.store_output = False

        return output
