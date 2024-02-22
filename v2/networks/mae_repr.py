from .mae import BaseNetwork
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from .modeling_vit_mae_custom import get_2d_sincos_pos_embed


class ProjOrdEmbBaseNetwork(BaseNetwork):
    def __init__(self, patch_size, *args, **kwargs):
        super().__init__(patch_size, 0, *args, **kwargs)

    def forward(self, history, state=-1, **kwargs):
        if self.stored_output is not None:
            stored_output = self.stored_output
            self.stored_output = None
            return stored_output

        kmask = history['kmask'][:, ::self.vit_patch_size,
                                 ::self.vit_patch_size].flatten(1).to(self.vit.device)
        canvas = history['history'].to(torch.float32).to(self.vit.device)
        out = self.vit(
            pixel_values=canvas,
            last_positions=None,
            padded_mask=None,
            kmask=kmask,
        )

        lhs = out['last_hidden_state'][:, 1:]
        
        return lhs
