import torch
import torch.nn as nn
import os
import tianshou as ts
import einops
from transformers import ViTImageProcessor, ViTModel

if os.environ['USER'] == 'server':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class ViTTrailEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit = ViTModel.from_pretrained('facebook/dino-vits8', use_mask_token=True,
                                            proxies={'http': '127.0.0.1:10809', 'https': '127.0.0.1:10809'}).to(device)
        self.vit_patch_size = self.vit.config.patch_size

    def forward(self, obs, **kwargs):
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
        # print(f"{out.last_hidden_state.shape=} {i=} {j=} {w=} {patch_w=} {patch_h=} {curr_index=} {history.curr_rel_col=} {history.curr_rel_row=} {history.history.shape=}")
        lhs = out.last_hidden_state
        indices = einops.repeat(curr_index, 'b -> b 1 h', h=lhs.shape[2])
        curr_enc = torch.gather(lhs, 1, indices.to(device))
        # curr_enc = lhs[torch.arange(lhs.shape[0]), curr_index, :]

        return curr_enc



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
        ), device=device)

        for param in self.vit_trail_encoder.parameters():
            param.requires_grad = False


    def forward(self, obs, **kwargs):
        curr_enc = self.vit_trail_encoder(obs, **kwargs)
        duel_out = self.dueling_head(curr_enc)
        # print(f"{curr_enc[:, :4]=} \n {duel_out=}")
        return duel_out