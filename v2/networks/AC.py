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


class PathViTModel(ViTModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = PathViTEmbedding(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class PathViTEmbedding(ViTEmbeddings):
    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__(config, use_mask_token)

        self.pad_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(
            self,
            pixel_values: torch.Tensor,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values,
                                           interpolate_pos_encoding=interpolate_pos_encoding)  # b x seq x h

        # unpack the fake bool_masked_pos
        kmask = bool_masked_pos['kmask']  # b x seq
        pmask = bool_masked_pos['pmask']  # b x seq
        indices = bool_masked_pos['indices']  # b x 5

        # insert patch embeddings
        seq_length = embeddings.shape[1]
        mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
        # replace the masked visual tokens by mask_tokens
        mask = pmask.unsqueeze(-1).type_as(mask_tokens)
        embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # b x seq+1 x h

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        # swap needed positions #todo use numba
        indices += 1  # to account for the [CLS] token
        kmask = nn.functional.pad(kmask, (1, 0), value=1)  # to account for the [CLS] token # b x seq+1
        arange_arr = torch.arange(embeddings.shape[1], device=indices.device).repeat((embeddings.shape[0], 1))
        bool_mask_remaining = torch.cat(
            [~torch.isin(arr, indicesr).unsqueeze(0) for arr, indicesr in zip(arange_arr, indices)], dim=0
        )
        remaining_arr = arange_arr[bool_mask_remaining].reshape((embeddings.shape[0], -1))
        new_arr = torch.cat([indices, remaining_arr], dim=1)
        embeddings = embeddings.gather(1, einops.repeat(new_arr, 'b seq -> b seq h', h=embeddings.shape[2]))
        kmask = kmask.gather(1, new_arr)


        # remove ~kmasked positions, pad to max seq len and stack to batch again # todo use numba
        max_len = kmask.sum(dim=1).max()
        # temp_alloc = einops.repeat(self.pad_token, 'h -> b s h', b=embeddings.shape[0], s=max_len).clone()
        temp_alloc = self.pad_token.expand(embeddings.shape[0], max_len, -1).clone()
        for i in range(embeddings.shape[0]):
            temp_alloc[i, :kmask[i].sum()] = embeddings[i, kmask[i]]
        embeddings = temp_alloc


        embeddings = self.dropout(embeddings)

        return embeddings


class BaseNetwork(nn.Module):
    def __init__(self, patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vit = PathViTModel.from_pretrained('facebook/dino-vits8', use_mask_token=True,
                                                proxies={'http': '127.0.0.1:10809', 'https': '127.0.0.1:10809'}).to(
            device)
        self.vit_patch_size = self.vit.config.patch_size
        self.patch_size = patch_size
        self.patch_h, self.patch_w = self.patch_size[0] // self.vit_patch_size, self.patch_size[
            1] // self.vit_patch_size
        self.layer_norm = nn.LayerNorm([self.vit.config.hidden_size], eps=self.vit.config.layer_norm_eps)

        for param in self.vit.parameters():
            param.requires_grad = False

    def build_indices(self, row, col, canvas):
        w = canvas.shape[3] // self.vit_patch_size
        i, j = row * self.patch_h, col * self.patch_w
        curr_index = i * w + j + (self.patch_h // 2) * w + (self.patch_w // 2)
        return curr_index

    def forward(self, obs, **kwargs):
        history = obs['history']
        indices = history['curdl_indices']
        kmask = history['kmask'][:, ::self.vit_patch_size, ::self.vit_patch_size].flatten(1)
        pmask = history['pmask'][:, ::self.vit_patch_size, ::self.vit_patch_size].flatten(1)
        canvas = history['history']
        indices = self.build_indices(indices[..., 0], indices[..., 1], canvas)
        out = self.vit(
            pixel_values=canvas.to(device),
            bool_masked_pos={'kmask': kmask.to(device), 'pmask': pmask.to(device), 'indices': indices.to(device)},
            interpolate_pos_encoding=True)
        lhs = out.last_hidden_state
        lhs = self.layer_norm(lhs)
        return lhs[:, 0], lhs[:, 1:5]


class Actor(nn.Module):
    def __init__(self, basenet):
        super().__init__()
        self.basenet = basenet
        # self.curr_linear = nn.Linear(basenet.vit.config.hidden_size, basenet.vit.config.hidden_size)
        self.curr_linear = nn.Linear(basenet.vit.config.hidden_size, 1024)
        # self.adj_linear = nn.Linear(basenet.vit.config.hidden_size, basenet.vit.config.hidden_size)
        self.adj_linear = nn.Linear(basenet.vit.config.hidden_size, 1024)
        self.t = nn.Parameter(torch.tensor(10, dtype=torch.float32))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs, **kwargs):
        curr, adj = self.basenet(obs, **kwargs)
        curr = self.curr_linear(curr)
        adj = self.adj_linear(adj)
        pi = einops.einsum(curr, adj, 'i k, i j k -> i j')
        probs = self.softmax(pi / self.t)
        return probs, None


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
