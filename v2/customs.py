import logging

from tianshou.data import Batch
from tianshou.trainer.offpolicy import *
import torch
from transformers.models.vit.modeling_vit import *
from tianshou.policy import DQNPolicy

def polyack_sync(self):
    current_state_dict = self.model.state_dict()
    old_state_dict = self.model_old.state_dict()

    for key in old_state_dict:
        old_state_dict[key] = self.polyak * old_state_dict[key] + (1 - self.polyak) * current_state_dict[key]


class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        self.polyak = kwargs.pop('polyak', None)
        if self.polyak:
            kwargs['target_update_freq'] = 1

            self.sync_weight = polyack_sync

        super().__init__(*args, **kwargs)

    def exploration_noise(
            self,
            act: Union[np.ndarray, Batch],
            batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        # print(f"{act=}", end='')
        return super().exploration_noise(act, batch)


class CustomOffpolicyTrainer(OffpolicyTrainer):
    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Perform off-policy updates."""
        # print(f"{data=}\n{result=}")
        assert self.train_collector is not None
        for _ in range(round(self.update_per_step * result["n/st"])):
            self.gradient_step += 1
            losses = self.policy.update(self.batch_size, self.train_collector.buffer)
            self.log_update_data(data, losses)


def custom_offpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    return CustomOffpolicyTrainer(*args, **kwargs).run()


class CustomViTEmbeddings(ViTEmbeddings):
    def forward(
            self,
            pixel_values: torch.Tensor,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # remove the masked tokens from the embeddings
            

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class CustomViTModel(ViTModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config, add_pooling_layer, use_mask_token)
        self.embeddings = CustomViTEmbeddings(config, use_mask_token=use_mask_token)
