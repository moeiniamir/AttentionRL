from typing import Any, Callable, List
from tianshou.data import Batch
from tianshou.env.utils import ENV_TYPE
from tianshou.trainer.offpolicy import *
from transformers.models.vit.modeling_vit import *
from tianshou.policy import DQNPolicy
import tianshou as ts
import random
import numpy

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
        return super().exploration_noise(act, batch)


class CustomOffpolicyTrainer(OffpolicyTrainer):
    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Perform off-policy updates."""
        assert self.train_collector is not None
        for _ in range(round(self.update_per_step * result["n/st"])):
            self.gradient_step += 1
            losses = self.policy.update(self.batch_size, self.train_collector.buffer)
            self.log_update_data(data, losses)

class CustomSubprocVectorEnv(ts.env.SubprocVectorEnv):
    def __init__(self, *args, wandb_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_logger = wandb_logger
    
    def mix_envs(self, action_list, episode_len):
        # build a list of numbers of len env_num whose numbers are evenly distributed in [0, episode_len)
        action_count_list = np.linspace(0, episode_len, self.env_num + 1, dtype=np.int32, endpoint=False)[1:]
        for i in range(self.env_num):
                for _ in range(action_count_list[i]):
                    self.workers[i].send(random.choice(action_list))
                    observation, reward, terminated, truncated, info = self.workers[i].recv()
                    assert not truncated, "The episode should not be truncated in mixing"
                    if terminated:
                        self.workers[i].send(None)
                        
    def step(
        self,
        *args,
        **kwargs
    ):
        out = super().step(*args, **kwargs)
        if self.wandb_logger and self.wandb_logger.use_wandb:
            metas = out[4]
            for meta in metas:
                if meta['logs']:
                    self.wandb_logger.wandb_run.log(meta['logs'])
        return out

