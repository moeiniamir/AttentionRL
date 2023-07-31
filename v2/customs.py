from tianshou.data import Batch
from tianshou.trainer.offpolicy import *
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

