from .mae_history import MAEHistory
from pack_existing_segs import *
from enum import Enum
from IPython import display
import matplotlib.pyplot as plt
import torch
import torch.utils.data as D
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import einops
from .history import History
from .mul_limited_history import LimitedHistory
from .mae_limited_history import MAELimitedHistory
from .mae_history_adj import MAEHistoryAdj


class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    END = 4


class Environment(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, dataset,
                 patch_size=(64, 64),
                 max_len=None,
                 n_last_positions=None,
                 history_type=None,
                 time_limit=None,
                 ):
        self.history_type = history_type
        self.dataloader = D.DataLoader(dataset, batch_size=1, shuffle=True)
        self.iterator = iter(self.dataloader)
        self.patch_size = patch_size
        self.max_len = max_len
        self.n_last_positions = n_last_positions

        self.img_emtpy_patch = torch.zeros(
            (patch_size[0] // 2, patch_size[1] // 2))
        self.img_emtpy_patch[::2, ::2] = 1
        self.seg_empty_patch = torch.zeros(
            (patch_size[0] // 2, patch_size[1] // 2))

        self.observation_space = spaces.Dict({
            'center': spaces.Box(low=0, high=255, shape=(3, self.patch_size[0], self.patch_size[1]), dtype=np.float16),
        })
        self.action_space = spaces.Discrete(len(Actions))
        self.time_limit = np.uint32(
            time_limit) if time_limit is not None else np.nan
        self.elapsed_steps = None
        self.ratio_quantiles = [.2, .4, .6, .8]
        self.step_quantiles = [10, 20, 30, 50, 80]

    def reset(self, **kwargs):
        try:
            # Samples the batch
            self.current_image, self.current_seg, self.image_id = next(
                self.iterator)
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.iterator = iter(self.dataloader)
            self.current_image, self.current_seg, self.image_id = next(
                self.iterator)

        # remove the batch dimension
        self.current_image = self.current_image[0]
        self.current_seg = self.current_seg[0]

        # cut the image and seg to multiples of patch size
        initial_height, initial_width = self.current_image.shape[1:]
        self.current_image = self.current_image[:, :initial_height // self.patch_size[0] * self.patch_size[0],
                                                :initial_width // self.patch_size[1] * self.patch_size[1]]
        self.current_seg = self.current_seg[:, :initial_height // self.patch_size[0] * self.patch_size[0],
                                            :initial_width // self.patch_size[1] * self.patch_size[1]]

        # add empty patch to all 4 edges of image and seg
        repeated_empty_patch = self.img_emtpy_patch.repeat(self.current_image.shape[0], 1,
                                                           self.current_image.shape[2] // self.img_emtpy_patch.shape[1])
        self.current_image = torch.cat(
            [repeated_empty_patch, self.current_image, repeated_empty_patch], dim=1)
        repeated_empty_patch = self.img_emtpy_patch.repeat(self.current_image.shape[0],
                                                           self.current_image.shape[1] // self.img_emtpy_patch.shape[0],
                                                           1)
        self.current_image = torch.cat(
            [repeated_empty_patch, self.current_image, repeated_empty_patch], dim=2)

        repeated_empty_patch = self.seg_empty_patch.repeat(self.current_seg.shape[0], 1,
                                                           self.current_seg.shape[2] // self.seg_empty_patch.shape[1])
        self.current_seg = torch.cat(
            [repeated_empty_patch, self.current_seg, repeated_empty_patch], dim=1)
        repeated_empty_patch = self.seg_empty_patch.repeat(self.current_seg.shape[0],
                                                           self.current_seg.shape[1] // self.seg_empty_patch.shape[0],
                                                           1)
        self.current_seg = torch.cat(
            [repeated_empty_patch, self.current_seg, repeated_empty_patch], dim=2)
        self.seg_sizes = self.current_seg.sum(dim=(1, 2))

        # init image metadata
        self.image_id = str(self.image_id.item())
        _, self.height, self.width = self.current_image.shape
        self.captions = self.dataloader.dataset.captions_dict[self.image_id]
        self.max_row, self.max_col = (self.height - self.patch_size[0]) // self.patch_size[0], (
            self.width - self.patch_size[1]) // self.patch_size[1]
        # self.row, self.col = self.max_row // 2, self.max_col // 2
        self.row, self.col = np.random.randint(0, self.max_row + 1), np.random.randint(
            0, self.max_col + 1)

        # init planes
        self.seen_patches = torch.zeros(
            (self.max_row + 1, self.max_col + 1)).to(torch.bool)
        if self.history_type == 'maehistory':
            assert self.max_len is None and self.n_last_positions
            self.history = MAEHistory(
                self.width, self.height, self.patch_size, self.n_last_positions)
        elif self.history_type == 'maelimitedhistory':
            assert self.max_len is not None and self.n_last_positions
            self.history = MAELimitedHistory(
                self.max_len, self.width, self.height, self.patch_size, self.n_last_positions)
        elif self.history_type == 'limitedhistory':
            self.history = LimitedHistory(
                self.max_len, self.width, self.height, self.patch_size)
        elif self.history_type == 'maehistoryadj':
            self.history = MAEHistoryAdj(
                self.width, self.height, self.patch_size, self.n_last_positions)
        else:
            raise Exception('wrong history type')

        # init timelimit
        self.elapsed_steps = 0

        # init stats
        self.rews = []
        self.ratio_quantile_idx = 0
        self.ratio_quantiles_vals = [None] * len(self.ratio_quantiles)
        self.step_quantile_idx = 0
        self.step_quantiles_vals = [None] * len(self.step_quantiles)

        # init render
        self.im = None
        self.render_mask = None

        _ = self._reward_return()  # to set the first patch as seen
        initial_obs = self._get_obs()  # to add the first patch to history
        _ = self._get_render_image()  # to set the first patch as seen

        return initial_obs, {}

    def _get_patch(self, base, row, col):
        start_row, end_row = row * \
            self.patch_size[0], (row + 1) * self.patch_size[0]
        start_col, end_col = col * \
            self.patch_size[1], (col + 1) * self.patch_size[1]
        return base[..., start_row: end_row, start_col: end_col]

    def _get_curr_patch(self, base):
        return self._get_patch(base, self.row, self.col)

    def _update_history(self):
        new_patch = self._get_curr_patch(self.current_image)
        self.history.append(new_patch, self.row, self.col,
                            right=self._get_patch(
                                self.current_image, self.row, self.col+1) if self.col < self.max_col else None,
                            left=self._get_patch(
                                self.current_image, self.row, self.col-1) if self.col > 0 else None,
                            bot=self._get_patch(
                                self.current_image, self.row+1, self.col) if self.row < self.max_row else None,
                            top=self._get_patch(
                                self.current_image, self.row-1, self.col) if self.row > 0 else None,
                            )
        return self.history

    def _get_obs(self, update_history=True):
        if update_history:
            self._update_history()
        return {
            'history': self.history.get_history_dict(),
        }

    def _adj_seg_zero_helper(self):
        self._get_patch(self.current_seg, self.row+1, self.col).zero_()
        self._get_patch(self.current_seg, self.row-1, self.col).zero_()
        self._get_patch(self.current_seg, self.row, self.col+1).zero_()
        self._get_patch(self.current_seg, self.row, self.col-1).zero_()

    def _reward_seg(self):
        '''
        incompatible with whoever uses `self.seg_sizes`
        '''
        patch_seg = self._get_curr_patch(self.current_seg)
        patch_seg.zero_()
        if type(self.history) == MAEHistoryAdj:
            self._adj_seg_zero_helper()
        not_seen = self.current_seg.sum(dim=(1, 2))
        rewarded = (not_seen/self.seg_sizes) <= .9
        self.seg_sizes[rewarded] = 0
        rewarded = rewarded.to(torch.int)
        reward = rewarded.sum().item()
        return reward

    def _reward_seg_dense(self):
        '''
        reward based on the portion of the seg seen in the current patch
        '''
        patch_seg = self._get_curr_patch(self.current_seg)
        seen = patch_seg.sum(dim=(1, 2))
        if type(self.history) == MAEHistoryAdj:  # for future projects: don't rely on selection to avoid this shitshow. use masks EVERYWHERE :(
            seen += self._get_patch(self.current_seg,
                                    self.row+1, self.col).sum(dim=(1, 2))
            seen += self._get_patch(self.current_seg,
                                    self.row-1, self.col).sum(dim=(1, 2))
            seen += self._get_patch(self.current_seg,
                                    self.row, self.col+1).sum(dim=(1, 2))
            seen += self._get_patch(self.current_seg,
                                    self.row, self.col-1).sum(dim=(1, 2))
        reward = ((seen/self.seg_sizes) * 1)
        reward.nan_to_num_()
        reward = reward.sum().item()
        patch_seg.zero_()
        if type(self.history) == MAEHistoryAdj:
            self._adj_seg_zero_helper()
        return reward

    def _reward_missed(self):
        return -(self.seg_sizes != 0).sum()

    def _reward_missed_soft(self):
        '''
        reward based on the missed portion of each segment
        '''
        not_seen = self.current_seg.sum(dim=(1, 2))
        reward = -(not_seen/self.seg_sizes) * 1
        reward.nan_to_num_()
        reward = reward.sum().item()
        return reward

    def _reward_return(self):
        reward = -1/6 if self.seen_patches[self.row, self.col] else 0
        return reward

    def _reward_step(self):
        reward = -1/12 + self._reward_return()
        return reward

    def _covered_done(self):
        if self.seen_patches.all():
            return True
        else:
            return False

    def log_eoe_stats(self):
        not_seen = self.current_seg.sum(dim=(1, 2))
        not_seen_ratio = not_seen/self.seg_sizes
        mean_not_seen_ratio = not_seen_ratio.mean()
        std_not_seen_ratio = not_seen_ratio.std()
        not_touched_segs_ratio = (
            not_seen == self.seg_sizes).sum()/len(not_seen)
        std_rewards = torch.tensor(self.rews).std()

        log = {
            **{
                str(k): self.ratio_quantiles_vals[i] for i, k in enumerate(self.ratio_quantiles) if self.ratio_quantiles_vals[i] is not None
            },
            **{
                str(k): self.step_quantiles_vals[i] for i, k in enumerate(self.step_quantiles) if self.step_quantiles_vals[i] is not None
            },
            'mean_not_seen_ratio': mean_not_seen_ratio.item(),
            'std_not_seen_ratio': std_not_seen_ratio.item(),
            'not_touched_segs_ratio': not_touched_segs_ratio.item(),
            'std_rewards_ep': std_rewards.item()
        }
        return log

    def store_quantile_stats(self):
        not_seen = self.current_seg.sum(dim=(1, 2))
        min_not_seen_ratio = (not_seen/self.seg_sizes).min()
        if self.ratio_quantile_idx < len(self.ratio_quantiles) and  min_not_seen_ratio >= self.ratio_quantiles[self.ratio_quantile_idx]:
            self.ratio_quantiles_vals[self.ratio_quantile_idx] = self.elapsed_steps
            self.ratio_quantile_idx += 1
        if self.step_quantile_idx < len(self.step_quantiles) and self.elapsed_steps >= self.step_quantiles[self.step_quantile_idx]:
            self.step_quantiles_vals[self.step_quantile_idx] = min_not_seen_ratio.item()
            self.step_quantile_idx += 1
            
    def get_logs(self, new_rew, finished):
        self.rews.append(new_rew)
        self.store_quantile_stats()
        if finished:
            return self.log_eoe_stats()        
        return {}

    def step(self, action):
        if Actions(action) == Actions.UP:
            self.row = self.row - 1 if self.row > 0 else self.row
        elif Actions(action) == Actions.RIGHT:
            self.col = self.col + 1 if self.col < self.max_col else self.col
        elif Actions(action) == Actions.DOWN:
            self.row = self.row + 1 if self.row < self.max_row else self.row
        elif Actions(action) == Actions.LEFT:
            self.col = self.col - 1 if self.col > 0 else self.col
        elif Actions(action) == Actions.END:
            reward_missed = self._reward_missed_soft()
            reward_missed = np.clip(reward_missed, -10, 10)
            logs = self.get_logs(reward_missed, True)            
            return self._get_obs(update_history=False), reward_missed, True, False, {'logs':logs}
        else:
            raise ValueError("Invalid action")

        # setup return values ---
        obs = self._get_obs()
        done = self._covered_done()
        reward_seg = self._reward_seg_dense()
        # reward_return = self._reward_return()
        reward_step = -1/12
        # reward_done = 100 if done else 0
        reward = reward_seg + reward_step
        reward = np.clip(reward, -10, 10)

        truncated = False
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.time_limit:
            truncated = True
        
        logs = self.get_logs(reward, done or truncated)
        # ---
        
        self.seen_patches[self.row, self.col] = True

        return obs, reward, done, truncated, {'logs':logs}

    def _get_render_image(self):
        # row = self.row
        # col = self.col
        # image = self.current_image

        history = self.history.get_history_dict()
        row = history['curr_rel_row']
        col = history['curr_rel_col']
        image = history['history']

        if self.render_mask is None:
            self.render_mask = torch.ones(image.shape[1:])
        patch = self._get_patch(self.render_mask, row, col)
        patch[...] = self._get_patch(self.render_mask, row, col) * 0.8

        if 'kmask' in history:
            kmask = history['kmask']
            image = image + (~kmask).to(torch.int32)

        # if 'pmask' in history:
        #     pmask = history['pmask']
        #     image[2] += pmask * 0.5

        curr_indicator = torch.zeros(image.shape[1:])
        curr_patch = self._get_patch(curr_indicator, row, col)
        curr_patch[...] = +.3

        image = image * self.render_mask
        image[0] += curr_indicator
        image.clamp_(0, 1)

        image = einops.rearrange(image, 'c h w -> h w c')
        return image

    def render(self):
        display.clear_output(wait=True)
        if self.im is None:
            self.im = plt.subplots()
            self.im[1].imshow(self._get_render_image())
            display.display(self.im[0])
        else:
            self.im[1].get_children()[0].set_data(self._get_render_image())
            display.display(self.im[0])
