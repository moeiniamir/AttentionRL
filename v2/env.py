from pack_existing_segs import *
from enum import Enum
from IPython import display
import matplotlib.pyplot as plt
import torch
import torch.utils.data as D
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import einops

if os.environ['USER'] == 'server':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    # STAY = 4

class History:
    def __init__(self, patch_size, max_row=None, max_col=None):
        self.patch_size = patch_size
        if max_row is None:
            self._min_row, self._min_col, self._max_row, self._max_col = None, None, None, None
            self.pos_mask = None
            self.history = None
        else:
            self._min_row, self._min_col, self._max_row, self._max_col = 0, 0, max_row, max_col
            self.pos_mask = torch.ones(((max_row + 1) * patch_size[0], (max_col + 1) * patch_size[1]), dtype=torch.bool)
            self.history = torch.zeros((3, (max_row + 1) * patch_size[0], (max_col + 1) * patch_size[1]), dtype=torch.uint8)

        self.curr_rel_row, self.curr_rel_col = None, None

    def append(self, patch, row, col):
        if self._min_row is None:
            self._min_row, self._min_col, self._max_row, self._max_col = row, col, row, col
            self.history = patch
            self.pos_mask = torch.ones(patch.shape[1:])
        else:
            if row < self._min_row:
                # pad history with zeros on top to account for row difference
                self.history = torch.cat(
                    (torch.zeros(3, self.patch_size[0] * (self._min_row - row), self.history.shape[2]), self.history),
                    dim=1)
                self.pos_mask = torch.cat(
                    (torch.ones(self.patch_size[0] * (self._min_row - row), self.history.shape[2]), self.pos_mask),
                    dim=0)
                self._min_row = row
            if row > self._max_row:
                # pad history with zeros on bottom to account for row difference
                self.history = torch.cat(
                    (self.history, torch.zeros(3, self.patch_size[0] * (row - self._max_row), self.history.shape[2])),
                    dim=1)
                self.pos_mask = torch.cat(
                    (self.pos_mask, torch.ones(self.patch_size[0] * (row - self._max_row), self.history.shape[2])),
                    dim=0)
                self._max_row = row
            if col < self._min_col:
                # pad history with zeros on left to account for col difference
                self.history = torch.cat(
                    (torch.zeros(3, self.history.shape[1], self.patch_size[1] * (self._min_col - col)), self.history),
                    dim=2)
                self.pos_mask = torch.cat(
                    (torch.ones(self.history.shape[1], self.patch_size[1] * (self._min_col - col)), self.pos_mask),
                    dim=1)
                self._min_col = col
            if col > self._max_col:
                # pad history with zeros on right to account for col difference
                self.history = torch.cat(
                    (self.history, torch.zeros(3, self.history.shape[1], self.patch_size[1] * (col - self._max_col))),
                    dim=2)
                self.pos_mask = torch.cat(
                    (self.pos_mask, torch.ones(self.history.shape[1], self.patch_size[1] * (col - self._max_col))),
                    dim=1)
                self._max_col = col
            # add patch to history
            top = self.patch_size[0] * (row - self._min_row)
            bottom = top + self.patch_size[0]
            left = self.patch_size[1] * (col - self._min_col)
            right = left + self.patch_size[1]

            # print(f"{self._min_row=}, {self._min_col=}, {self._max_row=}, {self._max_col=} {top=}, {bottom=}, {left=}, {right=}")
            self.history[:, top:bottom, left:right] = patch
            self.pos_mask[top:bottom, left:right] = 0
        self.curr_rel_col = col - self._min_col
        self.curr_rel_row = row - self._min_row

class Environment(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, dataset, patch_size=(64, 64)):
        self.dataloader = D.DataLoader(dataset, batch_size=1, shuffle=True)
        self.iterator = iter(self.dataloader)
        self.patch_size = patch_size

        self.img_emtpy_patch = torch.zeros((patch_size[0]//2, patch_size[1]//2))
        self.img_emtpy_patch[::2, ::2] = 1
        self.seg_empty_patch = torch.zeros((patch_size[0]//2, patch_size[1]//2))

        self.observation_space = spaces.Dict({
            'center': spaces.Box(low=0, high=255, shape=(3, self.patch_size[0], self.patch_size[1]), dtype=np.uint8),
        })
        self.action_space = spaces.Discrete(len(Actions))

        self.im = None

        self.reset()

    def reset(self, **kwargs):
        try:
            # Samples the batch
            self.current_image, self.current_seg, self.image_id = next(self.iterator)
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.iterator = iter(self.dataloader)
            self.current_image, self.current_seg, self.image_id = next(self.iterator)

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
        # here's an example:
        # empty_patch = torch.zeros(8, 8)
        # empty_patch[::2, ::2] = 1
        # image = torch.rand(3, 64, 64)
        # repeated_empty_patch = empty_patch.repeat(image.shape[0], 1, image.shape[2] // empty_patch.shape[1])
        # image = torch.cat([repeated_empty_patch, image, repeated_empty_patch], dim=1)
        # repeated_empty_patch = empty_patch.repeat(image.shape[0], image.shape[1] // empty_patch.shape[0], 1)
        # image = torch.cat([repeated_empty_patch, image, repeated_empty_patch], dim=2)
        repeated_empty_patch = self.img_emtpy_patch.repeat(self.current_image.shape[0], 1,
                                                           self.current_image.shape[2] // self.img_emtpy_patch.shape[1])
        self.current_image = torch.cat([repeated_empty_patch, self.current_image, repeated_empty_patch], dim=1)
        repeated_empty_patch = self.img_emtpy_patch.repeat(self.current_image.shape[0],
                                                           self.current_image.shape[1] // self.img_emtpy_patch.shape[0], 1)
        self.current_image = torch.cat([repeated_empty_patch, self.current_image, repeated_empty_patch], dim=2)

        repeated_empty_patch = self.seg_empty_patch.repeat(self.current_seg.shape[0], 1,
                                                           self.current_seg.shape[2] // self.seg_empty_patch.shape[1])
        self.current_seg = torch.cat([repeated_empty_patch, self.current_seg, repeated_empty_patch], dim=1)
        repeated_empty_patch = self.seg_empty_patch.repeat(self.current_seg.shape[0],
                                                           self.current_seg.shape[1] // self.seg_empty_patch.shape[0], 1)
        self.current_seg = torch.cat([repeated_empty_patch, self.current_seg, repeated_empty_patch], dim=2)


        self.image_id = str(self.image_id.item())
        _, self.height, self.width = self.current_image.shape
        self.captions = self.dataloader.dataset.captions_dict[self.image_id]
        self.max_row, self.max_col = (self.height - self.patch_size[0]) // self.patch_size[0], (
                self.width - self.patch_size[1]) // self.patch_size[1]
        self.row, self.col = self.max_row // 2, self.max_col // 2

        # self.seen_patches = torch.zeros((self.max_row + 1, self.max_col + 1))
        self.seen_patches = torch.full((self.max_row + 1, self.max_col + 1), -0.5)
        self.seen_masks = torch.zeros(self.current_seg.shape[0]).to(device)
        # self.history = History(self.patch_size)
        self.history = History(self.patch_size, self.max_row, self.max_col)

        return self._get_obs(), {}

    def _get_patch(self):
        start_row, end_row = self.row * self.patch_size[0], (self.row + 1) * self.patch_size[0]
        start_col, end_col = self.col * self.patch_size[1], (self.col + 1) * self.patch_size[1]
        return self.current_image[:, start_row: end_row, start_col: end_col]

    def _get_history(self, new_patch):
        self.history.append(new_patch, self.row, self.col)
        return self.history

    def _get_obs(self):
        patch = self._get_patch()
        history = self._get_history(patch)
        return {
            # 'center': patch,
            # 'surrounding': self._get_surrounding(),
            'history': {
                'history': history.history,
                'pos_mask': history.pos_mask,
                'curr_rel_row': torch.tensor(history.curr_rel_row),
                'curr_rel_col': torch.tensor(history.curr_rel_col),
                'patch_size': torch.tensor(history.patch_size),
            }
        }


    def _reward_seg(self):
        start_row, end_row = self.row * self.patch_size[0], (self.row + 1) * self.patch_size[0]
        start_col, end_col = self.col * self.patch_size[1], (self.col + 1) * self.patch_size[1]
        patch_seg = self.current_seg[:, start_row: end_row, start_col: end_col]
        patch_seg = patch_seg.sum(dim=(1, 2)) / self.current_seg.sum(dim=(1, 2))
        seen_threshold = 0.7
        seg_reward = (patch_seg > seen_threshold)
        seg_reward = seg_reward * (1 - self.seen_masks)
        self.seen_masks += seg_reward
        seg_reward = seg_reward.sum().item()

        return seg_reward

    def _reward_return(self):
        reward = -self.seen_patches[self.row, self.col]
        self.seen_patches[self.row, self.col] = 1
        return reward

    def _covered_done(self):
        if self.seen_patches.sum() == (self.max_row + 1) * (self.max_col + 1):
            return True
        else:
            return False

    def step(self, action):
        if Actions(action) == Actions.UP:
            self.row = self.row - 1 if self.row > 0 else self.row
        elif Actions(action) == Actions.RIGHT:
            self.col = self.col + 1 if self.col < self.max_col else self.col
        elif Actions(action) == Actions.DOWN:
            self.row = self.row + 1 if self.row < self.max_row else self.row
        elif Actions(action) == Actions.LEFT:
            self.col = self.col - 1 if self.col > 0 else self.col
        else:
            raise ValueError("Invalid action")

        obs = self._get_obs()
        done = self._covered_done()
        # reward_seg = self._reward_seg()
        reward_return = self._reward_return()
        reward_done = 100 if done else 0
        reward = reward_return + reward_done
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def _get_render_image(self):
        start_row, end_row = self.row * self.patch_size[0], (self.row + 1) * self.patch_size[0]
        start_col, end_col = self.col * self.patch_size[1], (self.col + 1) * self.patch_size[1]
        image = einops.rearrange(self.trail_image, 'c h w -> h w c')
        image[start_row: end_row, start_col: end_col] = 0.8 * image[start_row: end_row, start_col: end_col]
        return image

    def render(self):
        if self.im is None:
            self.trail_image = self.current_image.clone()
            self.im = plt.imshow(self._get_render_image())
            display.display(plt.gcf())
        else:
            display.clear_output(wait=True)
            self.im.set_data(self._get_render_image())
            self.im.axes.add_image(self.im)
            display.display(plt.gcf())