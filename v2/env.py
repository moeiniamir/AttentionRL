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
from abc import ABC, abstractmethod



class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    # STAY = 4


class AbstractHistory(ABC):
    @abstractmethod
    def append(self, patch, row, col):
        pass

    @abstractmethod
    def get_history_dict(self):
        pass


class History(AbstractHistory):
    def __init__(self, patch_size, max_row=None, max_col=None):
        self.patch_size = patch_size
        if max_row is None:
            self._min_row, self._min_col, self._max_row, self._max_col = None, None, None, None
            self.pos_mask = None
            self.history = None
        else:
            self._min_row, self._min_col, self._max_row, self._max_col = 0, 0, max_row, max_col
            self.pos_mask = torch.ones(((max_row + 1) * patch_size[0], (max_col + 1) * patch_size[1]), dtype=torch.bool)
            self.history = torch.zeros((3, (max_row + 1) * patch_size[0], (max_col + 1) * patch_size[1]),
                                       dtype=torch.float16)

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

            self.history[:, top:bottom, left:right] = patch
            self.pos_mask[top:bottom, left:right] = 0
        self.curr_rel_col = col - self._min_col
        self.curr_rel_row = row - self._min_row

    def get_history_dict(self):
        return {
            'history': self.history,
            'pos_mask': self.pos_mask,
            'curr_rel_row': torch.tensor(self.curr_rel_row),
            'curr_rel_col': torch.tensor(self.curr_rel_col),
            'patch_size': torch.tensor(self.patch_size),
        }


class LimitedHistory(AbstractHistory):
    def __init__(self, max_len, max_col, max_row, patch_size):
        self.max_len = max_len
        self.max_col = max_col
        self.max_row = max_row
        self.patch_size = patch_size
        self.loc_to_patch = {}
        self.loc_history = []

        self.canvas = torch.zeros((3, (max_row + 3) * patch_size[0], (max_col + 3) * patch_size[1]),
                                  dtype=torch.float16)
        self.kmask = torch.ones((max_row + 3) * patch_size[0], (max_col + 3) * patch_size[1], dtype=torch.bool)
        self.pmask = torch.ones((max_row + 3) * patch_size[0], (max_col + 3) * patch_size[1], dtype=torch.bool)
        self.indices = None

    def append(self, patch, row, col):
        # fix the row and col to be relative to the canvas
        row += 1
        col += 1
        # set the indices (current, top, right, bottom, left)
        self.indices = [(row, col), (row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)]
        self.loc_history.append((row, col))
        self.loc_to_patch[(row, col)] = patch

    def _set_patch(self, p, on, row, col):
        top = self.patch_size[0] * row
        bottom = top + self.patch_size[0]
        left = self.patch_size[1] * col
        right = left + self.patch_size[1]
        on[..., top:bottom, left:right] = p

    def _fill_canvas(self):
        self.kmask.fill_(0)
        self.pmask.fill_(0)
        self.canvas.fill_(0)  # todo why wasn't here at first?
        iterator = iter(self.loc_history[::-1])
        seen_indices = [self.indices[0]]
        adj_indices = [self.indices[1], self.indices[2], self.indices[3], self.indices[4]]
        # set the current and adjacent patches
        self._set_patch(self.loc_to_patch[next(iterator)], self.canvas, *seen_indices[0])
        self._set_patch(1, self.kmask, *seen_indices[0])
        self._set_patch(1, self.kmask, *adj_indices[0])
        self._set_patch(1, self.kmask, *adj_indices[1])
        self._set_patch(1, self.kmask, *adj_indices[2])
        self._set_patch(1, self.kmask, *adj_indices[3])
        curr_len = 5
        # set  additional patches till max_len
        while curr_len < self.max_len:
            try:
                loc = next(iterator)
            except StopIteration:
                break
            if loc not in seen_indices:
                seen_indices.append(loc)
                if loc not in adj_indices:
                    curr_len += 1
            self._set_patch(self.loc_to_patch[loc], self.canvas, *loc)
            self._set_patch(1, self.kmask, *loc)
        # set the patched patches
        for loc in adj_indices:
            if loc not in seen_indices:
                self._set_patch(1, self.pmask, *loc)

    def get_history_dict(self):
        self._fill_canvas()
        return {
            'history': self.canvas,
            'kmask': self.kmask,
            'pmask': self.pmask,
            'curdl_indices': torch.tensor(self.indices),
            'patch_size': torch.tensor(self.patch_size),  # for compatibility with the other history
            'pos_mask': ~self.kmask,  # for compatibility with the other history
            'curr_rel_row': torch.tensor(self.indices[0][0]),  # for compatibility with the other history
            'curr_rel_col': torch.tensor(self.indices[0][1]),  # for compatibility with the other history
        }


class Environment(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, dataset, patch_size=(64, 64), max_len=None):
        self.dataloader = D.DataLoader(dataset, batch_size=1, shuffle=True)
        self.iterator = iter(self.dataloader)
        self.patch_size = patch_size
        self.max_len = max_len

        self.img_emtpy_patch = torch.zeros((patch_size[0] // 2, patch_size[1] // 2))
        self.img_emtpy_patch[::2, ::2] = 1
        self.seg_empty_patch = torch.zeros((patch_size[0] // 2, patch_size[1] // 2))

        self.observation_space = spaces.Dict({
            'center': spaces.Box(low=0, high=255, shape=(3, self.patch_size[0], self.patch_size[1]), dtype=np.float16),
        })
        self.action_space = spaces.Discrete(len(Actions))

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
        repeated_empty_patch = self.img_emtpy_patch.repeat(self.current_image.shape[0], 1,
                                                           self.current_image.shape[2] // self.img_emtpy_patch.shape[1])
        self.current_image = torch.cat([repeated_empty_patch, self.current_image, repeated_empty_patch], dim=1)
        repeated_empty_patch = self.img_emtpy_patch.repeat(self.current_image.shape[0],
                                                           self.current_image.shape[1] // self.img_emtpy_patch.shape[0],
                                                           1)
        self.current_image = torch.cat([repeated_empty_patch, self.current_image, repeated_empty_patch], dim=2)

        repeated_empty_patch = self.seg_empty_patch.repeat(self.current_seg.shape[0], 1,
                                                           self.current_seg.shape[2] // self.seg_empty_patch.shape[1])
        self.current_seg = torch.cat([repeated_empty_patch, self.current_seg, repeated_empty_patch], dim=1)
        repeated_empty_patch = self.seg_empty_patch.repeat(self.current_seg.shape[0],
                                                           self.current_seg.shape[1] // self.seg_empty_patch.shape[0],
                                                           1)
        self.current_seg = torch.cat([repeated_empty_patch, self.current_seg, repeated_empty_patch], dim=2)

        self.image_id = str(self.image_id.item())
        _, self.height, self.width = self.current_image.shape
        self.captions = self.dataloader.dataset.captions_dict[self.image_id]
        self.max_row, self.max_col = (self.height - self.patch_size[0]) // self.patch_size[0], (
                self.width - self.patch_size[1]) // self.patch_size[1]
        self.row, self.col = self.max_row // 2, self.max_col // 2

        # self.seen_patches = torch.zeros((self.max_row + 1, self.max_col + 1))
        self.seen_patches = torch.full((self.max_row + 1, self.max_col + 1), -0.5)
        self.seen_masks = torch.zeros(self.current_seg.shape[0])
        if self.max_len is None:
            self.history = History(self.patch_size, self.max_row, self.max_col)
        else:
            self.history = LimitedHistory(self.max_len, self.max_col, self.max_row, self.patch_size)

        self.im = None
        self.render_mask = None

        return self._get_obs(), {}

    def _get_patch(self):
        start_row, end_row = self.row * self.patch_size[0], (self.row + 1) * self.patch_size[0]
        start_col, end_col = self.col * self.patch_size[1], (self.col + 1) * self.patch_size[1]
        return self.current_image[:, start_row: end_row, start_col: end_col]

    def _update_history(self, new_patch):
        self.history.append(new_patch, self.row, self.col)
        return self.history

    def _get_obs(self):
        patch = self._get_patch()
        self._update_history(patch)
        return {
            'history': self.history.get_history_dict(),
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
        # row = self.row
        # col = self.col
        # image = self.current_image

        history = self.history.get_history_dict()
        row = history['curr_rel_row']
        col = history['curr_rel_col']
        image = history['history']

        if self.render_mask is None:
            self.render_mask = torch.ones(image.shape[1:])

        if 'kmask' in history and 'pmask' in history:
            pmask = history['pmask']
            kmask = history['kmask']
            image[0] += (~kmask) * 0.5
            image[2] += pmask * 0.5

        start_row, end_row = row * self.patch_size[0], (row + 1) * self.patch_size[0]
        start_col, end_col = col * self.patch_size[1], (col + 1) * self.patch_size[1]
        self.render_mask[start_row: end_row, start_col: end_col] = 0.8 * self.render_mask[start_row: end_row,
                                                                         start_col: end_col]
        image = image * self.render_mask

        image = einops.rearrange(image, 'c h w -> h w c')
        return image

    def render(self):
        if self.im is None:
            self.im = plt.imshow(self._get_render_image())
            display.display(plt.gcf())
        else:
            display.clear_output(wait=True)
            self.im.set_data(self._get_render_image())
            # self.im.axes.add_image(self.im)
            display.display(plt.gcf())
