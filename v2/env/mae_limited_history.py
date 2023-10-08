from pack_existing_segs import *
import torch
from .abstract_history import AbstractHistory


class MAELimitedHistory(AbstractHistory):
    def __init__(self, max_len, width, height, patch_size, n_last_positions):
        assert n_last_positions > 0, "n_last_positions must be positive"
        self.max_len = max_len
        self.patch_size = patch_size
        self.loc_to_patch = {}
        self.loc_history = []
        self.n_last_positions = n_last_positions

        self.canvas = torch.zeros((3, height + 2 * patch_size[0], width + 2 * patch_size[1]),
                                  dtype=torch.float16)
        self.kmask = torch.ones(
            height + 2 * patch_size[0], width + 2 * patch_size[1], dtype=torch.bool)
        self.current_loc = None

    def append(self, patch, row, col, **kwargs):
        row += 1
        col += 1
        self.current_loc = (row, col)
        self.loc_history.append((row, col))
        self.loc_to_patch[(row, col)] = patch
        # self.loc_to_patch[(row, col + 1)] = kwargs['right']#! cheat
        # self.loc_to_patch[(row, col - 1)] = kwargs['left']#! cheat
        # self.loc_to_patch[(row + 1, col)] = kwargs['top']#! cheat
        # self.loc_to_patch[(row - 1, col)] = kwargs['bot']#! cheat
        

    def _set_patch(self, p, on, row, col):
        top = self.patch_size[0] * row
        bottom = top + self.patch_size[0]
        left = self.patch_size[1] * col
        right = left + self.patch_size[1]
        on[..., top:bottom, left:right] = p

    def _fill_canvas(self):
        self.kmask.fill_(0)
        self.canvas.fill_(0)
        iterator = iter(self.loc_history[::-1])
        seen_indices = []
        curr_len = 0
        while curr_len < self.max_len:
            try:
                loc = next(iterator)
            except StopIteration:
                break
            if loc not in seen_indices:
                seen_indices.append(loc)
                curr_len += 1
            self._set_patch(self.loc_to_patch[loc], self.canvas, *loc)
            self._set_patch(1, self.kmask, *loc)
        
        # if self.loc_to_patch[(self.current_loc[0] + 1, self.current_loc[1])] is not None:
        #     self._set_patch(self.loc_to_patch[(self.current_loc[0] + 1, self.current_loc[1])],
        #                     self.canvas, self.current_loc[0] + 1, self.current_loc[1]) #! cheat
        #     self._set_patch(1, self.kmask, self.current_loc[0] + 1, self.current_loc[1])
        # if self.loc_to_patch[(self.current_loc[0] - 1, self.current_loc[1])] is not None:
        #     self._set_patch(self.loc_to_patch[(self.current_loc[0] - 1, self.current_loc[1])],
        #                     self.canvas, self.current_loc[0] - 1, self.current_loc[1]) #! cheat
        #     self._set_patch(1, self.kmask, self.current_loc[0] - 1, self.current_loc[1])
        # if self.loc_to_patch[(self.current_loc[0], self.current_loc[1] + 1)] is not None:
        #     self._set_patch(self.loc_to_patch[(self.current_loc[0], self.current_loc[1] + 1)],
        #                     self.canvas, self.current_loc[0], self.current_loc[1] + 1) #! cheat
        #     self._set_patch(1, self.kmask, self.current_loc[0], self.current_loc[1] + 1)
        # if self.loc_to_patch[(self.current_loc[0], self.current_loc[1] - 1)] is not None:
        #     self._set_patch(self.loc_to_patch[(self.current_loc[0], self.current_loc[1] - 1)],
        #                     self.canvas, self.current_loc[0], self.current_loc[1] - 1) #! cheat
        #     self._set_patch(1, self.kmask, self.current_loc[0], self.current_loc[1] - 1)

    def get_history_dict(self):
        self._fill_canvas()

        last_positions = torch.tensor(
            self.loc_history[-self.n_last_positions:])
        padded_mask = torch.cat([
            torch.ones(self.n_last_positions -
                       last_positions.shape[0], dtype=torch.bool),
            torch.zeros(last_positions.shape[0], dtype=torch.bool)
        ])
        last_positions = torch.nn.functional.pad(
            last_positions, (0, 0, self.n_last_positions - last_positions.shape[0], 0), mode='constant', value=1)

        history_dict = {
            'history': self.canvas,
            'kmask': self.kmask,
            'last_positions': last_positions,
            'padded_mask': padded_mask,
            'curr_rel_row': self.current_loc[0],
            'curr_rel_col': self.current_loc[1],
        }
        return history_dict
