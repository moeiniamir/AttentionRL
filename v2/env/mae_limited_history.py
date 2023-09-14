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
        # self.pmask = torch.ones(
        #     height + 2 * patch_size[0], width + 2 * patch_size[1], dtype=torch.bool)
        # self.indices = None
        self.current_loc = None

    def append(self, patch, row, col):
        # fix the row and col to be relative to the canvas
        row += 1
        col += 1
        # set the indices (current, top, right, bottom, left)
        # self.indices = [(row, col), (row - 1, col),
        #                 (row, col + 1), (row + 1, col), (row, col - 1)]
        self.current_loc = (row, col)
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
        # self.pmask.fill_(0)
        self.canvas.fill_(0)
        iterator = iter(self.loc_history[::-1])
        # seen_indices = [self.indices[0]]
        seen_indices = []
        # adj_indices = [self.indices[1], self.indices[2],
        #                self.indices[3], self.indices[4]]
        # set the current and adjacent patches
        # self._set_patch(self.loc_to_patch[next(
        #     iterator)], self.canvas, *seen_indices[0])
        # self._set_patch(1, self.kmask, *seen_indices[0])
        # self._set_patch(1, self.kmask, *adj_indices[0])
        # self._set_patch(1, self.kmask, *adj_indices[1])
        # self._set_patch(1, self.kmask, *adj_indices[2])
        # self._set_patch(1, self.kmask, *adj_indices[3])
        # curr_len = 5
        curr_len = 0
        # set  additional patches till max_len
        while curr_len < self.max_len:
            try:
                loc = next(iterator)
            except StopIteration:
                break
            if loc not in seen_indices:
                seen_indices.append(loc)
                # if loc not in adj_indices:
                #     curr_len += 1
                curr_len += 1
            self._set_patch(self.loc_to_patch[loc], self.canvas, *loc)
            self._set_patch(1, self.kmask, *loc)
        # set the patched patches
        # for loc in adj_indices:
        #     if loc not in seen_indices:
        #         self._set_patch(1, self.pmask, *loc)

    def get_history_dict(self):
        self._fill_canvas()
        
        last_positions = torch.tensor(
            self.loc_history[-self.n_last_positions:])
        padded_mask = torch.cat([
            torch.ones(self.n_last_positions-last_positions.shape[0], dtype=torch.bool),
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
