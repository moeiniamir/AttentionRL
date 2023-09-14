from pack_existing_segs import *
import torch
from .abstract_history import AbstractHistory

class LimitedHistory(AbstractHistory):
    def __init__(self, max_len, width, height, patch_size):
        self.max_len = max_len
        self.patch_size = patch_size
        self.loc_to_patch = {}
        self.loc_history = []

        self.canvas = torch.zeros((3, height + 2 * patch_size[0], width + 2 * patch_size[1]),
                                  dtype=torch.float16)
        self.kmask = torch.ones(
            height + 2 * patch_size[0], width + 2 * patch_size[1], dtype=torch.bool)
        self.pmask = torch.ones(
            height + 2 * patch_size[0], width + 2 * patch_size[1], dtype=torch.bool)
        self.indices = None

    def append(self, patch, row, col):
        # fix the row and col to be relative to the canvas
        row += 1
        col += 1
        # set the indices (current, top, right, bottom, left)
        self.indices = [(row, col), (row - 1, col),
                        (row, col + 1), (row + 1, col), (row, col - 1)]
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
        self.canvas.fill_(0)
        iterator = iter(self.loc_history[::-1])
        seen_indices = [self.indices[0]]
        adj_indices = [self.indices[1], self.indices[2],
                       self.indices[3], self.indices[4]]
        # set the current and adjacent patches
        self._set_patch(self.loc_to_patch[next(
            iterator)], self.canvas, *seen_indices[0])
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
        history_dict = {
            'history': self.canvas,
            'kmask': self.kmask,
            'pmask': self.pmask,
            'curdl_indices': torch.tensor(self.indices),
            'curr_rel_row': torch.tensor(self.indices[0][0]),
            'curr_rel_col': torch.tensor(self.indices[0][1]),
            # for compatibility with the other history
            'patch_size': torch.tensor(self.patch_size),
            # for compatibility with the other history
            'pos_mask': ~self.kmask,  
        }
        return history_dict
