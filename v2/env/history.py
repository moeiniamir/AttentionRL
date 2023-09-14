from pack_existing_segs import *
import torch
from .abstract_history import AbstractHistory

class History(AbstractHistory):
    def __init__(self, patch_size, max_row=None, max_col=None):
        self.patch_size = patch_size
        if max_row is None:
            self._min_row, self._min_col, self._max_row, self._max_col = None, None, None, None
            self.pos_mask = None
            self.history = None
        else:
            self._min_row, self._min_col, self._max_row, self._max_col = 0, 0, max_row, max_col
            self.pos_mask = torch.ones(
                ((max_row + 1) * patch_size[0], (max_col + 1) * patch_size[1]), dtype=torch.bool)
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
                    (torch.zeros(
                        3, self.patch_size[0] * (self._min_row - row), self.history.shape[2]), self.history),
                    dim=1)
                self.pos_mask = torch.cat(
                    (torch.ones(
                        self.patch_size[0] * (self._min_row - row), self.history.shape[2]), self.pos_mask),
                    dim=0)
                self._min_row = row
            if row > self._max_row:
                # pad history with zeros on bottom to account for row difference
                self.history = torch.cat(
                    (self.history, torch.zeros(
                        3, self.patch_size[0] * (row - self._max_row), self.history.shape[2])),
                    dim=1)
                self.pos_mask = torch.cat(
                    (self.pos_mask, torch.ones(
                        self.patch_size[0] * (row - self._max_row), self.history.shape[2])),
                    dim=0)
                self._max_row = row
            if col < self._min_col:
                # pad history with zeros on left to account for col difference
                self.history = torch.cat(
                    (torch.zeros(
                        3, self.history.shape[1], self.patch_size[1] * (self._min_col - col)), self.history),
                    dim=2)
                self.pos_mask = torch.cat(
                    (torch.ones(
                        self.history.shape[1], self.patch_size[1] * (self._min_col - col)), self.pos_mask),
                    dim=1)
                self._min_col = col
            if col > self._max_col:
                # pad history with zeros on right to account for col difference
                self.history = torch.cat(
                    (self.history, torch.zeros(
                        3, self.history.shape[1], self.patch_size[1] * (col - self._max_col))),
                    dim=2)
                self.pos_mask = torch.cat(
                    (self.pos_mask, torch.ones(
                        self.history.shape[1], self.patch_size[1] * (col - self._max_col))),
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

