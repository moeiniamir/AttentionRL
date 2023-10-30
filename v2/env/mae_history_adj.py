from pack_existing_segs import *
import torch
from .abstract_history import AbstractHistory


class MAEHistoryAdj(AbstractHistory):
    def __init__(self, width, height, patch_size, n_last_positions):
        assert n_last_positions > 0, "n_last_positions must be positive"
        self.patch_size = patch_size
        self.loc_to_patch = {}
        self.loc_history = []
        self.n_last_positions = n_last_positions

        self.canvas = torch.zeros((3, height + 2 * patch_size[0], width + 2 * patch_size[1]),
                                  dtype=torch.float16)
        self.running_canvas = torch.zeros((3, height + 2 * patch_size[0], width + 2 * patch_size[1]),
                                          dtype=torch.float16)

        self.kmask = torch.empty(
            height + 2 * patch_size[0], width + 2 * patch_size[1], dtype=torch.bool)
        self.running_kmask = torch.zeros(
            height + 2 * patch_size[0], width + 2 * patch_size[1], dtype=torch.bool)
        self.current_loc = None

    def append(self, patch, row, col, **kwargs):
        row += 1
        col += 1
        self.current_loc = (row, col)
        self.loc_history.append((row, col))
        self.loc_to_patch[(row, col)] = patch
        self.loc_to_patch[(row, col + 1)] = kwargs['right']
        self.loc_to_patch[(row, col - 1)] = kwargs['left']
        self.loc_to_patch[(row + 1, col)] = kwargs['top']
        self.loc_to_patch[(row - 1, col)] = kwargs['bot']

        # center
        self._set_patch(patch, self.running_canvas, row, col)
        self._set_patch(1, self.running_kmask, row, col)
        # top
        if kwargs['top'] is not None:
            self._set_patch(kwargs['top'], self.running_canvas, row-1, col)
            self._set_patch(1, self.running_kmask, row-1, col)
        # right
        if kwargs['right'] is not None:
            self._set_patch(kwargs['right'], self.running_canvas, row, col+1)
            self._set_patch(1, self.running_kmask, row, col+1)
        # bottom
        if kwargs['bot'] is not None:
            self._set_patch(kwargs['bot'], self.running_canvas, row+1, col)
            self._set_patch(1, self.running_kmask, row+1, col)
        # left
        if kwargs['left'] is not None:
            self._set_patch(kwargs['left'], self.running_canvas, row, col-1)
            self._set_patch(1, self.running_kmask, row, col-1)

    def _set_patch(self, p, on, row, col):
        top = self.patch_size[0] * row
        bottom = top + self.patch_size[0]
        left = self.patch_size[1] * col
        right = left + self.patch_size[1]
        on[..., top:bottom, left:right] = p

    def _fill_canvas(self):
        self.kmask = self.running_kmask
        self.canvas = self.running_canvas

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

        curr_row = self.current_loc[0]
        curr_col = self.current_loc[1]
        history_dict = {
            'history': self.canvas,
            'kmask': self.kmask,
            'running_kmask': self.running_kmask,
            'last_positions': last_positions,
            'padded_mask': padded_mask,
            'curr_rel_row': curr_row,
            'curr_rel_col': curr_col,
            'urdl': torch.tensor(
                [(curr_row - 1, curr_col), (curr_row, curr_col + 1),
                 (curr_row + 1, curr_col), (curr_row, curr_col - 1)]
            )
        }
        return history_dict
