#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:30:29 2024

@author: bingg
"""

import numpy as np


class Utilities:
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)

    def update_neighbor_lists(self, force_update=False):
        if (self.step % self.neighbor == 0) | force_update:
            positions = np.asarray(self.atoms_positions, dtype=float)
            deltas = positions[:, None, :] - positions[None, :, :]
            box_size = getattr(self, "box_size", None)
            if box_size is not None:
                box_size = np.asarray(box_size, dtype=float)
                deltas -= box_size * np.round(deltas / box_size)
            matrix = np.linalg.norm(deltas, axis=-1) <= self.cut_off
            np.fill_diagonal(matrix, False)
            neighbor_lists = []
            for cpt, array in enumerate(matrix):
                list = np.where(array)[0].tolist()
                list = [ele for ele in list if ele > cpt]
                neighbor_lists.append(list)
            self.neighbor_lists = neighbor_lists
